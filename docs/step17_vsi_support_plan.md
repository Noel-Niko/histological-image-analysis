# Step 17: VSI Image Format Support

## Context

A collaborator has an Olympus/Evident microscope scanner that produces `.vsi` files. VSI is proprietary, not supported by OpenSlide or any maintained Python library. The only reliable reader is Bio-Formats (Java-based, OME consortium). Rather than adding a Java dependency to the Python package, we add a conversion step using the Bio-Formats CLI.

### VSI Format Background

VSI (Virtual Slide Image) is a two-part format:
- The `.vsi` file contains metadata only
- Actual pixel data lives in a companion directory named `_[filename]_/` containing `.ets` files
- **Both parts are required** — copying just the `.vsi` file without the companion directory will fail

VSI files are **multi-resolution pyramids** containing multiple "series":
- Macro/overview images (low-res thumbnails)
- Label images (slide label scan)
- Full-resolution pyramid levels (40x, 20x, 10x, etc.)
- The series order varies by scanner model and acquisition settings — **series 0 is NOT reliably the main image**

### Resolution Mismatch Problem

The trained models expect macro-scale images where brain region structure is visible:

| Training Data | Pixel Size | Typical Dimensions |
|---------------|------------|--------------------|
| Mouse CCFv3 | ~10 µm/pixel | ~800 × 1,140 |
| Human Allen | ~40 µm/pixel | ~1,700 × 2,100 |
| Human BigBrain | 200 µm/pixel | ~770 × 605 |

A typical Olympus scanner produces:

| Magnification | Pixel Size | Typical Dimensions |
|---------------|------------|--------------------|
| 40x | ~0.25 µm/pixel | 80,000–120,000+ per side |
| 20x | ~0.5 µm/pixel | 40,000–60,000+ per side |

At full scanner resolution, the model sees individual cells — not brain regions. **Downsampling by 20–80x is required** to match the training data's spatial scale. This downsampling must happen during conversion, not at inference time, because:
1. Loading a 100K × 100K image into memory requires ~30 GB (numpy array alone)
2. The sliding-window `logit_sum` array at full resolution with 1,328 labels would require ~12 TB
3. Even "tiled loading" cannot fix the logit accumulator memory requirement

The correct approach is to extract the appropriate pyramid level during conversion so the output TIFF matches the scale the model expects.

## Approach: `bfconvert` Conversion with Resolution Selection

Three new `make` commands: download bftools (one-time), inspect VSI metadata, and batch-convert at a target resolution. Then the existing annotation pipeline handles the converted files with **no changes to inference.py**.

### User Workflow

```bash
make download-bioformats-cli                                        # One-time: downloads bftools (~40MB, needs Java 11+)
make inspect-vsi IMAGES=/path/to/olympus/slides/                    # Shows series, dimensions, pixel sizes per file
make convert-vsi IMAGES=/path/to/olympus/slides/ RESOLUTION=10      # Convert at ~10 µm/pixel target
make annotate-mouse IMAGES=/path/to/olympus/slides/                 # Annotates the converted TIFFs
```

## Steps

### Step 1: Create `scripts/inspect_vsi.py`
- For each `.vsi` file in the input directory, run `showinf -nopix file.vsi` (bundled with bftools)
- Parse and display a summary table: series index, dimensions (W × H), pixel size (µm), channel count, description/type
- Validate `.ets` companion directory exists; print clear error if missing (explain the two-part format)
- Validate `showinf` is available (check `tools/bftools/showinf`), print install instructions if missing
- Validate Java is available (`java -version`), print instructions if missing

### Step 2: Create `scripts/convert_vsi.py`
- Scan input directory for `.vsi` files
- For each `.vsi` file:
  1. **Validate `.ets` companion directory** exists (`_[stem]_/` sibling directory containing `.ets` files). Produce a clear error explaining the two-part format if missing.
  2. **Run `showinf -nopix`** to enumerate all series with their dimensions and physical pixel sizes
  3. **Select the pyramid level** closest to the target resolution (default: 10 µm/pixel, configurable via `--resolution` / `RESOLUTION=` Make arg). Selection logic:
     - Filter out series that are clearly label/macro images (very small dimensions, e.g., < 1000 px on both sides)
     - From remaining series, pick the one whose pixel size is closest to the target
     - If no physical pixel size metadata is available, pick the series whose dimensions are closest to 1000–3000 px on the long side (matching training data scale)
     - If the closest available level is still >2x finer than the target, extract it and downsample with Pillow (`Image.LANCZOS`)
  4. **Run `bfconvert -compression LZW -series N input.vsi output.tiff`** with the selected series N
  5. **Validate output**: check the output TIFF exists, is non-zero bytes, and has reasonable dimensions (not 1×1 or 0×0)
  6. **Log the conversion details**: selected series index, input dimensions, output dimensions, pixel size, for reproducibility
- Place converted `.tiff` alongside originals with suffix `-converted.tiff` (skip if already exists)
- Report results: converted count, skipped count, any errors
- No BigTIFF flag needed — at target resolution, output files will be <100 MB each

### Step 3: Add Makefile targets
- `download-bioformats-cli`: curl + unzip bftools to `tools/bftools/`
- `inspect-vsi`: runs `scripts/inspect_vsi.py` with `IMAGES=` argument
- `convert-vsi`: runs `scripts/convert_vsi.py` with `IMAGES=` and `RESOLUTION=` arguments

### Step 4: Update documentation
- README: add "Working with Olympus VSI files" subsection under Supported Image Formats
  - Explain the two-part `.vsi` + `_name_/` format — users must copy both
  - Explain resolution selection and the default 10 µm/pixel target
  - Mention disk space expectations (~1–5 GB per file at full res, <100 MB at target resolution)
  - Mention Java 11+ prerequisite
- `docs/step16_cli_annotator_context.md`: add VSI context
- Add `tools/` to `.gitignore` (bftools is downloaded on demand)

### Step 5: Tests
- Create `tests/test_inspect_vsi.py`: test showinf output parsing, missing companion directory error, filename handling
- Create `tests/test_convert_vsi.py`: test directory scanning, series selection logic, skip logic, missing-bfconvert error, missing companion directory error, output validation, filename construction
- No real `.vsi` files needed — mock subprocess calls to `showinf`/`bfconvert`
- Test the resolution selection algorithm with various mock `showinf` outputs (with/without pixel size metadata, various series layouts)

### Step 6: Manual validation with real VSI slides
> **This step is a manual checkpoint — do not skip.**

Once Steps 1–5 are implemented:
1. Obtain 3–5 real `.vsi` files from the collaborator (ensure both `.vsi` and `_name_/` companion directories are included)
2. Run `make inspect-vsi IMAGES=/path/` to verify metadata parsing
3. Run `make convert-vsi IMAGES=/path/ RESOLUTION=10` to convert at ~10 µm/pixel
4. Visually inspect converted TIFFs — do they look like reasonable brain section images at the expected scale?
5. Run `make annotate-mouse IMAGES=/path/` (or `annotate-human-*` depending on species)
6. **Visually assess annotation quality**:
   - Are brain region boundaries roughly correct?
   - Does the model identify major structures (cortex, hippocampus, cerebellum, etc.)?
   - Are there large areas of obvious misclassification?
   - Compare against the atlas annotations the model was trained on — is the segmentation in the right ballpark?
7. Try different `RESOLUTION=` values (5, 10, 20) to see which produces the best annotations
8. **Document findings**: record the staining protocol, scanner settings, selected resolution, and annotation quality assessment

**Decision gate**: If annotation quality is acceptable for the collaborator's use case, the VSI support feature is complete. If quality is poor, proceed to the domain adaptation assessment (see below).

### Step 7 (Conditional): Domain Gap Assessment & Fine-Tuning Scope

> **Only proceed here if Step 6 reveals unacceptable annotation quality.**

This step defines the scope of a **separate project** — fine-tuning the model on real scanner images would be a new paper/study, not part of this VSI format support feature.

Assessment checklist:
- [ ] What staining protocol do the VSI slides use? (Nissl → smaller gap; H&E or other → larger gap)
- [ ] What species/brain regions are represented?
- [ ] How many slides are available? (Fine-tuning typically needs 10+ annotated images minimum)
- [ ] Is ground truth available? (Manual annotations, atlas registration, or expert review needed)
- [ ] What is the quality of available slides? (Artifacts, inconsistent staining, partial sections)

If fine-tuning is pursued, the approach would be:
- **Transfer learning** from the existing checkpoint (not training from scratch)
- Start with the frozen-backbone approach (only train the UperNet head) since data is limited
- Even 10–20 annotated slides can significantly improve domain-specific performance
- This constitutes a new study: "Domain adaptation of atlas-trained segmentation models to whole-slide scanner images"

**This is out of scope for Step 17** — track separately as a future paper/project.

## Removed from Original Plan

### ~~Step 3 (original): Large-image tiled loading in `inference.py`~~
**Removed.** By converting at the correct resolution during Step 2, output TIFFs are 1,000–3,000 px — well within PIL's and the existing inference pipeline's capacity. No changes to `inference.py` are needed. If a future use case requires full-resolution inference on whole-slide images, that would be a fundamentally different architecture (patch-based WSI classification) and a separate project.

### ~~Step 4 (original): Add `tifffile` explicit dependency~~
**Removed.** Not needed since we no longer do tiled TIFF loading. Standard Pillow TIFF support handles the converted files.

## Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `scripts/inspect_vsi.py` | Display VSI series/resolution metadata |
| Create | `scripts/convert_vsi.py` | VSI → TIFF batch conversion with resolution selection |
| Create | `tests/test_inspect_vsi.py` | Inspection script tests |
| Create | `tests/test_convert_vsi.py` | Conversion script tests |
| Edit | `Makefile` | `download-bioformats-cli`, `inspect-vsi`, `convert-vsi` targets |
| Edit | `README.md` | VSI workflow docs |
| Edit | `docs/step16_cli_annotator_context.md` | VSI context |
| Edit | `.gitignore` | Add `tools/` |

**Not modified**: `inference.py`, `pyproject.toml` — the conversion handles the format/resolution gap so the existing pipeline works unchanged.

## Verification
1. `make download-bioformats-cli` downloads bftools to `tools/bftools/`
2. `make inspect-vsi IMAGES=/path/` lists series, dimensions, pixel sizes for each `.vsi` file
3. `make convert-vsi IMAGES=/path/ RESOLUTION=10` selects correct pyramid level, converts to `.tiff`, reports details
4. Missing `.ets` companion directory produces clear error explaining the two-part format
5. Existing `make annotate-*` processes converted TIFFs without changes
6. `make test` — all existing + new tests pass
7. Missing Java or bftools produces clear error message with install instructions
8. Manual validation with real slides confirms annotation quality (Step 6)

## Prerequisites
- Java 11+ on the user's machine
- Internet access for bftools download (~40MB)
- Real `.vsi` files for manual validation (Step 6) — both `.vsi` and companion `_name_/` directory

## Progress
- [x] Step 1: Create `scripts/inspect_vsi.py` — done, plus shared `src/.../vsi.py` module
- [x] Step 2: Create `scripts/convert_vsi.py` — done, uses vsi.py for shared logic
- [x] Step 3: Add Makefile targets — `download-bioformats-cli`, `inspect-vsi`, `convert-vsi`
- [x] Step 4: Update documentation — README VSI section, step16 context doc, .gitignore
- [x] Step 5: Tests — 51 new tests (28 inspect + 23 convert), all 397 total pass
- [ ] Step 6: Manual validation with real VSI slides (CHECKPOINT — do not skip)
- [ ] Step 7: Domain gap assessment (only if Step 6 shows poor quality)
