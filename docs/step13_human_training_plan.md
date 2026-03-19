# Step 13: Human Brain Segmentation Model — Training Plan

**Date:** 2026-03-16
**Depends on:** `docs/data_download_plan_human.md`, `docs/human_data_search_results.md`, `docs/progress.md`
**Status:** STEP 6 IN PROGRESS — Both tracks trained. Allen 50ep complete, BigBrain 200ep complete. Next: Allen 200ep + depth-3 coarser-class experiment.
**Updated:** 2026-03-19 — Allen 50ep results: 25.8% CC mIoU (597 classes). BigBrain 200ep: 60.8% CC mIoU (10 classes). User chose: 200ep fine-grained + depth-3 (~92 classes) coarser model.

---

## Context

### What We Have (Data)

| Dataset | Images | Annotations | Ontology | Coverage |
|---------|--------|-------------|----------|----------|
| Allen Human SectionImage SVGs | 14,565 Nissl JPEGs (6 donors) | 4,463 SVGs with structure_id polygons | Graph 10 (1,839 structures) | 524+ unique struct IDs observed |
| Allen Developing Human Atlas (21 pcw) | 169 atlas plate JPEGs | 156/169 SVGs with annotations | Graph 16 (3,317 structures) | Not counted yet |
| BigBrain 9-class classified volume | 3D volume 696×770×605 (200μm) | 9 tissue classes | Custom (9 labels) | 35.1% non-zero |
| BigBrain histological volume 8-bit | 3D volume 696×770×605 (200μm) | — (intensity only) | — | 99.4% non-zero |
| BigBrain layer segmentation | 13 2D sections (raw + classified) | 6 cortical layers per section | Custom (6 layers) | 13 sections only |
| Julich-Brain v2.9 (BigBrain space) | 3D volume 357×463×411 (~320μm) | 122 cytoarchitectonic regions | Julich parcellation | 2.0% voxel coverage |
| Cortical layers (BigBrain space) | 3D volume 269×463×384 (~340μm) | 6 cortical layers (L1–L6) | Layer labels | 38.4% coverage |
| Isocortex segmentation | 3D volume 303×385×348 (400μm) | Binary: cortex vs subcortical | 2 labels | 25.8% coverage |

### What We Learned from Mouse Training

| Finding | Implication for Human |
|---------|----------------------|
| Run 9 (final): 74.8% mIoU (CC), 79.1% mIoU (SW), 200 epochs, unfrozen last 4 blocks | Same architecture + unfreezing strategy + 200 epochs |
| 657/1,328 classes had zero training pixels | Must use present-class-only mapping |
| Class pruning had no effect (Run 8a: +0.2%) | Don't waste effort pruning — just use present_mapping |
| Weighted loss regressed (Run 6: −1.6%) | Standard CE loss is fine |
| Extended augmentation regressed (Run 7: −6.5%) | Use baseline augmentation only (flip, rot±15°, jitter) |
| TTA catastrophically failed (−24%) | Don't use TTA |
| Pixel count ↔ IoU correlation: r=0.794 | More training data per class = better results |
| Model is orientation-specific (axial 3.2%, sagittal 0.5%) | Train on the orientation PhD researcher will use |
| Batch=1 crashes (PSP BatchNorm) | Must use batch≥2, `dataloader_drop_last=True` |
| 1,016 training samples → 74.8% mIoU (200ep) | Human will have ~3,188 training images (4 donors) — more data |

---

## Design Decisions (Finalized 2026-03-16)

### Q1: Primary data source → Parallel comparison: Track A (Allen) vs Track B (BigBrain)

**Decision:** Run both data sources as independent parallel tracks. Compare results at each step. Make data-driven choice based on observed performance.

**Approach:** Two separate notebooks with identical model architecture and training config (DINOv2-Large + UperNet, Run 9 recipe). The only variable is the data source. This isolates the effect of each dataset.

**Why not combine in a single training run:** The two sources have incompatible ontologies (Graph 10 structure IDs vs BigBrain tissue classes), different stains (Nissl vs Merker), and different annotation density (sparse SVGs vs dense 3D volume). Combining them would require a cross-ontology mapping that doesn't exist.

**Comparison criteria at each step:**
- Dataset statistics (samples, classes, annotation density, pixel distribution)
- Training loss curves (convergence speed)
- mIoU and per-class IoU (each track evaluated within its own class set)
- Training time and compute cost (3,188 2D images vs ~605 NIfTI slices — significant dataset size difference)
- Qualitative: which segmentations look more useful for PhD use case?
- After results: decide whether to pursue one track, both, or combine via sequential pre-training

### Q2: Class count → Full mapping with all 524+ observed structures

**Decision:** Use `build_full_mapping()` on Graph 10 (produces 1,839 classes — one per structure), then `build_present_mapping()` to filter down to only the ~524 structure IDs that actually appear in the training SVGs.

**Rationale:** Mouse findings show class pruning had negligible effect (+0.2%). Training all observed structures costs nothing in accuracy and preserves maximum granularity. `present_mapping` already handles absent classes cleanly. Analogous to mouse Run 5 approach (1,328 total → 503 valid). The ~524 count comes from scanning SVG polygons for unique structure IDs — `build_full_mapping()` alone would produce 1,839 classes (most absent from annotations).

### Q3: Split strategy → By donor (4 train / 1 val / 1 test)

**Decision:** Split by donor to test cross-individual generalization.

**Rationale:**
- Adjacent sections from the same donor share tissue architecture, cutting artifacts, and staining — random split would cause data leakage (same autocorrelation problem as mouse spatial split: 50.7% vs 88.0%)
- PhD collaborator will provide tissue from new, unseen donors — donor-split directly tests this use case
- 3,188 training images is 3× more than the mouse training set

**Specific split:**
| Role | Donor | Annotated Images |
|------|-------|-----------------|
| Train | H0351.2002 | 894 |
| Train | H0351.2001 | 830 |
| Train | H0351.1012 | 767 |
| Train | H0351.1009 | 697 |
| **Val** | **H0351.1016** | **641** |
| **Test** | **H0351.1015** | **634** |
| **Total** | | **4,463** |

Split ratio: 71% train / 14% val / 15% test.

### Q4: Sparse annotations → Masked loss (ignore_index=255)

**Decision:** Unlabeled pixels set to 255 in the mask, ignored by CE loss.

**Rationale:** Treating unlabeled pixels as background would be factually wrong — Allen SVGs annotate ~12 structures per image on average (mean=11.4, range 0-40 from 50-file smoke test), covering 17-33% of pixels. The remaining 67-83% of pixels are unlabeled brain tissue. Teaching the model that unlabeled brain tissue is "background" would poison the training signal. The existing `CrossEntropyLoss(ignore_index=255)` already supports this with no code changes to the loss function.

**Mechanism:** The rasterizer initializes the mask to 255 (ignore). For each polygon with a valid `structure_id`, the corresponding pixels are filled with the mapped class ID. This means: (a) annotated structure pixels get their class label, (b) everything else — true background outside the tissue, unlabeled brain tissue, "Non-sampled" regions — is 255 and ignored by loss. This avoids the ambiguity of distinguishing "actual background" from "unlabeled tissue" — both are simply excluded from training signal.

**Tradeoff:** Reduced effective training signal per image (only annotated pixels contribute to gradients). The model never learns to predict "background" (class 0), so at inference time on PhD slides, background pixels will be predicted as some structure. This is acceptable if PhD slides are pre-cropped to tissue regions. If background prediction is needed later, a simple intensity threshold or tissue detection step can be added as post-processing.

### Q5: Initialization → DINOv2 pretrained only (Phase 1), mouse checkpoint as Phase 2 ablation

**Decision:** Start from standard DINOv2 pretrained weights for Run 1.

**Rationale:**
- Clean baseline with no confounding variables from mouse-specific features
- Backbone unfreezing (last 4 blocks, differential LR) already handles domain adaptation — the single biggest improvement (+8.5%) in mouse experiments
- Mouse and human brains have different structural organization at fine granularity (524 human structures ≠ 1,328 mouse structures)
- If Run 1 underperforms, mouse init gives a clear comparison experiment
- Mouse findings doc recommends this as Phase 2: "Compare random init vs mouse init (both trained for 200 epochs)"

### Q6: Orientation → Standardize all slides to horizontal before feeding to model

**Decision:** All input slides rotated to consistent horizontal orientation.

**User input:** Slides start as VSI (converted to JPEG/PNG), may be horizontal or vertical but can be rotated to all be horizontal.

**Rationale:** Mouse model is strictly orientation-specific (coronal 68.9%, axial 3.2%, sagittal 0.5%). Allen Human SectionImage data has a natural orientation. Standardizing PhD slides to match training data orientation is the correct preprocessing step. No multi-orientation training needed.

---

## Architecture & Training Configuration (Finalized)

### Shared config — identical for both tracks (from mouse Run 5 recipe)

- **Backbone:** DINOv2-Large (304M params), initialized from standard pretrained weights
- **Decoder:** UperNet with PSP pooling
- **Input resolution:** 518×518 random crops (training), center crops (eval)
- **Optimizer:** AdamW, differential LR (backbone 1e-5, head 1e-4)
- **Backbone unfreezing:** Last 4 blocks (20-23) + gradient checkpointing (`use_reentrant=False`)
- **Batch:** 2 per GPU (effective 4 with `gradient_accumulation_steps=2`)
- **Augmentation:** Baseline only (horizontal flip 50%, rotation ±15°, color jitter ±0.2)
- **Loss:** CE with `ignore_index=255`
- **Epochs:** 200 (mouse val loss still declining at 100)
- **Constraints:** `batch≥2`, `dataloader_drop_last=True`, `ddp_find_unused_parameters=True`

### Track A: Allen Human SectionImage SVGs

| Property | Value |
|----------|-------|
| **Data source** | 4,463 annotated SVGs across 6 donors |
| **Stain** | Nissl (matches PhD tissue) |
| **Format** | 2D image + SVG pairs (lazy loading) |
| **Resolution** | ~1μm native (27K-48K pixels per axis, varies by donor/section), ~4μm at downsample=4 |
| **Ontology** | Graph 10 (1,839 structures, 524+ observed in SVGs) |
| **Classes** | `build_full_mapping()` → 1,839 → `build_present_mapping()` → ~524 present |
| **Annotation density** | ~12 structure polygons per image (mean=11.4, range 0-40 from 50-file sample); 17-33% pixel coverage at 512×512; ALL groups (Non-sampled, Spaces, Macrodissection) have valid structure_ids — no group filtering needed |
| **Loss behavior** | Masked — unlabeled pixels → 255 (ignored by CE) |
| **Split** | By donor: 4 train (3,188) / 1 val (641) / 1 test (634) |
| **Dataset class** | New `AllenHumanDataset` (2D image + SVG rasterization) |
| **Notebook** | `finetune_human_allen.ipynb` |

### Track B: BigBrain 3D Volume Slicing

| Property | Value |
|----------|-------|
| **Data source** | BigBrain histological volume (696×770×605, 200μm) |
| **Stain** | Merker (different from PhD Nissl tissue) |
| **Format** | 3D NIfTI volume slicing (like mouse CCFv3Slicer) |
| **Resolution** | 200μm isotropic |
| **Annotation options** | 9-class tissue classification (35.1% coverage, dense) — primary |
| | Julich-Brain v2.9 (122 regions, 2% coverage) — supplementary/separate run |
| **Classes** | 9 tissue classes (primary) or 122 Julich regions (supplementary) |
| **Annotation density** | Dense (every voxel labeled in 9-class volume) |
| **Loss behavior** | Standard CE (dense annotations, no masking needed) |
| **Split** | Interleaved (every 10th slice → val, like mouse) |
| **Dataset class** | New `BigBrainSlicer` (NIfTI loading, reuses `BrainSegmentationDataset` pattern) |
| **Notebook** | `finetune_human_bigbrain.ipynb` |

### Track comparison — key differences

| Dimension | Track A (Allen) | Track B (BigBrain) |
|-----------|----------------|-------------------|
| Stain match to PhD | Nissl = match | Merker = mismatch |
| Donors / subjects | 6 individuals | 1 brain |
| Annotation granularity | 524+ brain structures | 9 tissue classes (or 122 Julich) |
| Annotation density | Sparse per image | Dense per voxel |
| Training samples | ~3,188 (4 donors) | ~605 coronal slices (or ~2,000 multi-axis) |
| Split strategy | By donor (cross-individual) | Interleaved (spatial) |
| Effective training signal | Low (masked loss, few labeled pixels/image) | High (all voxels contribute) |

**Note on comparability:** Direct mIoU comparison between tracks is apples-to-oranges (different class counts and granularity). Comparison will focus on: (a) convergence behavior, (b) qualitative segmentation quality on held-out data, (c) which model produces more useful results for the PhD use case (identifying structures in novel tissue).

---

## Implementation Steps — Parallel Tracks

Steps are organized to build shared infrastructure first, then track-specific components, then compare.

### Step 1: Extend OntologyMapper for human (shared) — DONE (2026-03-17)

- **File:** `src/histological_image_analysis/ontology.py`
- **Changes:**
  - The existing `OntologyMapper.__init__()` already loads any `structure_graph_N.json` format — **Graph 10 loads with no code changes** (same `{"msg": [root_node]}` format). Root node: id=4005, name="brain". Depth-1 children: gray matter (4006, 1,424 descendants), white matter (9218, 336), sulci & spaces (9352, 78).
  - `build_full_mapping()` produces 1,839 contiguous class IDs (1..1839). `build_present_mapping()` filters correctly.
  - Added `build_bigbrain_9class_mapping()` → identity mapping {0:0, 1:1, ..., 9:9}
  - Added `BIGBRAIN_9CLASS_NAMES` dict: 0=Background, 1=Gray Matter, 2=White Matter, 3=CSF, 4=Meninges, 5=Blood Vessels, 6=Bone/Skull, 7=Muscle, 8=Artifact, 9=Other/Unknown
  - `build_coarse_mapping()` on Graph 10 correctly maps all structures to 0 (mouse coarse ancestors 567/343/512/1009/73 are absent from Graph 10)
  - For Julich-Brain 122 regions: label mapping from NIfTI volume metadata (deferred to Phase 2)
- **Tests:** `tests/test_ontology.py` — **17 new tests added (13 Graph 10 + 4 BigBrain), 228 total, all pass**

### Step 2A: Extend SVG rasterizer for human `<polygon>` elements (Track A — REQUIRED CODE CHANGE)

**This is NOT a "verify" task.** Mouse and human SVGs use fundamentally different element types:
- **Mouse:** `<path d="M3318.444,3094.128 c-59.808,76.077..." structure_id="...">` — bezier curves parsed via `svgpathtools.parse_path()`
- **Human:** `<polygon points="18432,8800,18432,8768,..." structure_id="...">` — comma-separated coordinate pairs, no bezier conversion needed

The existing `SVGRasterizer._parse_paths()` (`svg_rasterizer.py:124-132`) calls `findall(".//path")` and reads the `d` attribute. **It will find zero elements in any human SVG and return an empty mask.**

- **File:** `src/histological_image_analysis/svg_rasterizer.py`
- **Changes:**
  1. **Add `_parse_polygons()` method:** Search for `<polygon>` elements with `structure_id` attributes. Parse `points="x1,y1,x2,y2,..."` into `(x, y)` tuple pairs (simple string splitting — no bezier conversion). Return same `list[tuple[int, list[tuple[float, float]]]]` format as `_parse_paths()`.
  2. **Update `rasterize()` to call both:** Try `_parse_paths()` (mouse `<path>` elements) and `_parse_polygons()` (human `<polygon>` elements). Combine results. This keeps backward compatibility with mouse SVGs.
  3. **~~Filter by `graphic_group_label`~~** — **CORRECTED (2026-03-17):** Verified on 20 real SVGs: the "Default" group is always EMPTY (0 polygons). All annotations are in "Non-sampled" (brain structures like gyri, nuclei) and "Spaces" (white matter tracts, sulci, ventricles). Both have valid Graph 10 `structure_id` values. "Macrodissection" and "Macro Biological Replicate" also have valid structure IDs (coarser boundary polygons).
     - **Decision:** Include ALL `<polygon>` elements with non-empty `structure_id`. No group filtering needed. Document order handles layering (later polygons overwrite earlier). Polygons with `structure_id=""` are skipped by the existing `if not sid_str` check.
  4. **Handle empty `structure_id=""`:** Skip elements with empty or missing structure_id (already handled by existing `if not sid_str` check).
  5. **Handle variable SVG dimensions:** Human SVGs range from 27,184×34,240 to 39,808×48,384 (verified on 5-file sample across donors). The existing rasterizer reads `width`/`height` from the SVG root element and resizes to `target_width`/`target_height` — this handles the variability correctly.
  6. **Coordinate alignment:** SVG polygon coordinates are in native resolution (27K-48K pixels). Downloaded images at `downsample=4` are ~6K-12K pixels. The existing rasterizer handles this: it rasterizes at SVG native dimensions, then resizes to target dimensions via `Image.resize(NEAREST)`. The caller passes the actual image dimensions as `target_width`/`target_height`.
  7. **Initialize mask to 255 (not 0):** For human SVGs, the rasterizer should initialize the canvas to 255 (`ignore_index`) instead of 0 (background). Only pixels inside annotated polygons get structure IDs. This implements the masked loss strategy from Q4.
- **Tests:** `tests/test_svg_rasterizer.py` — **DONE (2026-03-17): 15 new tests, 25 total, all pass**
  - `TestParsePolygons` (5 tests): parse `<polygon>` elements, structure IDs, empty-sid skip, point values
  - `TestRasterizeHumanSVG` (5 tests): shape, structure IDs, overlap order, background, dtype
  - `TestRasterizeWithSparseAnnotations` (3 tests): sparse=True → 255 unlabeled, annotated correct, default=0
  - `TestMixedSVGBackwardCompat` (2 tests): mouse SVGs still work after changes

### Step 2A-validate: SVG rasterization smoke test (Track A) — DONE (2026-03-17)

Ran validation on 50 proportionally-sampled SVGs across all 6 donors.

**Results:**
| Metric | Value | Pass? |
|--------|-------|-------|
| Non-empty masks | 48/50 (96.0%) | PASS (>90%) |
| All structure IDs in Graph 10 | 247/247 (100%) | PASS |
| Polygons per image | min=0, max=40, mean=11.4 | Higher than expected |
| Structures per image | min=0, max=39, mean=11.4 | Higher than expected |
| Annotated pixel % (512×512) | 16.8%-32.8% (5 rasterized) | Reasonable |
| SVG dimensions | width 18,368-45,632, height 15,808-50,048 | Wider range than plan |
| Total unique structure IDs (50 files) | 247 | On track for ~524 at full dataset |

**Per-donor breakdown (50-file sample):**
| Donor | Files in sample | Unique structures |
|-------|----------------|-------------------|
| H0351.1009 | 8 | 54 |
| H0351.1012 | 9 | 51 |
| H0351.1015 | 7 | 74 |
| H0351.1016 | 7 | 77 |
| H0351.2001 | 9 | 104 |
| H0351.2002 | 10 | 80 |

**Key corrections to plan:**
1. Annotation density is **much higher** than initially estimated (11.4 structures/image, not "2-4")
2. 17-33% pixel coverage per image, not "very sparse" — effective training signal is reasonable
3. "Default" graphic_group is always EMPTY — all annotations are in "Non-sampled" (gyri, nuclei) + "Spaces" (white matter, sulci)
4. SVG dimensions range is wider: 18K-46K × 16K-50K (not "27K-48K")
5. 2/50 files (4%) had zero polygons — these are empty annotation SVGs that exist on disk but contain no structure annotations. The dataset loader should filter these out.

**2 empty files identified:** `H0351.1009_19372_102078034.svg`, `H0351.1012_9395_102284069.svg`

### Step 2B: Create BigBrainSlicer (Track B) — DONE (2026-03-17)

- **File:** `src/histological_image_analysis/bigbrain_slicer.py` (new)
- **Dependency:** Added `nibabel>=5.0` to `pyproject.toml`. The existing codebase uses `pynrrd` for NRRD files — `nibabel` is the new dependency for NIfTI support.
- **Changes:**
  - Analogous to `CCFv3Slicer` but loads NIfTI (`.nii.gz`) via `nibabel.load()` instead of NRRD
  - `load_volumes()` — loads BigBrain histological volume + 9-class annotation volume from NIfTI paths
  - `from_arrays(image, annotation)` — factory for testing without nibabel I/O
  - `get_slice(index, axis=0)` — returns (image, mask) 2D pair from 3D volume
  - `iter_slices(split, mapping, split_strategy="interleaved", multi_axis=False, gap=0)` — same interface as `CCFv3Slicer` plus `gap` param
  - `_normalize_image()` — static method, same logic as CCFv3Slicer (uint8 passthrough, float32 percentile clip, uint16 scale)
  - `_remap_mask()` — static method, LUT-based label remapping (no OntologyMapper dependency — BigBrain uses identity or caller-provided mapping)
  - `_get_valid_indices(axis)` — filters slices by MIN_BRAIN_FRACTION threshold
  - `get_split_indices(split_strategy, gap)` — interleaved (default) or spatial split with optional gap exclusion
  - **Gap-based exclusion:** `gap=N` removes train slices within ±N of val/test slices. At 200μm, `gap=2` creates a 1mm buffer to mitigate spatial leakage between adjacent coronal slices. Val/test indices are unaffected.
  - Volume dimensions: 696×770×605 → up to 605 coronal slices of 696×770
  - **Duck-typed interface:** Both `CCFv3Slicer` and `BigBrainSlicer` implement `iter_slices(split, mapping, split_strategy=..., multi_axis=...)`. The `BrainSegmentationDataset` type hint will be relaxed in Step 3B. No formal Protocol needed — Python duck typing is sufficient.
- **Tests:** `tests/test_bigbrain_slicer.py` — **31 new tests, 274 total, all pass**
  - `TestLoadVolumes` (4): from_arrays, shape mismatch, not-loaded error, num_slices
  - `TestNormalization` (3): uint8 passthrough, uint16 scaling, float32 percentile clip
  - `TestGetSlice` (8): coronal/axial/sagittal shapes, default axis, invalid axis, out-of-range, dtype, label preservation
  - `TestValidIndices` (2): background skip, valid count
  - `TestInterleavedSplit` (5): no overlap, coverage, distribution, spatial also works, invalid strategy
  - `TestGapExclusion` (3): removes adjacent, zero is default, reduces train count
  - `TestIterSlices` (6): yields tuples, remaps labels, dtypes, multi-axis increases train, multi-axis val coronal-only

### Step 3A: Create AllenHumanDataset (Track A) — DONE (2026-03-17)

- **File:** `src/histological_image_analysis/dataset.py` (new class + function)
- **Changes:**
  - New `AllenHumanDataset(Dataset)` class:
    1. Takes `image_svg_pairs: list[tuple[Path, Path]]` + `rasterizer` + `mapping`
    2. **Lazy loading** — loads RGB image + rasterizes SVG on each `__getitem__()` call
    3. Rasterizes SVG with `sparse=True` → unlabeled pixels are 255 (ignore_index)
    4. **LUT-based remapping:** Builds lookup table in `__init__()` mapping structure IDs → class IDs while preserving 255. Avoids OntologyMapper dependency at runtime — mapping is pre-computed by caller.
    5. **Mask padding uses 255** (not 0) — padded regions are treated as unlabeled, not background
    6. **Augmentation fill uses 255** — rotation fill for mask is 255 (ignore), preventing the model from learning that rotated borders are annotated background
    7. RGB normalization via `_normalize_rgb()` — transposes (H,W,3)→(3,H,W) then applies ImageNet mean/std. Does NOT replicate grayscale.
    8. Returns `{"pixel_values": (3,518,518), "labels": (518,518)}`
  - New `split_by_donor(pairs, train_donors, val_donors, test_donors)` function: takes `(image_path, svg_path, donor_id)` triples, returns `dict[str, list[tuple[Path, Path]]]` split by donor
  - **Test fixture:** `tests/fixtures/sample_human_image.jpg` — 200×100 RGB JPEG matching `sample_human.svg` dimensions
- **Tests:** `tests/test_allen_human_dataset.py` — **17 new tests, 294 total, all pass**
  - `TestAllenHumanDatasetInit` (2): length, empty pairs
  - `TestAllenHumanGetItem` (5): keys, shapes, dtypes
  - `TestAllenHumanSparseLabels` (2): ignore_index=255 present, class IDs (not raw structure IDs)
  - `TestAllenHumanRGBNormalization` (2): distinct channels, reasonable range
  - `TestAllenHumanAugmentation` (3): stochastic, deterministic, preserves ignore_index
  - `TestSplitByDonor` (3): correct counts, no overlap, returns Path tuples

### Step 3B: Adapt BrainSegmentationDataset for BigBrain (Track B) — DONE (2026-03-17)

- **File:** `src/histological_image_analysis/dataset.py`
- **Changes:**
  - **Type hint relaxed:** `slicer: CCFv3Slicer` → `slicer: Any` with docstring documenting duck-typed interface (`iter_slices(split, mapping, split_strategy=..., multi_axis=...) → Iterator[tuple[ndarray, ndarray]]`)
  - Removed `from histological_image_analysis.ccfv3_slicer import CCFv3Slicer` import (no longer needed)
  - Module docstring updated to reflect both volume slicers and image+SVG pairs
  - Existing `BrainSegmentationDataset` works with `BigBrainSlicer` unchanged — pre-loads all slices, applies padding/cropping/augmentation on the fly
  - Gap-based spatial leakage mitigation already implemented in `BigBrainSlicer.get_split_indices(gap=2)` (Step 2B) — caller passes `gap=2` when creating the slicer's split indices
- **Tests:** `tests/test_dataset.py` — **3 new tests (TestBigBrainSlicerCompat), 294 total, all pass**
  - Creates `BrainSegmentationDataset` with `BigBrainSlicer`, verifies shapes/dtypes/label ranges

### Step 4A: Create Allen training notebook (Track A) — DONE (2026-03-17)

- **File:** `notebooks/finetune_human_allen.ipynb`
- **Structure:** 9 cells (markdown header + 8 code cells), same pattern as mouse Run 9
  - Cell 0 (markdown): Track description, key differences from mouse
  - Cell 1: Install wheel + restartPython
  - Cell 2: Configuration — Graph 10 paths, donor split, MLflow experiment, 200 epochs
  - Cell 3: Download DINOv2-Large from JFrog Artifactory mirror
  - Cell 4: Build Allen datasets — load metadata JSON, filter annotated, build (image,svg,donor) triples, split_by_donor, scan training SVGs for structure IDs, build_full_mapping → build_present_mapping, create AllenHumanDataset + SVGRasterizer
  - Cell 5: Create model (num_labels from present_mapping) + freeze blocks 0-19 + gradient checkpointing + forward pass validation
  - Cell 6: Train — differential LR (backbone 1e-5, head 1e-4), AdamW, linear warmup, 200 epochs
  - Cell 7: Evaluate — center-crop (Trainer.evaluate) + sliding window (RGB `normalize_tile_rgb`, limited to 50 val images for speed). Per-class IoU. Logs to MLflow.
  - Cell 8: Save model + present_mapping.json + image processor + log to MLflow
- **Key differences from mouse notebook:**
  - RGB `normalize_tile_rgb()` for sliding window (transposes H,W,3 → 3,H,W, not grayscale replication)
  - Sliding window eval limited to 50 images (full val=641 would be very slow with lazy loading)
  - Saves `present_mapping.json` alongside model for later inference
  - Sliding window excludes ignore_index=255 pixels from metrics

### Step 4B: Create BigBrain training notebook (Track B) — DONE (2026-03-17)

- **File:** `notebooks/finetune_human_bigbrain.ipynb`
- **Structure:** 9 cells (markdown header + 8 code cells), same pattern as mouse Run 9
  - Cell 0 (markdown): Track description, key differences from mouse
  - Cell 1: Install wheel + restartPython
  - Cell 2: Configuration — BigBrain NIfTI paths, 10 classes, gap=2, MLflow experiment, 200 epochs
  - Cell 3: Download DINOv2-Large from JFrog Artifactory mirror
  - Cell 4: Build BigBrain datasets — BigBrainSlicer(histology, annotation), load_volumes(), build_bigbrain_9class_mapping(), BrainSegmentationDataset with interleaved split
  - Cell 5: Create model (num_labels=10) + freeze blocks 0-19 + gradient checkpointing + forward pass
  - Cell 6: Train — same recipe as Track A
  - Cell 7: Evaluate — center-crop + sliding window (grayscale `normalize_tile`, same as mouse). Per-class IoU for all 10 classes. All val slices (smaller dataset than Track A).
  - Cell 8: Save model + image processor + log to MLflow
- **Key differences from Track A:**
  - Grayscale normalization (same as mouse — BigBrain is single-channel)
  - Uses BrainSegmentationDataset (pre-loads all slices) not AllenHumanDataset (lazy)
  - Dense annotations — no ignore_index handling needed
  - gap=2 in split for spatial leakage mitigation
  - Full val set for sliding window (fewer slices than Track A's 641 images)

### Step 5: Deploy both notebooks

- **Makefile:** Add `deploy-notebook-human-allen` and `deploy-notebook-human-bigbrain` targets
- **Deploy:** `make deploy-notebook-human-allen deploy-notebook-human-bigbrain deploy-wheel`
- **Run both on Databricks** (can run simultaneously on separate clusters)

### Step 6: Compare and decide

- **Comparison table:** mIoU, accuracy, convergence speed, qualitative segmentation quality
- **Decision criteria:**
  - If Track A (Allen) clearly wins: proceed with Allen for final human model
  - If Track B (BigBrain) clearly wins: proceed with BigBrain, accept stain mismatch risk
  - If both have strengths: consider sequential training (pre-train on BigBrain → fine-tune on Allen)
  - If both underperform: investigate combined or alternative approaches
- **Phase 2 experiments** (after decision):
  - Mouse checkpoint initialization ablation
  - Winning track with Julich-Brain 122 regions (if BigBrain won) or coarse Allen mapping (if Allen won)
  - Cross-track pre-training (BigBrain → Allen fine-tune)

---

## Risk Assessment

| Risk | Track | Likelihood | Impact | Mitigation |
|------|-------|-----------|--------|------------|
| Sparse SVG annotations produce low mIoU | A (Allen) | HIGH | HIGH | Masked loss; BigBrain track as fallback |
| BigBrain stain mismatch hurts PhD generalization | B (BigBrain) | MEDIUM | HIGH | Allen track as fallback; stain noted in comparison |
| Human images too large for memory (27K-48K native, variable per donor) | A (Allen) | MEDIUM | MEDIUM | Downsample + random crop + lazy loading |
| BigBrain single brain overfits | B (BigBrain) | MEDIUM | MEDIUM | Only 1 subject — can't split by donor |
| Cross-donor generalization is poor | A (Allen) | MEDIUM | MEDIUM | Donor-split tests this explicitly |
| 9-class tissue model is too coarse for PhD use | B (BigBrain) | HIGH | LOW | Julich-Brain 122 regions as supplementary run |
| Neither track produces useful results | Both | LOW | HIGH | Sequential pre-training (BigBrain → Allen) or combined ontology |

---

## What We Do NOT Plan to Use in Initial Tracks (and why)

| Data source | Reason to exclude | Future potential |
|-------------|-------------------|-----------------|
| Developing Human Atlas (21 pcw) | Different ontology (Graph 16 ≠ Graph 10), developmental brain ≠ adult brain | Separate developmental brain model |
| Julich-Brain v2.9 (122 regions) | Only 2% voxel coverage — too sparse for primary Track B; different granularity than 9-class | Supplementary Track B run after 9-class baseline |
| Cortical layers (BigBrain space) | Different task (layer classification ≠ structure segmentation) | Separate cortical layer model (6 layers, 38.4% coverage) |
| Mouse Run 5 checkpoint | Want clean DINOv2 baseline for both tracks first | Phase 2 ablation on the winning track |

All downloaded data is on Databricks and available for follow-up experiments.

---

## Timeline Dependencies

```
Mouse Run 9 (200 epochs) ─── soft dependency (informs expectations, does NOT block) ───┐
                                                                                       ↓
All design decisions FINALIZED ──────────→ Step 1:  Extend OntologyMapper (shared)     (review Run 9 when available)
                                          → Step 2A: Extend SVG rasterizer for <polygon> (Track A — CODE CHANGE)
                                          → Step 2B: Create BigBrainSlicer (Track B)
                                          → Step 2A-validate: Smoke test SVG rasterization (50 samples)
                                          → Step 3A: Create AllenHumanDataset (Track A)
                                          → Step 3B: Adapt BrainSegmentationDataset (Track B)
                                          → Step 4A: Create finetune_human_allen.ipynb
                                          → Step 4B: Create finetune_human_bigbrain.ipynb
                                          → Step 5:  Deploy both + run on Databricks
                                          → Step 6:  Compare results → decide winning track
                                          ↓
                               Phase 2:   → Mouse checkpoint init ablation on winner
                                          → Cross-track pre-training (BigBrain → Allen) if useful
                                          → Julich-Brain 122 regions (if BigBrain wins)
```

---

## Resume Instructions

### Implementation Status (2026-03-17)

| Step | Status | Tests added | Notes |
|------|--------|-------------|-------|
| 0: Fix stale Run 5 refs | DONE | — | Lines 28, 37, 47 updated to Run 9 |
| 1: OntologyMapper | DONE | 17 (228 total) | Graph 10 works with no code changes. Added `build_bigbrain_9class_mapping()` + `BIGBRAIN_9CLASS_NAMES` |
| 2A: SVG rasterizer | DONE | 15 (243 total) | Added `_parse_polygons()`, `sparse` param. Mouse backward compat preserved |
| 2A-validate: Smoke test | DONE | — | 48/50 non-empty, 247 unique structure IDs, 11.4 structs/image mean |
| 2B: BigBrainSlicer | DONE | 31 (274 total) | New file `bigbrain_slicer.py` + nibabel dep. Gap-based exclusion for spatial leakage. |
| 3A: AllenHumanDataset | DONE | 17 (294 total) | Lazy load, LUT remap, RGB normalize, sparse mask=255, donor split |
| 3B: Adapt BrainSegDataset | DONE | 3 (294 total) | Type hint relaxed to `Any`, BigBrainSlicer compat verified |
| 4A: Allen notebook | DONE | — (notebook) | 9 cells, RGB sliding window, saves present_mapping.json |
| 4B: BigBrain notebook | DONE | — (notebook) | 9 cells, grayscale sliding window, gap=2, 10 classes |
| 5: Deploy both | DONE | — | Wheel + both notebooks deployed to Databricks (2026-03-17) |
| 5.1: Pre-cache fix | DONE | 7 (301 total) | Added `cache_dir` + `build_cache()` to AllenHumanDataset. 10.5x speedup. |
| 6: Compare | IN PROGRESS | — | BigBrain 200ep done (60.8% mIoU). Allen 50ep done (25.8% mIoU). Next: Allen 200ep + depth-3 |
| 6.1: Allen 200ep notebook | DONE | — | Updated `NUM_EPOCHS` 50→200, cleared stale outputs |
| 6.2: Allen depth-3 notebook | DONE | — | New `finetune_human_allen_depth3.ipynb`, depth-3 mapping (~92 classes), separate cache |

### Key corrections discovered during implementation
1. **"Default" graphic_group is always EMPTY.** All annotations are in "Non-sampled" (brain structures) + "Spaces" (white matter, sulci). No group filtering needed — include ALL polygons with non-empty `structure_id`.
2. **Annotation density is higher than estimated:** ~12 structures/image (not "2-4"), 17-33% pixel coverage (not "very sparse").
3. **SVG dimension range is wider:** 18K-46K × 16K-50K pixels.
4. **~4% of SVGs are empty** (have SVG file but zero annotation polygons). Dataset loader must filter these.
5. **BigBrain 9 classes identified:** Background, Gray Matter, White Matter, CSF, Meninges, Blood Vessels, Bone/Skull, Muscle, Artifact, Other/Unknown.

### Bugs found during first Databricks run (2026-03-17)

**Status: ALL 3 FIXED. Wheel + both notebooks redeployed (2026-03-17).**

1. **FIXED — Allen notebook `KeyError: 'donor_id'`**: Metadata JSON uses `_donor` (underscore prefix, added by download script), not `donor_id`. Changed `entry["donor_id"]` → `entry["_donor"]` in Cell 3.

2. **FIXED — BigBrain `IndexError: arrays used as indices must be of integer type`**: NIfTI annotation volume loaded by nibabel has float dtype. `_remap_mask()` uses `lut[mask]` which requires integer indices. Fixed by casting to `np.int64` in both `load_volumes()` and `from_arrays()` in `bigbrain_slicer.py`.

3. **FIXED — Allen sliding window eval OOM**: Allen images at downsample=4 are ~6K-12K pixels per axis. The sliding window eval allocates `logit_sum = np.zeros((num_labels, H, W), dtype=float32)`. With ~524 classes and 8560x6796 pixels: 524 * 8560 * 6796 * 4 bytes ≈ **122 GB per image**. Mouse slices were only 800x1140, making this tractable. Fix: added `resize_for_sliding_window()` helper that resizes images to `SW_MAX_DIM=1024` before tiling (logit_sum ≈ 2.1 GB, manageable). Also added CPU guard for `torch.amp.autocast` in both notebooks.

4. **FIXED — BigBrain sliding window CPU guard**: Added `use_cuda = device.type == "cuda"` check in BigBrain notebook's `sliding_window_predict()` to avoid crash on CPU.

### Second Databricks run results (2026-03-18)

**Track B (BigBrain) — COMPLETED:** 60.8% mIoU (CC), 61.3% mIoU (SW), 90% accuracy. 200 epochs in 74 min. Strong result for 10-class tissue segmentation. Convergence plateaued at ~epoch 60-80.

**Track A (Allen) — STOPPED at epoch 5.4/200 after 10 hours.** Training speed: 0.13 it/s. Estimated 14 days for 200 epochs. mIoU at epoch 5: 1.8% (expected — 597 classes with sparse annotations need many epochs). Loss was dropping (8.76 → 5.18), confirming the model was learning.

**Root cause:** `AllenHumanDataset` lazy-loads 6K-12K pixel JPEGs + rasterizes SVGs on every `__getitem__()` call. This is ~7.7s per training step vs ~0.3s for pre-loaded BigBrain slices.

### Fix: Pre-cache + reduced epochs (2026-03-17)

1. **Added `cache_dir` parameter to `AllenHumanDataset`** — when set, loads pre-resized `.npz` files instead of raw JPEGs + SVG rasterization. Added `build_cache()` static method to create the cache.
2. **Pre-cache cell in Allen notebook** — resizes all 4,463 images to max 1024px, rasterizes SVGs, remaps labels, saves as `.npz` files. One-time cost (~15-20 min).
3. **Reduced epochs from 200 → 50** — BigBrain plateaued at ~60 epochs; Allen has 3x more data per epoch.
4. **Expected training time:** ~12 hours for 50 epochs (vs 14 days before).
5. **Tests:** 301 pass (7 new cache tests).

### Third Databricks run: Allen 50 epochs with pre-cache (2026-03-19)

**Track A (Allen) — 50 epochs COMPLETE:**
- Cache build: 4,463 files in 468 min (SVG rasterization at full resolution was the bottleneck — much slower than the estimated 15-20 min)
- Training speed: 1.37 it/s (10.5x faster than uncached 0.13 it/s)
- Training time: 8.1 hours (29,169s) for 50 epochs / 39,850 steps
- CC mIoU: **25.8%**, CC accuracy: 63.8%
- SW mIoU: **21.0%** (50 images, resized to 1024px), SW accuracy: 75.9%
- 269/597 classes have valid IoU (45%)
- SW mIoU < CC mIoU (-4.9%) — resizing to 1024px loses fine-grained detail needed for 597 classes
- Loss still declining at epoch 50 (train 0.90, val 2.49) — model hadn't fully converged
- Top classes: medial parabrachial nucleus 100%, dentate nucleus 99.5%, pontine nuclei 99.0%
- Bottom classes: cerebellum lobules near zero (I-II: 0.01%, VIIAf: 0.1%)

**Track B (BigBrain) — already complete (previous run):**
- CC mIoU: **60.8%**, CC accuracy: 90.0%
- SW mIoU: **61.3%**, SW accuracy: 93.7%

**Comparison (Step 6 — initial):**

| Metric | Track A (Allen, 597 classes) | Track B (BigBrain, 10 classes) |
|--------|------------------------------|-------------------------------|
| CC mIoU | 25.8% | 60.8% |
| SW mIoU | 21.0% | 61.3% |
| CC accuracy | 63.8% | 90.0% |
| Training time | 8.1 hrs (50ep) | 74 min (200ep) |
| Classes with valid IoU | 269/597 (45%) | 10/10 (100%) |
| Convergence | Still improving | Plateaued at ~ep60 |

Direct mIoU comparison is apples-to-oranges (597 vs 10 classes). Key observations:
1. Allen model achieves high IoU on deep brain structures (thalamus, brainstem nuclei) — these have more consistent appearance across donors
2. Allen model struggles with cortical structures and cerebellum lobules — high inter-donor variability and fewer training examples
3. BigBrain model achieves strong tissue-level segmentation — useful as a coarse tissue type classifier
4. Allen training hadn't converged — more epochs would help

### User decision: more epochs + coarser classes (2026-03-19)

User chose **options 1+2** (confirmed via iMessage):
1. **More epochs** — Allen 597-class model to 200 total epochs (fresh run, /tmp checkpoint from 50ep is gone).
2. **Coarser classes** — Train a separate Allen model with **depth-3 mapping** (~92 total classes, ~50-80 after present filter). Depth-3 groups structures into brain regions: cerebral cortex, thalamus, cerebellum, pons, hypothalamus, midbrain tegmentum, etc.

**Implementation (approved 2026-03-19):**
1. Updated Allen notebook `NUM_EPOCHS` from 50 → 200. Cleared stale 50ep outputs.
2. Created `notebooks/finetune_human_allen_depth3.ipynb` — same structure as fine-grained notebook, but uses `build_depth_mapping(depth=3)` → `build_present_mapping()`. Separate cache dir `/tmp/allen_cache_depth3` (different mask labels). Separate output dir `/dbfs/FileStore/allen_brain_data/models/human-allen-depth3`.
3. Depth-3 mapping on Graph 10: 92 total classes (91 brain regions + background). After `build_present_mapping()` filter, expect ~50-80 classes (depending on which of the 596 observed structure IDs have depth-3 ancestors).
4. Both notebooks use 200 epochs, same Run 9 recipe, same donor split, same pre-cache infrastructure.
5. **Compare all three:** Allen 597-class (200ep) vs Allen depth-3 (~50-80 classes, 200ep) vs BigBrain 10-class (200ep).

**Cache note:** Depth-3 notebook requires its own cache (different mask labels). Cache build is ~468 min per notebook. Both need rebuilding if cluster was terminated (/tmp is ephemeral).

**Cluster requirements:** Single-GPU (L40S 48 GB recommended). Can run both Allen notebooks sequentially on the same cluster (share model weights + images, separate caches).

### Key files
- `docs/step13_human_training_plan.md` (this file — authoritative plan)
- `docs/data_download_plan_human.md` (data inventory, download/upload status)
- `docs/human_data_search_results.md` (search findings)
- `docs/mouse_model_findings.md` (mouse lessons applied to human decisions)
- `src/histological_image_analysis/ontology.py` (updated: BigBrain 9-class mapping)
- `src/histological_image_analysis/svg_rasterizer.py` (updated: `_parse_polygons()` + `sparse` param)
- `src/histological_image_analysis/bigbrain_slicer.py` (new: NIfTI volume slicer with gap-based exclusion)
- `src/histological_image_analysis/dataset.py` (updated: `AllenHumanDataset` class, `split_by_donor()`, relaxed type hint, `cache_dir` + `build_cache()`)
- `src/histological_image_analysis/training.py` (unchanged — `create_model()`, `get_training_args()`, `make_compute_metrics()`)
- `notebooks/finetune_human_allen.ipynb` (Track A: 597-class, 200 epochs, pre-cached)
- `notebooks/finetune_human_allen_depth3.ipynb` (Track A-depth3: ~50-80 classes, 200 epochs, separate cache)
- `notebooks/finetune_human_bigbrain.ipynb` (Track B: 10-class, 200 epochs)
- `tests/test_ontology.py` (228 tests)
- `tests/test_svg_rasterizer.py` (25 tests)
- `tests/test_bigbrain_slicer.py` (31 tests — new)
- `tests/test_dataset.py` (updated: +3 BigBrainSlicer compat tests)
- `tests/test_allen_human_dataset.py` (24 tests — 17 original + 7 cache tests)
- `tests/fixtures/sample_human.svg` (test fixture)
- `tests/fixtures/sample_human_image.jpg` (test fixture — 200×100 RGB)

### Data location (Databricks)
All at `/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/`
- Track A: `human_atlas/svgs/` (4,463 SVGs), `ontology/structure_graph_10.json`
- Track B: `bigbrain/classified_volume/`, `bigbrain/histological_volume/`, `bigbrain/siibra/`

### PhD slide workflow
VSI → JPEG/PNG conversion → rotate to horizontal → feed to winning model

### Implementation order
~~Steps 1 → 2A + 2B → 2A-validate → 3A+3B → 4A+4B → 5~~ → 6
