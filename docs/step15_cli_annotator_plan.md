# Step 15: CLI Annotator — Terminal-Only Brain Region Annotation

## Goal

Allow a PhD researcher to annotate mouse or human brain tissue slides from the terminal with two commands:

```bash
make download-models          # 1. Download models locally (one-time)
make annotate-mouse IMAGES=/path    # 2. Annotate images
```

Output: each input file gets a sibling file `{stem}-annotated-{YYYYMMDDTHHMMSS}.png` in the same directory, showing the original image with a color overlay of detected brain regions and a legend.

No IDE, no Databricks CLI, no notebooks — just `make` and a terminal.

---

## Current State

- **Models exist** on Databricks DBFS (mouse: `final-200ep`, 1,328 classes, 79.1% mIoU; human: `depth-3`, 44 classes, 65.5% mIoU)
- **Inference exists** in `scripts/run_inference.py` — produces masks + matplotlib side-by-side panels to a separate output dir
- **Models have `id2label`** in `config.json` — class index → human-readable region name (e.g., "Caudoputamen", "Cerebral cortex")
- **`huggingface-hub`** is already a dependency

### Gaps

1. Models are only downloadable via Databricks CLI (requires corporate auth) — no public download path
2. Output goes to a separate `results/` dir with `_mask.png` / `_visualization.png` naming — not `{name}-annotated-{timestamp}` alongside originals
3. Visualization is a 3-panel matplotlib figure, not a single annotated overlay image
4. No species selection (mouse vs. human)
5. No `make` targets for the end-user workflow

---

## Design Decisions

### Model Hosting: HuggingFace Hub

Upload trained models to a HuggingFace Hub repository (e.g., `{your-username}/histological-brain-segmentation`). Rationale:
- `huggingface-hub` is already a dependency — `snapshot_download()` works out of the box
- No auth needed for public repos
- Standard practice for distributing ML models
- Avoids requiring Databricks CLI for end users

**Repo structure** (single HF repo, two subdirectories):
```
{hf_repo_id}/
├── mouse/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors
└── human/
    ├── config.json
    ├── preprocessor_config.json
    └── model.safetensors
```

The HF repo ID will be configurable via `.env` (`HF_REPO_ID=your-username/histological-brain-segmentation`).

### Annotated Output Format

A single image per input file:
- Original image as background
- Semi-transparent color overlay (alpha ~0.4) showing segmented brain regions
- Each region gets a distinct color from a perceptually uniform colormap (`tab20` / `nipy_spectral`)
- Legend panel on the right side listing the top 15 detected regions by pixel area
- Region boundaries drawn as thin contour lines for clarity

File naming: `{stem}-annotated-{YYYYMMDDTHHMMSS}.png` placed in the **same directory** as the input file.

### Species Selection

- `--species mouse` (default) → loads mouse model (1,328 classes)
- `--species human` → loads human model (44 classes, depth-3)
- Model path auto-resolved: `models/mouse/` or `models/human/`

### CLI Entry Points

Two approaches (both implemented):
1. **Makefile targets** — simplest for terminal-only users
2. **Python scripts** — for users who want more control via flags

---

## Implementation Plan

### Step 1: Create `scripts/upload_to_hf.py` (one-time utility)

Uploads locally-downloaded models from `models/` to HuggingFace Hub. The developer (you) runs this once after downloading models from Databricks.

```python
# Usage:
python scripts/upload_to_hf.py --repo-id your-username/histological-brain-segmentation
```

- Reads mouse model from `models/dinov2-upernet-final/` (or custom path)
- Reads human model from `models/human-depth3/` (or custom path)
- Uploads to HF Hub under `mouse/` and `human/` subdirectories
- Requires `HF_TOKEN` env var for write access (one-time)

### Step 2: Create `scripts/download_models.py`

Downloads models from HuggingFace Hub to local `models/` directory.

```python
# Usage:
python scripts/download_models.py                     # Download all models
python scripts/download_models.py --species mouse      # Mouse only
python scripts/download_models.py --species human      # Human only
python scripts/download_models.py --repo-id custom/repo # Custom HF repo
```

Implementation:
- Uses `huggingface_hub.snapshot_download(repo_id, allow_patterns="mouse/*", local_dir="models/")`
- Verifies download: checks `config.json`, `preprocessor_config.json`, and `model.safetensors` exist
- Prints model metadata (num_labels, id2label count) after verification
- Handles errors: network failure, incomplete download, disk space

### Step 3: Create `scripts/annotate.py`

The core annotation script. Produces annotated overlay images.

```python
# Usage:
python scripts/annotate.py /path/to/slides/                      # Directory of images
python scripts/annotate.py /path/to/slide.jpg                     # Single file
python scripts/annotate.py /path/to/slides/ --species human       # Human model
python scripts/annotate.py /path/to/slides/ --sliding-window      # Full-res tiled inference
python scripts/annotate.py /path/to/slides/ --cpu                 # Force CPU
python scripts/annotate.py /path/to/slides/ --model ./models/custom  # Custom model path
```

Implementation:
- **Input handling**: Accept file path or directory path as positional arg. Auto-detect images by extension (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`).
- **Model loading**: Reuse `load_model()` logic from `run_inference.py`. Auto-resolve model path from `--species` flag → `models/{species}/`.
- **Inference**: Reuse `run_inference()` and `run_sliding_window_inference()` from existing code. Import directly from `run_inference.py` or extract shared functions into `src/`.
- **Overlay generation** (new function `create_annotated_overlay`):
  1. Run inference → get `prediction` array (H, W) of class IDs
  2. Resize prediction to original image dimensions (nearest-neighbor)
  3. Map class IDs → colors using a colormap
  4. Composite: `annotated = original * (1 - alpha) + color_overlay * alpha` where `alpha = 0.4` for non-background pixels
  5. Draw contour boundaries between adjacent regions (1px black lines)
  6. Add legend panel on the right: top 15 regions sorted by pixel area, with color swatches + names from `model.config.id2label`
  7. Save as PNG
- **Output naming**: `{stem}-annotated-{datetime.now().strftime('%Y%m%dT%H%M%S')}.png` in same directory as input.
- **Progress**: tqdm progress bar for batch processing.

### Step 4: Extract shared inference code into `src/`

To avoid code duplication between `run_inference.py` and `annotate.py`, extract shared functions into the package:

Create `src/histological_image_analysis/inference.py`:
- `load_model(model_path, device)` → `(model, processor)`
- `run_inference(image_path, model, processor, device)` → `(prediction, resized_prediction)`
- `run_sliding_window_inference(image_path, model, device, stride)` → `(prediction, resized_prediction)`
- Constants: `IMAGENET_MEAN`, `IMAGENET_STD`, `CROP_SIZE`

Update `scripts/run_inference.py` to import from `histological_image_analysis.inference` (preserving its existing behavior).

Update `scripts/annotate.py` to import from the same module.

### Step 5: Add Makefile targets

```makefile
# ── End-User Workflow ─────────────────────────────────────────────────

download-models: ## Download trained models from HuggingFace Hub (~2.5 GB)
	uv run python scripts/download_models.py --species all

download-models-mouse: ## Download mouse brain model only (~1.2 GB)
	uv run python scripts/download_models.py --species mouse

download-models-human-allen: ## Download human Allen depth-3 model only (~1.2 GB)
	uv run python scripts/download_models.py --species human

annotate-mouse: ## Annotate mouse brain images (set IMAGES=/path/to/folder)
	@test -n "$(IMAGES)" || (echo "ERROR: Set IMAGES path. Usage: make annotate-mouse IMAGES=/path/to/slides" && exit 1)
	uv run python scripts/annotate.py $(IMAGES)

annotate-human: ## Annotate with human model (set IMAGES=/path/to/folder)
	@test -n "$(IMAGES)" || (echo "ERROR: Set IMAGES path. Usage: make annotate-human IMAGES=/path/to/slides" && exit 1)
	uv run python scripts/annotate.py $(IMAGES) --species human

annotate-mouse-sliding: ## Annotate mouse with sliding window (slower, more accurate)
	@test -n "$(IMAGES)" || (echo "ERROR: Set IMAGES path." && exit 1)
	uv run python scripts/annotate.py $(IMAGES) --sliding-window
```

### Step 6: Update `.env.example`

Add:
```
# HuggingFace Hub repository for model downloads (no auth needed for public repos)
HF_REPO_ID=your-username/histological-brain-segmentation
```

### Step 7: Write tests (TDD)

New test file: `tests/test_annotate.py`

Tests:
1. **`test_create_annotated_overlay`** — Given a dummy image and prediction array, verify overlay is created with correct dimensions, alpha blending, and non-zero content.
2. **`test_annotated_filename_format`** — Verify output filename matches `{stem}-annotated-{YYYYMMDDTHHMMSS}.png` pattern.
3. **`test_annotated_file_placed_alongside_input`** — Verify output is in the same directory as input, not a separate output dir.
4. **`test_legend_top_regions`** — Verify legend shows top N regions sorted by area.
5. **`test_background_excluded_from_overlay`** — Verify class 0 (background) is not colored.

New test file: `tests/test_inference_module.py`

Tests:
1. **`test_load_model_missing_path`** — Verify clean error when model dir doesn't exist.
2. **`test_run_inference_returns_correct_shapes`** — Mock model, verify prediction shapes.

New test file: `tests/test_download_models.py`

Tests:
1. **`test_verify_download_detects_missing_files`** — Verify the verification function catches incomplete downloads.
2. **`test_download_creates_expected_directory_structure`** — Mock `snapshot_download`, verify files end up in `models/{species}/`.

### Step 8: Update README.md

Replace the "For PhD Researchers" section with the simplified workflow:

```markdown
## Quick Start: Annotate Brain Images

### 1. Install

```bash
git clone <repo-url>
cd histological-image-analysis
make install
```

### 2. Download models (~2.5 GB total, one-time)

```bash
make download-models
```

### 3. Annotate your images

```bash
# Mouse brain tissue (default)
make annotate-mouse IMAGES=/path/to/your/slides/

# Human brain tissue
make annotate-human IMAGES=/path/to/your/slides/

# Higher accuracy (slower — sliding window)
make annotate-mouse-sliding IMAGES=/path/to/your/slides/
```

### Output

For each input image, an annotated version appears in the same folder:

```
your-slides/
├── slide_001.jpg                              # Original (untouched)
├── slide_001-annotated-20260322T143052.png    # Annotated overlay
├── slide_002.tif                              # Original
└── slide_002-annotated-20260322T143055.png    # Annotated overlay
```

Each annotated image shows color-coded brain regions overlaid on the original,
with a legend identifying the detected structures.
```

---

## File Changes Summary

| Action | File | Description |
|--------|------|-------------|
| **Create** | `src/histological_image_analysis/inference.py` | Shared inference functions (extracted from `run_inference.py`) |
| **Create** | `scripts/annotate.py` | Main annotation CLI script |
| **Create** | `scripts/download_models.py` | Model download from HuggingFace Hub |
| **Create** | `scripts/upload_to_hf.py` | One-time model upload utility |
| **Create** | `tests/test_annotate.py` | Annotation overlay + filename tests |
| **Create** | `tests/test_inference_module.py` | Shared inference module tests |
| **Create** | `tests/test_download_models.py` | Download/verification tests |
| **Edit** | `scripts/run_inference.py` | Refactor to import from `inference.py` |
| **Edit** | `Makefile` | Add `download-models`, `annotate-mouse`, `annotate-human`, `annotate-mouse-sliding` targets |
| **Edit** | `.env.example` | Add `HF_REPO_ID` |
| **Edit** | `pyproject.toml` | Add `tqdm` to explicit deps (currently transitive) |
| **Edit** | `README.md` | Updated quick-start section |
| **Edit** | `.gitignore` | No changes needed — `models/` already ignored |

---

## Order of Execution

1. Write tests first (TDD) — `test_annotate.py`, `test_inference_module.py`, `test_download_models.py`
2. Extract `src/histological_image_analysis/inference.py` from `run_inference.py`
3. Update `scripts/run_inference.py` to import from new module
4. Implement `scripts/annotate.py` with overlay generation
5. Implement `scripts/download_models.py`
6. Implement `scripts/upload_to_hf.py`
7. Update Makefile with new targets
8. Update `.env.example` and `pyproject.toml`
9. Update `README.md`
10. Run full test suite: `make test`

---

## HuggingFace Configuration (Resolved)

- **Username:** `Noel-Niko`
- **Repo naming:** `{foundation-model}-{date}-histology-annotation-{species}`
- **Mouse repo:** `Noel-Niko/dinov2-upernet-20260322-histology-annotation-mouse`
- **Human repo:** `Noel-Niko/dinov2-upernet-20260322-histology-annotation-human`
- **Token env var:** `HUGGING_FACE_TOKEN` (in `.env`)

---

## Status — ALL COMPLETE

- [x] Step 1: Tests (TDD) — `test_annotate.py`, `test_inference_module.py`, `test_download_models.py`
- [x] Step 2: Extract `src/histological_image_analysis/inference.py`
- [x] Step 3: Create `src/histological_image_analysis/annotation.py`
- [x] Step 4: Create `src/histological_image_analysis/download.py`
- [x] Step 5: Refactor `scripts/run_inference.py` to use shared module
- [x] Step 6: Create `scripts/annotate.py`
- [x] Step 7: Create `scripts/download_models.py`
- [x] Step 8: Create `scripts/upload_to_hf.py`
- [x] Step 9: Makefile targets (`download-models`, `annotate-mouse`, `annotate-human`, `annotate-mouse-sliding`, `upload-models`)
- [x] Step 10: `.env.example` updated with `HUGGING_FACE_TOKEN`
- [x] Step 11: `pyproject.toml` updated with `tqdm` explicit dep
- [x] Step 12: `README.md` updated with 3-command quick-start
- [x] Step 13: Full test suite — **342 tests passing**

## Next Steps (Manual)

1. **Upload models to HuggingFace Hub:**
   ```bash
   # Download mouse model from Databricks (if not already local)
   databricks fs cp -r dbfs:/FileStore/allen_brain_data/models/final-200ep ./models/dinov2-upernet-final

   # Download human model from Databricks
   # (path depends on where the depth-3 model was saved)

   # Set token and upload
   export HUGGING_FACE_TOKEN=hf_your_token
   make upload-models
   ```

2. **Test the full end-user workflow:**
   ```bash
   make download-models
   make annotate-mouse IMAGES=/path/to/test/slides/
   ```
