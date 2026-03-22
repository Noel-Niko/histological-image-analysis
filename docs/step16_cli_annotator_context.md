# CLI Annotator — Full Context for New Contributors

This document provides complete context for anyone (human or LLM) working on the
annotation CLI tool with no prior knowledge of this codebase. Start here.

---

## What This Project Does

This project fine-tunes a DINOv2-Large + UperNet semantic segmentation model to
identify anatomical brain regions in Nissl-stained histological tissue sections.
It supports both **mouse** and **human** brain tissue.

A PhD researcher can take photos/scans of brain tissue slides, point the tool at
a folder of those images, and get back annotated versions showing which brain
regions the model detected — with color overlays and a legend.

---

## Architecture Overview

```
User's brain images (JPG/PNG/TIFF/BMP)
        │
        ▼
scripts/annotate.py          ← CLI entry point (argparse)
        │
        ├── histological_image_analysis/inference.py
        │       load_model()              ← loads HuggingFace UperNet model
        │       run_inference()           ← center-crop: resize to 518x518, predict
        │       run_sliding_window_inference()  ← tile at native res, average logits
        │       get_image_files()         ← find supported images in a directory
        │
        ├── histological_image_analysis/annotation.py
        │       create_annotated_overlay()  ← alpha-blend color overlay on original
        │       build_annotated_filename()  ← {stem}-annotated-{species}-{timestamp}.png
        │       _draw_contours()            ← boundary lines between regions
        │       _draw_legend()              ← top-15 regions with color swatches
        │
        └── Output: {stem}-annotated-{species}-{YYYYMMDDTHHMMSS}.png
            (saved in SAME folder as input, originals never modified)
```

### Model Details

| Model | Species | Classes | What it identifies | mIoU | HuggingFace Repo |
|-------|---------|---------|-------------------|------|------------------|
| Mouse (final, Run 9) | Mouse | 1,328 | Anatomical brain structures | 79.1% | `Noel-Niko/dinov2-upernet-20260322-histology-annotation-mouse` |
| Human Allen depth-3 | Human | 44 | Brain regions (cortex, thalamus, etc.) | 65.5% | `Noel-Niko/dinov2-upernet-20260322-histology-annotation-human` |
| Human BigBrain | Human | 10 | Tissue types (gray matter, white matter, CSF, etc.) | 60.8% | `Noel-Niko/dinov2-upernet-20260322-histology-annotation-human-bigbrain` |

Both models use the same architecture:
- **Backbone:** DINOv2-Large (304M params, 24 transformer blocks, pretrained on ImageNet-22k)
- **Head:** UperNet (Pyramid Pooling Module, 38M params)
- **Input:** 518x518 RGB images (ImageNet-normalized)
- **Output:** Per-pixel class IDs
- **Framework:** PyTorch via HuggingFace `transformers` (`UperNetForSemanticSegmentation`)
- **Config:** Each model's `config.json` contains `id2label` mapping class index → human-readable name

### Inference Modes

1. **Center-crop** (default): Resize entire image to 518x518, run model once, resize prediction back. Fast.
2. **Sliding window**: Tile the image with 518x518 windows at stride 259 (50% overlap), average logits in overlap regions, argmax. Full resolution. Slower but +4.4% mIoU on mouse.

### File Formats

Supported input: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`

All images are converted to RGB on load. Grayscale works. Output is always `.png`.

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/annotate.py` | Main CLI — guided mode (no args) or direct mode (with path) |
| `scripts/download_models.py` | Downloads models from HuggingFace Hub → `models/` |
| `scripts/upload_to_hf.py` | One-time upload from local → HuggingFace Hub |
| `scripts/run_inference.py` | Older inference script — raw masks + matplotlib panels |
| `src/.../inference.py` | Shared: model loading, inference, file discovery |
| `src/.../annotation.py` | Overlay generation: colors, contours, legend |
| `src/.../download.py` | Download verification, HF repo ID resolution |
| `tests/test_inference_module.py` | 14 tests for inference utilities |
| `tests/test_annotate.py` | 16 tests for overlay + filename |
| `tests/test_download_models.py` | 11 tests for download verification |
| `Makefile` | User-facing commands: `annotate-mouse`, `annotate-human-allen`, etc. |

---

## Makefile Targets (User-Facing)

```
make download-models                       # Download all models from HuggingFace (~3.6 GB)
make download-models-mouse                 # Mouse model only (~1.2 GB)
make download-models-human-allen           # Human Allen depth-3 model only (~1.2 GB)
make download-models-human-bigbrain        # Human BigBrain model only (~1.2 GB)

make annotate-mouse                              # Mouse brain, guided mode
make annotate-mouse IMAGES=/path                 # Mouse brain, direct mode
make annotate-human-allen                        # Human brain regions (44 classes), guided
make annotate-human-allen IMAGES=/path           # Human brain regions, direct mode
make annotate-human-bigbrain                     # Human tissue types (10 classes), guided
make annotate-human-bigbrain IMAGES=/path        # Human tissue types, direct mode
make annotate-mouse-sliding                      # Mouse, sliding window, guided
make annotate-human-allen-sliding                 # Human Allen regions, sliding window, guided
make annotate-human-bigbrain-sliding             # Human tissue, sliding window, guided
```

---

## Current State (as of 2026-03-22)

### What Works

- Full end-to-end pipeline: download → annotate → output
- Both mouse and human models uploaded to HuggingFace Hub and downloadable
- Guided mode with file-type instructions and path prompts
- Direct mode for scripted/repeated use
- Unsupported files are warned about before processing begins
- 342 tests passing (41 new for CLI tool + 301 existing)
- Original files are never modified (verified by tests + code audit)

### What's Missing / Known Gaps

1. **Human model accuracy is lower than mouse (65.5% vs 79.1% mIoU).**
   The human model was trained on Allen Human Brain Atlas depth-3 mapping (44 classes)
   with only ~3,500 training images from 4 donors. The mouse model had denser annotations
   from 3D volume slicing. Improving the human model is a research priority.

2. **No species auto-detection.**
   The user must choose `make annotate-mouse` (mouse) or `make annotate-human-allen` (human).
   There is no automatic detection of whether an image is mouse or human tissue.
   This could be added as a lightweight classifier in the future.

3. **No whole-slide image (WSI) support.**
   The tool works on standard image files. Whole-slide imaging formats (`.svs`, `.ndpi`,
   `.mrxs`) used in digital pathology are not supported. These require specialized
   libraries like `openslide` to read tiled pyramid images at different zoom levels.

4. **Font fallback on non-macOS systems.**
   The legend panel (`annotation.py:152`) tries to load Helvetica from macOS system fonts.
   It falls back to PIL's default bitmap font on Linux/Windows, which looks worse.
   A bundled TTF font would improve cross-platform appearance.

5. **No batch progress reporting beyond tqdm.**
   For very large batches (hundreds of slides), there's no summary log file or
   CSV output listing which files were processed, how many regions were detected, etc.

6. **Training data is not included in the repo.**
   Training data (Allen Brain CCFv3 volumes, human atlas images) lives on Databricks.
   The `data/` directory has download scripts but is mostly `.gitignore`d. A future
   contributor would need Databricks access to retrain models.

---

## How Models Are Trained (Summary)

Training happens on Databricks GPU clusters (NVIDIA L40S), not locally. The pipeline:

1. **Data preparation:** 3D brain volumes (NRRD for mouse, NIfTI for BigBrain, JPEG+SVG
   for human atlas) are sliced into 2D coronal sections with per-pixel annotation labels.
2. **Dataset:** PyTorch `Dataset` class pads/crops to 518x518, applies augmentation
   (rotation, flip, color jitter), normalizes to ImageNet stats.
3. **Model:** DINOv2-Large backbone (frozen or last 4 blocks unfrozen) + UperNet head.
   Created via `training.py:create_model(num_labels=N)`.
4. **Training:** HuggingFace `Trainer` with MLflow logging. 200 epochs for final mouse,
   50 epochs for human models.
5. **Saving:** `model.save_pretrained(path)` + `processor.save_pretrained(path)` → DBFS.
   Then `make fetch-models-from-dbfs` to pull locally, `make upload-models` to push to HF Hub.

### Key training notebooks:
- `notebooks/finetune_final_200ep.ipynb` — Mouse final model (Run 9)
- `notebooks/finetune_human_allen_depth3.ipynb` — Human depth-3 model
- `notebooks/finetune_human_bigbrain.ipynb` — Human BigBrain 9-class model

---

## How to Add a New Model

If you train a new model (e.g., improved human model, different species):

1. **Train** on Databricks using the existing notebook pattern.
2. **Save** with `model.save_pretrained()` — ensure `id2label`/`label2id` are set in
   `model.config` before saving (see any training notebook, cell 7).
3. **Download** from Databricks: update `DBFS_*_MODEL` paths in `Makefile` if needed,
   run `make fetch-models-from-dbfs`.
4. **Upload** to HuggingFace: update `DEFAULT_MODEL_BASE` in `src/.../download.py` if
   the naming convention changes, then `make upload-models`.
5. **Update** `resolve_repo_ids()` in `download.py` if you add a new species.
6. **Update** `Makefile` with a new `annotate-{species}` target.
7. **Update** `README.md` with the new model in the command table.
8. **Run tests:** `make test` — all 342+ should still pass.

---

## Environment Setup

```bash
# Prerequisites: uv (Python package manager), Python >= 3.11
git clone https://github.com/Noel-Niko/histological-image-analysis.git
cd histological-image-analysis
make install          # uv sync --all-extras
make test             # 342 tests
make download-models  # pull models from HuggingFace Hub
```

For Databricks deployment (training): see `docs/databricks_connectivity.md`.

HuggingFace credentials: `HUGGING_FACE_TOKEN` in `.env` (write access needed only for uploads).
