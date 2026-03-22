# Histological Image Analysis

Fine-tune DINOv2-Large + UperNet for semantic segmentation of brain structures in Nissl-stained histological sections. Uses Allen Brain Institute CCFv3 volumetric data (10µm Nissl + annotations) to train a model that identifies anatomical brain regions at multiple granularities (coarse 6-class through fine 1,328-class), with the eventual goal of generalizing to autofluorescence imaging.

---

## Annotate Brain Images

Annotate mouse or human brain tissue slides from the terminal — **no GPU, no Databricks, no IDE needed**. The tool identifies anatomical brain regions in Nissl-stained histological sections and produces color-coded overlays with a legend of detected structures.

### 1. Install

```bash
git clone https://github.com/Noel-Niko/histological-image-analysis.git
cd histological-image-analysis
make install
```

Requires [uv](https://docs.astral.sh/uv/) (Python package manager) and Python >= 3.11.

### 2. Download models (~2.5 GB total, one-time)

```bash
make download-models
```

Or download individually:

```bash
make download-models-mouse           # Mouse brain model (~1.2 GB, 1,328 structures)
make download-models-human-allen     # Human brain regions model (~1.2 GB, 44 regions)
make download-models-human-bigbrain  # Human tissue types model (~1.2 GB, 10 types)
```

### 3. Annotate your images

**Choose the command that matches your tissue and analysis goal:**

| Your tissue | What you want to identify | Command |
|-------------|--------------------------|---------|
| **Mouse** brain | 1,328 anatomical structures (79.1% mIoU) | `make annotate-mouse` |
| **Human** brain | 44 brain regions — cortex, thalamus, etc. (65.5% mIoU) | `make annotate-human` |
| **Human** brain | 10 tissue types — gray matter, white matter, CSF, etc. (60.8% mIoU) | `make annotate-human-bigbrain` |

The **human** and **human-bigbrain** models answer different questions about the same tissue:
- `annotate-human` identifies *which brain region* (e.g., "cerebral cortex", "thalamus", "hippocampus")
- `annotate-human-bigbrain` identifies *what tissue type* (e.g., "gray matter", "white matter", "CSF")

Using a mouse model on human tissue (or vice versa) will produce incorrect results.

**First time? Just run the command** — it will walk you through supported file types,
ask for your folder path, and show a shortcut for next time:

```bash
make annotate-mouse           # Mouse
make annotate-human           # Human — brain regions
make annotate-human-bigbrain  # Human — tissue types
```

**Returning user?** Provide the path directly to skip the prompts:

```bash
make annotate-mouse IMAGES=/Users/yourname/Desktop/mouse_slides/
make annotate-human IMAGES=/Users/yourname/Desktop/human_slides/
make annotate-human-bigbrain IMAGES=/Users/yourname/Desktop/human_slides/
```

**Want higher accuracy?** Sliding window mode tiles across the full image at native
resolution instead of resizing to 518x518. Slower, but more accurate at image edges:

```bash
make annotate-mouse-sliding IMAGES=/Users/yourname/Desktop/mouse_slides/
make annotate-human-sliding IMAGES=/Users/yourname/Desktop/human_slides/
make annotate-human-bigbrain-sliding IMAGES=/Users/yourname/Desktop/human_slides/
```

### Supported image formats

| Format | Extensions |
|--------|------------|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| TIFF | `.tif`, `.tiff` |
| BMP | `.bmp` |

**Image requirements:**
- **Color mode:** RGB or grayscale — both work (grayscale is auto-converted to RGB)
- **Resolution:** Any size — the model resizes internally (518x518 for center-crop, or tiles at native resolution for sliding-window mode)
- **Best results:** Nissl-stained histological sections, similar to Allen Brain Institute atlas images
- **File size:** No limit, but very large files (e.g., whole-slide images > 50,000 px) will be slow on CPU

**NOT supported** (convert these to PNG or TIFF first):
- DICOM (`.dcm`) — use a DICOM viewer to export as PNG
- NRRD (`.nrrd`) / NIfTI (`.nii.gz`) — 3D volumes, not 2D images
- PDF / SVG — export as raster images
- WebP, HEIC — re-save as JPEG or PNG

When you point to a folder, the tool **immediately warns** about any files that will be
skipped due to unsupported format, listing each one by name before processing begins.

### Output

For each input image, an annotated version appears in the **same folder**.
Your original files are never modified.

```
/Users/yourname/Desktop/brain_slides/
    slide_001.jpg                              # Original (untouched)
    slide_001-annotated-20260322T143052.png    # NEW — annotated overlay
    slide_002.tif                              # Original (untouched)
    slide_002-annotated-20260322T143055.png    # NEW — annotated overlay
    notes.txt                                  # Ignored (not an image)
```

Each annotated image contains:
- The original image with a **semi-transparent color overlay** showing detected brain regions
- **Contour lines** at region boundaries
- A **legend panel** on the right listing the top 15 detected regions by area, with color swatches and percentage labels

### Available models

| Model | Species | Classes | What it identifies | mIoU | Command |
|-------|---------|---------|-------------------|------|---------|
| Mouse (final) | Mouse | 1,328 | Anatomical brain structures | 79.1% | `make annotate-mouse` |
| Human Allen depth-3 | Human | 44 | Brain regions (cortex, thalamus, etc.) | 65.5% | `make annotate-human` |
| Human BigBrain | Human | 10 | Tissue types (gray/white matter, CSF, etc.) | 60.8% | `make annotate-human-bigbrain` |

### Inference modes

| Mode | Command | Description |
|------|---------|-------------|
| **Center-crop** (default) | `make annotate-mouse` | Fast. Resizes image to 518x518, runs once. |
| **Sliding window** | `make annotate-mouse-sliding` | Slower but more accurate. Tiles at native resolution with 50% overlap. Better for structures at image edges. |

### Compute requirements

| Hardware | Speed per image |
|----------|----------------|
| **CPU only (laptop)** | ~10-30 seconds |
| Laptop GPU (MX/GTX) | ~2-5 seconds |
| Desktop GPU (RTX 3060+) | ~1-2 seconds |

- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** ~1.2 GB per model + space for output images
- **GPU:** Optional, auto-detected. Not required.

### Advanced usage

For raw segmentation masks and multi-panel visualizations (not overlays):

```bash
python scripts/run_inference.py --image-dir images/ --output inference_results/
```

All CLI options:

```bash
python scripts/annotate.py --help
python scripts/run_inference.py --help
```

### More info

- [Model download guide](docs/model_download_guide.md) — download, verify, troubleshoot
- [CLI annotator plan](docs/step15_cli_annotator_plan.md) — full design document
- [Paper draft](docs/paper_draft.md) — full ablation study (79.1% mIoU, 9 runs)

---

## Developer Setup

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [uv](https://docs.astral.sh/uv/) | >= 0.4 | Python package manager |
| Python | >= 3.11 | Runtime |
| [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) | >= 0.278 | Deployment to Databricks |
| JFrog Artifactory access | -- | Model weights mirror (`facebook/dinov2-large`) |

**Databricks workspace:** `grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com`

**Available clusters:**

| Cluster | Instance | GPU | Use Case |
|---------|----------|-----|----------|
| Single GPU | g6e.16xlarge | 1x NVIDIA L40S 48GB | Primary training (verified for all granularities) |
| Multi-GPU | g6e.48xlarge | 8x NVIDIA L40S 48GB | Speed via DDP (requires reduced batch size for full mapping) |

**Note:** Cluster IDs are ephemeral and will differ at execution time.

### Databricks CLI Setup

```bash
# Verify CLI is configured
databricks auth profiles
# Should show 'dev' profile targeting the workspace above
```

## Local Development

```bash
# Clone and install
git clone https://github.com/Noel-Niko/histological-image-analysis.git
cd histological-image-analysis
cp .env.example .env        # Edit with your values
make install                # uv sync --all-extras

# Run tests
make test                   # 133 tests

# Lint
make lint                   # ruff check

# Build wheel
make build                  # -> dist/histological_image_analysis-0.1.0-py3-none-any.whl
```

## Training Granularities

The pipeline supports three mapping granularities, with notebooks for frozen and unfrozen backbone:

| Granularity | Classes | Notebook | Backbone | Model Output Dir |
|-------------|---------|----------|----------|------------------|
| **Coarse** | 6 | `finetune_coarse.ipynb` | Frozen | `/dbfs/.../models/coarse_6class` |
| **Depth-2** | 19 | `finetune_depth2.ipynb` | Frozen | `/dbfs/.../models/depth2` |
| **Full** | 1,328 | `finetune_full.ipynb` | Frozen | `/dbfs/.../models/full` |
| **Full (unfrozen)** | 1,328 | `finetune_unfrozen.ipynb` | Last 4 blocks unfrozen | `/dbfs/.../models/unfrozen` |
| **Final (200ep)** | 1,328 | `finetune_final_200ep.ipynb` | Last 4 blocks unfrozen | `/dbfs/.../models/final-200ep` |

All notebooks use the same pipeline code (`ontology.py`, `ccfv3_slicer.py`, `dataset.py`, `training.py`). The final model uses differential learning rate (backbone 1e-5, head 1e-4) and trains for 200 epochs. See `docs/paper_draft.md` for the full ablation study.

## Databricks Deployment

### Step 1: Deploy

```bash
# Deploy everything (build wheel, upload to DBFS, upload all 3 notebooks)
make deploy

# Or individually:
make deploy-wheel              # Build + upload wheel to DBFS
make deploy-notebook           # Upload coarse notebook
make deploy-notebook-depth2    # Upload depth-2 notebook
make deploy-notebook-full      # Upload full-mapping notebook
make deploy-notebook-unfrozen  # Upload unfrozen-backbone notebook
```

### Step 2: Staged Validation (Cells 0-4)

Open the notebook on Databricks and run cells one at a time:

| Cell | What It Does | Success Criteria |
|------|-------------|-----------------|
| 0 | Install wheel from DBFS | No pip errors, kernel restarts |
| 1 | Load configuration | Paths print correctly, mapping type shown |
| 2 | Download DINOv2-Large weights | `snapshot_download` completes |
| 3 | Build data pipeline | Classes and split sizes print, sample shape `(3, 518, 518)` |
| 4 | Create model + forward pass | Logits shape `(1, NUM_LABELS, 518, 518)`, "Forward pass OK" |

Run `make validate` for a detailed checklist with expected outputs.

### Step 3: Full Training (Cells 5-7)

After cells 0-4 pass:

| Cell | What It Does |
|------|-------------|
| 5 | Train with MLflow logging (50 epochs frozen, 100 epochs unfrozen) |
| 6 | Evaluate on val set, per-class IoU report, visualize predictions vs ground truth |
| 7 | Save final model to DBFS + close MLflow run |

### Multi-GPU (DDP)

No code changes needed. Just run any notebook on a multi-GPU cluster. HF Trainer auto-detects
`WORLD_SIZE > 1` and wraps the model in DDP.

**Important:** For the full 1,328-class mapping, use `batch_size=2` with `gradient_accumulation_steps=2`
on multi-GPU to avoid OOM from per-GPU DDP overhead. See [docs/step10_gpu_memory_review.md](docs/step10_gpu_memory_review.md)
for details. Single-GPU runs with `batch_size=4` have been verified to work without issues.

## Training Results

| Run | Config | mIoU (CC) | mIoU (SW) | Accuracy | Runtime | Cluster |
|-----|--------|-----------|-----------|----------|---------|---------|
| 1 | Coarse 6-class, spatial split | 50.7% | — | 78.8% | 23 min | 1x L40S |
| 2 | Coarse 6-class, interleaved | 88.0% | — | 95.8% | 23 min | 1x L40S |
| 3 | Depth-2 19-class | 69.6% | — | 95.5% | 25 min | 1x L40S |
| 4 | Full 1,328-class (frozen) | 60.3% | — | 88.9% | 5.5 hrs | 1x L40S |
| 5 | Full 1,328-class (unfrozen) | 68.8% | — | 92.5% | 11.3 hrs | 1x L40S |
| 6 | Weighted Dice+CE loss | 67.2% | — | 89.7% | ~33 hrs | 1x L40S |
| 7 | Extended augmentation | 62.3% | — | 90.1% | ~12 hrs | 1x L40S |
| TTA | Eval-only: 6-variant TTA on Run 5 | 44.4% | — | 84.4% | ~20 min | — |
| 8a | Pruned 673-class | 69.0% | — | 92.4% | ~9.5 hrs | 1x L40S |
| **9** | **Final: unfrozen, 200 epochs** | **74.8%** | **79.1%** | **96.9%** | **23.0 hrs** | **1x L40S** |

**Runs 1→2:** Contiguous spatial split along AP axis excluded all Cerebrum pixels from val/test. Interleaved split fixed this (+37.3% mIoU on 6-class task).

**Run 3 (depth-2):** 15 of 19 classes had valid IoU. Top: Cerebrum 95.8%, Brain stem 94.5%, Cerebellum 93.0%.

**Run 4 (full, frozen):** 503 of 1,328 classes had valid IoU. Top: Caudoputamen 95.2%, Background 94.3%, Main olfactory bulb 93.8%.

**Run 5 (unfrozen backbone):** +8.5% mIoU from unfreezing last 4 DINOv2 blocks (20-23) with differential LR (backbone 1e-5, head 1e-4). Single largest training improvement.

**Runs 6-7 (negative results):** Weighted Dice+CE loss (−1.6%) and extended augmentation (−6.5%) both degraded performance. TTA was catastrophic (−24.4%) due to lack of rotational equivariance.

**Run 8a (class pruning):** Removing 655 zero-pixel classes from the output head had no meaningful effect (+0.2% mIoU). Useful only for halving logits memory.

**Run 9 (final model):** Doubling training from 100→200 epochs yielded +6.0% mIoU — second largest improvement. Sliding window evaluation added +4.4% mIoU and revealed 168 additional valid classes. See [docs/paper_draft.md](docs/paper_draft.md) for the full ablation study.

## Project Structure

```
histological-image-analysis/
├── src/histological_image_analysis/     # Installable Python package
│   ├── ontology.py                      # Allen Brain structure ontology mapper
│   ├── ccfv3_slicer.py                  # CCFv3 3D volume -> 2D slice extraction
│   ├── svg_rasterizer.py                # SVG annotation -> pixel mask
│   ├── dataset.py                       # PyTorch Dataset (pad, crop, augment)
│   ├── training.py                      # DINOv2 + UperNet model/trainer factories
│   ├── inference.py                     # Shared inference (model loading, prediction)
│   ├── annotation.py                    # Overlay generation (color regions + legend)
│   └── download.py                      # Model download/verification utilities
├── tests/                               # pytest suite
│   ├── test_ontology.py                 # 51 tests (incl. real ontology + annotation)
│   ├── test_ccfv3_slicer.py             # 22 tests
│   ├── test_svg_rasterizer.py           # 10 tests
│   ├── test_dataset.py                  # 15 tests
│   ├── test_training.py                 # 35 tests
│   ├── test_inference_module.py         # 14 tests (inference utilities)
│   ├── test_annotate.py                 # 16 tests (overlay + filename)
│   ├── test_download_models.py          # 11 tests (download verification)
│   ├── conftest.py                      # Shared fixtures
│   └── fixtures/minimal_ontology.json   # 15-structure test fixture
├── scripts/
│   ├── annotate.py                      # Annotate images with brain regions
│   ├── download_models.py               # Download models from HuggingFace Hub
│   ├── upload_to_hf.py                  # Upload models to HuggingFace (one-time)
│   ├── run_inference.py                 # Raw inference: masks + visualizations
│   ├── download_allen_data.py           # Download Allen Brain CCFv3 data
│   └── generate_paper_figures.py        # Generate paper figures locally
├── docs/                                # Design docs and progress tracking
├── Makefile                             # Build, test, deploy, annotate commands
├── .env.example                         # Environment variable template
├── pyproject.toml                       # Dependencies and build config
└── uv.lock                              # Locked dependencies
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Databricks Cluster                             │
│                                                                  │
│  Notebooks (thin orchestration):                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ finetune_coarse.ipynb          (6 classes, coarse)        │  │
│  │ finetune_depth2.ipynb          (19 classes, depth-2)      │  │
│  │ finetune_full.ipynb            (1,328 classes, frozen)    │  │
│  │ finetune_unfrozen.ipynb        (1,328 classes, unfrozen)  │  │
│  │                                                            │  │
│  │ Cell 0: %pip install wheel from DBFS                      │  │
│  │ Cell 1: Configuration (paths, hyperparams, mapping type)  │  │
│  │ Cell 2: Download DINOv2-Large from JFrog/HF               │  │
│  │ Cell 3: OntologyMapper -> CCFv3Slicer -> Dataset          │  │
│  │ Cell 4: create_model(num_labels=N) + forward pass check   │  │
│  │ Cell 5: Trainer.train() (HF Trainer + MLflow)             │  │
│  │ Cell 6: Evaluate + per-class IoU + visualization          │  │
│  │ Cell 7: Save model to DBFS + close MLflow run             │  │
│  └──────────────┬─────────────────────────────────────────────┘  │
│                 │ imports                                         │
│  ┌──────────────▼─────────────────────────────────────────────┐  │
│  │  histological_image_analysis (wheel on DBFS)               │  │
│  │                                                            │  │
│  │  ontology.py    -> Structure ID -> class mapping           │  │
│  │                    build_coarse_mapping()   -> 6 classes    │  │
│  │                    build_depth_mapping(2)   -> 19 classes   │  │
│  │                    build_full_mapping()     -> 1,328 classes│  │
│  │  ccfv3_slicer.py -> 3D NRRD -> 2D coronal slices          │  │
│  │  dataset.py     -> PyTorch Dataset (crop, augment)         │  │
│  │  training.py    -> DINOv2 + UperNet + Trainer              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Data on Databricks:                                             │
│  ├── /Workspace/.../ontology/structure_graph_1.json              │
│  ├── /Workspace/.../ccfv3/annotation_10.nrrd                     │
│  └── /dbfs/FileStore/.../ara_nissl_10.nrrd (2.17 GB)            │
└──────────────────────────────────────────────────────────────────┘
```

## Training Data Sources

All models are trained on publicly available neuroanatomical atlas data. No proprietary data is used.

### Mouse model — Allen Brain CCFv3

The mouse model is trained on the [Allen Mouse Brain Common Coordinate Framework v3](https://mouse.brain-map.org/) (CCFv3), provided by the Allen Institute for Brain Science.

| Dataset | File | Size | Resolution | Format | Description |
|---------|------|------|------------|--------|-------------|
| Nissl image volume | `ara_nissl_10.nrrd` | 2.17 GB | 10µm isotropic | NRRD float32 | 1,320 coronal slices of Nissl-stained mouse brain (1320×800×1140 voxels) |
| Annotation volume | `annotation_10.nrrd` | 33 MB | 10µm isotropic | NRRD uint32 | Per-voxel structure IDs — 672 unique structures observed |
| Structure ontology | `structure_graph_1.json` | 880 KB | — | JSON | Hierarchical tree of 1,327 brain structures |

- **Species:** Mouse (P56)
- **Stain:** Nissl (cell body, grayscale)
- **Annotation density:** 100% — every voxel has a structure label
- **Training split:** 1,016 train / 127 val / 127 test (interleaved every 10 slices along AP axis)
- **Classes:** 1,328 (1,327 structures + background)
- **Download:** `scripts/download_allen_data.py` or direct from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/

**Citation:** Allen Institute for Brain Science. Allen Mouse Brain Atlas — Common Coordinate Framework v3. https://atlas.brain-map.org

### Human Allen model — Allen Human Brain Atlas

The human region model is trained on the [Allen Human Brain Atlas](https://human.brain-map.org/) Nissl-stained section images with SVG structure annotations, mapped to depth-3 of the ontology hierarchy (44 classes).

| Dataset | Source | Count | Description |
|---------|--------|-------|-------------|
| Section images | 6 post-mortem donors | 14,566 total | Nissl-stained coronal sections (JPEG, ~1.7K×2.1K px at downsample=4) |
| SVG annotations | ISH sampling regions | 4,463 annotated | Polygon paths marking brain structures (~12 structures per image, 17–33% pixel coverage) |
| Structure ontology | Graph 10 | 1,839 structures | Adult human brain hierarchy (depth-3 mapping collapses to 44 classes) |

- **Species:** Human (adult, 6 donors)
- **Stain:** Nissl (cell body, color photography)
- **Annotation density:** Sparse — 17–33% of pixels labeled per image (ISH sampling regions only)
- **Training split:** ~3,500 annotated images from 4 donors (train) / held-out donors (val/test)
- **Classes:** 44 brain regions at ontology depth 3
- **API:** https://api.brain-map.org/api/v2/ (section images + SVG downloads)

**Citation:** Hawrylycz, M. J., Lein, E. S., Guillozet-Bongaarts, A. L., et al. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome. *Nature*, 489(7416), 391–399. https://human.brain-map.org

### Human BigBrain model — BigBrain 3D Volume

The BigBrain tissue model is trained on the [BigBrain](https://bigbrainproject.org/) ultra-high-resolution 3D human brain volume with voxel-level tissue classification.

| Dataset | File | Size | Resolution | Format | Description |
|---------|------|------|------------|--------|-------------|
| Tissue classification | `full_cls_200um_9classes.nii.gz` | 16 MB | 200µm isotropic | NIfTI gzip | 9-class tissue segmentation (696×770×605 voxels) |
| Histological volume | `full8_200um_optbal.nii.gz` | 74 MB | 200µm isotropic | NIfTI gzip | Grayscale intensity volume (696×770×605 voxels) |

- **Species:** Human (single 65-year-old post-mortem specimen)
- **Stain:** Merker silver stain (cell body, grayscale — different from Nissl)
- **Annotation density:** 100% of non-background voxels (35.1% of total volume is brain tissue)
- **Training split:** 468 train / 59 val / 59 test (interleaved every 10 slices, 1mm gaps)
- **Classes:** 10 (9 tissue types + background): gray matter, white matter, CSF, meninges, blood vessels, bone, muscle, artifact, other
- **Download:** https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/

**Citation:** Amunts, K., Lepage, C., Borgeat, L., et al. (2013). BigBrain: An ultrahigh-resolution 3D human brain model. *Science*, 340(6139), 1472–1475. https://bigbrainproject.org

### Data source comparison

| Property | Mouse (CCFv3) | Human (Allen Atlas) | Human (BigBrain) |
|----------|--------------|--------------------|--------------------|
| Annotation type | Dense 3D volume | Sparse 2D SVG polygons | Dense 3D volume |
| Pixel coverage | 100% | 17–33% | 35% (non-background) |
| Stain | Nissl (gray) | Nissl (color) | Merker silver (gray) |
| Resolution | 10µm | ~1.7K×2.1K px per section | 200µm |
| Specimens | 1 reference brain | 6 donors | 1 specimen |
| Anatomical plane | Coronal only | Coronal only | Coronal only |

Data was downloaded locally via `scripts/download_allen_data.py` and uploaded to Databricks Workspace/DBFS for training. See `docs/step5_6_completion_report.md` for details.

## References

See [docs/references.md](docs/references.md) for links to Allen Brain API, DINOv2, UperNet, and HuggingFace documentation.
