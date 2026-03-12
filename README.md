# Histological Image Analysis

Fine-tune DINOv2-Large + UperNet for semantic segmentation of brain structures in Nissl-stained histological sections. Uses Allen Brain Institute CCFv3 volumetric data (10µm Nissl + annotations) to train a model that identifies anatomical brain regions at multiple granularities (coarse 6-class through fine 1,328-class), with the eventual goal of generalizing to autofluorescence imaging.

## Prerequisites

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
git clone <repo-url>
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

All notebooks use the same pipeline code (`ontology.py`, `ccfv3_slicer.py`, `dataset.py`, `training.py`). The unfrozen notebook adds differential learning rate (backbone 1e-5, head 1e-4) and trains for 100 epochs with early stopping.

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

| Run | Granularity | Split | mIoU | Overall Acc | Runtime | Cluster |
|-----|-------------|-------|------|-------------|---------|---------|
| 1 | Coarse 6-class | spatial | 50.7% | 78.8% | 23 min | 1x L40S |
| 2 | Coarse 6-class | interleaved | **88.0%** | **95.8%** | 23 min | 1x L40S |
| 3 | Depth-2 19-class | interleaved | **69.6%** | **95.5%** | 25 min | 1x L40S |
| 4 | Full 1,328-class (frozen) | interleaved | **60.3%** | **88.9%** | 5.5 hrs | 1x L40S |
| 5 | Full 1,328-class (unfrozen) | interleaved | **68.8%** | — | 11.3 hrs | 1x L40S |

**Run 1 → Run 2 lesson:** Contiguous spatial split along AP axis excluded all Cerebrum pixels from val/test (Cerebrum is anterior-only in mouse brain). Interleaved split fixed this — every 10th slice goes to val/test, ensuring all brain regions appear in all splits.

**Run 3 (depth-2):** 15 of 19 classes had valid IoU (4 classes absent from val set). Top classes: Cerebrum 95.8%, Brain stem 94.5%, Cerebellum 93.0%. Lowest: extrapyramidal fiber systems 44.4%.

**Run 4 (full, frozen):** 503 of 1,328 classes had valid IoU (825 absent from val). Top: Caudoputamen 95.2%, Background 94.3%, Main olfactory bulb 93.8%. See [docs/dinov2_model_research.md](docs/dinov2_model_research.md) for analysis and next-step recommendations.

**Run 5 (full, unfrozen backbone):** +8.5% mIoU gain from unfreezing last 4 DINOv2 blocks (20-23) with differential LR (backbone 1e-5, head 1e-4). 100 epochs, batch=2 + grad_accum=2. See [docs/step11_plan.md](docs/step11_plan.md) for deployment lessons (OOM, BatchNorm, gradient checkpointing fixes).

## Using Trained Models

### Quick Start (PhD Researchers)

**Run inference on your brain tissue images:**

```bash
# 1. Download model (~1.2 GB, requires Databricks CLI)
databricks fs cp -r dbfs:/FileStore/allen_brain_data/models/unfrozen ./models/dinov2-upernet-unfrozen

# 2. Run inference on a single image
python scripts/run_inference.py --image path/to/brain_slice.jpg --output results/

# 3. Or batch process a directory
python scripts/run_inference.py --image-dir images/ --output results/
```

**Compute Requirements:**
- ✅ **Runs on laptop** (CPU or GPU)
- RAM: 8 GB minimum, 16 GB recommended
- CPU inference: ~10-30 seconds per image
- GPU inference: ~1-5 seconds per image (optional)

**Output:** Segmentation masks + visualizations showing predicted brain regions

### Advanced Usage

```bash
# Load directly from MLflow (no download)
python -c "
import mlflow.transformers
model = mlflow.transformers.load_model('runs:/6cc49e1ccb0d4b30b371e9a071dcbe6f/model')
"

# Custom model path
python scripts/run_inference.py --image image.jpg --model ./models/custom-model --output results/

# Force CPU (no GPU)
python scripts/run_inference.py --image image.png --cpu --output results/
```

**Model size:** ~1.2-1.5 GB (excluded from git via `.gitignore`)

**See also:**
- [docs/model_download_guide.md](docs/model_download_guide.md) - Complete download guide, verification, troubleshooting
- [scripts/run_inference.py](scripts/run_inference.py) - CLI inference tool with batch processing
- [verify_model.py](verify_model.py) - Quick model verification script

## Project Structure

```
histological-image-analysis/
├── src/histological_image_analysis/     # Installable Python package
│   ├── ontology.py                      # Allen Brain structure ontology mapper
│   ├── ccfv3_slicer.py                  # CCFv3 3D volume -> 2D slice extraction
│   ├── svg_rasterizer.py                # SVG annotation -> pixel mask
│   ├── dataset.py                       # PyTorch Dataset (pad, crop, augment)
│   └── training.py                      # DINOv2 + UperNet model/trainer factories
├── tests/                               # 133 tests (pytest)
│   ├── test_ontology.py                 # 51 tests (incl. real ontology + annotation)
│   ├── test_ccfv3_slicer.py             # 22 tests
│   ├── test_svg_rasterizer.py           # 10 tests
│   ├── test_dataset.py                  # 15 tests
│   ├── test_training.py                 # 35 tests
│   ├── conftest.py                      # Shared fixtures
│   └── fixtures/minimal_ontology.json   # 15-structure test fixture
├── notebooks/
│   ├── finetune_coarse.ipynb            # Coarse 6-class training
│   ├── finetune_depth2.ipynb            # Depth-2 19-class training
│   ├── finetune_full.ipynb              # Full 1,328-class training (frozen backbone)
│   └── finetune_unfrozen.ipynb          # Full 1,328-class training (unfrozen backbone)
├── scripts/
│   └── download_allen_data.py           # Local download of Allen Brain data
├── docs/                                # Design docs and progress tracking
│   ├── progress.md                      # Full project history
│   ├── step10_plan.md                   # Step 10 plan (for LLM continuity)
│   ├── step11_plan.md                   # Step 11 plan (backbone unfreezing)
│   ├── dinov2_model_research.md         # Backbone size analysis and recommendations
│   ├── finetuning_recommendations.md    # Comprehensive fine-tuning roadmap
│   ├── step10_gpu_memory_review.md      # GPU memory lessons from full-mapping runs
│   └── joyful-popping-planet.md         # Step 9 tracker
├── Makefile                             # Build, test, deploy commands (14 targets)
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

## Data

Training data comes from the Allen Brain Institute CCFv3 (Common Coordinate Framework v3):

- **Image volume:** `ara_nissl_10.nrrd` -- 1320 coronal slices of Nissl-stained mouse brain at 10µm (float32, 1320x800x1140)
- **Annotation volume:** `annotation_10.nrrd` -- per-voxel structure IDs, 672 unique structures (uint32)
- **Ontology:** `structure_graph_1.json` -- hierarchical structure tree (1,327 structures)

Data was downloaded locally via `scripts/download_allen_data.py` and uploaded to Databricks Workspace/DBFS. See `docs/step5_6_completion_report.md` for details.

## References

See [docs/references.md](docs/references.md) for links to Allen Brain API, DINOv2, UperNet, and HuggingFace documentation.
