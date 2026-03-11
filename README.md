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
| Single GPU | g6e.16xlarge | 1x NVIDIA L40S 48GB | Fallback / staged validation |
| Multi-GPU | g6e.48xlarge | 8x NVIDIA L40S 48GB | Primary training (DDP, ~6-7x faster) |

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

The pipeline supports three mapping granularities, each with its own notebook:

| Granularity | Classes | Notebook | Model Output Dir |
|-------------|---------|----------|------------------|
| **Coarse** | 6 (Cerebrum, Brain stem, Cerebellum, fiber tracts, VS, background) | `step9_finetune_coarse.ipynb` | `/dbfs/.../models/coarse_6class` |
| **Depth-2** | 19 (18 brain structures at ontology depth 2 + background) | `step10_finetune_depth2.ipynb` | `/dbfs/.../models/depth2` |
| **Full** | 1,328 (every structure in the Allen Brain ontology + background) | `step10_finetune_full.ipynb` | `/dbfs/.../models/full` |

All three use the same pipeline code (`ontology.py`, `ccfv3_slicer.py`, `dataset.py`, `training.py`). The only differences are which mapping function is called and the batch size.

## Databricks Deployment

### Step 1: Deploy

```bash
# Deploy everything (build wheel, upload to DBFS, upload all 3 notebooks)
make deploy

# Or individually:
make deploy-wheel              # Build + upload wheel to DBFS
make deploy-notebook           # Upload coarse notebook (step 9)
make deploy-notebook-depth2    # Upload depth-2 notebook (step 10)
make deploy-notebook-full      # Upload full-mapping notebook (step 10)
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

Run `make validate` for a detailed checklist with expected outputs (coarse notebook).

### Step 3: Full Training (Cells 5-7)

After cells 0-4 pass:

| Cell | What It Does |
|------|-------------|
| 5 | Train for 50 epochs with MLflow logging |
| 6 | Evaluate on val set, per-class IoU report, visualize predictions vs ground truth |
| 7 | Save final model to DBFS + close MLflow run |

### Multi-GPU (DDP)

No code changes needed. Just run any notebook on the multi-GPU cluster (g6e.48xlarge, 8x L40S). HF Trainer auto-detects `WORLD_SIZE > 1` and wraps the model in DDP. Expect ~6-7x speedup.

## Training Results

| Run | Granularity | Split | mIoU | Overall Acc | Runtime | Cluster |
|-----|-------------|-------|------|-------------|---------|---------|
| 1 | Coarse 6-class | spatial | 50.7% | 78.8% | 23 min | 1x L40S |
| 2 | Coarse 6-class | interleaved | **88.0%** | **95.8%** | 23 min | 1x L40S |
| 3 | Depth-2 19-class | interleaved | *pending* | *pending* | ~4-5 min est. | 8x L40S |
| 4 | Full 1,328-class | interleaved | *pending* | *pending* | ~5-10 min est. | 8x L40S |

**Run 1 → Run 2 lesson:** Contiguous spatial split along AP axis excluded all Cerebrum pixels from val/test (Cerebrum is anterior-only in mouse brain). Interleaved split fixed this — every 10th slice goes to val/test, ensuring all brain regions appear in all splits.

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
│   ├── step9_finetune_coarse.ipynb      # Coarse 6-class training
│   ├── step10_finetune_depth2.ipynb     # Depth-2 19-class training
│   └── step10_finetune_full.ipynb       # Full 1,328-class training
├── scripts/
│   └── download_allen_data.py           # Local download of Allen Brain data
├── docs/                                # Design docs and progress tracking
│   ├── progress.md                      # Full project history
│   ├── step10_plan.md                   # Current step plan (for LLM continuity)
│   └── joyful-popping-planet.md         # Step 9 tracker
├── Makefile                             # Build, test, deploy commands (12 targets)
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
│  │ step9_finetune_coarse.ipynb    (6 classes, coarse)        │  │
│  │ step10_finetune_depth2.ipynb   (19 classes, depth-2)      │  │
│  │ step10_finetune_full.ipynb     (1,328 classes, full)      │  │
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
