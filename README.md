# Histological Image Analysis

Fine-tune DINOv2-Large + UperNet for semantic segmentation of brain structures in Nissl-stained histological sections. Uses Allen Brain Institute CCFv3 volumetric data (10um Nissl + annotations) to train a model that identifies anatomical brain regions at multiple granularities (coarse 6-class through fine 1,327-class), with the eventual goal of generalizing to autofluorescence imaging.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [uv](https://docs.astral.sh/uv/) | >= 0.4 | Python package manager |
| Python | >= 3.11 | Runtime |
| [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) | >= 0.278 | Deployment to Databricks |
| JFrog Artifactory access | -- | Model weights mirror (`facebook/dinov2-large`) |

**Databricks workspace:** `grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com`
**Cluster:** `0306-215929-ai2l0t8w` (g6e.16xlarge, NVIDIA L40S 48GB, 512GB RAM)

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
make test                   # 102 tests

# Lint
make lint                   # ruff check

# Build wheel
make build                  # -> dist/histological_image_analysis-0.1.0-py3-none-any.whl
```

## Databricks Deployment

### Step 1: Deploy

```bash
# Deploy everything (build wheel, upload to DBFS, upload notebook)
make deploy

# Or individually:
make deploy-wheel           # Build + upload wheel to DBFS
make deploy-notebook        # Upload notebook to workspace
```

### Step 2: Staged Validation (Cells 0-4)

Open the notebook on Databricks and run cells one at a time:

| Cell | What It Does | Success Criteria |
|------|-------------|-----------------|
| 0 | Install wheel from DBFS | No pip errors, kernel restarts |
| 1 | Load configuration | Paths print correctly |
| 2 | Download DINOv2-Large weights | `snapshot_download` completes |
| 3 | Build data pipeline | 6 coarse classes, slices load, sample shape `(3, 518, 518)` |
| 4 | Create model + forward pass | Logits shape `(1, 6, 518, 518)`, "Forward pass OK" |

Run `make validate` for a detailed checklist with expected outputs.

### Step 3: Full Training (Cells 5-7)

After cells 0-4 pass:

| Cell | What It Does |
|------|-------------|
| 5 | Train for 50 epochs with MLflow logging |
| 6 | Evaluate on val set, visualize predictions vs ground truth |
| 7 | Save final model to workspace |

## Project Structure

```
histological-image-analysis/
├── src/histological_image_analysis/     # Installable Python package
│   ├── ontology.py                      # Allen Brain structure ontology mapper
│   ├── ccfv3_slicer.py                  # CCFv3 3D volume -> 2D slice extraction
│   ├── svg_rasterizer.py                # SVG annotation -> pixel mask
│   ├── dataset.py                       # PyTorch Dataset (pad, crop, augment)
│   └── training.py                      # DINOv2 + UperNet model/trainer factories
├── tests/                               # 102 tests (pytest)
│   ├── test_ontology.py
│   ├── test_ccfv3_slicer.py
│   ├── test_svg_rasterizer.py
│   ├── test_dataset.py
│   └── test_training.py
├── notebooks/
│   └── step9_finetune_coarse.ipynb      # Databricks training notebook
├── scripts/
│   └── download_allen_data.py           # Local download of Allen Brain data
├── docs/                                # Design docs and progress tracking
├── Makefile                             # Build, test, deploy commands
├── .env.example                         # Environment variable template
├── pyproject.toml                       # Dependencies and build config
└── uv.lock                              # Locked dependencies
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Databricks Cluster                        │
│                                                             │
│  step9_finetune_coarse.ipynb (thin orchestration)           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Cell 0: %pip install wheel from DBFS                │    │
│  │ Cell 1: Configuration (paths, hyperparams)          │    │
│  │ Cell 2: Download DINOv2-Large from JFrog/HF         │    │
│  │ Cell 3: OntologyMapper -> CCFv3Slicer -> Dataset    │    │
│  │ Cell 4: create_model() + forward pass check         │    │
│  │ Cell 5: Trainer.train() (HF Trainer + MLflow)       │    │
│  │ Cell 6: Evaluate + matplotlib visualization         │    │
│  │ Cell 7: Save model to workspace                     │    │
│  └──────────────┬──────────────────────────────────────┘    │
│                 │ imports                                    │
│  ┌──────────────▼──────────────────────────────────────┐    │
│  │  histological_image_analysis (wheel on DBFS)        │    │
│  │                                                     │    │
│  │  ontology.py    -> Structure ID -> class mapping    │    │
│  │  ccfv3_slicer.py -> 3D NRRD -> 2D coronal slices   │    │
│  │  dataset.py     -> PyTorch Dataset (crop, augment)  │    │
│  │  training.py    -> DINOv2 + UperNet + Trainer       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Data on Databricks:                                        │
│  ├── /Workspace/.../ontology/structure_graph_1.json         │
│  ├── /Workspace/.../ccfv3/annotation_10.nrrd                │
│  └── /dbfs/FileStore/.../ara_nissl_10.nrrd (2.17 GB)       │
└─────────────────────────────────────────────────────────────┘
```

## Data

Training data comes from the Allen Brain Institute CCFv3 (Common Coordinate Framework v3):

- **Image volume:** `ara_nissl_10.nrrd` -- 1320 coronal slices of Nissl-stained mouse brain at 10um (float32, 1320x800x1140)
- **Annotation volume:** `annotation_10.nrrd` -- per-voxel structure IDs, 672 unique structures (uint32)
- **Ontology:** `structure_graph_1.json` -- hierarchical structure tree (1,327 structures)

Data was downloaded locally via `scripts/download_allen_data.py` and uploaded to Databricks Workspace/DBFS. See `docs/step5_6_completion_report.md` for details.

## References

See [docs/references.md](docs/references.md) for links to Allen Brain API, DINOv2, UperNet, and HuggingFace documentation.
