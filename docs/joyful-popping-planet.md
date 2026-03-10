# Step 9 Implementation Plan: Fine-tuning DINOv2-Large + UperNet

## Context

Step 8 (training data pipeline) is COMPLETE — 66 tests pass, 4 components built. Now we need the training infrastructure: a reusable `training.py` module with model creation, metrics, and trainer wiring, plus a thin Databricks notebook for orchestration.

**User decisions:** Coarse 6 classes first, 10μm Nissl data, `facebook/dinov2-large` added to JFrog.

---

## Implementation Checklist

### 1. ✅ DONE — Add `huggingface-hub` to `pyproject.toml`

Added `huggingface-hub>=0.20` to core dependencies. `uv lock && uv sync` ran.

### 2. ✅ DONE — Write tests first: `tests/test_training.py` (36 tests)

**Tiny model config (no downloads):**
```python
Dinov2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
             intermediate_size=128, image_size=64, patch_size=8,
             out_features=["stage1","stage2","stage3","stage4"],
             reshape_hidden_states=True, apply_layernorm=True)
```

**Test classes:**
- `TestCreateModel` — returns `UperNetForSemanticSegmentation`, correct `num_labels`, has auxiliary head, correct `loss_ignore_index`
- `TestBackboneFreeze` — frozen backbone has `requires_grad=False`, head always trainable, param count decreases when frozen
- `TestForwardPass` — produces loss + finite loss, logits shape `(B, num_labels, H, W)`, no loss without labels, ignore_index=255 pixels produce zero loss
- `TestComputeMetrics` — perfect mIoU=1.0, zero mIoU=0.0, known hand-calculated IoU values, ignore_index exclusion, absent classes excluded from mean, overall accuracy, per-class keys
- `TestPreprocessLogitsForMetrics` — reduces `(B,C,H,W)` → `(B,H,W)`, correct argmax, int64 output
- `TestMakeComputeMetrics` — returns callable, closure captures num_labels
- `TestGetTrainingArgs` — returns `TrainingArguments`, `remove_unused_columns=False`, `fp16=True`, custom LR, kwargs passthrough
- `TestCreateTrainer` — returns `Trainer` instance

### 3. ✅ DONE — Implement `src/histological_image_analysis/training.py`

All 6 functions implemented. **36/36 tests pass, 102/102 full suite passes.**

Bugs fixed during implementation:
- `auxiliary_in_channels` in `UperNetConfig` must match backbone `hidden_size` (was defaulting to 384)
- BatchNorm in UperNet PPM layer requires batch_size>1 in training mode — tests use `model.eval()` for single-sample tests
- accelerate `PartialState` singleton retains fp16 state across tests — trainer test uses `use_cpu=True`

**Functions (in implementation order):**

**`preprocess_logits_for_metrics(logits, labels) -> Tensor`**
- `logits.argmax(dim=1)` — reduces `(B,C,H,W)` to `(B,H,W)` to prevent eval OOM

**`compute_metrics(eval_pred, num_labels, ignore_index=255) -> dict`**
- Manual numpy mIoU: flatten, mask ignore_index, per-class intersection/union/IoU
- Returns `{"mean_iou", "overall_accuracy", "iou_class_0", ..., "iou_class_N"}`
- Classes absent from ground truth excluded from mean (standard mIoU convention)

**`make_compute_metrics(num_labels, ignore_index) -> callable`**
- Closure factory — HF Trainer expects `f(EvalPrediction) -> dict` (single arg)
- Captures `num_labels` without global variables

**`create_model(num_labels, freeze_backbone=True, pretrained_backbone_path=None, backbone_config=None, ...) -> UperNetForSemanticSegmentation`**
- Two paths: `backbone_config` (testing, no download) vs `pretrained_backbone_path` (production, from JFrog)
- Production: `Dinov2Config.from_pretrained(path, out_features=["stage6","stage12","stage18","stage24"], reshape_hidden_states=True)`
- Creates `UperNetConfig(backbone_config=..., num_labels=N, hidden_size=512, use_auxiliary_head=True, auxiliary_loss_weight=0.4, loss_ignore_index=255)`
- Creates model with random weights, then loads pretrained backbone: `model.backbone.load_state_dict(pretrained_backbone.state_dict())`
- If `freeze_backbone`: sets `requires_grad=False` on all backbone params
- Logs total/trainable/frozen param counts

**`get_training_args(output_dir, ...) -> TrainingArguments`**
- Defaults: `batch_size=8, lr=1e-4, epochs=50, fp16=True, warmup_ratio=0.1`
- **Critical:** `remove_unused_columns=False`, `label_names=["labels"]`
- `metric_for_best_model="mean_iou"`, `report_to="mlflow"`, `save_total_limit=3`

**`create_trainer(model, training_args, train_dataset, eval_dataset, num_labels) -> Trainer`**
- Wires `make_compute_metrics` closure + `preprocess_logits_for_metrics`

### 4. ✅ DONE — Create `notebooks/step9_finetune_coarse.ipynb`

Thin Databricks notebook — 8 cells (0-7). Revised per `ml-workflow-tool/training_templates/LESSONS_LEARNED.md`:

| Cell | Content |
|------|---------|
| 0 | `%pip install` wheel from DBFS + `dbutils.library.restartPython()` |
| 1 | Config: MLflow experiment setup, `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`, paths, `HYPERPARAMS` dict, JFrog Artifactory URL as default |
| 2 | JFrog download with **retry loop** (3 attempts, exponential backoff) + `etag_timeout=86400` |
| 3 | Pipeline: `OntologyMapper` → `CCFv3Slicer.load_volumes()` → `BrainSegmentationDataset` (train + val) |
| 4 | Model: `create_model(num_labels=6, freeze_backbone=True, pretrained_backbone_path=model_path)` |
| 5 | Training: `mlflow.start_run()` + `mlflow.log_params(HYPERPARAMS)` → `trainer.train()` |
| 6 | Eval: `trainer.evaluate()` + `mlflow.log_metrics()` for final metrics + matplotlib pred vs GT grid |
| 7 | Save: `trainer.save_model()` + `mlflow.end_run()` to close the run |

**Fixes applied from LESSONS_LEARNED.md:**
- Retry pattern on `snapshot_download` (Artifactory drops connections)
- `etag_timeout=86400` (avoid unnecessary HEAD requests)
- JFrog Artifactory URL as default (not generic `huggingface.co`)
- `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true` (GPU/CPU/memory tracking)
- Single `mlflow.start_run()` spanning cells 5-7 (no split runs)
- `mlflow.log_params(HYPERPARAMS)` (log all hyperparams)
- `mlflow.end_run()` in Cell 7 (per "HF Trainer leaves active runs" lesson)
- Removed duplicate stale config cell with wrong `1.json` path

### 5. ✅ DONE — Create `docs/references.md`

Reference URLs for Allen Brain API, HuggingFace models, DINOv2 docs, UperNet source code.

### 6. ✅ DONE — Update `docs/progress.md`

Mark Step 9 as in-progress, update compact context.

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `pyproject.toml` | Add `huggingface-hub>=0.20` |
| `tests/test_training.py` | Create (~30 tests, TDD) |
| `src/histological_image_analysis/training.py` | Create (6 functions) |
| `notebooks/step9_finetune_coarse.ipynb` | Create (7 cells) |
| `docs/references.md` | Create (URL reference list) |
| `docs/progress.md` | Update Step 9 status |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `backbone_config` param for tests | Avoids HuggingFace downloads in unit tests |
| Manual mIoU (no `evaluate` library) | 15-line numpy function vs adding a dependency |
| `make_compute_metrics` closure | Injects `num_labels` without global variables (per CLAUDE.md) |
| `preprocess_logits_for_metrics` | Prevents eval OOM by argmaxing before accumulating |
| JFrog download in notebook only | Environment-specific; not reusable across environments |
| `remove_unused_columns=False` | Critical — Trainer defaults to True and strips dict keys |

## Post-Implementation: Deployment Infrastructure

Added after Steps 1-6 were complete:

### ✅ DONE — Bug fix: ONTOLOGY_PATH in notebook
Changed `1.json` to `structure_graph_1.json` (matches actual uploaded filename).

### ✅ DONE — Add %pip install cell to notebook
Inserted Cell 0 with `%pip install /dbfs/FileStore/wheels/...whl` + `dbutils.library.restartPython()`.

### ✅ DONE — Create `.env.example`
Variables: `DATABRICKS_PROFILE`, `HF_ENDPOINT`, `WORKSPACE_BASE`, `DBFS_WHEEL_DIR`.

### ✅ DONE — Create `Makefile`
Targets: `install`, `test`, `lint`, `build`, `clean`, `deploy-wheel`, `deploy-notebook`, `deploy`, `validate`, `help`.

### ✅ DONE — Rewrite `README.md`
Full project docs: overview, prerequisites, local dev, Databricks deployment, staged validation, project structure, architecture diagram.

## Databricks Validation

Staged validation via `make deploy` then running cells 0-4 on the cluster:
- Cell 0: `%pip install` wheel from DBFS + kernel restart
- Cell 1: Configuration paths + hyperparameters
- Cell 2: JFrog/HuggingFace model download
- Cell 3: Data pipeline (ontology → slicer → dataset)
- Cell 4: Model creation + forward pass sanity check

Run `make validate` for the full checklist with expected outputs.

## Verification

1. `make test` — 102/102 tests pass (36 training + 66 existing)
2. `make build` — wheel builds with all 5 modules
3. `make deploy` — uploads wheel to DBFS + notebook to workspace
4. Cells 0-4 on Databricks — staged validation (pending)
