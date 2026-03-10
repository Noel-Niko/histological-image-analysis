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

## Databricks Deployment & Validation

### Staged Validation — ✅ PASSED (2026-03-10)

All cells 0-4 passed on cluster `0306-215929-ai2l0t8w` (g6e.16xlarge, L40S 48GB):
- Cell 0: `%pip install` — wheel installed, kernel restarted
- Cell 1: Config — all paths correct, MLflow experiment created (ID: 1345391216675532)
- Cell 2: JFrog download — DINOv2-Large downloaded via Artifactory, cached at `/tmp/dinov2-large/`
- Cell 3: Data pipeline — 6 coarse classes, 1320 slices, train=1016 / val=127 / test=127, sample shape `(3,518,518)`
- Cell 4: Forward pass — logits `(1,6,518,518)`, "Forward pass OK"

### Deployment Issues Encountered & Fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `databricks workspace import` error: "accepts 1 arg" | CLI v0.278 uses `--file` flag, not positional arg | `--file $(NOTEBOOK_SRC)` |
| "parent folder does not exist" | `/histology/notebooks/` dir didn't exist on workspace | Added `databricks workspace mkdirs` before import |
| `SafetensorError: File too large (os error 27)` | Workspace 500 MB per-file limit; DINOv2+UperNet ~1.2 GB | `OUTPUT_DIR=/tmp/...`, `FINAL_MODEL_DIR=/dbfs/...` |
| `skip logging GPU metrics` | `pynvml` not accessible for MLflow system metrics | Informational only, harmless |
| `find_unused_parameters=True` DDP warning | Trainer wraps model in DDP even on single GPU | Harmless warning, standard behavior |

---

## First Training Run — Results (2026-03-10)

**Config:** Coarse 6-class, frozen DINOv2-Large backbone, UperNet head-only, 50 epochs, batch_size=8, lr=1e-4, fp16=True

**Training:**
- 6,350 global steps, training loss: 0.217
- Runtime: ~23 minutes on L40S 48GB
- Throughput: 36.3 samples/sec, 4.5 steps/sec

**Evaluation:**

| Metric | Value |
|--------|-------|
| Overall accuracy | 78.8% |
| Mean IoU | 50.7% |
| Eval loss | 1.238 |

| Class | Name | IoU |
|-------|------|-----|
| 0 | Background | 53.5% |
| 1 | **Cerebrum** | **NaN** |
| 2 | Brain stem | 74.1% |
| 3 | Cerebellum | 70.1% |
| 4 | Fiber tracts | 41.1% |
| 5 | Ventricular systems | 14.7% |

**Model saved to:** `/dbfs/FileStore/allen_brain_data/models/coarse_6class`
**MLflow experiment:** `/Users/noel.nosse@grainger.com/histology-brain-segmentation`

### Critical Issue: Cerebrum (class 1) IoU is NaN

Cerebrum is the **largest brain region** — 637 of 1,327 structures map to it in the coarse mapping. NaN IoU means the model either never predicts class 1, or never correctly predicts it. Since Cerebrum dominates the volume, the most likely cause is that the model labels all cerebrum pixels as Background (class 0). This is a class imbalance / representation problem — the frozen backbone features may not discriminate cerebrum from background well enough with head-only training.

Other weak classes: Ventricular systems (14.7%) and fiber tracts (41.1%) — both are small structures that are hard to segment at coarse level.

---

## Lessons Learned (Databricks-specific)

See also `ml-workflow-tool/training_templates/LESSONS_LEARNED.md` for general patterns.

1. **Workspace has 500 MB per-file limit** — checkpoints and model saves MUST use `/tmp/` (local, ephemeral) or `/dbfs/` (persistent, no limit). Workspace paths fail with `SafetensorError: File too large`.
2. **`databricks workspace import` CLI syntax** — v0.278+ uses `TARGET_PATH --file LOCAL_PATH` (not two positional args). Parent dirs must exist — use `databricks workspace mkdirs` first.
3. **JFrog Artifactory** — route all model downloads through `https://graingerreadonly.jfrog.io/...`. Always use retry loop + `etag_timeout=86400`.
4. **MLflow single run** — use `mlflow.start_run()` before training, `mlflow.end_run()` after save. HF Trainer's `MLflowCallback` can leave runs open.
5. **GPU metrics** — `pynvml` may not be accessible; "skip logging GPU metrics" is informational only.
6. **DDP warning** — `find_unused_parameters=True` is standard HF Trainer behavior on single GPU, harmless.

---

## Cerebrum NaN Diagnosis — ROOT CAUSE FOUND (2026-03-10)

### Finding: Contiguous spatial split excludes Cerebrum from val/test

The `get_split_indices()` method in `ccfv3_slicer.py` splits valid AP indices into contiguous blocks (first 80% → train, next 10% → val, last 10% → test). Because the mouse brain cerebrum is concentrated in the **anterior** portion:

| Split | Cerebrum pixels | Brain stem | Cerebellum |
|-------|----------------|------------|------------|
| train (406 slices) | **17.5M** | 5.7M | 261K |
| val (50 slices) | **0** | 1.1M | 1.8M |
| test (52 slices) | **0** | 879K | 1.4M |

- **Cerebrum (class 1):** 100% in train, 0% in val/test → NaN IoU during evaluation
- **Cerebellum (class 3):** Only 261K in train, 3.2M in val/test → model gets good eval IoU despite tiny training set

### Verified facts:
- `compute_metrics` NaN means class absent from ground truth (confirmed by 3 new tests)
- Real ontology mapping is correct: 637 structures → Cerebrum, 322 of those appear in annotation volume
- Real annotation volume has 17.5M Cerebrum pixels (22.9% of total volume)
- The contiguous spatial split is the sole cause

### Fix: Interleaved split strategy (approved by user)
Every 10th valid slice → val, every 10th+1 → test, rest → train. This ensures ALL coarse classes appear in ALL splits. Minor spatial leakage between adjacent slices (accepted tradeoff).

### Implementation:
- ✅ DONE — Diagnostic tests confirming root cause (`tests/test_ontology.py`)
- ✅ DONE — New `compute_metrics` NaN behavior tests (`tests/test_training.py`)
- ✅ DONE — Add `split_strategy="interleaved"` to `get_split_indices()` + `iter_slices()` + `BrainSegmentationDataset`
- ✅ DONE — 7 new interleaved split tests in `tests/test_ccfv3_slicer.py`
- ✅ DONE — Real-data validation: interleaved split gives 1.75M Cerebrum pixels in val (vs 0 with spatial)
- ✅ DONE — Updated notebook Cell 1 (added `split_strategy` to HYPERPARAMS) and Cell 3 (interleaved split + class distribution check)
- ✅ DONE — Redeployed wheel + retrained on Databricks (2026-03-10)

---

## Second Training Run — Results (2026-03-10, interleaved split)

**Config:** Same as Run 1 but with `split_strategy="interleaved"`.

**Training:**
- 6,350 global steps, training loss: 0.240
- Runtime: ~23 minutes on L40S 48GB (1395.5s)
- Throughput: 36.4 samples/sec, 4.6 steps/sec

**Evaluation:**

| Metric | Run 1 (spatial) | Run 2 (interleaved) | Change |
|--------|----------------|---------------------|--------|
| Overall accuracy | 78.8% | **95.8%** | +17.0 |
| Mean IoU | 50.7% | **88.0%** | +37.3 |
| Eval loss | 1.238 | **0.175** | -85.9% |

| Class | Name | Run 1 IoU | Run 2 IoU | Change |
|-------|------|-----------|-----------|--------|
| 0 | Background | 53.5% | **92.1%** | +38.6 |
| 1 | Cerebrum | **NaN** | **95.8%** | Fixed |
| 2 | Brain stem | 74.1% | **94.3%** | +20.2 |
| 3 | Cerebellum | 70.1% | **92.9%** | +22.8 |
| 4 | Fiber tracts | 41.1% | **73.0%** | +31.9 |
| 5 | Ventricular systems | 14.7% | **80.1%** | +65.4 |

**Val class distribution (verified in Cell 3):**
- Background: 4,561,560 | Cerebrum: 12,052,492 | Brain stem: 10,452,870
- Cerebellum: 3,229,700 | Fiber tracts: 3,288,162 | VS: 492,364

**Key findings:**
- Training loss was similar (0.240 vs 0.217) — the model learned comparably. Run 1's poor eval metrics were **entirely caused by the broken spatial split**, not by model quality.
- Cerebrum is now the **best-performing class** (95.8% IoU), consistent with it being the largest brain region.
- Fiber tracts (73.0%) and VS (80.1%) still weakest — these are small, thin structures that are inherently harder to segment.
- Eval loss dropped from 1.238 to 0.175 — the model generalizes well when train/val have similar class distributions.

**Model saved to:** `/dbfs/FileStore/allen_brain_data/models/coarse_6class` (overwritten)

---

## Next Steps

### Immediate: Evaluate + decide direction
1. ~~Diagnose Cerebrum NaN~~ **DONE**
2. ~~Implement interleaved split~~ **DONE**
3. ~~Retrain with interleaved split~~ **DONE — mIoU 88.0%**
4. **Evaluate on test split** — 127 held-out slices, compare with val metrics
5. **Decide: improve coarse or move to fine granularity?**
   - Coarse model is strong (88% mIoU). May be diminishing returns to push higher.
   - Weak spots: fiber tracts 73%, VS 80% — class-weighted loss could help.
   - Moving to finer granularity tests pipeline scaling and is closer to the final goal.

### Phase 2: Improve Coarse Model (optional)
6. **Class-weighted loss** — inverse-frequency weighting for fiber tracts and VS
7. **Unfreeze backbone** — lower LR for backbone (1e-5) vs head (1e-4)

### Phase 3: Fine Granularity
8. **Increase to depth-2 mapping** (~20-50 classes) — test the pipeline's scaling
9. **Full mapping** (~1,327 classes) — the final target per Step 7 plan
10. **Autofluorescence domain** — train on 25μm template volume (the ultimate target domain)

### Phase 4: Evaluation
11. **Mouse Atlas 2D validation** — evaluate on held-out 2D atlas sections with SVG annotations
12. **Per-structure IoU analysis** — identify which structures are well/poorly segmented

## Verification

1. `make test` — **124/124 tests pass** (39 training + 29 slicer + 38 ontology + 18 dataset+svg)
2. `make build` — wheel builds with all 5 modules
3. `make deploy` — uploads wheel to DBFS + notebook to workspace
4. Staged validation cells 0-4 — ✅ PASSED
5. Full training run 1 (spatial split, 50 epochs) — ✅ COMPLETED, mIoU 50.7%, Cerebrum NaN
6. Cerebrum NaN diagnosis — ✅ ROOT CAUSE FOUND, fix implemented and validated with real data
7. Full training run 2 (interleaved split, 50 epochs) — ✅ COMPLETED, **mIoU 88.0%**, all classes present

### Test additions for Cerebrum NaN diagnosis:
- `tests/test_training.py`: 3 new tests (NaN behavior: class absent→NaN, class present but unpredicted→0.0, 0.0 included in mean)
- `tests/test_ontology.py`: 12 new tests (7 real ontology mapping, 5 real annotation distribution incl. interleaved validation)
- `tests/test_ccfv3_slicer.py`: 7 new tests (interleaved split: no overlap, coverage, fractions, distribution, regularity, invalid strategy, backwards compat)
