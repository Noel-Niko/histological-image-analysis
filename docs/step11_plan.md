# Plan: Step 11a — Unfreeze Backbone with Differential Learning Rate

## Context

Step 10 complete. Training results with frozen DINOv2-Large backbone, head-only training:

| Run | Granularity | mIoU | Overall Acc | Runtime | Cluster |
|-----|-------------|------|-------------|---------|---------|
| 2 | Coarse 6-class | 88.0% | 95.8% | 23 min | 1x L40S |
| 3 | Depth-2 19-class | 69.6% | 95.5% | 25 min | 1x L40S |
| 4 | Full 1,328-class | 60.3% | 88.9% | 5.5 hrs | 1x L40S |

**Primary bottleneck:** DINOv2 was pretrained on ImageNet-scale natural images. Nissl-stained brain tissue has fundamentally different visual characteristics — cell density patterns, laminar organization, staining gradients. Frozen features cannot capture these domain-specific patterns.

**Expected improvement:** 5-15% mIoU gain based on histology transfer learning literature. This is consistently the largest single factor in domain-shifted segmentation tasks.

---

## Strategy: Partial Fine-Tuning (Last 4 Blocks)

DINOv2-Large has 24 transformer blocks. Unfreeze only blocks 20-23 to:
- Adapt high-level features to the Nissl histology domain
- Preserve low-level feature extraction (edges, textures) in early layers
- Minimize overfitting risk on 1,016 training samples

| Layer range | Trainable? | Rationale |
|-------------|-----------|-----------|
| `backbone.embeddings.*` | Frozen | Generic patch/position embeddings |
| `backbone.encoder.layer.{0-19}.*` | Frozen | General visual features |
| `backbone.encoder.layer.{20-23}.*` | **Trainable** | Domain-specific adaptation |
| `backbone.layernorm.*` | **Trainable** | Adapts to new feature distribution |
| `decode_head.*` | **Trainable** | Task-specific decoding |
| `auxiliary_head.*` | **Trainable** | Auxiliary loss head |

### Differential Learning Rate

Two parameter groups with 10x LR difference:

| Group | Parameters | LR |
|-------|-----------|-----|
| Backbone (fine-tune) | `backbone.encoder.layer.{20-23}.*`, `backbone.layernorm.*` | 1e-5 |
| Head (full training) | `decode_head.*`, `auxiliary_head.*` | 1e-4 |

### Training Configuration

```
optimizer: AdamW
backbone LR: 1e-5
head LR: 1e-4
weight_decay: 0.01
warmup_ratio: 0.1
num_epochs: 100 (doubled from 50)
early_stopping: load_best_model_at_end=True, metric_for_best_model="mean_iou"
fp16: True
batch_size: 2 (minimum — batch=1 fails on UperNet PSP BatchNorm)
gradient_accumulation_steps: 2 (effective batch = 4)
gradient_checkpointing: backbone only, use_reentrant=False (REQUIRED for frozen/unfrozen boundary)
ddp_find_unused_parameters: True (Databricks wraps in DDP even on single-GPU)
mapping: full (1,328 classes)
split: interleaved
cluster: single GPU (1x L40S 48 GB)
```

### Memory Budget (Full Mapping, Unfrozen Last 4 Blocks)

| Component | Frozen Backbone (Run 4) | + Unfrozen Blocks 20-23 |
|-----------|------------------------|------------------------|
| Backbone weights (fp16) | 0.6 GB | 0.6 GB |
| Backbone activations (blocks 20-23) | 0 (not stored) | ~0.8 GB |
| Backbone gradients (blocks 20-23) | 0 | ~0.2 GB |
| Optimizer states (blocks 20-23) | 0 | ~0.4 GB |
| UperNet head + logits + gradients | ~23 GB | ~23 GB |
| **Total estimate** | **~24 GB** | **~25-26 GB** |
| **Headroom on L40S 48 GB** | **~24 GB** | **~22-23 GB** |

**Actual results (L40S 48 GB):**
- Batch=4: **OOM** — driver killed (Py4J ConnectionResetError)
- Batch=1 + grad_accum=4: **ValueError** — UperNet PSP module uses 1x1 adaptive
  average pool → BatchNorm receives `[1, 512, 1, 1]` → only 1 value per channel →
  PyTorch raises `Expected more than 1 value per channel when training`
- **Batch=2 + grad_accum=2: works** — effective batch=4, BatchNorm gets 2 values per channel

Additional memory optimizations applied:
- `model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`
  - Backbone only — UperNet raises ValueError if called on outer model
  - **MUST use `use_reentrant=False`** — the default reentrant checkpoint breaks gradient
    flow at the frozen/unfrozen boundary (block 19→20). Frozen blocks produce outputs
    without `requires_grad`, and the reentrant checkpoint silently drops gradient tracking
    for the unfrozen blocks. Symptom: DDP RuntimeError "parameters not used in producing
    loss" with 72 parameter indices (= the unfrozen backbone params).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `ddp_find_unused_parameters=True` — Databricks wraps models in DDP even on single-GPU
  clusters; frozen params need this tolerance

### Overfitting Mitigation

Risk: 1,016 training samples with ~50M additional trainable parameters (blocks 20-23).

Mitigations already in place:
1. **Early stopping** — `load_best_model_at_end=True` on val mIoU
2. **Weight decay** — 0.01 (AdamW)
3. **Low backbone LR** — 1e-5 (10x lower than head)
4. **Only 4 blocks unfrozen** — 83% of backbone remains frozen
5. **Existing augmentation** — horizontal flip, rotation ±15°, color jitter

---

## Implementation

### Approach: Notebook-only (no source file changes)

All customization happens in the new notebook `finetune_unfrozen.ipynb`. The existing `training.py` is not modified — the user's preference is to create new files rather than change working code.

The key technique: HF Trainer accepts an `optimizers` tuple `(optimizer, lr_scheduler)`. By building the optimizer in the notebook with parameter groups, we get differential LR without touching any library code.

### What changes from `finetune_full.ipynb`

| Cell | Change |
|------|--------|
| 0 (Install) | No change |
| 1 (Config) | `freeze_backbone=False`, `backbone_lr=1e-5`, `head_lr=1e-4`, `num_epochs=100`, output dirs → `unfrozen`, MLflow run name updated |
| 2 (Download) | No change |
| 3 (Data) | No change |
| 4 (Model) | `create_model(freeze_backbone=False)`, then selectively freeze blocks 0-19, print trainable param counts, `torch.cuda.empty_cache()` after forward pass sanity check |
| 5 (Train) | Build custom AdamW optimizer with 2 param groups, build linear warmup scheduler, pass `optimizers=(optimizer, scheduler)` to `create_trainer()` |
| 6 (Eval) | No change |
| 7 (Save) | Update model dir to `unfrozen` |

### LESSONS_LEARNED.md Patterns Applied

- [x] Route downloads through JFrog Artifactory with retry pattern
- [x] `etag_timeout=86400`
- [x] Single `mlflow.start_run()` spanning all cells
- [x] `mlflow.end_run()` at notebook end
- [x] `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true`
- [x] Checkpoints to `/tmp/`, final model to `/dbfs/`
- [x] **Gradient checkpointing on backbone only** with `use_reentrant=False` (UperNet outer model doesn't support it, but DINOv2 backbone does; must use non-reentrant for frozen/unfrozen boundary)
- [x] **Single GPU target** (multi-GPU DDP adds per-GPU overhead)
- [x] `torch.cuda.empty_cache()` after sanity-check forward pass
- [x] `remove_unused_columns=False` for dict-style datasets

---

## Files to Create/Modify

| File | Action | What Changes |
|------|--------|--------------|
| `notebooks/finetune_unfrozen.ipynb` | Create | 8 cells, unfrozen backbone, differential LR, 100 epochs |
| `Makefile` | Modify | Add `deploy-notebook-unfrozen` target, update `deploy` |
| `README.md` | Modify | Add unfrozen notebook to training granularities table |
| `docs/progress.md` | Modify | Step 11 status |
| `docs/step11_plan.md` | Create | This plan |

**Files NOT modified:** `training.py`, `dataset.py`, `ccfv3_slicer.py`, `ontology.py`, `pyproject.toml`.

---

## Implementation Status

| Step | Task | Status |
|------|------|--------|
| 0 | Create step 11 plan | DONE |
| 1 | Create `finetune_unfrozen.ipynb` notebook | DONE — 8 cells |
| 2 | Update Makefile | DONE — `deploy-notebook-unfrozen` target added |
| 3 | Update README.md | DONE — notebook table, deploy commands, project structure, architecture |
| 4 | Update docs/progress.md | DONE |
| 5 | Run `make test` — verify 133/133 pass | DONE — **133/133 pass** |
| 6 | Deploy + train on Databricks | DONE — **Run 5: mIoU 68.8% (+8.5%)** |
| 7 | Review results, update docs | DONE |
| 8 | Fix missing MLflow artifact logging | DONE — all 4 notebooks |

---

## Actual Results (Run 5)

| Metric | Frozen (Run 4) | Unfrozen (Run 5) | Delta |
|--------|---------------|-------------------|-------|
| **mIoU** | 60.3% | **68.8%** | **+8.5%** |
| Overall Accuracy | 88.9% | — | — |
| Classes with valid IoU | 503 / 1,328 | 503 / 1,328 | — |
| Training time | 5.5 hrs | 11.3 hrs | ~2x |
| Training loss | — | 0.730 | — |
| Epochs | 50 | 100 | — |

+8.5% mIoU gain — within the expected 5-15% range. Backbone unfreezing confirmed
as the highest-impact single change for domain-shifted segmentation.

### Deployment Issues Encountered (3 attempts)

1. **OOM at batch=4** — Py4J ConnectionResetError (Databricks driver killed)
2. **ValueError at batch=1** — UperNet PSP BatchNorm needs ≥2 values per channel
3. **DDP RuntimeError at batch=2** — `use_reentrant=True` (default) broke gradient flow at frozen/unfrozen boundary

All resolved. Final config: batch=2, grad_accum=2, `use_reentrant=False`, `ddp_find_unused_parameters=True`.

### Missing MLflow Artifact (post-training fix)

Cell 7 saved model to DBFS but never called `mlflow.log_artifacts()`. Model was on
disk but invisible in MLflow. Retroactively logged via separate notebook
(`mlflow_log_unfrozen_model_with-failed-register-step.ipynb`). All 4 notebooks
now fixed with `mlflow.log_artifacts()` before `mlflow.end_run()`.

## Next Steps

- **Step 11b:** Class-weighted Dice + CE loss — address class imbalance (Background/class 0 dominates)
- **Step 11c:** Extended augmentation (elastic deformation, stain normalization)
- **Step 11d:** Hierarchical loss
- See `docs/finetuning_recommendations.md` for full Phase 1-4 roadmap
