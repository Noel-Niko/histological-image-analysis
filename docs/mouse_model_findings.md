# Mouse Model Findings — Transfer Learnings for Human Finetuning

**Document Purpose:** Comprehensive record of all findings from mouse brain segmentation experiments (Runs 1-8a) to inform the human brain model training strategy. Written 2026-03-16 after completing the full ablation study.

**Context:** This project trains two separate models: (1) a mouse model using Allen Mouse Brain Atlas CCFv3 Nissl data (1,320 coronal slices, 1,328 brain structures), and (2) a human model using Allen Human Brain Atlas data (14,565 Nissl sections, 6 donors, ground truth TBD). This document captures the mouse model lessons to accelerate human model development.

---

## Summary: What Works and What Doesn't

### Critical Success Factors ✓

1. **Backbone unfreezing is essential** — Frozen backbone: 60.3% mIoU. Unfrozen (last 4 of 24 blocks): 68.8% mIoU (+8.5%). This is the single largest improvement.
2. **Simple augmentation is sufficient** — Baseline (flip + rot15° + jitter) works. Extended augmentation (rot90, elastic, blur) hurts (−6.5%).
3. **Cross-entropy loss is optimal** — Weighted Dice+CE underperforms plain CE (−1.6%).
4. **Interleaved train/val/test split** — Spatial split causes distribution shift (50.7% → 88.0% mIoU switch).
5. **100+ epochs needed** — Val loss still declining at epoch 100 for all unfrozen runs.

### Proven Failures ✗

1. **Rotational augmentation** — TTA with rot90: −24% mIoU. Model is orientation-specific.
2. **Class pruning** — Removing 655 zero-pixel classes: +0.2% mIoU (within noise). Softmax dilution is not a problem.
3. **Multi-axis training (high risk)** — Cross-axis diagnostic: coronal 68.9%, axial 3.2%, sagittal 0.5%. Non-coronal data would act as noise.
4. **Aggressive augmentation** — Extended preset (rot90, elastic deformation α=50, blur): −6.5% mIoU.

---

## Architecture & Training Configuration

### Model: DINOv2-Large + UperNet

- **Backbone:** `facebook/dinov2-large` (304M params, 24 transformer blocks)
- **Decode head:** UperNet with PSP pooling (37.2M params for 673 classes, 37.7M for 1,328)
- **Total params:** 342M (341M for pruned)
- **Trainable (unfrozen):** 87.6M (25.7% of total) — last 4 backbone blocks + full decode head

**Why this architecture:**
- DINOv2 is pretrained on natural images with self-supervised learning (strong feature extraction)
- UperNet handles multi-scale features via PSP and FPN pathways
- No domain-specific pretraining on histology — this is pure transfer learning

### Optimal Hyperparameters (Run 5 config)

```python
# Optimizer
AdamW with 2 parameter groups:
  backbone_lr = 1e-5
  decode_head_lr = 1e-4
weight_decay = 0.01
warmup_ratio = 0.1

# Training
batch_size = 2 (per GPU)
gradient_accumulation_steps = 2  # Effective batch = 4
epochs = 100  # Run 9 extends to 200
fp16 = True
gradient_checkpointing = True (use_reentrant=False)

# Data
train_samples = 1,016 (coronal only, interleaved split)
val_samples = 127
image_size = 518×518 (random crops from 800×1140 slices)

# Loss
loss_fn = CrossEntropyLoss (plain, unweighted)

# Augmentation (baseline preset)
- Horizontal flip: 50%
- Rotation: ±15° (always)
- Color jitter: brightness/contrast/saturation ±0.1 (always)
```

### Critical Training Constraints

These are hard requirements that must be followed for UperNet to train successfully on Databricks:

| Constraint | Reason | Solution |
|------------|--------|----------|
| `batch_size ≥ 2` | PSP BatchNorm fails with `batch_size=1` (single-sample batch norm is undefined) | Always use `per_device_train_batch_size=2` minimum |
| `dataloader_drop_last=True` for odd sample counts | When `total_samples % batch_size == 1`, last batch has 1 sample → same PSP crash | Add to `TrainingArguments` when training samples are odd |
| `ddp_find_unused_parameters=True` | Databricks wraps in DDP even on single GPU; auxiliary head params unused during training | Always pass in `TrainingArguments` |
| `use_reentrant=False` for gradient checkpointing | At frozen/unfrozen boundary, reentrant mode breaks gradient flow | Use when calling `backbone.gradient_checkpointing_enable()` |
| NEVER use `mlflow.transformers.log_model()` | UperNet is not a Pipeline and not in AutoModel registry | Use `mlflow.log_artifacts(MODEL_DIR, artifact_path="model")` instead |

---

## Empirical Findings by Category

### 1. Backbone Unfreezing (Critical)

**Runs 4 → 5:** Frozen backbone (60.3%) → Last 4 blocks unfrozen (68.8%) = **+8.5% mIoU**

| Config | mIoU | Accuracy | Trainable Params | Notes |
|--------|------|----------|------------------|-------|
| Frozen | 60.3% | 91.3% | 15M (4.4%) | Only decode head trained |
| Unfrozen (last 4) | 68.8% | 92.5% | 88M (25.7%) | Best config |

**Learning rate strategy:**
- Backbone blocks: 1e-5 (10× lower than head)
- Decode head: 1e-4
- This 10:1 ratio prevents catastrophic forgetting of DINOv2 features

**For human model:**
- Start with unfrozen backbone from the beginning (skip the frozen baseline unless compute is very limited)
- Consider starting from Run 5 mouse weights (`UperNetForSemanticSegmentation.from_pretrained()`) if human structures have anatomical overlap with mouse (e.g., cortical layers, hippocampus, cerebellum)

### 2. Loss Functions

**Run 6 (weighted Dice+CE):** 67.2% mIoU = **−1.6%** vs Run 5 (plain CE at 68.8%)

```python
# Run 5 (best): Plain CE
loss = CrossEntropyLoss()

# Run 6 (worse): Weighted combo
dice_loss = DiceLoss(mode='multiclass', ignore_index=IGNORE_INDEX)
ce_loss = CrossEntropyLoss(weight=class_weights)  # inverse frequency weights
loss = 0.5 * dice_loss + 0.5 * ce_loss
```

**Why weighted loss failed:**
- Dice loss dilutes gradients on high-frequency classes (which are already well-learned)
- Class weights based on inverse frequency over-penalize errors on tiny structures
- The dataset is already balanced enough for plain CE to work

**For human model:**
- Start with plain CE
- Only try weighted loss if initial runs show severe class imbalance (>100:1 ratio in pixel counts)

### 3. Augmentation Strategy

**Run 7 (extended aug):** 62.3% mIoU = **−6.5%** vs Run 5 (baseline at 68.8%)

| Preset | Transforms | mIoU | Delta |
|--------|-----------|------|-------|
| Baseline (Run 5) | flip (50%), rot ±15°, jitter | 68.8% | — |
| Extended (Run 7) | + rot90 (50%), elastic α=50, blur σ=0.5-2.0 | 62.3% | −6.5% |

**Why extended augmentation failed:**
1. **Rotation 90°:** Introduces orientations the model never sees in real data (coronal slices have consistent dorsal-ventral axis). Model lacks rotational equivariance (confirmed by TTA: −24% with rot90).
2. **Elastic deformation (α=50, σ=5):** Warps anatomical boundaries beyond biologically plausible variation. Small structures become unrecognizable.
3. **Too many transforms:** 6 transforms applied sequentially heavily distort every sample.

**For human model:**
- Use baseline augmentation preset (flip + small rotation + jitter)
- Human coronal sections have the same directional priors as mouse (up=dorsal, down=ventral)
- Avoid rot90, elastic deformation, and blur unless human data is much larger (>5,000 slices)

### 4. Test-Time Augmentation (TTA)

**Eval-only experiment on Run 5:** 44.4% mIoU = **−24% catastrophic failure**

6 TTA variants tested: original, flip, rot90, rot180, rot270, flip+rot90

**Findings:**
- Original (no aug): 68.8% mIoU
- With TTA: 44.4% mIoU
- Model completely fails on rot90/180/270 variants
- Averaging predictions across incompatible orientations produces random noise

**Root cause:** DINOv2 + UperNet has **zero rotational equivariance**. The model learns directional priors (e.g., "cortex is at the top of the image") that break under rotation.

**For human model:**
- Do NOT use TTA with rotations
- Flip-only TTA might work (+1-2%) if bilateral symmetry holds
- Multi-axis eval (coronal/sagittal/axial) requires multi-axis training, not TTA

### 5. Class Pruning (Output Head Size)

**Run 8a (pruned):** 69.0% mIoU vs Run 5 (full) 68.8% = **+0.2% (within noise)**

| Metric | Run 5 (1,328 classes) | Run 8a (673 classes) | Delta |
|--------|----------------------|---------------------|-------|
| mIoU | 68.8% | 69.0% | +0.2% |
| Accuracy | 92.5% | 92.4% | −0.1% |
| Valid classes | 503 | 503 | 0 |
| Logits memory | 1,425 MB | 722 MB | −49% |
| Zero-IoU classes | 7 | 9 | +2 |

**Hypothesis (rejected):** 655 zero-pixel classes in the softmax output dilute gradients on valid classes.

**Reality:** The model assigns near-zero probability to zero-pixel classes within the first few epochs. They don't compete with valid classes during optimization.

**Practical benefit:** Logits memory halved (useful for larger batch sizes on GPU-constrained systems).

**For human model:**
- Do NOT prune classes unless GPU memory is severely limited
- Training all classes (even zero-pixel) does not harm accuracy
- Only prune if you need the memory for larger batches or higher resolution

### 6. Cross-Axis Generalization

**Run 8a diagnostic:** Trained on coronal only, evaluated on all 3 axes

| Axis | Samples | mIoU | Accuracy | Status |
|------|---------|------|----------|--------|
| Coronal | 127 | 68.9% | 92.4% | TRAINED |
| Axial | 68 | 3.2% | 22.9% | Unseen |
| Sagittal | 96 | 0.5% | 12.8% | Unseen |

**Finding:** Model is **strictly orientation-specific**. Cannot generalize to unseen planes.

**Implication:** Multi-axis training is high-risk. Non-coronal slices would act as noise (similar to rot90 augmentation failure). The 2,649-sample multi-axis dataset (1,016 coronal + 677 axial + 956 sagittal) would likely underperform the 1,016 coronal-only baseline.

**For human model:**
- Train on **one primary plane only** (likely coronal, which has the most annotated slices)
- Do NOT mix planes unless you have 3,000+ slices per plane
- If you need cross-plane inference, train 3 separate models (one per plane)

### 7. Per-Class Performance Predictors

**Run 8a diagnostic:** Correlation(log₁₀(pixel_count), IoU) = **0.794** (strong positive)

**IoU distribution (503 valid classes):**
- 424 classes (84%): IoU ≥0.5 (well-segmented)
- 59 classes (12%): IoU 0.1–0.5 (partial)
- 20 classes (4%): IoU <0.1 (failure)

**Key insight:** Class pixel count is the dominant predictor of segmentation quality. Small structures (<1,000 pixels total in val set) are systematically under-learned.

**Bottom 10 classes (non-zero IoU):**
All have <10,000 pixels in the validation set. Examples:
- trochlear nerve: IoU 0.09
- oculomotor nerve: IoU 0.03
- medial corticohypothalamic tract: IoU 0.06

**For human model:**
- Expect similar pixel-count correlation
- Small white matter tracts and cranial nerves will be hard to segment
- If human annotations are sparse (e.g., only 50 major structures), expect higher per-class IoU than mouse (because human classes will be larger on average)

---

## Dataset Characteristics

### Mouse Data (Allen CCFv3)

- **Volume:** `ara_nissl_10.nrrd` (Nissl stain, 10μm isotropic)
- **Dimensions:** 1320 coronal × 800 × 1140 pixels (Z × H × W)
- **Structures:** 1,328 classes (but only 503 have >0 pixels in val set)
- **Train/val/test split:** Interleaved (80/10/10) to avoid spatial autocorrelation
- **Preprocessing:** Random 518×518 crops, downsample-4 (40μm effective resolution)

### Train/Val Split Strategy

**Run 1 (spatial split):** 50.7% mIoU — model overfits to anterior brain
**Run 2 (interleaved split):** 88.0% mIoU — **+37% improvement**

Interleaved split ensures train and val slices sample the full rostral-caudal axis. Spatial split (train=anterior 80%, val=posterior 10%) causes severe distribution shift because brain structures vary dramatically along the rostral-caudal axis.

**For human model:**
- Use interleaved split if slices are sequential (e.g., coronal sections from a single brain)
- If data is from multiple subjects, split by subject (not by slice) to avoid data leakage

---

## Convergence & Training Duration

### Epoch Requirements

All unfrozen runs (5, 6, 7, 8a) show **declining validation loss at epoch 100**. The model has not converged yet.

**Estimated convergence:** 150–200 epochs based on loss trajectory.

**Training time (NVIDIA L40S 48GB):**
- 100 epochs: ~9-11 hours (unfrozen, batch=4)
- 200 epochs: ~18-22 hours (estimated)

**For human model:**
- Budget 200 epochs minimum
- Monitor val loss — if it plateaus before 200, can stop early
- If dataset is 10× larger (e.g., 10,000 slices), may need 300+ epochs

### Learning Rate Schedule

**Run 5 (best):** Linear warmup (10% of steps) + linear decay to 0

```python
warmup_ratio = 0.1  # First 10% of training
lr_scheduler_type = "linear"  # Linear decay after warmup
```

No learning rate spikes or divergence observed. This schedule works reliably.

**For human model:** Use the same schedule.

---

## MLflow Artifact Management (Critical)

**Pattern for saving UperNet models:**

```python
import os
import mlflow
from transformers import AutoImageProcessor

# 1. Set id2label/label2id BEFORE saving
model.config.id2label = {i: name for i, name in enumerate(class_names)}
model.config.label2id = {name: i for i, name in enumerate(class_names)}

# 2. Save model + processor to DBFS
trainer.save_model(FINAL_MODEL_DIR)
processor = AutoImageProcessor.from_pretrained(model_path)
processor.save_pretrained(FINAL_MODEL_DIR)

# 3. Log to MLflow — use log_artifacts(), NOT transformers.log_model()
mlflow.log_artifacts(FINAL_MODEL_DIR, artifact_path="model")

# 4. Close run
mlflow.end_run()
```

**CRITICAL:** `mlflow.transformers.log_model()` does NOT work with UperNet. It only accepts:
1. A transformers Pipeline
2. A dict of pipeline components
3. A path to a checkpoint that `AutoModel.from_pretrained()` can load

UperNet is none of these. Passing the model object → `MlflowException`. Passing the path → `KeyError: UperNetConfig`.

**This bug occurred 4 times** before being permanently fixed. Use `mlflow.log_artifacts()` for all custom architectures.

**Loading model later:**
```python
from transformers import UperNetForSemanticSegmentation
model = UperNetForSemanticSegmentation.from_pretrained(artifact_path)
```

---

## Recommendations for Human Model Training

### Phase 1: Baseline (Priority: HIGH, Cost: LOW)

1. **Start with unfrozen backbone** — Skip the frozen baseline. Go straight to last-4-blocks unfrozen.
2. **Use Run 5 config exactly** — Same hyperparameters, same augmentation, same loss function.
3. **Train 200 epochs** — Don't stop at 100 (val loss still declining).
4. **Use interleaved split** — If slices are sequential. If multi-subject, split by subject.
5. **Coronal plane only** — Do NOT mix planes in training.

**Expected mIoU:** Depends on annotation quality and class granularity. If human annotations are for ~50 major structures (vs mouse 1,328), expect higher per-class IoU (70-80%) because classes will be larger.

### Phase 2: Model Initialization (Priority: MEDIUM, Cost: LOW)

Test whether initializing from mouse weights helps:

```python
# Load mouse Run 5 checkpoint
base_model = UperNetForSemanticSegmentation.from_pretrained(
    "/dbfs/FileStore/allen_brain_data/models/unfrozen"
)

# Reinitialize the output projection layer for new num_labels
# (keep backbone + most of decode head, only change final classifier)
```

**Hypothesis:** Mouse and human brains share cytoarchitectural patterns (cortical layers, hippocampus, cerebellum). Pretrained features might transfer.

**Test:** Compare random init vs mouse init (both trained for 200 epochs). If mouse init converges faster or reaches higher mIoU, use it.

### Phase 3: NOT Recommended (Priority: LOW / SKIP)

1. **DO NOT use weighted loss** — Only try if class imbalance is extreme (>100:1).
2. **DO NOT use extended augmentation** — Stick to baseline preset.
3. **DO NOT prune output classes** — Unless GPU memory is critically limited.
4. **DO NOT use multi-axis training** — High risk of rot90-style failure.
5. **DO NOT use TTA** — Model is orientation-specific.

### Phase 4: Advanced Techniques (Priority: LOW, Cost: MEDIUM)

Only pursue these if baseline human model plateaus below 65% mIoU:

1. **Sliding window evaluation** — Use overlapping crops at inference time (no retraining needed). May recover 1-3% mIoU if structures near image edges are being missed.
2. **Higher resolution crops** — 518×518 → 768×768. Requires more GPU memory (reduce batch size to 1, use grad accumulation).
3. **Unfreeze more blocks** — Last 4 → last 6 or last 8. Increases trainable params but may improve domain adaptation.

---

## Open Questions for Human Model

1. **Ground truth availability:** Do human annotations exist? In what format (SVG, pixel masks, 3D volumes)?
2. **Class ontology:** How many structures? Is it the Allen Human Brain Atlas ontology (fewer than 1,328) or something else?
3. **Multi-subject variance:** 6 donors' worth of images (14,565 slices) — are annotations consistent across subjects?
4. **Slice plane:** Are human slices coronal? If mixed planes, need to filter to one primary orientation.
5. **Resolution:** What is the pixel size of human images? Mouse was downsampled to 40μm effective. Human cortex is thicker, so may need less downsampling.

**Next steps for human team:**
- Confirm ground truth exists and is accessible (Step 8 of step12_plan.md)
- Download and preprocess human data (Step 9)
- Run baseline training with Run 5 config (Step 10)
- Compare mouse-init vs random-init

---

## Metrics for Success

### Mouse Model (Final — Run 9)

Target metrics for the 200-epoch run:
- **mIoU:** 70-72% (current 68.8% + expected 1-2% from more epochs)
- **Pixel accuracy:** 92.5%+ (already achieved)
- **Per-class IoU:** 424/503 classes ≥0.5 IoU (84% of valid classes)

### Human Model (Target)

Success criteria depend on annotation granularity:
- If ~50 major structures: 75%+ mIoU, 93%+ accuracy
- If ~500 structures (similar to mouse): 65-70% mIoU, 90%+ accuracy
- If annotations are sparse (e.g., only cortical regions): 80%+ mIoU

**Key deliverable for paper:**
- Two separate models (mouse + human)
- Ablation study showing backbone unfreezing is critical
- Cross-architecture comparison (DINOv2 vs SegFormer vs MAE)
- Analysis of per-class performance vs anatomical priors (pixel count, laminar structure, etc.)

---

## Conclusion

The mouse model experiments establish a clear recipe:
- **Architecture:** DINOv2-Large + UperNet (proven to work)
- **Training:** Unfrozen backbone, plain CE loss, baseline augmentation, 200 epochs
- **Data:** Coronal plane only, interleaved split, 518×518 crops
- **Constraints:** Batch ≥2, drop_last=True, DDP unused params, gradient checkpointing

Three experiments tried to improve on the baseline (weighted loss, extended augmentation, class pruning) and all failed. The performance ceiling is determined by dataset characteristics (class pixel counts, training set size), not model capacity.

**For human model:** Start with this exact config and only deviate if baseline results warrant it. The most important unknowns are ground truth quality and annotation density — these will dominate model performance more than any architectural choice.

All code, notebooks, and full experimental results are in this repo. Run 9 (200 epochs) is the final mouse model. After that, pivot to human data collection and training.