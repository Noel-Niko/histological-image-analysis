# Mouse Model Findings — Transfer Learnings for Human Finetuning

**Document Purpose:** Comprehensive record of all findings from mouse brain segmentation experiments (Runs 1-9) to inform the human brain model training strategy. Originally written 2026-03-16 after completing the full ablation study. Updated 2026-03-17 with Run 9 (final 200-epoch model) results.

**Context:** This project trains two separate models: (1) a mouse model using Allen Mouse Brain Atlas CCFv3 Nissl data (1,320 coronal slices, 1,328 brain structures), and (2) a human model using Allen Human Brain Atlas data (14,565 Nissl sections, 6 donors, ground truth TBD). This document captures the mouse model lessons to accelerate human model development.

**Final mouse model:** Run 9 (200 epochs) — **74.8% mIoU** (center-crop), **79.1% mIoU** (sliding window), **96.9% accuracy** (sliding window). Saved to `/dbfs/FileStore/allen_brain_data/models/final-200ep`.

---

## Summary: What Works and What Doesn't

### Critical Success Factors ✓

1. **Backbone unfreezing is essential** — Frozen backbone: 60.3% mIoU. Unfrozen (last 4 of 24 blocks): 68.8% mIoU (+8.5%). This is the single largest improvement.
2. **200 epochs required for convergence** — Run 5 (100ep): 68.8% mIoU. Run 9 (200ep): 74.8% mIoU (+6.0%). The model was substantially underfit at 100 epochs. This is the second-largest improvement after backbone unfreezing.
3. **Simple augmentation is sufficient** — Baseline (flip + rot15° + jitter) works. Extended augmentation (rot90, elastic, blur) hurts (−6.5%).
4. **Cross-entropy loss is optimal** — Weighted Dice+CE underperforms plain CE (−1.6%).
5. **Interleaved train/val/test split** — Spatial split causes distribution shift (50.7% → 88.0% mIoU switch).
6. **Sliding window eval captures true performance** — Center-crop (518×518) underestimates mIoU by ~4% and misses 168 classes that only appear at slice edges. Sliding window: 79.1% mIoU, 671 valid classes (vs center-crop: 74.8%, 503 classes).

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
| Unfrozen (last 4), 100ep | 68.8% | 92.5% | 88M (25.7%) | Run 5 |
| Unfrozen (last 4), 200ep | **74.8%** | **94.1%** | 88M (25.7%) | **Run 9 (final)** |

**Learning rate strategy:**
- Backbone blocks: 1e-5 (10× lower than head)
- Decode head: 1e-4
- This 10:1 ratio prevents catastrophic forgetting of DINOv2 features

**For human model:**
- Start with unfrozen backbone from the beginning (skip the frozen baseline unless compute is very limited)
- Consider starting from Run 9 mouse weights (`UperNetForSemanticSegmentation.from_pretrained("/dbfs/FileStore/allen_brain_data/models/final-200ep")`) if human structures have anatomical overlap with mouse (e.g., cortical layers, hippocampus, cerebellum)

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

**Confirmed by Run 9:** 100 epochs is insufficient. 200 epochs yields +6.0% mIoU over 100 epochs with identical config.

| Epochs | mIoU (CC) | Accuracy (CC) | Eval Loss | Training Loss |
|--------|-----------|---------------|-----------|---------------|
| 100 (Run 5) | 68.8% | 92.5% | 0.363 | 0.730 |
| 200 (Run 9) | **74.8%** | **94.1%** | **0.300** | **0.557** |
| Delta | +6.0% | +1.7% | −17.3% | −23.7% |

Eval loss dropped from 0.363 → 0.300 at epoch 200 — still declining but the rate of improvement is slowing. A 300-epoch run would likely yield diminishing returns (+1-2% mIoU at most, at a cost of ~11 additional hours). The model is approaching but has not fully reached convergence.

**Training time (NVIDIA L40S 48GB):**
- 100 epochs: ~9-11 hours (unfrozen, batch=4)
- 200 epochs: ~23 hours (measured, Run 9)

**For human model:**
- Budget 200 epochs minimum
- Monitor val loss — if it plateaus before 200, can stop early
- If dataset is 10× larger (e.g., 10,000 slices), fewer epochs per sample may suffice but total training time will increase

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

## Run 9: Final Mouse Model Results (200 Epochs)

**Run date:** 2026-03-16 to 2026-03-17 | **Runtime:** 23.0 hours on 1× NVIDIA L40S 48GB

### Center-Crop Evaluation (comparable to all previous runs)

| Metric | Run 5 (100ep) | Run 9 (200ep) | Delta |
|--------|---------------|---------------|-------|
| mIoU | 68.8% | **74.8%** | **+6.0%** |
| Accuracy | 92.5% | **94.1%** | +1.7% |
| Eval loss | 0.363 | **0.300** | −17.3% |
| Training loss | 0.730 | **0.557** | −23.7% |
| Valid classes | 503 | 503 | 0 |

### Sliding Window Evaluation (new — full-resolution tiled inference)

| Metric | Center-Crop | Sliding Window | Delta |
|--------|-------------|----------------|-------|
| mIoU | 74.8% | **79.1%** | **+4.4%** |
| Accuracy | 94.1% | **96.9%** | +2.8% |
| Valid classes | 503 | **671** | **+168** |

Sliding window uses 518×518 tiles with 50% overlap (stride 259), averaging logits in overlapping regions. This evaluates the full 800×1140 coronal slice instead of a single center crop.

The +168 valid classes from sliding window eval means 168 structures exist only at slice edges (outside the 518×518 center crop) and were invisible to all previous evaluations. These are predominantly small, peripheral structures.

### Top 10 Classes by IoU (center-crop)

| Class | IoU |
|-------|-----|
| Caudoputamen | 97.9% |
| Main olfactory bulb | 97.5% |
| Background | 97.3% |
| Nodulus (X) | 96.4% |
| Anterior olfactory nucleus | 95.8% |
| Field CA1 | 95.2% |
| Uvula (IX) | 95.1% |
| Lobule III | 95.1% |
| Nucleus accumbens | 94.7% |
| Periaqueductal gray | 94.5% |

### Bottom 10 Classes by IoU (non-zero, center-crop)

| Class | IoU |
|-------|-----|
| Primary somatosensory area, lower limb, layer 2/3 | 2.3% |
| trochlear nerve | 5.9% |
| Primary somatosensory area, trunk, layer 5 | 6.4% |
| Interpeduncular nucleus | 6.8% |
| Rostrolateral area, layer 6b | 9.5% |
| Midbrain trigeminal nucleus | 12.0% |
| supraoptic commissures | 12.0% |
| Posterolateral visual area, layer 6a | 12.8% |
| Retrosplenial area, lateral agranular part, layer 6b | 14.5% |
| Primary somatosensory area, unassigned, layer 4 | 15.9% |

Bottom classes are all fine-grained cortical layers and small cranial nerves — consistent with the pixel-count correlation (r=0.794) observed in Run 8a.

### Convergence Assessment

The +6.0% mIoU gain from 100→200 epochs was 3× larger than the predicted +1-2%. This means Run 5 was significantly underfit. At epoch 200, eval loss (0.300) is still declining but the rate of change is slowing. Estimated remaining headroom from further training: +1-3% mIoU (diminishing returns). See [Should We Train Longer?](#should-we-train-longer-300-epochs) below.

### Model Saved

- **DBFS:** `/dbfs/FileStore/allen_brain_data/models/final-200ep`
- **MLflow:** Run `final-200ep-1328class-20260316-1405`, experiment ID `1345391216675532`

---

## Sliding Window Evaluation — Methodology Finding

Sliding window evaluation is a significant methodological improvement discovered in Run 9.

**Why it matters:**
- Center-crop eval (518×518 from 800×1140 slices) covers only ~29% of each slice
- 168 brain structures exist exclusively outside the center crop area
- All previous runs (1-8a) were evaluated with center-crop, systematically underestimating model quality
- Sliding window should be the standard evaluation method going forward

**Implementation:** 518×518 tiles, stride 259 (50% overlap), logit averaging in overlap regions, full-resolution argmax. Implemented in `notebooks/finetune_final_200ep.ipynb` Cell 7.

**For human model:** Always evaluate with sliding window. Center-crop can be used as a quick sanity check during training, but final reported metrics should use sliding window.

**Retroactive note:** Run 5's true performance was likely ~73% mIoU sliding window (estimated by adding the ~4% delta). The gap between Runs 5 and 9 in sliding window terms is still substantial.

---

## Should We Train Longer? (300+ Epochs)

**Evidence for:**
- Eval loss still declining at epoch 200 (0.300)
- Training loss still declining (0.557)
- The 100→200 gain (+6.0%) was much larger than predicted, so the 200→300 gain might also surprise

**Evidence against:**
- Diminishing returns: the loss curve is flattening
- 23 hours per 200 epochs; a 300-epoch run would be ~34.5 hours
- The model at 79.1% sliding window mIoU already achieves strong segmentation quality
- The binding constraint remains data scarcity (1,016 training slices, 657 zero-pixel classes) — more epochs cannot fix absent classes
- Compute time is better invested in the human model pipeline, which has more unknowns and higher marginal value

**Recommendation:** Do NOT train 300 epochs. The mouse model is complete at 79.1% mIoU (sliding window). The expected +1-3% from 300 epochs does not justify the compute vs. starting human ground truth investigation (Step 8), which is the critical path for the project. If a 300-epoch run is desired later, it can be queued as a low-priority background job.

---

## Recommendations for Human Model Training

### Phase 1: Baseline (Priority: HIGH, Cost: LOW)

1. **Start with unfrozen backbone** — Skip the frozen baseline. Go straight to last-4-blocks unfrozen.
2. **Use Run 9 config exactly** — Same hyperparameters, same augmentation, same loss function, 200 epochs.
3. **Use interleaved split** — If slices are sequential. If multi-subject, split by subject.
4. **Coronal plane only** — Do NOT mix planes in training.
5. **Evaluate with sliding window** — Do not rely on center-crop metrics.

**Expected mIoU:** Depends on annotation quality and class granularity. If human annotations are for ~50 major structures (vs mouse 1,328), expect higher per-class IoU (75-85%) because classes will be larger. If ~500+ structures, expect 70-75% with 200 epochs.

### Phase 2: Model Initialization (Priority: MEDIUM, Cost: LOW)

Test whether initializing from mouse weights helps:

```python
# Load mouse Run 9 checkpoint (final 200-epoch model)
base_model = UperNetForSemanticSegmentation.from_pretrained(
    "/dbfs/FileStore/allen_brain_data/models/final-200ep"
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

Only pursue these if baseline human model plateaus below 70% mIoU (sliding window):

1. **Higher resolution crops** — 518×518 → 768×768. Requires more GPU memory (reduce batch size to 1, use grad accumulation).
2. **Unfreeze more blocks** — Last 4 → last 6 or last 8. Increases trainable params but may improve domain adaptation.
3. **300+ epochs** — If eval loss is still declining at 200. Diminishing returns expected.

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
- Run baseline training with Run 9 config (Step 10)
- Compare mouse-init vs random-init

---

## Metrics for Success

### Mouse Model (Final — Run 9) — ACHIEVED

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| mIoU (center-crop) | 70-72% | **74.8%** | Exceeded |
| mIoU (sliding window) | — | **79.1%** | New metric |
| Pixel accuracy | 92.5%+ | **94.1% / 96.9%** | Exceeded |
| Valid classes | 503 | **503 CC / 671 SW** | +168 (SW) |

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

## Complete Training History

| Run | Config | mIoU (CC) | mIoU (SW) | Accuracy (CC) | Runtime | Notebook |
|-----|--------|-----------|-----------|---------------|---------|----------|
| 1 | Coarse 6-class, frozen, spatial split | 50.7% | — | 78.8% | 23 min | `finetune_coarse.ipynb` |
| 2 | Coarse 6-class, frozen, interleaved | 88.0% | — | 95.8% | 23 min | `finetune_coarse.ipynb` |
| 3 | Depth-2 19-class, frozen | 69.6% | — | 95.5% | 25 min | `finetune_depth2.ipynb` |
| 4 | Full 1,328-class, frozen | 60.3% | — | 88.9% | 5.5 hrs | `finetune_full.ipynb` |
| 5 | Full 1,328-class, unfrozen, 100ep | 68.8% | — | 92.5% | 11.3 hrs | `finetune_unfrozen.ipynb` |
| 6 | Full 1,328-class, weighted Dice+CE | 67.2% | — | 89.7% | ~33 hrs | `finetune_weighted_loss.ipynb` |
| 7 | Full 1,328-class, extended augmentation | 62.3% | — | 90.1% | ~11-14 hrs | `finetune_augmented.ipynb` |
| TTA | Eval-only: 6-variant TTA on Run 5 | 44.4% | — | 84.4% | ~20 min | `eval_tta.ipynb` |
| 8a | Pruned 673-class, unfrozen, coronal-only | 69.0% | — | 92.4% | ~9.5 hrs | `finetune_pruned_ablation.ipynb` |
| **9** | **Full 1,328-class, unfrozen, 200ep** | **74.8%** | **79.1%** | **94.1% / 96.9%** | **23.0 hrs** | **`finetune_final_200ep.ipynb`** |

All model dirs under `/dbfs/FileStore/allen_brain_data/models/`. Final model: **`final-200ep`**.

---

## Conclusion

The mouse model experiments (9 training runs + 2 eval-only experiments) establish a clear, proven recipe:
- **Architecture:** DINOv2-Large + UperNet (proven to work)
- **Training:** Unfrozen backbone (last 4 blocks), plain CE loss, baseline augmentation, 200 epochs
- **Data:** Coronal plane only, interleaved split, 518×518 crops
- **Evaluation:** Sliding window (518×518 tiles, stride 259) for accurate metrics
- **Constraints:** Batch ≥2, drop_last=True, DDP unused params, gradient checkpointing

**Final mouse model: 79.1% mIoU (sliding window), 96.9% accuracy, 671 valid classes out of 1,328.**

The two most impactful interventions were backbone unfreezing (+8.5%) and extended training to 200 epochs (+6.0%). Three experiments tried to improve on the unfrozen baseline (weighted loss, extended augmentation, class pruning) and all failed. The performance ceiling is determined by dataset characteristics (class pixel counts, training set size), not model capacity or training tricks.

**For human model:** Start with this exact config (Run 9) and only deviate if baseline results warrant it. The most important unknowns are ground truth quality and annotation density — these will dominate model performance more than any architectural choice.

**Next step:** Pivot to human ground truth investigation (Step 8 of step12_plan.md). The mouse model is complete.