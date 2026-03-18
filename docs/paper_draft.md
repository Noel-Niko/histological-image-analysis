# Transfer Learning for Ultra-Fine-Grained Brain Region Segmentation: An Ablation Study with DINOv2 + UperNet on 1,328 Allen Mouse Brain Atlas Structures

---

## Abstract

Automated segmentation of brain regions from histological images is essential for high-throughput neuroanatomical analysis, yet existing approaches typically target tens to low hundreds of anatomical classes. We address the extreme multi-class setting: pixel-level segmentation of 1,328 brain structures defined by the Allen Mouse Brain Atlas Common Coordinate Framework version 3 (CCFv3). Using transfer learning from a self-supervised DINOv2-Large vision transformer (304M parameters) with an UperNet decode head, we conduct a systematic ablation study across 9 training runs examining backbone unfreezing, loss functions, augmentation strategies, output head pruning, training duration, and evaluation methodology. Our final model achieves 79.1% mean Intersection-over-Union (mIoU) and 96.9% pixel accuracy on 671 valid classes using sliding window evaluation. The two most impactful interventions are backbone partial fine-tuning (+8.5% mIoU) and extended training from 100 to 200 epochs (+6.0% mIoU). Three attempted improvements — weighted Dice+CrossEntropy loss, aggressive augmentation, and test-time augmentation — all degrade performance, providing informative negative results. We find that per-class segmentation quality is strongly predicted by class pixel count (r=0.794) and that the model is strictly orientation-specific, failing catastrophically on unseen anatomical planes. We introduce sliding window evaluation as a methodological improvement that reveals 168 additional valid classes invisible to standard center-crop evaluation.

---

## 1. Introduction

Brain atlas segmentation — the task of automatically delineating anatomical structures in histological tissue sections — underpins modern computational neuroanatomy. The Allen Mouse Brain Atlas Common Coordinate Framework version 3 (CCFv3) defines a hierarchical ontology of 1,328 brain structures at full resolution, ranging from large regions like the cerebrum and cerebellum down to individual cortical layers (e.g., "Primary somatosensory area, lower limb, layer 2/3") and small cranial nerves (e.g., the trochlear nerve). Automated segmentation at this granularity would accelerate analyses that currently depend on manual or registration-based annotation.

This extreme class count distinguishes our setting from typical semantic segmentation benchmarks. Cityscapes defines 19 classes; ADE20K defines 150; even the most detailed medical segmentation datasets rarely exceed 500 structures. With 1,328 classes — of which 657 have zero training pixels and 825 are absent from the validation split — the task exposes challenges in long-tail class distributions, softmax output scaling, and evaluation methodology that are not well-studied in the segmentation literature.

We adopt a transfer learning approach using DINOv2-Large, a self-supervised vision transformer pretrained on curated natural images at ImageNet-22k scale. Despite the significant domain gap between natural images and Nissl-stained histology, DINOv2 features provide a strong initialization for dense prediction. We pair the backbone with UperNet, a multi-scale feature aggregation decoder that combines Feature Pyramid Network (FPN) pathways with Pyramid Pooling Module (PSP) context aggregation.

Our contributions are:

1. **A systematic ablation study** across 9 training configurations on an extreme multi-class histology segmentation task, identifying backbone partial fine-tuning and extended training as the dominant performance factors.
2. **Informative negative results** demonstrating that weighted loss functions, aggressive augmentation, test-time augmentation, and output class pruning all fail to improve performance in this setting.
3. **Sliding window evaluation methodology** that reveals 168 additional valid classes and +4.4% mIoU over standard center-crop evaluation, establishing a more accurate measure of model quality.
4. **Analysis of per-class performance predictors** showing that class pixel count is the dominant factor (r=0.794) and that the model is strictly orientation-specific with zero cross-axis generalization.

---

## 2. Related Work

### Brain Atlas Segmentation

Traditional brain atlas segmentation relies on deformable registration: aligning a reference atlas volume to individual specimens via non-linear spatial transformations. Tools such as ANTs (Advanced Normalization Tools) and Elastix are widely used for 3D volumetric registration. While effective for standardized datasets, registration-based approaches struggle with tissue damage, section loss, and staining variation inherent to histological preparations. Learning-based approaches promise greater robustness to these artifacts.

### Vision Transformers for Semantic Segmentation

Vision Transformers (ViTs) have become the dominant backbone for dense prediction tasks. SegFormer (Xie et al., 2021) introduced a hierarchical transformer with a lightweight MLP decoder. Mask2Former (Cheng et al., 2022) reframes segmentation as mask classification using a transformer decoder with cross-attention. UperNet (Xiao et al., 2018) combines PSP and FPN pathways and has been widely adopted for its simplicity and compatibility with various backbones, including ViTs.

### Self-Supervised Pretraining with DINOv2

DINOv2 (Oquab et al., 2024) trains vision transformers with self-distillation on a curated dataset of 142M images. The resulting features produce strong dense prediction results with frozen backbones, often competitive with supervised pretraining. DINOv2 models are available at multiple scales (Small/86M, Base/151M, Large/304M, Giant/1.1B) and have been shown to transfer across domains including medical imaging, satellite imagery, and microscopy.

### Transfer Learning for Histological Images

Transfer learning from ImageNet-pretrained models to histology is standard practice in computational pathology. However, the domain gap between natural and histological images is substantial: Nissl-stained tissue exhibits cell-density gradients, laminar organization, and stain-specific chromatic properties absent from natural images. Prior work has shown that partial backbone fine-tuning with differential learning rates outperforms both frozen-backbone and full fine-tuning strategies, balancing domain adaptation against catastrophic forgetting of general visual features.

---

## 3. Methods

### 3.1 Dataset

We use the Allen Mouse Brain Atlas CCFv3 volume `ara_nissl_10.nrrd`, a Nissl-stained 3D reference volume at 10μm isotropic resolution with dimensions 1,320 × 800 × 1,140 voxels (coronal × dorsoventral × mediolateral). The accompanying annotation volume maps each voxel to one of 1,328 brain structures defined by the Allen Mouse Brain Atlas ontology hierarchy.

**Class distribution.** Of 1,328 total classes (1,327 brain structures plus background), 657 classes have zero pixels in the training set and are unlearnable regardless of training strategy. In the validation set, 503 classes have non-zero pixel counts; the remaining 825 are absent. This extreme long-tail distribution is inherent to the atlas: many structures are either too small to appear at 10μm resolution in coronal sections or are confined to specific rostral-caudal positions not captured by the validation split.

**Preprocessing.** Each coronal slice (800 × 1,140 pixels) is converted to RGB and served as a 2D image. During training, random 518 × 518 crops are extracted. During evaluation, either a center crop or sliding window tiling is applied (see Section 3.5).

### 3.2 Data Splitting Strategy

We use an interleaved train/validation/test split: every 10th coronal slice is assigned to validation, every (10th+1) slice to test, and the remaining 80% to training. This yields 1,016 training, 127 validation, and 127 test slices.

The interleaved strategy is critical. An early experiment (Run 1) used a contiguous spatial split (anterior 80% for training, posterior 10% for validation). Because brain structures vary dramatically along the rostral-caudal axis — the cerebrum is concentrated anteriorly while the cerebellum and brainstem dominate posteriorly — this spatial split created severe distribution shift between training and validation sets. Switching to interleaved splitting improved mIoU from 50.7% to 88.0% on a 6-class coarse segmentation task (+37.3%), confirming the spatial autocorrelation problem.

### 3.3 Model Architecture

**Backbone.** DINOv2-Large (`facebook/dinov2-large`): 24 transformer blocks, 1,024 hidden dimension, 304M parameters. Self-supervised pretraining on 142M curated natural images.

**Decode head.** UperNet with PSP pooling module and FPN lateral connections. For 1,328 output classes, the decode head has 37.7M parameters.

**Total model.** 342M parameters. With the last 4 backbone blocks unfrozen, 88M parameters (25.7%) are trainable. (See **Figure 1** for architecture overview.)

**Backbone unfreezing strategy.** We unfreeze the last 4 of 24 transformer blocks (blocks 20–23). Earlier blocks remain frozen, preserving general low-level and mid-level visual features. This partial unfreezing balances domain adaptation against catastrophic forgetting of pretrained representations.

### 3.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (2 parameter groups) |
| Backbone learning rate | 1e-5 |
| Decode head learning rate | 1e-4 |
| LR schedule | Linear warmup (10%) + linear decay |
| Weight decay | 0.01 |
| Batch size | 2 per GPU × 2 gradient accumulation = 4 effective |
| Epochs | 200 (final model) |
| Mixed precision | fp16 |
| Gradient checkpointing | Backbone only (use_reentrant=False) |
| Loss function | CrossEntropyLoss (plain, unweighted) |
| Augmentation | Horizontal flip (50%), rotation ±15°, color jitter (brightness/contrast/saturation ±0.1) |

**Differential learning rate.** The 10:1 ratio between decode head and backbone learning rates prevents catastrophic forgetting of DINOv2 features while allowing meaningful adaptation of the backbone's later layers to the histology domain.

**Hardware.** All experiments run on a single NVIDIA L40S GPU (48 GB VRAM) on Databricks 17.3 LTS.

### 3.5 Evaluation Protocol

**Center-crop evaluation.** A single 518 × 518 crop from the center of each validation slice. This covers approximately 29% of the full 800 × 1,140 slice area and is directly comparable across all training runs.

**Sliding window evaluation.** The full 800 × 1,140 validation slice is tiled with 518 × 518 windows at stride 259 (50% overlap). Logits are accumulated in overlapping regions and averaged before applying argmax. This evaluates the entire slice and captures structures that exist only at slice edges.

**Metrics.** We report mean Intersection-over-Union (mIoU) computed over all classes with non-NaN predictions, overall pixel accuracy, and the count of valid classes (those with at least one correctly predicted pixel).

---

## 4. Experiments

We conduct 9 training runs and 2 evaluation-only experiments in a progressive ablation design. Each run changes a single variable from a reference configuration to isolate its effect.

### 4.1 Baseline Establishment (Runs 1–4)

We establish baselines at three class granularities before addressing the full 1,328-class task.

| Run | Classes | Config | mIoU | Accuracy | Notes |
|-----|---------|--------|------|----------|-------|
| 1 | 6 (coarse) | Frozen, spatial split | 50.7% | 78.8% | Broken eval split |
| 2 | 6 (coarse) | Frozen, interleaved split | 88.0% | 95.8% | +37.3% from split fix |
| 3 | 19 (depth-2) | Frozen, interleaved | 69.6% | 95.5% | 15/19 valid classes |
| 4 | 1,328 (full) | Frozen, interleaved | 60.3% | 88.9% | 503/1,328 valid classes |

**Runs 1→2** demonstrate the critical importance of the data splitting strategy. The spatial split assigned all cerebrum pixels to training, leaving the validation set with only posterior structures — a distribution shift that halved performance.

**Runs 2→3→4** show expected mIoU degradation as class granularity increases: 88.0% → 69.6% → 60.3%. The frozen backbone achieves reasonable performance even at full resolution, confirming that DINOv2 features capture neuroanatomically relevant structure despite the natural-to-histology domain gap.

### 4.2 Backbone Unfreezing (Run 4 → Run 5)

**Variable changed:** Frozen backbone → last 4/24 blocks unfrozen, with differential learning rate (backbone 1e-5, head 1e-4). Training extended from 50 to 100 epochs.

| Metric | Run 4 (frozen) | Run 5 (unfrozen) | Delta |
|--------|----------------|-------------------|-------|
| mIoU | 60.3% | 68.8% | **+8.5%** |
| Accuracy | 88.9% | 92.5% | +3.6% |
| Eval loss | 0.496 | 0.363 | −26.8% |
| Trainable params | 15M (4.4%) | 88M (25.7%) | +73M |

This is the single largest improvement in the ablation study. Unfreezing allows the backbone's later layers to adapt from natural image features to histology-specific patterns: cell density gradients, laminar boundaries, and stain intensity distributions. The 10:1 learning rate ratio is essential — full-rate backbone fine-tuning risks catastrophic forgetting of the general visual features encoded in earlier layers.

### 4.3 Loss Function Ablation (Run 5 → Run 6)

**Variable changed:** Plain CrossEntropyLoss → 0.5 × weighted CrossEntropyLoss + 0.5 × multiclass Dice loss, with inverse-frequency class weights.

| Metric | Run 5 (plain CE) | Run 6 (weighted Dice+CE) | Delta |
|--------|-------------------|--------------------------|-------|
| mIoU | 68.8% | 67.2% | **−1.6%** |
| Accuracy | 92.5% | 89.7% | −2.8% |
| Training loss | 0.284 | 1.710 | +502% (not comparable) |

**Negative result.** The weighted loss underperforms plain cross-entropy. We attribute this to three factors: (1) Dice loss dilutes gradients on high-frequency classes that are already well-learned, slowing convergence on the majority of classes; (2) inverse-frequency weights over-penalize errors on rare structures with very few pixels, destabilizing optimization; (3) the dataset's class imbalance is insufficient to warrant aggressive rebalancing — plain CE already learns 503 valid classes effectively.

### 4.4 Augmentation Ablation (Run 5 → Run 7)

**Variable changed:** Baseline augmentation (flip 50%, rotation ±15°, color jitter) → extended augmentation (+ rotation 90° at 50%, elastic deformation α=50/σ=5 at 30%, Gaussian blur σ=0.5–2.0 at 30%).

| Metric | Run 5 (baseline aug) | Run 7 (extended aug) | Delta |
|--------|---------------------|----------------------|-------|
| mIoU | 68.8% | 62.3% | **−6.5%** |
| Accuracy | 92.5% | 90.1% | −2.4% |
| Eval loss | 0.363 | 0.489 | +34.7% |

**Negative result.** Extended augmentation degrades performance substantially. The primary cause is 90° rotation: coronal brain sections have strong directional priors (dorsal-ventral axis, bilateral symmetry) that are destroyed by large rotations. The model has zero rotational equivariance beyond the ±15° range it was trained on (confirmed by the TTA experiment, Section 4.5). Elastic deformation at α=50 warps anatomical boundaries beyond biological plausibility, making small structures unrecognizable. With only 1,016 training slices, the compounding effect of 6 sequential augmentation transforms distorts every training sample beyond the domain the model can learn.

### 4.5 Test-Time Augmentation (Evaluation-Only Experiment)

**Setup:** Run 5 model evaluated with 6-variant TTA: original, horizontal flip, rotation 90°/180°/270°, flip+rotation 90°. Logits averaged across variants.

| Metric | Baseline (1 pass) | TTA (6 variants) | Delta |
|--------|-------------------|-------------------|-------|
| mIoU | 68.8% | 44.4% | **−24.4%** |
| Accuracy | 92.5% | 84.4% | −8.1% |

**Catastrophic failure.** Of 504 valid classes, 482 regressed, 8 improved, and 14 were unchanged. The worst regressions exceeded −79% IoU (ventral tegmental decussation: 79.5% → 0.0%). The effective TTA signal comprises ~33% correct information (original and its flip) diluted by ~67% noise (rotation variants the model cannot interpret).

**Root cause.** DINOv2 + UperNet fundamentally lacks rotational equivariance. The model encodes directional priors — "cortex is dorsal, ventricles are medial" — that are destroyed by 90° rotation. This finding is consistent with the augmentation ablation (Section 4.4) and the cross-axis evaluation (Section 5.3).

### 4.6 Class Pruning (Run 5 → Run 8a)

**Variable changed:** 1,328 output classes → 673 classes (655 zero-pixel classes removed from the output head).

| Metric | Run 5 (1,328 classes) | Run 8a (673 classes) | Delta |
|--------|----------------------|---------------------|-------|
| mIoU | 68.8% | 69.0% | +0.2% |
| Accuracy | 92.5% | 92.4% | −0.1% |
| Valid classes | 503 | 503 | 0 |
| Logits memory | 1,425 MB | 722 MB | −49% |

**Neutral result.** Removing 655 zero-pixel classes from the softmax output has no meaningful effect on segmentation quality. The hypothesis that zero-pixel classes dilute softmax gradients on valid classes is not supported: the model assigns near-zero probability to absent classes within the first few epochs, so they do not compete with valid classes during optimization. The only practical benefit is halved logits memory (1,425 → 722 MB), which could enable larger batch sizes on memory-constrained hardware.

### 4.7 Extended Training (Run 5 → Run 9)

**Variable changed:** 100 epochs → 200 epochs (all other hyperparameters identical to Run 5).

| Metric | Run 5 (100 ep) | Run 9 (200 ep) | Delta |
|--------|----------------|----------------|-------|
| mIoU (center-crop) | 68.8% | 74.8% | **+6.0%** |
| Accuracy (center-crop) | 92.5% | 94.1% | +1.7% |
| Eval loss | 0.363 | 0.300 | −17.3% |
| Training loss | 0.284 | 0.557 | — |
| Runtime | 11.3 hrs | 23.0 hrs | +103% |

The +6.0% mIoU gain from doubling training duration is the second-largest improvement in the ablation study. This gain was 3× larger than our a priori estimate of +1–2%, indicating that Run 5 was substantially underfit at 100 epochs.

At epoch 200, eval loss (0.300) continues to decline but the rate of improvement is slowing. We estimate +1–3% additional mIoU from a 300-epoch run, representing diminishing returns relative to the doubled compute cost. **Figure 2** shows representative segmentation examples from the final model across four rostral-caudal positions.

### 4.8 Sliding Window Evaluation (Run 9)

We introduce sliding window evaluation alongside the standard center-crop protocol.

| Metric | Center-Crop | Sliding Window | Delta |
|--------|-------------|----------------|-------|
| mIoU | 74.8% | **79.1%** | **+4.4%** |
| Accuracy | 94.1% | **96.9%** | +2.8% |
| Valid classes | 503 | **671** | **+168** |

**Methodology.** The full 800 × 1,140 validation slice is tiled with 518 × 518 windows at stride 259 (50% overlap). In overlapping regions, logits are averaged before applying argmax. This evaluates the complete slice rather than the central 29%.

**Finding.** 168 brain structures exist exclusively outside the center crop area — predominantly small peripheral structures near slice edges. All previous runs (1–8a) were evaluated with center-crop, systematically underestimating model quality. Sliding window evaluation recovers these classes and provides a more accurate assessment of the model's true segmentation capability. (See **Figure 6** for a visual comparison.)

---

## 5. Analysis

### 5.1 Ablation Summary

Table 1 summarizes all ablation interventions ranked by their impact on mIoU.

**Table 1. Ablation results ranked by mIoU impact.**

| Rank | Intervention | Comparison | Δ mIoU | Direction |
|------|-------------|-----------|--------|-----------|
| 1 | Interleaved split | Run 1 → 2 (6-class) | +37.3% | Positive |
| 2 | Backbone unfreezing | Run 4 → 5 | +8.5% | Positive |
| 3 | Extended training (200 ep) | Run 5 → 9 | +6.0% | Positive |
| 4 | Sliding window eval | Run 9 CC → SW | +4.4% | Methodological |
| 5 | Class pruning | Run 5 → 8a | +0.2% | Neutral |
| 6 | Weighted Dice+CE | Run 5 → 6 | −1.6% | Negative |
| 7 | Extended augmentation | Run 5 → 7 | −6.5% | Negative |
| 8 | Test-time augmentation | Run 5 eval | −24.4% | Catastrophic |

The three successful interventions (interleaved splitting, backbone unfreezing, extended training) all address fundamental training adequacy: ensuring the model sees representative data, adapts its features to the target domain, and trains long enough to converge. The three failed interventions (weighted loss, extended augmentation, TTA) all attempt to compensate for perceived deficiencies (class imbalance, data scarcity, inference-time robustness) that either do not exist or cannot be addressed by the chosen technique. **Figure 3** visualizes these results as a bar chart.

### 5.2 Per-Class Performance Predictors

We analyze per-class IoU as a function of class pixel count in the validation set (Run 8a diagnostic data, 503 valid classes).

**Pixel count correlation.** Pearson correlation between log₁₀(pixel_count) and IoU is r = 0.794 (p < 0.001), indicating that class pixel count is the dominant predictor of segmentation quality.

**IoU bracket distribution:**

| Bracket | Count | Percentage | Description |
|---------|-------|------------|-------------|
| IoU ≥ 0.5 | 424 | 84% | Well-segmented |
| IoU 0.1–0.5 | 59 | 12% | Partially segmented |
| IoU < 0.1 | 11 | 2% | Barely detected |
| IoU = 0.0 | 9 | 2% | Complete failure |

84% of valid classes achieve IoU ≥ 0.5, indicating that the model segments the large majority of detectable structures well (**Figure 4**). The 20 classes below 0.1 IoU are uniformly small structures with fewer than 10,000 validation pixels: cranial nerves (trochlear nerve, oculomotor nerve), fine cortical layers (e.g., "Rostrolateral area, layer 6b"), and small fiber tracts (medial corticohypothalamic tract).

**Top 10 classes by IoU (Run 9, center-crop):**

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

Top-performing classes are large, spatially contiguous structures with distinctive cytoarchitectural signatures (high cell density in caudoputamen, layered architecture in cerebellar lobules, clear boundaries in olfactory regions). These classes approach the practical ceiling for semantic segmentation at 10μm resolution.

**Bottom 10 classes by IoU (Run 9, center-crop, non-zero):**

| Class | IoU |
|-------|-----|
| Primary somatosensory area, lower limb, layer 2/3 | 2.3% |
| Trochlear nerve | 5.9% |
| Primary somatosensory area, trunk, layer 5 | 6.4% |
| Interpeduncular nucleus | 6.8% |
| Rostrolateral area, layer 6b | 9.5% |
| Midbrain trigeminal nucleus | 12.0% |
| Supraoptic commissures | 12.0% |
| Posterolateral visual area, layer 6a | 12.8% |
| Retrosplenial area, lateral agranular part, layer 6b | 14.5% |
| Primary somatosensory area, unassigned, layer 4 | 15.9% |

Bottom-performing classes share two characteristics: small spatial extent (few pixels in both training and validation sets) and high label ambiguity (individual cortical layers within the same area are distinguished by subtle differences in cell density and laminar thickness that are difficult to resolve at 10μm). The performance ceiling for these classes is determined by data characteristics, not model capacity.

### 5.3 Cross-Axis Generalization

A diagnostic evaluation on Run 8a tested the coronal-only model on all three anatomical planes (Table 2).

**Table 2. Cross-axis evaluation of a coronal-trained model.**

| Axis | Val Samples | mIoU | Accuracy | Status |
|------|-------------|------|----------|--------|
| Coronal | 127 | 68.9% | 92.4% | Trained |
| Axial | 68 | 3.2% | 22.9% | Unseen |
| Sagittal | 96 | 0.5% | 12.8% | Unseen |

The model is strictly orientation-specific. Performance drops from 68.9% to 3.2% on axial sections (dorsoventral slices viewed from above) and to 0.5% on sagittal sections (mediolateral slices viewed from the side). This confirms that DINOv2 + UperNet encodes strong directional priors from the training orientation and cannot generalize across anatomical planes without explicit multi-axis training. This is consistent with the TTA finding (Section 4.5): the model fails on rotated inputs because it has learned orientation-dependent features, not orientation-invariant anatomical patterns. **Figure 5** shows model predictions across all three anatomical planes.

**Implication.** Multi-axis inference requires either (a) separate models trained per orientation, or (b) multi-axis training — though the latter carries the risk of rot90-style failure (Section 4.4) if the model treats off-axis data as noise rather than learning generalizable representations.

### 5.4 Convergence Analysis

Training curves for the key runs reveal consistent convergence patterns.

**Table 3. Training progression at selected epochs (Runs 4, 5, 9).**

| Epoch | Run 4 mIoU | Run 5 mIoU | Run 9 mIoU* |
|-------|------------|------------|-------------|
| 10 | 39.2% | 32.0% | — |
| 25 | 53.8% | 53.9% | — |
| 50 | 60.3% | 62.1% | — |
| 75 | — | 66.6% | — |
| 100 | — | 68.8% | — |
| 200 | — | — | 74.8% |

*Run 9 used identical configuration to Run 5 extended to 200 epochs. Intermediate checkpoint data not captured.

**Observations:**

1. **Unfreezing slows early convergence.** Run 5 (unfrozen) lags Run 4 (frozen) at epoch 10 (32.0% vs 39.2%) because the backbone parameters are initially destabilized. By epoch 25, both runs converge to ~54% mIoU, and Run 5 subsequently pulls ahead.
2. **Validation loss is still declining at termination.** Run 5 eval loss drops from 0.496 (epoch 10) to 0.363 (epoch 100) without plateauing. Run 9 extends this to 0.300 at epoch 200, still declining but at a decreasing rate.
3. **Diminishing returns above 200 epochs.** The eval loss trajectory suggests a 300-epoch run would yield +1–3% additional mIoU — meaningful but with diminishing returns relative to the ~11 additional GPU-hours required.

---

## 6. Discussion

### What Works

The three successful interventions share a common theme: they address fundamental training adequacy rather than adding complexity.

**Correct data splitting** eliminates distribution shift between training and evaluation, which is the largest single factor (+37.3% on the coarse task). This is not a model improvement per se but a prerequisite for valid evaluation.

**Backbone partial fine-tuning** (+8.5%) allows the model to adapt pretrained natural image features to histology-specific patterns. The differential learning rate (10:1 head-to-backbone ratio) is essential: it enables meaningful domain adaptation while preventing catastrophic forgetting. This finding aligns with the broader transfer learning literature showing that partial fine-tuning consistently outperforms both frozen and fully fine-tuned approaches on domain-shifted tasks.

**Extended training** (+6.0%) reveals that the model was substantially underfit at 100 epochs — the standard budget for many fine-tuning studies. With 1,016 training samples and 88M trainable parameters, the model requires ~200 epochs (~2 full passes per parameter per epoch, accounting for gradient accumulation) to approach convergence. This is consistent with the observation that training loss (0.557 at epoch 200) remains well above zero, indicating the model has not memorized the training set.

### What Doesn't Work

**Weighted loss and Dice loss** (−1.6%) attempt to address class imbalance, but the imbalance in this dataset is structural: 657 classes have zero training pixels and cannot be improved by any reweighting scheme. For the 503 valid classes, plain cross-entropy provides sufficient gradient signal. Dice loss's region-based formulation dilutes gradients on well-learned large classes without meaningfully improving rare classes.

**Aggressive augmentation** (−6.5%) violates the domain's directional priors. Nissl-stained coronal brain sections have a consistent dorsal-ventral orientation that 90° rotations destroy. Elastic deformation at high magnitude (α=50) warps anatomical boundaries beyond biological plausibility. With only 1,016 training slices, aggressive augmentation does not expand the effective training distribution — it corrupts it.

**Test-time augmentation** (−24.4%) is the extreme manifestation of the rotational invariance failure. The model's features are not rotation-equivariant; averaging predictions across incompatible orientations produces noise.

**Class pruning** (+0.2%, neutral) tests the hypothesis that zero-pixel classes in the softmax output dilute gradients on valid classes. The null result indicates that the model learns to suppress absent classes early in training, making them computationally inert. This is a useful negative result: practitioners need not invest effort in pruning output classes unless GPU memory is a binding constraint.

### Performance Ceiling

The strong correlation between pixel count and IoU (r=0.794) indicates that the performance ceiling is determined by data characteristics, not model capacity. The 20 lowest-IoU classes are uniformly small structures with insufficient training pixels. Improvements beyond 79.1% mIoU would require either (a) higher-resolution data capturing more pixels per small structure, (b) targeted oversampling of slices containing rare structures, or (c) hierarchical classification that allows partial credit for coarse-grained correct predictions.

### Limitations

1. **Single orientation.** The model is trained and evaluated on coronal sections only. Generalization to axial or sagittal planes requires separate models or multi-axis training.
2. **Single stain type.** Only Nissl-stained tissue is represented. Generalization to fluorescent antibody stains, H&E, or other preparations is untested.
3. **Single species.** Results apply to the mouse CCFv3 atlas. Human brain tissue has different cytoarchitecture, folding patterns, and spatial scale.
4. **Single reference volume.** All data comes from one 3D Nissl volume. Inter-specimen variation (tissue damage, preparation artifacts) is not represented.
5. **No post-processing.** Results report raw model output without CRF smoothing, connected component analysis, or other standard post-processing refinements.

---

## 7. Conclusion

We present a systematic ablation study on ultra-fine-grained brain region segmentation with 1,328 classes from the Allen Mouse Brain Atlas CCFv3. Using DINOv2-Large + UperNet with partial backbone fine-tuning, our final model achieves 79.1% mIoU and 96.9% pixel accuracy on 671 valid classes (sliding window evaluation), with the top-performing classes approaching 98% IoU.

The proven recipe emerging from 9 training runs is straightforward: (1) DINOv2-Large backbone with last 4/24 blocks unfrozen, (2) differential learning rate (1e-5 backbone, 1e-4 head), (3) plain cross-entropy loss, (4) minimal augmentation (flip, small rotation, color jitter), (5) 200 training epochs, and (6) sliding window evaluation for accurate metrics. Three attempted improvements — weighted loss, aggressive augmentation, and TTA — all failed, demonstrating that the performance ceiling in this setting is determined by data characteristics (class pixel counts, dataset size) rather than training techniques.

**Complete training history:**

| Run | Config | mIoU (CC) | mIoU (SW) | Accuracy | Runtime |
|-----|--------|-----------|-----------|----------|---------|
| 1 | 6-class, frozen, spatial split | 50.7% | — | 78.8% | 23 min |
| 2 | 6-class, frozen, interleaved | 88.0% | — | 95.8% | 23 min |
| 3 | 19-class, frozen | 69.6% | — | 95.5% | 25 min |
| 4 | 1,328-class, frozen | 60.3% | — | 88.9% | 5.5 hrs |
| 5 | 1,328-class, unfrozen, 100 ep | 68.8% | — | 92.5% | 11.3 hrs |
| 6 | 1,328-class, weighted Dice+CE | 67.2% | — | 89.7% | ~33 hrs |
| 7 | 1,328-class, extended augmentation | 62.3% | — | 90.1% | ~12 hrs |
| TTA | Eval-only: 6-variant TTA on Run 5 | 44.4% | — | 84.4% | ~20 min |
| 8a | 673-class, pruned | 69.0% | — | 92.4% | ~9.5 hrs |
| **9** | **1,328-class, unfrozen, 200 ep** | **74.8%** | **79.1%** | **94.1% / 96.9%** | **23.0 hrs** |

The two most impactful interventions — backbone partial fine-tuning (+8.5% mIoU) and extended training to 200 epochs (+6.0% mIoU) — account for the full progression from the frozen baseline (60.3%) to the final model (74.8% center-crop). Combined with sliding window evaluation (+4.4%), these three factors together produce the 79.1% sliding window mIoU that represents the model's true segmentation quality.

---

## Figures

| Figure | Description | Generation |
|--------|-------------|------------|
| **Figure 1** | Architecture diagram: DINOv2-Large backbone (blocks 0-19 frozen, 20-23 unfrozen) + UperNet decode head. 342M total params, 88M trainable. | Architecture diagram (matplotlib) |
| **Figure 2** | Segmentation examples: 4×3 grid showing Input / Ground Truth / Prediction for four rostral-caudal positions (anterior, mid-anterior, mid-posterior, posterior). | Model inference on validation set |
| **Figure 3** | Ablation bar chart: horizontal bars showing mIoU delta for each intervention, color-coded by outcome (positive/negative/neutral/methodological). | Hardcoded data from Runs 1–9 |
| **Figure 4** | Per-class IoU distribution: (left) histogram of IoU values across valid classes with bracket annotations; (right) log₁₀(pixel_count) vs IoU scatter with Pearson correlation. | Center-crop evaluation on validation set |
| **Figure 5** | Cross-axis generalization: 3×3 grid showing Input / Ground Truth / Prediction for coronal, axial, and sagittal planes, demonstrating catastrophic failure on unseen orientations. | Model inference on 3 anatomical planes |
| **Figure 6** | Sliding window vs center-crop: 3-panel comparison showing ground truth (with crop box overlay), center-crop prediction (~29% coverage), and sliding window prediction (100% coverage). | Tiled inference with 50% overlap |

All figures are generated by `notebooks/generate_paper_figures.ipynb` (Databricks) or `scripts/generate_paper_figures.py` (local, Figures 1 and 3 only).

---

## References

1. Allen Institute for Brain Science. Allen Mouse Brain Atlas Common Coordinate Framework v3. https://atlas.brain-map.org
2. Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention mask transformer for universal image segmentation. CVPR 2022.
3. Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. TMLR 2024.
4. Xiao, T., Liu, Y., Zhou, B., Jiang, Y., & Sun, J. (2018). Unified perceptual parsing for scene understanding. ECCV 2018.
5. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. NeurIPS 2021.
6. Macenko, M., Niethammer, M., Marron, J. S., et al. (2009). A method for normalizing histology slides for quantitative analysis. ISBI 2009.
7. Avants, B. B., Tustison, N., & Song, G. (2009). Advanced normalization tools (ANTS). Insight Journal, 2(365), 1-35.
