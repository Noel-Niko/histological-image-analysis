# Final Recommendations: DINOv2 Fine-Tuning for Brain Region Segmentation

## Goal

Enable PhD researchers to pass in images of **Nissl-stained** or **fluorescent antibody-stained** human and mouse brain tissue and have brain regions **automatically identified and outlined** at multiple granularities, aligned with the Allen Brain Institute atlas ontology.

---

## Current State

The pipeline uses **DINOv2-Large (304M params) + UperNet** trained on Allen Brain Institute CCFv3 10um Nissl mouse brain data with a frozen backbone and head-only training.

| Granularity | Classes | mIoU | Overall Acc | Notes |
|-------------|---------|------|-------------|-------|
| Coarse | 6 | 88.0% | 95.8% | Strong baseline |
| Depth-2 | 19 | 69.6% | 95.5% | 15/19 classes with valid IoU |
| Full | 1,328 | 60.3% | 88.9% | 503/1,328 classes present in val |

**Key limitations of the current system:**

1. Backbone is frozen — features are not adapted to histology domain
2. Trained only on mouse Nissl CCFv3 data — no human tissue, no fluorescent staining
3. No stain normalization — stain variation between labs will degrade performance
4. No class balancing — 825 classes absent from validation, many rare structures at 0% IoU
5. Single-scale inference — no tile overlap or multi-scale fusion
6. Flat classification — no hierarchical loss exploiting atlas ontology structure

---

## Recommendations Overview

Ordered by expected impact, not by implementation effort.

| Priority | Intervention | Expected Impact | Effort |
|----------|-------------|-----------------|--------|
| 1 | Unfreeze backbone with differential LR | High — domain adaptation | Low |
| 2 | Stain normalization pipeline | High — cross-lab generalization | Medium |
| 3 | Hierarchical loss (coarse + fine) | High — reduces misclassification | Medium |
| 4 | Class-weighted loss + balanced sampling | Medium — improves rare classes | Low |
| 5 | Multi-scale training and inference | Medium — better boundaries | Medium |
| 6 | Cross-species and cross-stain data | High — enables target use case | High |
| 7 | Segmentation head upgrade (Mask2Former) | Medium — sharper boundaries | High |
| 8 | Self-supervised atlas discovery | High when labels sparse | High |
| 9 | Larger backbone (DINOv2-Giant) | Low-Medium — marginal gains | Low |

---

## 1. Unfreeze Backbone with Differential Learning Rate

**Why:** The single highest-impact change. DINOv2 was pretrained on natural images (ImageNet-22k scale). Nissl-stained and fluorescent brain tissue looks nothing like natural images — cell density patterns, laminar organization, and staining gradients are domain-specific. Frozen features cannot capture these.

**Strategy: Two-stage partial fine-tuning.**

### Stage 1 — Linear probe (already done)

Freeze backbone, train only the UperNet head. This establishes whether the pretrained features align with the task at all. Current results confirm they do (88% mIoU coarse).

### Stage 2 — Partial fine-tuning

Unfreeze the last 2-4 transformer blocks of DINOv2-Large. Keep early layers frozen to preserve low-level feature extraction and prevent catastrophic forgetting.

```
DINOv2-Large has 24 transformer blocks.

| Layer range | Train? | Rationale |
|-------------|--------|-----------|
| Patch embedding | Frozen | Generic edge/texture detection |
| Blocks 0-19 | Frozen | General visual features |
| Blocks 20-23 | Trainable | Domain-specific adaptation |
| UperNet head | Trainable | Task-specific decoding |
```

**Training configuration:**

```
optimizer: AdamW
backbone LR (blocks 20-23): 1e-5
head LR (UperNet): 1e-4
weight_decay: 0.01
warmup_ratio: 0.1
epochs: 50-100 with early stopping (load_best_model_at_end=True)
```

**Implementation:** Already possible via `get_training_args(**kwargs)` with `freeze_backbone=False` in `create_model()`. The differential LR requires a parameter group setup — either modify `create_trainer()` to accept parameter groups or build the optimizer in the notebook.

**Risk:** Overfitting on 1,016 training slices. Mitigate with:
- Early stopping on validation mIoU
- Weight decay (0.01)
- Stronger augmentation (see section 5)
- Only unfreezing 2-4 blocks, not the full backbone

**Expected improvement:** 5-15% mIoU gain based on histology transfer learning literature. The gap between frozen and fine-tuned backbones is consistently the largest single factor in domain-shifted segmentation tasks.

---

## 2. Stain Normalization Pipeline

**Why:** This is critical for the target use case. PhD researchers will submit images from different labs, scanners, and staining protocols. Nissl stain intensity, antibody fluorescence brightness, and tissue preparation vary dramatically. Without normalization, a model trained on one lab's images will fail on another's.

**Recommended approaches by stain type:**

### Nissl and H&E-like brightfield stains

Use **Macenko normalization** — decomposes staining into optical density space and normalizes to a reference image. Well-established in digital pathology.

Alternative: **Reinhard normalization** — simpler color transfer in Lab color space. Faster but less robust for extreme stain variation.

### Fluorescent antibody stains

Fluorescent images have fundamentally different characteristics (dark background, channel-specific signal). Standard stain normalization does not apply. Instead:
- Per-channel percentile normalization (clip at 1st and 99th percentile, scale to [0, 1])
- Intensity histogram matching to a reference image per channel
- Background subtraction using rolling ball or morphological opening

### Implementation approach

Add a `StainNormalizer` class to the data pipeline that:
1. Detects stain type (brightfield vs. fluorescent) based on image statistics or user specification
2. Applies the appropriate normalization
3. Feeds normalized images to the existing `BrainSegmentationDataset`

**Impact:** Stain normalization routinely improves cross-lab generalization by 10-20% in digital pathology benchmarks. Without it, the model will appear to work in development but fail in production when researchers submit images from different preparation protocols.

---

## 3. Hierarchical Loss Exploiting Atlas Ontology

**Why:** The Allen Brain Atlas is inherently hierarchical. Predicting "Somatosensory cortex" when the ground truth is "Primary somatosensory cortex, layer 4" should be penalized less than predicting "Cerebellum." Flat cross-entropy treats all misclassifications equally.

**Architecture:**

```
DINOv2 backbone
    |
    v
Shared feature map
    |
    +---> Coarse head (6 classes) ---> CE loss (weight: 1.0)
    |
    +---> Depth-2 head (19 classes) ---> CE loss (weight: 0.5)
    |
    +---> Full head (1,328 classes) ---> CE loss (weight: 0.5)
```

**Total loss:**

```
L = CE_coarse + 0.5 * CE_depth2 + 0.5 * CE_full
```

The coarse head acts as a regularizer — the model must get the major region right before trying to identify fine subregions. This is a form of curriculum learning built into the loss function.

**Why this matters for the full mapping:**

Current full-mapping mIoU is 60.3%. Many failures are cases where the model confuses structures within the same parent region (e.g., confusing CA1 with CA3 within hippocampus). A hierarchical loss forces the model to learn the coarse grouping first, constraining the search space for fine classification.

**Implementation:** Add a `HierarchicalHead` module that wraps the existing UperNet decoder with additional classification heads at coarse and depth-2 levels. The ontology mapper already provides all three mappings.

---

## 4. Class-Weighted Loss and Balanced Sampling

**Why:** The dataset is severely imbalanced. Background and Cerebrum dominate pixel counts, while 825 classes have zero validation pixels. The model has no incentive to learn rare structures.

**Two complementary approaches:**

### Inverse-frequency class weights

Weight each class in the cross-entropy loss inversely proportional to its pixel frequency in the training set.

```python
# Compute pixel counts per class across training set
class_counts = count_pixels_per_class(train_dataset)
weights = 1.0 / (class_counts + epsilon)
weights = weights / weights.sum() * num_classes  # normalize
loss_fn = CrossEntropyLoss(weight=weights)
```

### Balanced sampling

Instead of uniform sampling over slices, oversample slices that contain rare structures. This ensures the model sees rare classes more frequently during training.

**Combined with Dice loss:**

Replace pure cross-entropy with a combined loss:

```
L = 0.5 * weighted_CE + 0.5 * Dice
```

Dice loss inherently handles class imbalance because it operates on per-class overlap ratios rather than per-pixel predictions.

**Expected impact:** 5-10% mIoU improvement on rare classes, with minimal degradation on common classes.

---

## 5. Multi-Scale Training and Inference

**Why:** Brain structures exist at wildly different spatial scales. The hippocampal formation spans thousands of pixels, while individual nuclei in the hypothalamus may occupy only a few dozen pixels in a 518x518 crop. A single crop size cannot capture both.

### Multi-scale training

Train with random crop sizes drawn from {384, 518, 672}, resized to the model's input resolution (518x518). This teaches the model to recognize structures at different magnifications.

### Overlapping tile inference

At inference time, use overlapping tiles with blending to avoid boundary artifacts:

```
tile size: 512
stride: 256 (50% overlap)
blending: Gaussian weighting (center pixels weighted higher)
```

This is standard practice in digital pathology and consistently improves boundary accuracy by 3-5% mIoU.

### Augmentation additions

The current augmentation pipeline (horizontal flip, rotation +/-15, color jitter) is reasonable but should be extended for histology:

| Augmentation | Rationale |
|-------------|-----------|
| Elastic deformation | Brain tissue is physically deformable; sections warp during preparation |
| Gaussian blur (sigma 0.5-2.0) | Simulates focus variation in microscopy |
| Stain jitter (if using Macenko) | Simulates stain concentration variation |
| Random 90-degree rotations | Brain sections can be mounted at any orientation |

**Avoid:** Heavy color augmentations (saturation >0.5, hue shifts) that break the relationship between stain color and tissue composition.

---

## 6. Cross-Species and Cross-Stain Training Data

**Why:** The current model is trained exclusively on mouse CCFv3 Nissl data. The target use case requires generalization to:
- Human brain tissue
- Fluorescent antibody stains
- Different tissue preparation protocols

This is the largest gap between current capability and the stated goal.

### Data sources to integrate

| Source | Species | Stain | Resolution | Use |
|--------|---------|-------|------------|-----|
| Allen CCFv3 (current) | Mouse | Nissl | 10um | Primary training |
| Allen Human Brain Atlas | Human | Nissl | 1um | Human generalization |
| Allen Mouse ISH | Mouse | ISH (various) | 25um | Stain diversity |
| BICCN (Brain Initiative Cell Census Network) | Mouse + Human | Fluorescent | Varies | Fluorescent stain training |
| Human Protein Atlas (brain section) | Human | IHC / IF | Varies | Antibody stain training |

### Cross-stain domain adaptation strategy

Train a single backbone that generalizes across stain types using:

1. **Stain-agnostic preprocessing:** Convert all images to a common representation (e.g., grayscale intensity or normalized optical density) as an additional input channel alongside the original RGB
2. **Stain-type conditioning:** Add a learned stain-type embedding to the model that modulates features based on the input stain type (Nissl, H&E, fluorescent, etc.)
3. **Domain-adversarial training:** Add a discriminator head that tries to predict the stain type from features. Train the backbone to fool it, forcing stain-invariant representations

**Practical minimum:** Even without formal domain adaptation, adding human Nissl data from the Allen Human Brain Atlas to the training set would substantially improve human tissue performance. The atlas ontology provides compatible region labels.

---

## 7. Segmentation Head: UperNet vs. Mask2Former

**Current architecture:** DINOv2-Large + UperNet

**Recommended upgrade:** DINOv2-Large + Mask2Former

### Why Mask2Former

Mask2Former treats segmentation as **mask classification** rather than per-pixel classification. It predicts a set of binary masks, each with a class label, using a transformer decoder with cross-attention to multi-scale features.

Advantages for brain region segmentation:
- **Sharper boundaries** — mask-based prediction produces cleaner region outlines than per-pixel classification
- **Better handling of small structures** — the query-based mechanism can attend to small regions that get lost in per-pixel approaches
- **Multi-scale features** — the pixel decoder produces features at multiple resolutions, critical for structures spanning different scales

### Architecture

```
DINOv2-Large backbone
    |
    v
Feature Pyramid / Pixel Decoder
(converts transformer tokens to multi-scale feature maps)
    |
    v
Mask2Former Transformer Decoder
(100 learnable queries cross-attend to features)
    |
    v
Per-query outputs:
  - Binary mask prediction (one per query)
  - Class prediction (one per query)
    |
    v
Assembled segmentation map
```

### Implementation considerations

Mask2Former is available in the `transformers` library as `Mask2FormerForUniversalSegmentation`, but it does not natively support DINOv2 as a backbone. Integration requires:

1. A custom pixel decoder that converts DINOv2 patch tokens into multi-scale feature maps
2. The standard Mask2Former transformer decoder and loss computation
3. Combined loss: cross-entropy + dice + mask loss

**Alternative: ViT-Adapter + Mask2Former** — a published architecture that specifically adapts vision transformer backbones for dense prediction with multi-scale features. This is the current state-of-the-art stack for ViT-based segmentation.

### When to upgrade

UperNet is adequate for the coarse and depth-2 granularities (88% and 69.6% mIoU). Upgrade to Mask2Former when:
- Full-mapping mIoU plateaus after implementing recommendations 1-4
- Boundary quality becomes a concern for the end-user annotation workflow
- The project moves to production-grade inference

---

## 8. Self-Supervised Atlas Discovery

**Why:** When labels are sparse — which is the case for human brain tissue and fluorescent stains where curated atlas annotations are limited — supervised training alone will not suffice.

### Concept

DINOv2 patch tokens encode spatial information about tissue structure. Without any labels, these tokens can be clustered to discover anatomical regions that correspond to real neuroanatomy.

### Pipeline

```
Step 1: Extract DINOv2 embeddings for all tiles across a brain slice
Step 2: Build a dense feature map (token grid covering the whole section)
Step 3: Cluster embeddings (K-means or HDBSCAN)
Step 4: Spatially smooth clusters (CRF or graph cuts)
Step 5: Register cluster map to Allen atlas via deformable registration
Step 6: Assign atlas labels to clusters by maximum overlap
Step 7: Use cluster-atlas assignments as pseudo-labels
Step 8: Train segmentation network on pseudo-labels
Step 9: Iterate (retrain, re-cluster, refine)
```

### When to use this

- **Human brain tissue** where Allen Human Brain Atlas annotations are coarser than CCFv3
- **Fluorescent stains** where no dense pixel-level annotations exist
- **New species or preparations** where manual annotation is prohibitively expensive
- **Discovering subregions** that may not be in the reference atlas

### Clustering DINOv2 tokens

```python
# Extract patch tokens (exclude CLS token)
outputs = backbone(pixel_values=image_tensor)
tokens = outputs.last_hidden_state[:, 1:, :]  # shape: [B, N, 1024]

# Reshape to spatial grid
B, N, C = tokens.shape
H = W = int(N ** 0.5)  # 37 for 518x518 input
feature_map = tokens.reshape(B, H, W, C)
```

These feature maps can be PCA-reduced to 3 dimensions and visualized directly — the resulting images often show clear anatomical boundaries without any training.

### Expected accuracy

In histology datasets, self-supervised discovery followed by atlas alignment typically achieves 85-95% of fully supervised performance, and sometimes exceeds it when supervised labels are noisy or sparse.

---

## 9. Backbone Size: When to Use DINOv2-Giant

### Current recommendation: Stay with DINOv2-Large

Based on the project's dataset size (~1,016 training slices) and current bottlenecks, switching to Giant is **not the highest-priority intervention**.

| Factor | Assessment |
|--------|-----------|
| Dataset size | ~1,016 slices — below the ~200k patch threshold where Giant typically outperforms Large |
| Current bottleneck | Frozen backbone and flat loss, not feature capacity |
| Memory cost | Giant uses ~22-28 GB vs Large's ~18-24 GB on L40S 48 GB — fits, but less headroom |
| Training cost | 4-5x slower than Large |
| Expected mIoU gain | 1-3% (based on DINOv2 benchmark scaling curves) |

### When Giant becomes worthwhile

Move to Giant **after** implementing recommendations 1-6, if:
- Feature quality (not training dynamics) is the remaining bottleneck
- Dataset has grown to >200k labeled patches
- Fine-grained classes plateau below expected levels even with unfrozen Large backbone
- Multi-GPU training is stable and well-characterized

### Giant-specific configuration

```
model: facebook/dinov2-giant (40 layers, 1536 hidden dim, 1.1B params)
out_features: ["stage10", "stage20", "stage30", "stage40"]
backbone LR: 5e-6 (lower than Large due to more parameters)
gradient checkpointing: ON (model.backbone.gradient_checkpointing_enable())
mixed precision: bf16
batch size: 2 per GPU + gradient accumulation
```

### "With registers" variants

DINOv2 models with register tokens (`facebook/dinov2-large-with-registers`) may produce slightly cleaner feature maps for dense prediction. These are drop-in replacements with the same config structure. Worth testing as a zero-effort experiment before moving to Giant.

---

## 10. Whole-Slide Inference Pipeline

**Why:** PhD researchers will submit whole brain sections, not pre-cropped 518x518 tiles. The system needs an end-to-end pipeline from raw slide to annotated output.

### Pipeline

```
Raw brain section image (potentially gigapixel)
    |
    v
Tissue detection (threshold background)
    |
    v
Multi-resolution tiling (512x512, stride 256)
    |
    v
Stain normalization (per tile)
    |
    v
Model inference (DINOv2 + segmentation head)
    |
    v
Prediction merging (Gaussian-weighted blending)
    |
    v
CRF post-processing (boundary refinement)
    |
    v
Atlas overlay (region labels + outlines)
    |
    v
Output: annotated image + region table
```

### Tissue detection

Skip background tiles to reduce inference by 80-90%:

```python
def is_tissue(tile, threshold=220):
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    tissue_fraction = (gray < threshold).mean()
    return tissue_fraction > 0.1  # at least 10% tissue
```

### CRF post-processing

Dense CRF smoothing enforces spatial consistency in the final segmentation map, reducing salt-and-pepper noise at tile boundaries. This is standard in digital pathology pipelines and typically improves boundary quality without retraining.

### Output format

The end user should receive:
1. An annotated image with region boundaries overlaid
2. A table mapping each region to its Allen atlas identifier, name, and area in pixels/square microns
3. A confidence score per region (mean softmax probability)

---

## 11. Implementation Roadmap

The recommendations above should be implemented in phases, where each phase builds on the previous one and produces a measurable improvement.

### Phase 1 — Maximize performance on existing data

**Goal:** Push full-mapping mIoU from 60% toward 75%+ without new data.

| Step | Intervention | Builds on |
|------|-------------|-----------|
| 1a | Unfreeze backbone (last 4 blocks), differential LR | Existing pipeline |
| 1b | Class-weighted cross-entropy + Dice loss | 1a |
| 1c | Hierarchical loss (coarse + depth-2 + full) | 1a, 1b |
| 1d | Extended augmentation (elastic deformation, blur, rotation) | 1a |
| 1e | More epochs (100-200) with early stopping | 1a-1d |

### Phase 2 — Cross-stain and cross-species generalization

**Goal:** Enable the model to handle fluorescent and human tissue.

| Step | Intervention | Builds on |
|------|-------------|-----------|
| 2a | Stain normalization pipeline (Macenko + fluorescent) | Phase 1 |
| 2b | Integrate Allen Human Brain Atlas data | 2a |
| 2c | Integrate fluorescent antibody stain datasets | 2a |
| 2d | Self-supervised atlas discovery for label-sparse domains | 2c |
| 2e | Domain-adversarial training for stain invariance | 2a-2d |

### Phase 3 — Production-grade inference

**Goal:** End-to-end whole-slide pipeline usable by PhD researchers.

| Step | Intervention | Builds on |
|------|-------------|-----------|
| 3a | Whole-slide tiling and inference pipeline | Phase 1-2 |
| 3b | Overlapping tile inference with Gaussian blending | 3a |
| 3c | CRF post-processing | 3b |
| 3d | Atlas overlay and output generation | 3c |
| 3e | Mask2Former segmentation head (if boundary quality insufficient) | 3a |

### Phase 4 — Scale and optimize

**Goal:** Push accuracy ceiling and handle edge cases.

| Step | Intervention | Builds on |
|------|-------------|-----------|
| 4a | DINOv2-Giant backbone (if feature quality is bottleneck) | Phase 1-3 |
| 4b | Active learning loop (uncertainty-based annotation) | Phase 2-3 |
| 4c | Multi-scale training (random crop sizes) | Phase 1 |
| 4d | Deformable atlas registration for per-sample alignment | Phase 3 |

---

## 12. What Improves Accuracy More Than Model Size

Based on digital pathology and histology segmentation literature, the factors that most improve accuracy — in order — are:

1. **Better region labels** (atlas alignment quality, annotation consistency)
2. **Domain adaptation** (unfreezing backbone, stain normalization)
3. **Balanced sampling and loss weighting** (rare class handling)
4. **Tile overlap and multi-scale inference** (boundary quality)
5. **Hierarchical classification** (exploiting ontology structure)
6. **More diverse training data** (cross-stain, cross-species)
7. **Segmentation head architecture** (Mask2Former vs UperNet)
8. **Backbone model size** (Large vs Giant)

Model scaling (item 8) is consistently the **least impactful** factor when the other seven are not optimized. The current project has substantial room for improvement in items 1-6 before model scaling becomes relevant.

---

## 13. GPU Infrastructure Requirements

### Current infrastructure (sufficient for Phase 1)

| Cluster | GPU | VRAM | Use |
|---------|-----|------|-----|
| g6e.16xlarge | 1x L40S | 48 GB | All granularities, proven working |
| g6e.48xlarge | 8x L40S | 48 GB each | DDP for speed (requires batch=2 + grad_accum=2) |

### Phase 2-3 infrastructure

| Model | Backbone training | VRAM per GPU | Recommended cluster |
|-------|-------------------|-------------|---------------------|
| DINOv2-Large + UperNet (unfrozen) | Partial fine-tune | ~30-35 GB | 1x L40S (fits) |
| DINOv2-Large + Mask2Former | Full pipeline | ~35-40 GB | 1x L40S (tight) or A100 |
| DINOv2-Giant + UperNet | Partial fine-tune | ~45-55 GB | 1x A100 80 GB |
| DINOv2-Giant + Mask2Former | Full pipeline | ~60-70 GB | 1x A100 80 GB |

### Memory reduction techniques

```
gradient checkpointing (on backbone, not UperNet — see docs/step10_gpu_memory_review.md)
bf16 mixed precision
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torch.cuda.empty_cache() between pipeline stages
batch=2 + gradient_accumulation_steps=N for effective larger batch
```

---

## 14. Key Risk: Generalization Gap

The largest risk to the stated goal is the **generalization gap** between training data (mouse CCFv3 Nissl) and target data (human + mouse, Nissl + fluorescent antibody).

### Mouse Nissl to Human Nissl

- Cytoarchitecture differs between species (layer thickness, cell density, folding patterns)
- Region boundaries are in different absolute positions
- Allen Human Brain Atlas provides Nissl sections with region annotations, but at coarser granularity than CCFv3
- **Mitigation:** Train on both mouse and human data; use hierarchical labels at the resolution available for each species

### Nissl to Fluorescent Antibody

- Fundamentally different imaging modality (absorption vs. emission)
- Different contrast mechanism (all cells vs. specific protein expression)
- Region boundaries may differ (antibody expression patterns do not always align with cytoarchitectural boundaries)
- **Mitigation:** Self-supervised atlas discovery on fluorescent data; pseudo-label generation; stain-agnostic preprocessing

### Lab-to-lab variation

- Tissue thickness, fixation protocol, staining time, scanner calibration all vary
- **Mitigation:** Stain normalization (Macenko/Reinhard for brightfield, percentile normalization for fluorescent); aggressive augmentation during training

---

## 15. Summary of Recommendations

| # | Recommendation | Impact | When |
|---|---------------|--------|------|
| 1 | Unfreeze last 4 backbone blocks with differential LR (1e-5 backbone, 1e-4 head) | High | Immediate — no code changes beyond optimizer setup |
| 2 | Add stain normalization (Macenko for Nissl, percentile for fluorescent) | High | Before any cross-lab deployment |
| 3 | Hierarchical loss (coarse + depth-2 + full heads) | High | After backbone unfreezing proves effective |
| 4 | Class-weighted cross-entropy + Dice loss | Medium | Alongside backbone unfreezing |
| 5 | Extended augmentation (elastic deformation, blur, 90-degree rotations) | Medium | Alongside backbone unfreezing |
| 6 | Integrate human Nissl data from Allen Human Brain Atlas | High | Phase 2 |
| 7 | Self-supervised atlas discovery for label-sparse domains | High | Phase 2, for fluorescent and new species |
| 8 | Overlapping tile inference with Gaussian blending | Medium | Phase 3, for whole-slide inference |
| 9 | Upgrade to Mask2Former segmentation head | Medium | Phase 3, if boundary quality insufficient |
| 10 | DINOv2-Giant backbone | Low-Medium | Phase 4, only if feature quality remains the bottleneck |

The single most important insight: **the biggest gains come from domain adaptation (unfreezing, stain normalization, cross-stain data) and loss design (hierarchical, class-weighted), not from model scaling.**

---

## References

### Models

- DINOv2: [github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) — Self-distilled Vision Transformers (Meta AI)
- Mask2Former: [arxiv.org/abs/2112.01527](https://arxiv.org/abs/2112.01527) — Masked-attention Mask Transformer for Universal Image Segmentation
- MaskDINO: [arxiv.org/abs/2206.02777](https://arxiv.org/abs/2206.02777) — Unified object detection and segmentation with DINO features

### Data Sources

- Allen Mouse Brain Atlas CCFv3: [atlas.brain-map.org](https://atlas.brain-map.org)
- Allen Human Brain Atlas: [human.brain-map.org](https://human.brain-map.org)
- BICCN: [biccn.org](https://biccn.org)

### Techniques

- Macenko stain normalization: Macenko et al., "A method for normalizing histology slides for quantitative analysis," ISBI 2009
- Dense CRF: Krahenbuhl & Koltun, "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials," NeurIPS 2011
- ViT-Adapter: Chen et al., "Vision Transformer Adapter for Dense Predictions," ICLR 2023
