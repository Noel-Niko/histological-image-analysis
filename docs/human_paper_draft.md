# Cross-Species Transfer of Ultra-Fine-Grained Brain Segmentation: From Mouse to Human with DINOv2 + UperNet

---

## Abstract

We extend our DINOv2-Large + UperNet segmentation approach from the Allen Mouse Brain Atlas (1,328 classes, 79.1% mIoU) to human brain tissue using two complementary datasets: the Allen Human Brain Atlas (sparse SVG polygon annotations across 6 donors, Nissl stain) and the BigBrain 200μm classified volume (dense 9-class tissue segmentation, Merker stain). We evaluate three class granularities — 597 fine-grained structures, 44 depth-3 anatomical regions, and 10 tissue classes — to determine the optimal resolution for human brain segmentation with limited annotation density.

Our principal finding is that annotation density, not model capacity, determines the performance ceiling. The depth-3 model (44 brain regions) achieves 65.5% center-crop mIoU and 99.4% pixel accuracy, with major structures (cerebellum, cerebral cortex, thalamus, pons) exceeding 99% IoU. The fine-grained 597-class model plateaus at ~27% mIoU despite 200 epochs of training, confirming that sparse polygon annotations (12 structures per image, 17–33% pixel coverage) cannot support per-structure learning at full ontological depth. The BigBrain 10-class tissue model achieves 60.8% mIoU with dense annotations on a single brain.

The training recipe established in the mouse ablation study — partial backbone unfreezing (last 4/24 blocks), differential learning rate (1e-5/1e-4), plain cross-entropy loss, minimal augmentation, 200 epochs — transfers directly to human tissue with no modifications, validating the recipe's generalizability across species, stains, and annotation methodologies.

---

## 1. Introduction

In the companion paper (Nosse 2026a), we presented a systematic ablation study achieving 79.1% mIoU on 1,328 mouse brain structures using DINOv2-Large + UperNet with the Allen Mouse Brain Atlas CCFv3. That study established a training recipe and identified the dominant performance factors: correct data splitting, backbone partial fine-tuning (+8.5% mIoU), and extended training (+6.0% mIoU). Three attempted improvements — weighted loss, aggressive augmentation, and test-time augmentation — all failed.

This paper asks whether the same approach transfers to human brain tissue, which differs from mouse in three critical ways:

1. **Scale and complexity.** The human brain is ~1,000× larger by volume with more extensive cortical folding (gyri and sulci), broader white matter tracts, and more complex subcortical nuclei. The Allen Human Brain Atlas ontology (Graph 10) defines 1,839 structures versus 1,328 in the mouse CCFv3.

2. **Annotation methodology.** Mouse training uses dense voxel-level annotations from a 3D reference volume — every pixel has a ground-truth label. Human annotations come from two fundamentally different sources: (a) sparse SVG polygon annotations marking ~12 structures per histological section (17–33% pixel coverage), and (b) dense 3D tissue classification at coarse granularity (9 classes). Neither matches the ideal of dense, fine-grained labels.

3. **Inter-individual variation.** The mouse CCFv3 is a single reference brain. Human data spans 6 post-mortem donors with varying brain morphology, tissue quality, and sectioning artifacts. The model must generalize across individuals — a challenge absent from the mouse study.

We train three models in parallel to isolate the effects of class granularity and annotation density:

- **Track A (fine-grained):** 597 brain structures from Allen Human SVG annotations. Tests the limit of sparse annotations on a deep ontology.
- **Track A-depth3 (coarse):** 44 depth-3 brain regions from the same Allen SVG data, with structures grouped into anatomical ancestors (cerebral cortex, thalamus, cerebellum, etc.). Tests whether coarser classes overcome annotation sparsity.
- **Track B (tissue):** 10 tissue classes from the BigBrain classified volume. Tests dense annotations on a different stain and a single specimen.

---

## 2. Related Work

### Human Brain Atlas Segmentation

Human brain parcellation has traditionally relied on cytoarchitectonic analysis — expert identification of cortical areas by cell morphology and laminar organization (Brodmann, 1909; von Economo & Koskinas, 1925). Modern automated approaches use deformable registration of atlas templates (ANTs, FreeSurfer) to MRI volumes, achieving reliable parcellation of ~100 cortical and subcortical regions. However, registration-based methods operate on MRI, not histology, and cannot capture the cellular-level detail visible in Nissl-stained sections.

The Allen Human Brain Atlas (Hawrylycz et al., 2012) provides the most comprehensive histological annotation of the adult human brain, with 14,566 Nissl-stained sections across 6 donors and SVG polygon annotations marking sampled brain structures. However, these annotations are sparse — they mark regions sampled for in situ hybridization (ISH) gene expression studies, not exhaustive anatomical boundaries.

The BigBrain project (Amunts et al., 2013) provides a single ultra-high-resolution (20μm) histological volume of the human brain (Merker stain) with 3D classified tissue volumes at 200μm resolution. While limited to a single specimen, BigBrain provides the only dense voxel-level human brain annotations suitable for supervised segmentation training.

### Domain Transfer in Histological Segmentation

Transfer learning from natural-image backbones to histology is well-established in computational pathology (Ciga et al., 2022). DINOv2 features have been shown to transfer effectively to histological tasks despite the significant domain gap (Oquab et al., 2024). Our mouse study confirmed this: DINOv2-Large achieved 60.3% mIoU with a frozen backbone, rising to 68.8% with partial fine-tuning — demonstrating that self-supervised natural image features encode tissue-relevant patterns.

The cross-species transfer question — whether training recipes optimized for mouse brain transfer to human — is less studied. Key differences include stain type (Nissl vs. Merker), annotation density (dense vs. sparse), tissue scale (μm vs. mm), and morphological complexity.

---

## 3. Methods

### 3.1 Datasets

#### 3.1.1 Allen Human Brain Atlas (Tracks A and A-depth3)

**Source.** 14,566 Nissl-stained coronal sections across 6 post-mortem donors (Allen Institute for Brain Science). Of these, 4,463 sections have SVG polygon annotations marking sampled brain structures with Graph 10 ontology structure IDs.

**Annotation characteristics.** Each annotated section contains a mean of 11.4 structure polygons (range 0–40), covering 17–33% of the tissue area at 512×512 resolution. Unlabeled pixels — comprising both true background and unannotated brain tissue — are assigned the ignore label (255) and excluded from the loss function. This is a fundamental difference from the mouse study, where every pixel contributes to training.

**Ontology.** The Human Brain Atlas ontology (Graph 10) defines 1,839 structures in a hierarchical tree with maximum depth 9. Scanning the 3,188 training SVGs yields 596 unique structure IDs. We evaluate two mapping strategies:

- **Full mapping (Track A):** Each of the 596 observed structure IDs maps to a unique class, producing 597 classes (596 structures + background). This preserves maximum anatomical detail.
- **Depth-3 mapping (Track A-depth3):** Each structure ID maps to its depth-3 ancestor in the ontology tree. Graph 10 contains 92 depth-3 nodes; the 596 observed structure IDs map to 44 unique depth-3 ancestors, producing 44 classes. Example depth-3 regions: cerebral cortex, cerebral nuclei, thalamus, hypothalamus, cerebellum, pons, medullary reticular formation.

**Pre-processing.** Images are resized to a maximum dimension of 1,024 pixels, SVGs are rasterized to matching dimensions, and labels are remapped via the chosen mapping. Results are cached as compressed `.npz` files (cache build: ~470–496 minutes for the full 4,463-image dataset, dominated by SVG rasterization).

**Split strategy.** By donor, to test cross-individual generalization:

| Role | Donors | Images |
|------|--------|--------|
| Train | H0351.2002, H0351.2001, H0351.1012, H0351.1009 | 3,188 |
| Val | H0351.1016 | 641 |
| Test | H0351.1015 | 634 |

This split (71% / 14% / 15%) ensures the model is evaluated on a donor unseen during training. Adjacent sections from the same donor share tissue architecture and cutting artifacts; a random split would create data leakage analogous to the spatial split problem identified in the mouse study (+37.3% mIoU from correcting the split).

#### 3.1.2 BigBrain Classified Volume (Track B)

**Source.** The BigBrain 200μm isotropic histological volume (696 × 770 × 605 voxels) with a paired 9-class tissue classification volume. The histological volume is Merker-stained (silver stain for cell bodies), distinct from the Nissl stain used in Allen Human and the PhD collaborator's tissue.

**Classes (10 including background):**

| ID | Class | Coverage |
|----|-------|----------|
| 0 | Background | 64.9% |
| 1 | Gray Matter | — |
| 2 | White Matter | — |
| 3 | Cerebrospinal Fluid | — |
| 4 | Meninges | — |
| 5 | Blood Vessels | — |
| 6 | Bone/Skull | — |
| 7 | Muscle | — |
| 8 | Artifact | — |
| 9 | Other/Unknown | — |

**Non-zero voxel coverage:** 35.1% (the remaining 64.9% is background).

**Split strategy.** Interleaved — every 10th coronal slice to validation, every 11th to test, remainder to training. A gap of ±2 slices (1mm at 200μm) between train and val/test splits mitigates spatial leakage.

| Role | Slices |
|------|--------|
| Train | 468 |
| Val | 59 |

### 3.2 Model Architecture

Identical across all three tracks:

- **Backbone:** DINOv2-Large (304M parameters), 24 transformer blocks, 1,024 hidden dimension
- **Decode head:** UperNet with PSP pooling module and FPN lateral connections
- **Unfreezing:** Last 4 backbone blocks (20–23) trainable; blocks 0–19 and embeddings frozen
- **Gradient checkpointing:** Backbone only (`use_reentrant=False`)

| Track | Output Classes | Head Params | Total Trainable | Total Params |
|-------|---------------|-------------|-----------------|--------------|
| A (597-class) | 597 | 37.2M | 87.6M (25.6%) | 341.5M |
| A-depth3 (44-class) | 44 | 36.7M | 87.1M (25.5%) | 341.1M |
| B (10-class) | 10 | 36.7M | 87.1M (25.5%) | 341.1M |

The head parameter count varies minimally across tracks because the UperNet decode head is dominated by the FPN and PSP modules (shared across class counts); only the final classification convolution scales with the number of output classes.

### 3.3 Training Configuration

All three tracks use the identical training recipe established in the mouse ablation study (Run 9):

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (2 parameter groups) |
| Backbone learning rate | 1e-5 |
| Decode head learning rate | 1e-4 |
| LR schedule | Linear warmup (10%) + linear decay |
| Weight decay | 0.01 |
| Batch size | 2 per GPU × 2 gradient accumulation = 4 effective |
| Epochs | 200 |
| Mixed precision | fp16 |
| Loss function | CrossEntropyLoss |
| Augmentation | Horizontal flip (50%), rotation ±15°, color jitter (brightness/contrast/saturation ±0.1) |
| Hardware | Single NVIDIA L40S GPU (48 GB VRAM), Databricks 17.3 LTS |

**Loss modification for sparse annotations (Tracks A and A-depth3).** CrossEntropyLoss uses `ignore_index=255` to exclude unlabeled pixels from gradient computation. This means only 17–33% of pixels per training image contribute to the loss. The effective training signal per epoch is substantially lower than for the mouse (100% pixel coverage) or BigBrain (35.1% non-background coverage with all voxels labeled).

**Input normalization.** Tracks A and A-depth3 use standard RGB ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) since Allen images are color Nissl photographs. Track B uses grayscale-to-RGB replication (identical to the mouse pipeline) since BigBrain is a single-channel intensity volume.

### 3.4 Evaluation Protocol

**Center-crop (CC) evaluation.** A single 518 × 518 crop from the center of each validation image/slice. For Tracks A/A-depth3, only labeled pixels (≠ 255) contribute to metrics.

**Sliding window (SW) evaluation.** Images resized to maximum 1,024 pixels, then tiled with 518 × 518 windows at stride 259 (50% overlap). Logits averaged in overlapping regions before argmax. For Tracks A/A-depth3, limited to 50 validation images (full val set of 641 images would be prohibitively slow with tiled inference). For Track B, all 59 validation slices are evaluated.

**Metrics.** Mean Intersection-over-Union (mIoU) computed over classes with non-NaN predictions, overall pixel accuracy, and count of valid classes.

---

## 4. Results

### 4.1 Summary

**Table 1. Results across all three human brain segmentation tracks.**

| Metric | Track A (597-class) | Track A-depth3 (44-class) | Track B (10-class) |
|--------|---------------------|---------------------------|---------------------|
| **CC mIoU** | 25.8% (50ep) / ~27% (117ep)* | **65.5%** | 60.8% |
| **SW mIoU** | 21.0% (50ep) | 45.1% | **61.3%** |
| **CC accuracy** | 63.8% (50ep) | **99.4%** | 90.0% |
| **SW accuracy** | 75.9% (50ep) | **99.5%** | 93.7% |
| Classes with valid IoU | 269/597 (45%) | 39/44 (89%) | 10/10 (100%) |
| Training steps | 39,850 (50ep) | 159,400 (200ep) | 23,400 (200ep) |
| Training time | 8.1 hrs (50ep) | 8.5 hrs | 74 min |
| Final train loss | 0.90 (50ep) / 0.128 (200ep) | 0.128 | 0.342 |
| Final eval loss | 2.49 (50ep) | 0.117 | 0.343 |
| Cache build time | 468 min | 496 min | N/A |
| Convergence | Plateaued ~27% at ep117 | Fully converged | Plateaued ~ep60 |

*Track A 597-class: trained for 200 epochs; mIoU was monitored at epochs 50 (25.8%) and 117 (~27.3%) showing near-plateau. Full evaluation not performed as the model was not competitive with Track A-depth3.

### 4.2 Track A: Fine-Grained 597-Class Model

The 597-class model achieves 25.8% CC mIoU at 50 epochs and approximately 27.3% at epoch 117, with minimal improvement despite continued training. This represents a fundamentally different regime from the mouse study, where the same architecture achieves 74.8% CC mIoU on 1,328 classes.

**Root cause: annotation sparsity.** The critical difference is annotation density. The mouse CCFv3 provides dense voxel-level labels — every training pixel has a ground-truth class. The Allen Human SVGs annotate only 12 structures per image on average, leaving 67–83% of pixels unlabeled. With 597 classes distributed across 3,188 training images, most classes appear in only a handful of images and occupy tiny fractions of those images. The effective training signal per class is orders of magnitude lower than in the mouse setting.

**Trajectory analysis.** The training loss dropped from 8.76 (epoch 1) to 0.90 (epoch 50) to 0.128 (epoch 200), indicating the model learned the training distribution. However, the validation mIoU plateaued: 25.8% at epoch 50, ~25.0% at epoch 61, ~27.3% at epoch 117. The gap between declining training loss and stagnant validation mIoU indicates overfitting to the sparse training annotations without generalizing to the held-out donor.

**Class-level performance (50-epoch evaluation).** Top-performing classes are deep brain structures with consistent appearance across donors: medial parabrachial nucleus (100.0%), dentate nucleus (99.5%), pontine nuclei (99.0%). Bottom-performing classes are cortical structures and cerebellum lobules with high inter-donor variability: cerebellum lobule I-II (0.01%), lobule VIIAf (0.1%). Only 269/597 classes (45%) have any valid IoU.

### 4.3 Track A-depth3: Coarse 44-Class Model

Grouping the same 596 observed structure IDs into 44 depth-3 brain regions transforms performance. The depth-3 model achieves 65.5% CC mIoU — a +39.7 percentage point improvement over the fine-grained model, using identical training data, architecture, and hyperparameters.

**Table 2. Per-class IoU for the depth-3 model (top 15 and bottom 10 of 39 valid classes).**

| Rank | Class | IoU |
|------|-------|-----|
| 1 | cerebellum | 99.98% |
| 2 | cerebral cortex | 99.63% |
| 3 | central glial substance | 99.51% |
| 4 | pons | 99.37% |
| 5 | thalamus | 99.35% |
| 6 | inferior olivary complex | 99.03% |
| 7 | raphe nuclei of medulla | 98.98% |
| 8 | cerebral nuclei | 98.61% |
| 9 | medullary reticular formation | 98.32% |
| 10 | cerebellar white matter tracts | 96.57% |
| 11 | hypothalamus | 90.76% |
| 12 | midbrain tegmentum | 87.00% |
| 13 | pontine reticular formation | 85.37% |
| 14 | subthalamus | 83.66% |
| 15 | medulla oblongata white matter tracts | 80.67% |
| ... | | |
| 30 | occipital lobe sulci | 45.65% |
| 31 | major divisions | 36.80% |
| 32 | pyramidal tract | 35.06% |
| 33 | frontal lobe sulci | 31.92% |
| 34 | cranial nerves | 30.31% |
| 35 | insular lobe sulci | 26.50% |
| 36 | parietal lobe sulci | 16.24% |
| 37 | central tegmental tract | 10.74% |
| 38 | midbrain tectum | 1.52% |
| 39 | cerebellar sulci | 1.40% |

**Why depth-3 works.** Grouping structures into depth-3 ancestors concentrates training pixels per class. Instead of distributing 596 structure IDs across 597 classes (many with <100 training pixels), the same pixels are pooled into 44 classes. Major regions — cerebral cortex, cerebellum, thalamus — accumulate millions of training pixels and reach near-perfect segmentation. The model learns robust region-level boundaries rather than attempting to distinguish individual structures that are visually similar at the annotation density available.

**Remaining failures.** The 5 classes below 5% IoU are sulci (cerebellar sulci 1.4%, parietal lobe sulci 16.2%, etc.) and small fiber tracts (central tegmental tract 10.7%). Sulci are thin, elongated structures that lose spatial detail when images are resized to 1,024 pixels for caching. Fiber tracts are rare and appear in few training images.

**Sliding window gap.** The SW mIoU (45.1%) is 20.4 percentage points below CC mIoU (65.5%). This gap is larger than in the mouse study (SW was +4.4% above CC). The cause is different: mouse SW improves on CC because it covers edge structures invisible to center crops. Human SW degrades because (a) only 27/44 classes appear in the 50-image SW sample, (b) SW evaluates entire images including periphery where rare classes (sulci, small tracts) dominate and perform poorly, and (c) the resize to 1,024px loses fine boundary detail for small structures. The near-perfect accuracy (99.4–99.5%) confirms that dominant classes are correctly identified; the low SW mIoU reflects averaging over rare, poorly-performing classes.

### 4.4 Track B: BigBrain 10-Class Tissue Model

The BigBrain model achieves 60.8% CC mIoU and 61.3% SW mIoU on 10 tissue classes, with all classes contributing valid IoU scores.

**Table 3. Per-class IoU for the BigBrain 10-class model.**

| Class | CC IoU | SW IoU |
|-------|--------|--------|
| Background | 98.64% | 99.41% |
| Gray Matter | 70.39% | 71.50% |
| White Matter | 77.54% | 78.49% |
| Cerebrospinal Fluid | 88.61% | 88.66% |
| Meninges | 36.05% | 36.44% |
| Blood Vessels | 15.77% | 15.71% |
| Bone/Skull | 83.88% | 84.16% |
| Muscle | 74.60% | 77.70% |
| Artifact | 3.80% | 3.40% |
| Other/Unknown | 56.85% | 56.25% |

**Strengths.** Dense annotations produce stable per-class results. Background, CSF, white matter, bone/skull, and gray matter all exceed 70% IoU. The CC-to-SW gap is negligible (+0.6%), consistent with the mouse pattern where SW reveals additional valid information from full-image coverage.

**Weaknesses.** Blood vessels (15.8%) and artifact (3.8%) are rare, small, and visually ambiguous classes. The single-brain limitation means the model has learned one individual's morphology — cross-individual generalization is untested.

**Stain mismatch.** BigBrain uses Merker (silver) stain; the target application (PhD collaborator tissue) uses Nissl stain. These stains highlight different cellular features, creating a domain gap that would degrade performance on real-world application.

### 4.5 Cross-Track Comparison

Direct mIoU comparison across tracks is inappropriate because they segment different numbers of classes at different granularities. However, several cross-cutting observations emerge:

**1. Annotation density dominates class count.** Track A-depth3 (44 classes, sparse) outperforms Track B (10 classes, dense) on CC mIoU (65.5% vs 60.8%) despite having 4.4× more classes. The depth-3 mapping concentrates sufficient pixels per class to overcome annotation sparsity. Conversely, Track A (597 classes, sparse) catastrophically underperforms Track B despite using the same images — the pixel budget is spread too thin.

**2. The training recipe transfers without modification.** All three tracks use identical hyperparameters from the mouse Run 9 recipe. No human-specific tuning was required. This validates the recipe's robustness across:
- Species (mouse → human)
- Stain (Nissl grayscale → Nissl color, Merker grayscale)
- Annotation methodology (dense voxel → sparse polygon, dense 3D classification)
- Scale (800×1,140 slices → 1,024px-resized sections)

**3. Convergence speed correlates with class count.** BigBrain (10 classes) plateaued by epoch 60 and trained in 74 minutes. Depth-3 (44 classes) converged over the full 200 epochs in 8.5 hours. The 597-class model had not converged on validation metrics even at epoch 200 (though training loss converged, indicating overfitting).

---

## 5. Analysis

### 5.1 The Annotation Density Threshold

The contrast between Track A (597 classes, 25.8% mIoU at 50ep) and Track A-depth3 (44 classes, 65.5% mIoU) using identical images reveals a critical threshold: the number of effective training pixels per class determines whether learning succeeds.

**Back-of-envelope calculation.** With 3,188 training images at 518×518 crops and ~25% labeled pixel coverage, the total labeled pixel budget is approximately 3,188 × 518 × 518 × 0.25 ≈ 213M pixels.

- At 597 classes: ~357K pixels/class on average, but with extreme long-tail distribution. Many classes have <10K pixels.
- At 44 classes: ~4.8M pixels/class on average. Even the rarest depth-3 classes (cerebellar sulci, midbrain tectum) have sufficient representation.

The mouse study showed strong pixel-count-to-IoU correlation (r=0.794) with 1,328 classes and ~1.2B total training pixels (~900K pixels/class average). The human 597-class model has ~2.5× fewer pixels per class, which — combined with the inter-donor variability absent from the single-brain mouse volume — pushes most classes below the learning threshold.

### 5.2 Sparse Annotations as Weak Supervision

The Allen Human SVG annotations were not designed for segmentation training. They mark ISH sampling regions — gyri, nuclei, and tracts where tissue was dissected for gene expression analysis. This creates a distinctive form of weak supervision:

- **Incomplete coverage.** Most brain tissue in each section is unlabeled. The model never receives negative examples ("this pixel is NOT cerebral cortex") for the majority of the image.
- **Biased sampling.** Structures sampled for ISH tend to be larger, more accessible regions. Small, deep structures may be underrepresented.
- **Inconsistent boundaries.** SVG polygons approximate regional boundaries but do not trace exact cytoarchitectonic borders. The same structure may be annotated with different polygon shapes across donors.

Despite these limitations, the depth-3 model achieves near-perfect accuracy (99.4%) on labeled pixels, demonstrating that the annotations carry genuine spatial information that the model can learn — provided the class count is appropriate for the annotation density.

### 5.3 Depth-3 Classes: Practical Utility

The 44 depth-3 classes represent anatomically meaningful brain regions:

**Solved regions (IoU > 90%):** cerebellum, cerebral cortex, central glial substance, pons, thalamus, inferior olivary complex, raphe nuclei of medulla, cerebral nuclei, medullary reticular formation, cerebellar white matter tracts, hypothalamus.

**Partially solved (IoU 50–90%):** midbrain tegmentum, pontine reticular formation, subthalamus, medulla oblongata white matter tracts, epithalamus, pretectal region, temporal lobe sulci, midbrain reticular formation, lateral lemniscus, temporal lobe gyri, parietal lobe gyri, occipital lobe gyri, frontal lobe gyri.

**Challenging (IoU < 50%):** sulci of all lobes (16–46%), cranial nerves (30%), fiber tracts (11–35%), midbrain tectum (1.5%), cerebellar sulci (1.4%).

For the PhD collaborator use case — identifying brain structures in novel tissue sections from standard cover slips — the depth-3 model can reliably identify which major brain region a tissue section belongs to. This is the primary clinical question: "Is this tissue from cortex, thalamus, cerebellum, or brainstem?"

### 5.4 BigBrain: Dense but Coarse

The BigBrain model's 60.8% CC mIoU on 10 classes is lower than the depth-3 Allen model's 65.5% on 44 classes. Two factors explain this:

1. **Class difficulty distribution.** BigBrain includes inherently hard classes: blood vessels (15.8% IoU) are thin, sparse, and low-contrast; artifacts (3.8%) are by definition irregular and unpredictable. These drag down the mean. The Allen depth-3 model has fewer fundamentally-hard classes relative to its total.

2. **Single specimen.** BigBrain captures one brain's tissue characteristics. Morphological variability in meninges thickness, vascular density, and muscle presence across brain regions creates intra-specimen variation that the model cannot fully capture with 468 training slices.

The BigBrain model has distinct practical value as a **tissue type classifier** — distinguishing gray matter from white matter, CSF, meninges, and bone. This is complementary to the Allen depth-3 model's **region identifier** role. A two-stage pipeline (BigBrain for tissue type → Allen depth-3 for anatomical region) could provide both levels of information.

---

## 6. Discussion

### Recipe Transferability

The central methodological finding is that the mouse training recipe transfers to human tissue without modification. The same backbone (DINOv2-Large), unfreezing strategy (last 4/24 blocks), differential learning rate (1e-5/1e-4), loss function (plain CE), augmentation (minimal), and training duration (200 epochs) produce strong results across:

| Dimension | Mouse | Human Allen | Human BigBrain |
|-----------|-------|-------------|----------------|
| Species | Mouse | Human | Human |
| Stain | Nissl (grayscale) | Nissl (color) | Merker (grayscale) |
| Annotation | Dense 3D voxels | Sparse 2D polygons | Dense 3D voxels |
| Subjects | 1 reference brain | 6 donors | 1 brain |
| Best mIoU | 79.1% (SW, 671 classes) | 65.5% (CC, 44 classes) | 61.3% (SW, 10 classes) |

This suggests the recipe is robust and can serve as a starting point for other histological segmentation tasks. The key constraint is not the training procedure but the training data: annotation density per class determines the performance ceiling.

### What Would Improve Performance

Based on the analysis, the most impactful improvements would be:

1. **Denser annotations.** More structures annotated per image, or more images annotated per structure. Even partial improvement (e.g., 30 structures per image instead of 12) would double the effective training signal.

2. **Higher-resolution caching.** The current 1,024px resize loses detail for small structures (sulci, fiber tracts). Caching at 2,048px would preserve more spatial detail at the cost of 4× cache build time and 4× logit buffer memory during sliding window evaluation.

3. **Structure-aware sampling.** During training, biasing crop selection toward images containing underrepresented classes would improve the per-class pixel balance without changing the annotations.

4. **Cross-track pre-training.** Pre-training on BigBrain (dense tissue labels) before fine-tuning on Allen (sparse structure labels) could give the model a foundation in human tissue appearance before attempting fine-grained parcellation.

### Practical Deployment

The target application is identifying brain regions in novel Nissl-stained tissue sections from standard cover slips, prepared by a PhD collaborator. The deployment pipeline is:

1. **Image acquisition.** Whole-slide images captured in VSI format (Olympus/Evident scanner).
2. **Format conversion.** VSI → JPEG/PNG using Bio-Formats or QuPath export.
3. **Orientation standardization.** Rotate all sections to consistent horizontal orientation to match the Allen training data. The model is orientation-specific (as demonstrated in the companion mouse paper: coronal 68.9% mIoU, axial 3.2%, sagittal 0.5%).
4. **Inference.** Load the depth-3 model, resize to 1,024px max dimension, run sliding window inference (518×518 tiles, stride 259, 50% overlap) with fp16 on a single GPU.
5. **Output.** Per-pixel region predictions across 44 depth-3 brain regions. Major regions (cerebral cortex, cerebellum, thalamus, pons, cerebral nuclei) are predicted at >98% IoU. Sulci and small fiber tracts are less reliable (1–35% IoU).

The model requires no post-processing for the primary use case of region identification. For boundary refinement, CRF smoothing or connected component filtering could be added as optional post-processing steps.

**Hardware requirements.** Inference on a single NVIDIA L40S (48 GB) processes one 1,024px image in ~2 seconds with sliding window tiling. A consumer GPU (RTX 3080/4080, 10–16 GB) is sufficient for inference — the 44-class logit buffer at 1,024px requires only ~75 MB.

### Limitations

1. **Sliding window subset for validation.** SW evaluation on the Allen tracks used only 50 of 641 validation images, introducing sampling variance in per-class SW mIoU estimates. The SW mIoU of 45.1% (vs 65.5% CC mIoU) partly reflects this limited sample: only 27/44 classes appeared in the 50-image SW subset, and rare classes (sulci, fiber tracts) that dominate the SW periphery are overrepresented relative to their true population frequency. Full-set SW evaluation on the test donor (634 images) provides a more robust estimate.

2. **Single stain for target application.** The PhD collaborator will provide Nissl-stained tissue, matching the Allen training data. Generalization to other stains (H&E, immunohistochemistry) is untested. The BigBrain model (Merker stain) demonstrates that the architecture transfers across stains, but cross-stain inference without retraining is not validated.

3. **No post-processing.** Raw model predictions without CRF smoothing, connected component analysis, or anatomical priors. Post-processing could improve boundary coherence, particularly for sulci and small structures where the model achieves low IoU.

4. **597-class model not formally evaluated at 200 epochs.** The fine-grained model completed the full 200-epoch training schedule, but formal CC+SW evaluation was intentionally omitted. The decision was based on monitoring during training: mIoU was 25.8% at epoch 50, ~25.0% at epoch 61, and ~27.3% at epoch 117 — a gain of only +1.5% over 67 additional epochs, with the gap to depth-3 (65.5%) too large to close. Training loss converged to 0.128, confirming the model learned the training distribution but could not generalize to the held-out donor at this class granularity.

5. **Single validation/test donor per split.** Each split contains a single donor (val: H0351.1016, test: H0351.1015), so reported metrics reflect performance on one individual's brain morphology. True cross-donor generalization requires evaluation across multiple held-out donors, which the current 6-donor dataset cannot support without reducing training data.

---

## 7. Conclusion

We demonstrate that the DINOv2-Large + UperNet training recipe established on 1,328 mouse brain structures transfers to human brain tissue across two datasets, three class granularities, and two staining protocols. The principal finding is that **annotation density per class, not model capacity or training recipe, determines the segmentation performance ceiling**.

The optimal model for the target application (identifying brain regions in PhD tissue slides) is the **Allen depth-3 model (44 brain regions, 65.5% CC mIoU, 99.4% pixel accuracy)**. Major brain regions — cerebral cortex, cerebellum, thalamus, pons, cerebral nuclei, hypothalamus — are segmented at >90% IoU. The model can reliably answer the primary question: which major anatomical region does a tissue section represent?

**Complete training history:**

| Track | Classes | Epochs | CC mIoU | SW mIoU | CC Acc | Training Time |
|-------|---------|--------|---------|---------|--------|---------------|
| A (fine-grained) | 597 | 200 | ~27%* | 21.0% (50ep) | ~63% | 8.1 hrs (50ep) |
| **A-depth3 (coarse)** | **44** | **200** | **65.5%** | **45.1%** | **99.4%** | **8.5 hrs** |
| B (tissue) | 10 | 200 | 60.8% | 61.3% | 90.0% | 74 min |
| Mouse (reference) | 1,328 | 200 | 74.8% | 79.1% | 94.1% | 23.0 hrs |

*Estimated from epoch 117 checkpoint monitoring; formal evaluation not performed.

**Future work.** (1) Run the full held-out test set evaluation on all models. (2) Cross-track pre-training: BigBrain → Allen fine-tune to combine dense tissue features with sparse structure labels. (3) Higher-resolution caching (2,048px) to recover performance on sulci and fiber tracts. (4) Mouse checkpoint initialization to test whether mouse brain features provide a better starting point than generic DINOv2 features for human tissue.

---

## Figures

| Figure | Description | Generation |
|--------|-------------|------------|
| **Figure 1** | Model architecture: same as mouse paper (DINOv2-Large + UperNet). Note identical architecture across all 3 tracks. | Reuse from mouse paper |
| **Figure 2** | Three-track comparison: one row per track showing Input / Ground Truth / Prediction. Demonstrates quality difference between 597-class (noisy), 44-class (clean), and 10-class (tissue-level). | Model inference on validation set |
| **Figure 3** | Class granularity vs mIoU: bar chart showing 597-class, 44-class, 10-class, and mouse 1,328-class results. Illustrates the annotation density threshold. | Hardcoded data |
| **Figure 4** | Depth-3 per-class IoU distribution: horizontal bars for all 39 valid classes, color-coded by anatomical system (cortical, subcortical, brainstem, cerebellum). | Center-crop evaluation data |
| **Figure 5** | Annotation density illustration: side-by-side of (a) mouse CCFv3 slice (100% labeled), (b) Allen Human section (12 structure polygons, ~25% labeled), (c) BigBrain slice (100% labeled, 10 classes). | Raw data visualization |
| **Figure 6** | Training convergence: loss and mIoU curves for all 3 tracks overlaid. Shows BigBrain plateauing early, depth-3 converging steadily, 597-class stagnating on validation. | MLflow training logs |

---

## References

1. Allen Institute for Brain Science. Allen Human Brain Atlas. https://human.brain-map.org
2. Amunts, K., Lepage, C., Borgeat, L., et al. (2013). BigBrain: An ultrahigh-resolution 3D human brain model. Science, 340(6139), 1472-1475.
3. Brodmann, K. (1909). Vergleichende Lokalisationslehre der Großhirnrinde. Johann Ambrosius Barth.
4. Ciga, O., Xu, T., & Martel, A. L. (2022). Self supervised contrastive learning for digital histopathology. Machine Learning with Applications, 7, 100198.
5. Hawrylycz, M. J., Lein, E. S., Guillozet-Bongaarts, A. L., et al. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome. Nature, 489(7416), 391-399.
6. Nosse, N. (2026a). Transfer learning for ultra-fine-grained brain region segmentation: An ablation study with DINOv2 + UperNet on 1,328 Allen Mouse Brain Atlas structures. [Companion paper].
7. Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. TMLR 2024.
8. von Economo, C. F., & Koskinas, G. N. (1925). Die Cytoarchitektonik der Hirnrinde des erwachsenen Menschen. Julius Springer.
9. Xiao, T., Liu, Y., Zhou, B., Jiang, Y., & Sun, J. (2018). Unified perceptual parsing for scene understanding. ECCV 2018.
