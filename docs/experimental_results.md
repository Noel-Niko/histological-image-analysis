# Experimental Results — All Training Runs

This document consolidates all training run data for the DINOv2-Large + UperNet semantic segmentation model on the CCFv3 Allen Brain Atlas. Data extracted from Databricks notebook outputs on 2026-03-14.

**Data correction (2026-03-14):** Run 5 overall accuracy was previously documented as 89.4% across multiple .md files. The notebook output shows **92.48%**. All references have been corrected. The eval_loss for Run 5 is **0.3631**, not "~0.41".

---

## Summary Table

| Run | Date | Config | Classes | Epochs | mIoU | Accuracy | Eval Loss | Train Loss | Runtime | Valid Classes |
|-----|------|--------|---------|--------|------|----------|-----------|------------|---------|---------------|
| 1 | 2026-03-10 | Coarse, frozen, spatial split | 6 | 50 | 50.7% | 78.8% | — | — | 23 min | 5/6 |
| 2 | 2026-03-10 | Coarse, frozen, interleaved | 6 | 50 | **88.0%** | **95.8%** | 0.175 | 0.240 | 23 min | 6/6 |
| 3 | 2026-03-10 | Depth-2, frozen | 19 | 50 | 69.6% | 95.5% | 0.194 | 0.154 | 25 min | 15/19 |
| 4 | 2026-03-11 | Full, frozen | 1,328 | 50 | 60.3% | 88.9% | 0.496 | 0.352 | 5.5 hrs | 503/1,328 |
| **5** | **2026-03-12** | **Full, unfrozen (4 blocks)** | **1,328** | **100** | **68.8%** | **92.5%** | **0.363** | **0.284** | **11.3 hrs** | **503/1,328** |
| 6 | 2026-03-12 | Full, weighted Dice+CE | 1,328 | 100 | 67.2% | 89.7% | 0.444 | 1.710 | ~33 hrs | 503/1,328 |
| 7 | 2026-03-15 | Full, extended augmentation | 1,328 | 100 | 62.3% | 90.1% | 0.489 | ~0.55 | ~11-14 hrs | 503/1,328 |

**Best model: Run 5** — 68.8% mIoU, 92.5% accuracy on 503/1,328 valid classes.

All runs on NVIDIA L40S 48 GB, single GPU, Databricks 17.3 LTS. Model: DINOv2-Large backbone (304M params) + UperNet decode head. Dataset: CCFv3 ara_nissl_10.nrrd, 1,320 coronal slices at 10μm, 518×518 random crops, interleaved train/val/test split (80/10/10).

---

## Test-Time Augmentation (TTA) — Eval-Only Experiment (2026-03-15)

**NEGATIVE RESULT.** 6-variant TTA on Run 5 model catastrophically degraded performance.

| Metric | Baseline (1 pass) | TTA (6 variants) | Delta |
|--------|-------------------|-------------------|-------|
| mIoU | 68.44% | 44.39% | **−24.05%** |
| Accuracy | 92.12% | 84.35% | −7.77% |
| Valid classes | 504 | 504 | — |

**TTA strategy:** Average logits from 6 variants: original, horizontal flip, rot90 k=0,1,2,3. fp16 inference on GPU, fp32 accumulation on CPU.

**Per-class impact:** 482/504 classes regressed, 8 improved, 14 unchanged. Mean delta: −24.05%, median: −18.24%, std: 20.91%.

**Top improvements (rare):**
| Class | Name | Baseline | TTA | Delta |
|-------|------|----------|-----|-------|
| 261 | Posterolateral visual area, layer 2/3 | 5.4% | 43.5% | +38.1% |
| 892 | Posterolateral visual area, layer 5 | 7.9% | 22.8% | +14.9% |
| 1059 | Primary somatosensory area, barrel field, layer 5 | 41.9% | 54.8% | +13.0% |

**Worst regressions:**
| Class | Name | Baseline | TTA | Delta |
|-------|------|----------|-----|-------|
| 387 | ventral tegmental decussation | 79.5% | 0.0% | −79.5% |
| 190 | Rostral linear nucleus raphe | 78.6% | 0.4% | −78.2% |
| 581 | Central linear nucleus raphe | 86.2% | 8.4% | −77.8% |

**Root cause:** The model was trained with only flip + rot15° + jitter. rot90 variants produce images the model has never seen. Brain coronal sections have strong directional priors (dorsal-ventral, medial-lateral) that are destroyed by 90° rotation. Effective TTA budget: 33% correct signal (original ×2), 17% plausible (hflip), 50% noise (rot90 k=1,2,3).

**Key insight:** Reinforces Run 7's −6.5% regression with rot90 augmentation during training. The model fundamentally lacks rotational equivariance beyond small angles.

---

## Hyperparameter Comparison

| Parameter | Runs 1-2 (Coarse) | Run 3 (Depth-2) | Run 4 (Full) | Runs 5,7 (Unfrozen) | Run 6 (Weighted) |
|-----------|-------------------|-----------------|--------------|---------------------|-----------------|
| Backbone | Frozen | Frozen | Frozen | Last 4 blocks unfrozen | Last 4 blocks unfrozen |
| Optimizer | AdamW | AdamW | AdamW | AdamW (2 param groups) | AdamW (2 param groups) |
| Backbone LR | — | — | — | 1e-5 | 1e-5 |
| Head LR | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Batch size | 8 | 8 | 4 | 2 (×2 grad accum = 4) | 2 (×2 grad accum = 4) |
| Epochs | 50 | 50 | 50 | 100 | 100 |
| Warmup ratio | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| FP16 | Yes | Yes | Yes | Yes | Yes |
| Grad checkpointing | No | No | No | Backbone only (use_reentrant=False) | Backbone only (use_reentrant=False) |
| Loss | CE | CE | CE | CE | 0.5×weighted_CE + 0.5×Dice |
| Augmentation | flip, rot15°, jitter | flip, rot15°, jitter | flip, rot15°, jitter | flip, rot15°, jitter | flip, rot15°, jitter |
| Split strategy | Run 1: spatial; Run 2: interleaved | interleaved | interleaved | interleaved | interleaved |

**Run 7 augmentation (extended):** flip (50%), rot90 (50%), rot15° (always), elastic deformation (30%, α=50, σ=5), Gaussian blur (30%, σ=0.5-2.0), color jitter (always).

**Trainable parameters:**
- Frozen backbone: ~15M (UperNet head only)
- Unfrozen (last 4 blocks): 88.1M / 342.1M (25.8%)

---

## Per-Class IoU Results

### Run 2 — Coarse (6 classes)

| Class ID | Name | IoU |
|----------|------|-----|
| 0 | Background | 0.921 |
| 1 | Cerebrum | 0.958 |
| 2 | Brain stem | 0.943 |
| 3 | Cerebellum | 0.929 |
| 4 | Fiber tracts | 0.730 |
| 5 | Ventricular systems | 0.801 |

### Run 3 — Depth-2 (19 classes, 15 valid)

| Class ID | Name | IoU |
|----------|------|-----|
| 0 | Background | 0.9119 |
| 1 | lateral ventricle | 0.8087 |
| 2 | interventricular foramen | NaN |
| 3 | third ventricle | 0.6946 |
| 4 | cerebral aqueduct | 0.7953 |
| 5 | fourth ventricle | 0.5787 |
| 6 | central canal, spinal cord/medulla | 0.0000 |
| 7 | Brain stem | 0.9447 |
| 8 | Cerebellum | 0.9301 |
| 9 | Cerebrum | 0.9580 |
| 10 | Interpeduncular fossa | NaN |
| 11 | cerebellum related fiber tracts | 0.7258 |
| 12 | cranial nerves | 0.6462 |
| 13 | lateral forebrain bundle system | 0.7712 |
| 14 | medial forebrain bundle system | 0.6499 |
| 15 | extrapyramidal fiber systems | 0.4443 |
| 16 | grooves of the cerebral cortex | NaN |
| 17 | grooves of the cerebellar cortex | NaN |
| 18 | supra-callosal cerebral white matter | 0.5815 |

### Run 4 — Full 1,328-class (frozen backbone), Top/Bottom 10

**Top 10:**

| Class ID | Name | IoU |
|----------|------|-----|
| 662 | Caudoputamen | 0.9524 |
| 0 | Background | 0.9429 |
| 497 | Main olfactory bulb | 0.9376 |
| 958 | Nodulus (X) | 0.9289 |
| 974 | Lobule III | 0.9198 |
| 1080 | Lobules IV-V | 0.9153 |
| 372 | Field CA1 | 0.9136 |
| 947 | Uvula (IX) | 0.9109 |
| 926 | Declive (VI) | 0.9073 |
| 785 | Periaqueductal gray | 0.8997 |

**Bottom 10:**

| Class ID | Name | IoU |
|----------|------|-----|
| 322 | Accessory supraoptic group | 0.0000 |
| 541 | Central amygdalar nucleus, lateral part | 0.0000 |
| 566 | Accessory facial motor nucleus | 0.0000 |
| 822 | oculomotor nerve | 0.0000 |
| 859 | Posterolateral visual area, layer 4 | 0.0000 |
| 945 | Lateral reticular nucleus, magnocellular part | 0.0000 |
| 968 | Paragigantocellular reticular nucleus, lateral part | 0.0000 |
| 1100 | Primary somatosensory area, trunk, layer 5 | 0.0000 |
| 1203 | Primary somatosensory area, unassigned, layer 4 | 0.0000 |
| 1290 | Frontal pole, layer 6b | 0.0000 |

### Run 5 — Full 1,328-class (unfrozen backbone, BEST), Top/Bottom 10

**Top 10:**

| Class ID | Name | IoU |
|----------|------|-----|
| 662 | Caudoputamen | 0.9717 |
| 497 | Main olfactory bulb | 0.9672 |
| 0 | Background | 0.9640 |
| 958 | Nodulus (X) | 0.9535 |
| 974 | Lobule III | 0.9395 |
| 947 | Uvula (IX) | 0.9363 |
| 154 | Anterior olfactory nucleus | 0.9353 |
| 372 | Field CA1 | 0.9341 |
| 195 | Medial vestibular nucleus | 0.9332 |
| 1080 | Lobules IV-V | 0.9322 |

**Bottom 10:**

| Class ID | Name | IoU |
|----------|------|-----|
| 270 | Retrosplenial area, lateral agranular part, layer 6b | 0.0000 |
| 541 | Central amygdalar nucleus, lateral part | 0.0000 |
| 566 | Accessory facial motor nucleus | 0.0000 |
| 740 | Posterolateral visual area, layer 1 | 0.0000 |
| 859 | Posterolateral visual area, layer 4 | 0.0000 |
| 945 | Lateral reticular nucleus, magnocellular part | 0.0000 |
| 968 | Paragigantocellular reticular nucleus, lateral part | 0.0000 |
| 1100 | Primary somatosensory area, trunk, layer 5 | 0.0000 |
| 1203 | Primary somatosensory area, unassigned, layer 4 | 0.0000 |
| 1290 | Frontal pole, layer 6b | 0.0000 |

### Run 6 — Full 1,328-class (weighted Dice+CE), Top/Bottom 10

**Top 10:**

| Class ID | Name | IoU |
|----------|------|-----|
| 662 | Caudoputamen | 0.9469 |
| 958 | Nodulus (X) | 0.9450 |
| 497 | Main olfactory bulb | 0.9331 |
| 0 | Background | 0.9303 |
| 926 | Declive (VI) | 0.9276 |
| 974 | Lobule III | 0.9263 |
| 947 | Uvula (IX) | 0.9202 |
| 372 | Field CA1 | 0.9164 |
| 1080 | Lobules IV-V | 0.9157 |
| 785 | Periaqueductal gray | 0.9078 |

**Bottom 10:**

| Class ID | Name | IoU |
|----------|------|-----|
| 322 | Accessory supraoptic group | 0.0000 |
| 541 | Central amygdalar nucleus, lateral part | 0.0000 |
| 566 | Accessory facial motor nucleus | 0.0000 |
| 740 | Posterolateral visual area, layer 1 | 0.0000 |
| 859 | Posterolateral visual area, layer 4 | 0.0000 |
| 945 | Lateral reticular nucleus, magnocellular part | 0.0000 |
| 968 | Paragigantocellular reticular nucleus, lateral part | 0.0000 |
| 1100 | Primary somatosensory area, trunk, layer 5 | 0.0000 |
| 1203 | Primary somatosensory area, unassigned, layer 4 | 0.0000 |
| 1290 | Frontal pole, layer 6b | 0.0000 |

### Run 7 — Full 1,328-class (extended augmentation), Top/Bottom 10

**Top 10:**

| Class ID | Name | IoU |
|----------|------|-----|
| 662 | Caudoputamen | 0.9595 |
| 497 | Main olfactory bulb | 0.9517 |
| 0 | Background | 0.9504 |
| 958 | Nodulus (X) | 0.9334 |
| 974 | Lobule III | 0.9238 |
| 947 | Uvula (IX) | 0.9193 |
| 926 | Declive (VI) | 0.9119 |
| 372 | Field CA1 | 0.9118 |
| 1080 | Lobules IV-V | 0.9108 |
| 785 | Periaqueductal gray | 0.9078 |

**Bottom 10:**

| Class ID | Name | IoU |
|----------|------|-----|
| 566 | Accessory facial motor nucleus | 0.0000 |
| 740 | Posterolateral visual area, layer 1 | 0.0000 |
| 822 | oculomotor nerve | 0.0000 |
| 859 | Posterolateral visual area, layer 4 | 0.0000 |
| 892 | Posterolateral visual area, layer 5 | 0.0000 |
| 968 | Paragigantocellular reticular nucleus, lateral part | 0.0000 |
| 1100 | Primary somatosensory area, trunk, layer 5 | 0.0000 |
| 1203 | Primary somatosensory area, unassigned, layer 4 | 0.0000 |
| 1213 | Anterior area, layer 6a | 0.0000 |
| 1290 | Frontal pole, layer 6b | 0.0000 |

---

## Training Curves

### Run 3 — Depth-2 (19 classes, frozen backbone, 50 epochs)

| Epoch | Train Loss | Val Loss | mIoU | Accuracy |
|-------|-----------|----------|------|----------|
| 1 | 1.911 | 1.991 | 18.9% | 79.0% |
| 5 | 0.398 | 0.484 | 48.8% | 90.7% |
| 10 | 0.268 | 0.318 | 57.9% | 93.1% |
| 15 | 0.235 | 0.274 | 61.8% | 93.8% |
| 20 | 0.197 | 0.249 | 63.6% | 94.1% |
| 25 | 0.182 | 0.228 | 67.0% | 94.7% |
| 30 | 0.175 | 0.216 | 67.0% | 94.9% |
| 35 | 0.175 | 0.209 | 67.8% | 95.1% |
| 40 | 0.184 | 0.203 | 68.8% | 95.2% |
| 45 | 0.163 | 0.197 | 69.2% | 95.4% |
| 50 | 0.154 | 0.194 | 69.6% | 95.5% |

### Run 4 — Full 1,328-class (frozen backbone, 50 epochs)

| Epoch | Train Loss | Val Loss | mIoU | Accuracy |
|-------|-----------|----------|------|----------|
| 1 | 7.352 | 7.953 | 1.1% | 31.7% |
| 5 | 1.638 | 1.875 | 21.9% | 71.1% |
| 10 | 0.818 | 0.972 | 39.2% | 80.7% |
| 15 | 0.661 | 0.765 | 47.7% | 83.7% |
| 20 | 0.514 | 0.669 | 51.3% | 85.1% |
| 25 | 0.539 | 0.602 | 53.8% | 86.5% |
| 30 | 0.406 | 0.564 | 56.2% | 87.2% |
| 35 | 0.397 | 0.535 | 58.2% | 87.9% |
| 40 | 0.460 | 0.521 | 58.7% | 88.3% |
| 45 | 0.386 | 0.507 | 59.5% | 88.6% |
| 50 | 0.352 | 0.496 | 60.3% | 88.9% |

### Run 5 — Full 1,328-class (unfrozen backbone, 100 epochs, BEST)

| Epoch | Train Loss | Val Loss | mIoU | Accuracy |
|-------|-----------|----------|------|----------|
| 1 | 7.838 | 8.581 | 1.1% | 28.4% |
| 5 | 2.508 | 2.919 | 13.2% | 63.9% |
| 10 | 1.133 | 1.308 | 32.0% | 77.8% |
| 15 | 0.730 | 0.847 | 44.8% | 82.8% |
| 20 | 0.540 | 0.687 | 49.7% | 84.9% |
| 25 | 0.536 | 0.609 | 53.9% | 86.6% |
| 30 | 0.394 | 0.545 | 57.0% | 87.8% |
| 40 | 0.417 | 0.491 | 60.0% | 88.9% |
| 50 | 0.310 | 0.451 | 62.1% | 90.1% |
| 60 | 0.351 | 0.426 | 63.5% | 90.7% |
| 70 | 0.321 | 0.399 | 66.0% | 91.5% |
| 75 | 0.296 | 0.391 | 66.6% | 91.6% |
| 80 | 0.284 | 0.382 | 67.0% | 91.8% |
| 90 | 0.270 | 0.368 | 68.3% | 92.3% |
| 95 | 0.302 | 0.365 | 68.5% | 92.4% |
| 100 | 0.284 | 0.363 | 68.7% | 92.5% |

### Run 6 — Full 1,328-class (weighted Dice+CE, 100 epochs)

| Epoch | Train Loss | Val Loss | mIoU | Accuracy |
|-------|-----------|----------|------|----------|
| 1 | 10.982 | 5.434 | 0.7% | 3.6% |
| 5 | 5.741 | 2.961 | 18.4% | 33.6% |
| 10 | 3.009 | 1.511 | 35.6% | 62.7% |
| 25 | 1.493 | 0.747 | 51.8% | 81.2% |
| 50 | 1.047 | 0.537 | 60.5% | 86.4% |
| 75 | 0.907 | 0.470 | 65.0% | 88.8% |
| 100 | 0.861 | 0.448 | 67.1% | 89.7% |

Note: Run 6 uses combined Dice+CE loss, so training loss values are not directly comparable with CE-only runs.

### Run 7 — Full 1,328-class (extended augmentation, 100 epochs)

| Epoch | Train Loss | Val Loss | mIoU | Accuracy |
|-------|-----------|----------|------|----------|
| 1 | 7.913 | 8.758 | 0.6% | 23.7% |
| 5 | 3.165 | 3.473 | 8.7% | 57.9% |
| 10 | 1.604 | 1.714 | 24.1% | 71.8% |
| 25 | 0.828 | 0.806 | 46.0% | 82.9% |
| 50 | 0.525 | 0.596 | 56.6% | 87.5% |
| 75 | 0.496 | 0.528 | 59.9% | 89.0% |
| 100 | 0.553 | 0.491 | 62.3% | 90.1% |

---

## Runs 1 & 2 — Coarse (6 classes)

### Run 1 — Spatial Split (BROKEN)

mIoU 50.7%, accuracy 78.8%, 50 epochs, 23 min. Per-class IoU: Background 53.5%, Cerebrum **NaN**, Brain stem 74.1%, Cerebellum 70.1%, Fiber tracts 41.1%, Ventricular systems 14.7%.

**Root cause:** Contiguous spatial split along AP axis put ALL cerebrum pixels in training (0 in val/test). The mouse cerebrum is concentrated anteriorly; the posterior 20% (val+test) contains only brain stem, cerebellum, fiber tracts, and ventricular systems.

**Fix:** `split_strategy="interleaved"` — every 10th slice → val, every 10th+1 → test. All 6 classes present in all splits.

Training curve data: NOT AVAILABLE (notebook outputs not preserved).

### Run 2 — Interleaved Split (CORRECT)

mIoU 88.0%, accuracy 95.8%, eval_loss 0.175, training_loss 0.240, 50 epochs, 23 min. All 6/6 classes valid.

Training curve data: NOT AVAILABLE (notebook outputs not preserved).

---

## Key Observations

### Class Distribution

- **1,328 total classes** (1,327 brain structures + background)
- **503 classes** have non-NaN IoU in validation (consistent across Runs 4-7)
- **657 classes** have zero training pixels — unlearnable regardless of technique
- **825 classes** absent from validation split
- Top classes (Caudoputamen, Main olfactory bulb, Background) are consistently >93% IoU across all runs
- Bottom classes are consistently 0.0% — small structures with insufficient pixels

### Ablation Insights

| Comparison | Delta mIoU | Interpretation |
|-----------|-----------|----------------|
| Frozen → Unfrozen (Run 4→5) | **+8.5%** | Largest single improvement. Backbone adaptation critical for domain shift. |
| 50 → 100 epochs (Run 4 still improving at 50) | ~+2-3% est. | Run 4 was still improving; Run 5 had more epochs AND unfreezing. |
| CE → Weighted Dice+CE (Run 5→6) | **−1.6%** | Dice diluted CE gradient on common classes. |
| Baseline aug → Extended aug (Run 5→7) | **−6.5%** | Augmentation too aggressive for 1,016 samples. |

### Convergence Behavior

- **Runs 4, 5:** Val loss still declining at final epoch — both could benefit from more epochs (especially Run 4).
- **Run 6:** Converged slowly (combined loss makes optimization harder). Still improving at epoch 100.
- **Run 7:** Val loss plateaued ~epoch 90. mIoU plateau at ~62.3%.
- **Run 5 vs 7 initial loss:** Both start at ~7.9-8.6 val loss, but Run 5 converges faster. Extended augmentation slows early convergence.

---

## Data Gaps

| Data Point | Status |
|-----------|--------|
| Runs 1-2 training curves | LOST — notebook outputs not preserved |
| Run 1 eval_loss | LOST |
| Run 3 full per-class IoU (all 19) | CAPTURED above |
| Run 4 full per-class IoU (503) | Only top/bottom 10 captured; full data in notebook |
| Run 5 full per-class IoU (503) | Only top/bottom 10 captured; full data in notebook |
| Class pixel distribution (train/val/test) | NOT YET COMPUTED — needs data scan |
| finetune_coarse.ipynb, finetune_depth2.ipynb | Were never run on Databricks; actual runs used step10_*.ipynb and step9 notebooks |

---

## Source Notebooks

| Run | Notebook with outputs | Location |
|-----|----------------------|----------|
| 1 | Original step9 notebook (outputs lost) | — |
| 2 | Original step9 notebook (outputs lost) | — |
| 3 | `step10_depth2.ipynb` | `notebooks/historical/` |
| 4 | `step10_full.ipynb` | `notebooks/historical/` |
| 5 | `finetune_unfrozen.ipynb` | `notebooks/` (active) |
| 6 | `finetune_weighted_loss.ipynb` | `notebooks/historical/` |
| 7 | `finetune_augmented.ipynb` | `notebooks/` (active) |
