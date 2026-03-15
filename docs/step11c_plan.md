# Plan: Step 11c — Extended Augmentation

## Context

The DINOv2-Large + UperNet model achieves 68.8% mIoU on the full 1,328-class Allen Brain Atlas mapping (Run 5, unfrozen backbone). Step 11b (weighted Dice+CE loss) regressed to 67.2% — the loss reweighting approach did not help. The best model remains Run 5.

The current augmentation pipeline (`dataset.py:155-190`) has three transforms: horizontal flip (50%), rotation ±15°, and color jitter (brightness/contrast ±0.2). The `docs/finetuning_recommendations.md:217-228` recommends extending this with elastic deformation, Gaussian blur, and random 90° rotations — all physically motivated for histological brain tissue.

With only 1,016 training samples at 518×518 crops, extended augmentation provides more training diversity at zero extra data cost. This is orthogonal to loss function changes and the most promising next step.

**Why no stain jitter?** The `finetuning_recommendations.md:225` also recommends "Stain jitter (if using Macenko)". This is excluded because the CCFv3 Nissl data is grayscale — Macenko color deconvolution requires H&E or similar multi-stain RGB images. The existing color jitter (brightness/contrast ±0.2) already covers the relevant intensity variation for grayscale Nissl.

---

## Files to Create/Modify

| File | Action | What |
|------|--------|------|
| `tests/test_dataset.py` | **Modify** | Add `TestExtendedAugmentation` class (~12 tests, TDD — write first) |
| `src/histological_image_analysis/dataset.py` | **Modify** | Add 3 helper methods + update `_apply_augmentation()` pipeline + docstring |
| `pyproject.toml` | **Modify** | Add `scipy>=1.10` as explicit dependency |
| `notebooks/finetune_augmented.ipynb` | **Create** | Training notebook based on `finetune_unfrozen.ipynb` |
| `Makefile` | **Modify** | Add `deploy-notebook-augmented` target |
| `docs/step11c_plan.md` | **Create** | This plan |
| `docs/progress.md` | **Modify** | Step 11c status |

**Files NOT modified:** `training.py`, `losses.py`, `ontology.py`, `ccfv3_slicer.py`.

---

## Implementation Details

### 1. New augmentation helpers in `dataset.py`

All methods are private on `BrainSegmentationDataset`, consistent with existing `_random_crop`, `_center_crop`, `_pad_if_needed` pattern.

**New imports:** `from scipy.ndimage import gaussian_filter, map_coordinates`

**`_random_rot90(self, image, mask)` → (image, mask)**
- `np.rot90(image, k=randint(0,3))` — exact transform, no interpolation, no fill artifacts
- `.copy()` for contiguous memory (same pattern as existing flip at line 166-167)
- `k=0` → short-circuit, no copy needed

**`_gaussian_blur(self, image, sigma_range=(0.5, 2.0))` → image**
- `scipy.ndimage.gaussian_filter()` with random sigma from range
- Image only — mask is never blurred
- Cast to float32 before filter, clip to [0,255], cast back to uint8

**`_elastic_deform(self, image, mask, alpha=50.0, sigma=5.0)` → (image, mask)**
- Random displacement fields `dx, dy` from `np.random.standard_normal`, smoothed with `gaussian_filter(sigma=sigma)`, scaled by `alpha`
- `scipy.ndimage.map_coordinates` with `order=1` (bilinear) for image, `order=0` (nearest-neighbor) for mask
- `mode='constant', cval=0` — out-of-bounds fills with 0 (Background class)
- `alpha=50, sigma=5` are standard for 512×512 medical images

### 2. Updated `_apply_augmentation()` pipeline

Order: geometric transforms first → intensity transforms last.

```
1. Horizontal flip          — 50% probability  (existing)
2. Random 90° rotation      — 50% probability  (NEW)
3. Rotation ±15°            — always sampled, skip if |angle| ≤ 0.5°  (existing)
4. Elastic deformation      — 30% probability  (NEW)
5. Gaussian blur            — 30% probability  (NEW, image only)
6. Color jitter             — always applied    (existing, image only)
```

90° rotation before ±15° continuous rotation: establishes coarse orientation first, then fine angular variation. Elastic after rotation: adds local warping on top of global orientation.

**Docstring update:** The existing `_apply_augmentation()` docstring (`dataset.py:158-162`) lists only the original 3 transforms. This must be updated to document all 6 transforms in the pipeline.

### 3. `pyproject.toml` change

Add `"scipy>=1.10"` to `dependencies`. Already transitively available via svgpathtools, but per 12-Factor App principle 2 (explicit dependencies), it should be declared directly. Both local venv (1.17.1) and Databricks cluster (1.15.1) already have it.

### 4. Notebook (`finetune_augmented.ipynb`)

Based on `finetune_unfrozen.ipynb`. This is Run 7 — same hyperparameters as Run 5 but with extended augmentation. The only way augmentation changes affect the notebook is through the wheel (augmentation is in `dataset.py`).

| Cell | Change from `finetune_unfrozen.ipynb` |
|------|---------------------------------------|
| Markdown header | Title updated: "Extended Augmentation" |
| Cell 1 (Config) | `HYPERPARAMS["augmentation"]` = description string; `OUTPUT_DIR` → `augmented`; `FINAL_MODEL_DIR` → `augmented`; run_name → `augmented-...` |
| Cell 3 (Data) | No change — `augment=True` triggers all augmentation |
| Cell 6 (Eval) | Comparison baseline: Run 5 = 68.8% mIoU |
| **Cell 7 (Save)** | **CRITICAL:** Use corrected MLflow save pattern (see below) |

**Cell 7 corrected save pattern** (per artifact recovery notes):

1. Set `model.config.id2label` and `model.config.label2id` from class names
2. `trainer.save_model(FINAL_MODEL_DIR)` as backup
3. `processor.save_pretrained(FINAL_MODEL_DIR)` — save image processor
4. `mlflow.transformers.log_model()` with model + image_processor, `name="model"`, `task="image-segmentation"`
5. Do NOT use `mlflow.log_artifacts()` or `registered_model_name`

---

## Test Plan (TDD — Write First)

### `TestExtendedAugmentation` in `tests/test_dataset.py` (~12 tests)

**Elastic deformation (4 tests):**
- `test_elastic_deform_preserves_shape` — output shape == input shape
- `test_elastic_deform_preserves_mask_dtype` — mask remains int64
- `test_elastic_deform_preserves_mask_values` — no new class IDs introduced (10 iterations). Output IDs ⊆ (original IDs ∪ {0})
- `test_elastic_deform_changes_image` — output differs from input

**Gaussian blur (3 tests):**
- `test_gaussian_blur_preserves_shape` — shape unchanged
- `test_gaussian_blur_returns_uint8` — dtype preserved
- `test_gaussian_blur_reduces_high_frequency` — Laplacian magnitude decreases

**90° rotation (3 tests):**
- `test_rot90_preserves_shape` — square crop shape unchanged
- `test_rot90_preserves_mask_integrity` — exact transform, class IDs identical (10 iterations)
- `test_rot90_applies_all_orientations` — over 100 trials, ≥3 of 4 orientations seen

**Full pipeline integration (2 tests):**
- `test_extended_augmentation_preserves_mask_integrity_full_pipeline` — 20 iterations through `ds[0]`, dtype=long, min≥0
- `test_extended_augmentation_stochastic_variation` — 10 calls produce different outputs

Tests access helper methods via `ds._elastic_deform(image, mask)` etc., getting raw arrays from `ds._slices[0]` then padding/cropping to 518×518. Integration tests use `ds[0]` through the full pipeline.

---

## Performance Impact

| Operation | Time per sample | Probability | Expected time |
|-----------|----------------|-------------|---------------|
| Horizontal flip | ~0.3 ms | 50% | 0.15 ms |
| 90° rotation | ~0.01 ms | 50% | 0.005 ms |
| Rotation ±15° | ~2.5 ms | ~47% | 1.2 ms |
| Elastic deformation | ~21 ms | 30% | 6.3 ms |
| Gaussian blur | ~1.8 ms | 30% | 0.54 ms |
| Color jitter | ~0.2 ms | 100% | 0.2 ms |
| **Total expected** | | | **~8.4 ms** |

With `dataloader_num_workers=4` (default in `get_training_args()` at `training.py:380`), effective per-sample overhead is ~2.1 ms. GPU training step takes ~1600 ms. Augmentation adds <1% overhead.

---

## Implementation Status

| Step | Task | Status |
|------|------|--------|
| 0 | Create `docs/step11c_plan.md` | DONE |
| 1 | Write `TestExtendedAugmentation` tests (12 tests) | DONE — 12 tests |
| 2 | Add `scipy>=1.10` + `accelerate>=1.13.0` to `pyproject.toml` | DONE |
| 3 | Add `from scipy.ndimage import ...` to `dataset.py` | DONE |
| 4 | Implement `_random_rot90()` | DONE |
| 5 | Implement `_gaussian_blur()` | DONE |
| 6 | Implement `_elastic_deform()` | DONE |
| 7 | Update `_apply_augmentation()` pipeline + docstring | DONE |
| 8 | Run full test suite | DONE — 181/181 pass |
| 9 | Create `notebooks/finetune_augmented.ipynb` | DONE |
| 10 | Update Makefile | DONE — `deploy-notebook-augmented` target |
| 11 | Update `docs/progress.md` | DONE |
| 12 | Final `make test` | DONE — 181/181 pass |
| 13 | Deploy + run on Databricks (Run 7) | DONE — **NEGATIVE RESULT** (62.3% mIoU, −6.5% vs Run 5) |

---

## Run 7 Results (2026-03-15)

**NEGATIVE RESULT — extended augmentation regressed mIoU by 6.5%.**

### Key Metrics

| Metric | Run 7 (augmented) | Run 5 (unfrozen baseline) | Delta |
|--------|-------------------|---------------------------|-------|
| mIoU | **62.35%** | **68.8%** | **−6.5%** |
| Overall accuracy | 90.1% | 92.5% | −2.4% |
| Eval loss | 0.4894 | 0.3631 | +0.13 |
| Valid classes | 503 / 1,328 | 503 / 1,328 | 0 |
| Training loss (final) | ~0.55 | 0.284 | — |
| Epochs | 100 (no early stop) | 100 | — |

### Training Curve Summary

| Epoch | Train Loss | Val Loss | mIoU | Accuracy |
|-------|-----------|----------|------|----------|
| 1 | 7.913 | 8.758 | 0.6% | 23.7% |
| 10 | 1.604 | 1.714 | 24.1% | 71.8% |
| 25 | 0.828 | 0.806 | 46.0% | 82.9% |
| 50 | 0.525 | 0.596 | 56.6% | 87.5% |
| 74 | 0.509 | 0.520 | 60.6% | 89.3% |
| 88 | 0.486 | 0.496 | 62.1% | 90.0% |
| 99 | 0.506 | 0.489 | **62.3%** | 90.1% |
| 100 | 0.553 | 0.491 | 62.3% | 90.1% |

Model was still improving marginally at epoch 100 (val loss declined from 0.496 at epoch 88 to 0.489 at epoch 99), but mIoU plateaued at ~62.3%. More epochs unlikely to close the 6.5% gap.

### Top 10 Classes by IoU

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

### Bottom 10 Classes by IoU

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

### Analysis: Why Extended Augmentation Regressed

1. **Augmentation too aggressive for small dataset.** With only 1,016 training samples and 6 simultaneous augmentations (elastic deformation, blur, rot90, flip, rot15°, jitter), the model rarely sees the same sample in a recognizable form across epochs. This makes it harder to build consistent feature representations.

2. **Elastic deformation distorts anatomical boundaries.** α=50 with σ=5 creates significant local warping. Unlike cell-level pathology (where elastic deformation helps because cells have random shapes), brain anatomy has precise spatial relationships. Warping a cortical layer boundary by ±50 pixels can make the segmentation label ambiguous — the model learns conflicting signals.

3. **High combined augmentation probability.** Any given sample has a ~30% chance of elastic + ~30% blur + ~50% rot90 + ~50% flip + always rot15° + always jitter. The compounding effect is severe — most training samples are heavily distorted.

4. **Accuracy and mIoU both dropped.** Overall accuracy dropped from 92.5% to 90.1% (−2.4%) alongside the mIoU regression (−6.5%). The augmentation noise degraded both coarse and fine-grained predictions.

5. **Initial loss much higher.** Epoch 1 train loss was 7.91 — substantially higher than Run 5 would have been from the same pretrained weights. The augmentation made early training signal much noisier, slowing convergence from the start.

### MLflow Artifacts

Corrected save pattern confirmed working:
- Model + image processor saved to `/dbfs/FileStore/allen_brain_data/models/augmented`
- MLflow model logged: [link](https://grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com/ml/experiments/1345391216675532/models/m-afe1ecc9de134bb680d9ecf40e913682?o=2648165776397546)

### Comparison Table (All Full-Mapping Runs)

| Run | Config | mIoU | Accuracy | Eval Loss | Valid Classes |
|-----|--------|------|----------|-----------|---------------|
| 4 | Frozen backbone | 60.3% | 88.9% | — | 503 |
| **5** | **Unfrozen backbone** | **68.8%** | **92.5%** | **0.3631** | **503** |
| 6 | Weighted Dice+CE | 67.2% | 89.7% | 0.4444 | 503 |
| 7 | Extended augmentation | 62.3% | 90.1% | 0.4894 | 503 |

**Best model remains Run 5 (unfrozen baseline): 68.8% mIoU.**
