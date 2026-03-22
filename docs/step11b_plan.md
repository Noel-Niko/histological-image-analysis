# Plan: Step 11b — Class-Weighted Dice + CE Loss

## Context

The DINOv2-Large + UperNet model achieves 68.8% mIoU on the full 1,328-class Allen Brain Atlas mapping (Run 5, unfrozen backbone). The dataset is severely imbalanced: Background and Cerebrum dominate pixel counts, 825 classes have zero validation pixels, and the flat CrossEntropyLoss provides no incentive to learn rare structures. A combined weighted-CE + Dice loss directly addresses this. Expected improvement: +5-10% mIoU on rare classes with minimal degradation on common classes.

---

## Architecture Decision: Subclass UperNet

**Why not override Trainer's `compute_loss()`?** UperNet's `forward()` computes auxiliary logits internally and discards them — `SemanticSegmenterOutput` only returns main `logits`, not `auxiliary_logits`. Overriding the Trainer would only apply custom loss to the main head (60% of loss signal), leaving the auxiliary head on standard CE.

**Solution:** Subclass `UperNetForSemanticSegmentation`, override `forward()` to replace the 5-line loss computation block with `CombinedDiceCELoss` applied to both main and auxiliary logits. The Trainer reads `output.loss` as before — no Trainer changes needed.

---

## Files to Create/Modify

| File | Action | What |
|------|--------|------|
| `tests/test_losses.py` | **Create** | TDD tests for all loss functions (write first) |
| `src/histological_image_analysis/losses.py` | **Create** | `DiceLoss`, `CombinedDiceCELoss`, `compute_class_weights_from_dataset()` |
| `tests/test_training.py` | **Modify** | Add `TestWeightedLossUperNet` class (~9 tests) |
| `src/histological_image_analysis/training.py` | **Modify** | Add `WeightedLossUperNet` subclass, update `create_model()` with `loss_fn` param |
| `notebooks/finetune_weighted_loss.ipynb` | **Create** | Training notebook (based on `finetune_unfrozen.ipynb`) |
| `Makefile` | **Modify** | Add `deploy-notebook-weighted-loss` target |
| `docs/step11b_plan.md` | **Create** | This plan |
| `docs/progress.md` | **Modify** | Update Step 11b status |

---

## Implementation Details

### 1. `losses.py` — Three Components

**`compute_class_weights_from_dataset(dataset, num_labels, ignore_index, clip_percentile)`**
- Single pass over training dataset, counting pixels per class via `np.bincount()`
- Inverse-frequency: `weight = 1.0 / (count + epsilon)`
- Zero-count classes get weight 0.0, clip at percentile, normalize to sum=num_labels

**`DiceLoss(ignore_index, smooth, chunk_size)`**
- Soft Dice using softmax probabilities (differentiable)
- Chunked computation to avoid materializing full one-hot tensor
- Only classes present in batch contribute to mean Dice

**`CombinedDiceCELoss(class_weights, alpha, ignore_index, dice_smooth, dice_chunk_size)`**
- `L = alpha * weighted_CE + (1 - alpha) * Dice`
- Both components respect `ignore_index`

### 2. `WeightedLossUperNet` in `training.py`

Override only the loss block in `forward()`:
```python
# Replaces:
loss_fct = CrossEntropyLoss(ignore_index=self.config.loss_ignore_index)
loss = loss_fct(logits, labels)

# With:
loss = self._custom_loss_fn(logits, labels)
```

Applied to both main and auxiliary heads.

### 3. Memory Analysis

Additional ~1.4 GB for softmax probabilities (fp16, batch=2). Fits within 22 GB headroom on L40S 48 GB.

---

## Implementation Status

| Step | Task | Status |
|------|------|--------|
| 0 | Create step11b plan | DONE |
| 1 | Write `tests/test_losses.py` (TDD) | DONE — 27 tests |
| 2 | Implement `losses.py` | DONE — 27/27 pass |
| 3 | Add `TestWeightedLossUperNet` to `test_training.py` | DONE — 9 tests |
| 4 | Implement `WeightedLossUperNet` in `training.py` | DONE — 48/48 pass |
| 5 | Update `create_model()` | DONE — backward compatible |
| 6 | Run full test suite | DONE — 169/169 pass |
| 7 | Create notebook | DONE — `finetune_weighted_loss.ipynb` |
| 8 | Update Makefile | DONE — `deploy-notebook-weighted-loss` target |
| 9 | Update docs/progress.md | DONE |
| 10 | Deploy + run on Databricks | DONE — Run 6 completed |

---

## Run 6 Results — NEGATIVE RESULT

**Run 6 (weighted Dice+CE, 2026-03-12):** mIoU **67.2%** (−1.6% vs unfrozen 68.8%)

| Metric | Frozen (Run 4) | Unfrozen (Run 5) | Weighted Loss (Run 6) |
|--------|----------------|-------------------|----------------------|
| mIoU | 60.3% | 68.8% | **67.2%** |
| Overall Accuracy | 88.9% | 92.5% | 89.7% |
| Eval Loss | 0.4964 | 0.3631 | 0.4444 |
| Training Loss | 0.352 | 0.284 | 1.710 |
| Valid classes (non-NaN IoU) | 503 | 503 | 503 |
| Runtime | 5.5 hrs | 11.3 hrs | ~33 hrs |
| Epochs | 50 | 100 | 100 |

### Top 10 classes by IoU
| Class | IoU |
|-------|-----|
| Caudoputamen | 94.7% |
| Nodulus (X) | 94.5% |
| Main olfactory bulb | 93.3% |
| Background | 92.0% |
| Uvula (IX) | 91.7% |
| Lobule III | 90.8% |
| Pontine reticular nucleus | 90.1% |
| Medial vestibular nucleus | 89.9% |
| Declive (VI) | 89.8% |
| Lobules IV-V | 89.8% |

### Bottom 10 classes by IoU
| Class | IoU |
|-------|-----|
| Primary visual area, layer 2/3 | 15.3% |
| Retrosplenial area, lateral agranular part, layer 6b | 14.7% |
| Posterolateral visual area, layer 2/3 | 14.0% |
| Primary somatosensory area, trunk, layer 5 | 13.2% |
| Central canal, spinal cord/medulla | 9.1% |
| Posterolateral visual area, layer 4 | 9.0% |
| Inferior olivary complex | 0.0% |
| Central amygdalar nucleus, lateral part | 0.0% |
| Paragigantocellular reticular nucleus, lateral part | 0.0% |
| Frontal pole, layer 6b | 0.0% |

### Analysis

The weighted Dice+CE loss **regressed −1.6% mIoU** from the unfrozen baseline. Key observations:

1. **No new classes discovered:** 503 classes with valid IoU — identical to Run 5. The 657 zero-pixel classes cannot be learned regardless of loss function.
2. **Common classes degraded slightly:** Background dropped from ~94% to 92%, suggesting the inverse-frequency weights and Dice component pulled gradient signal away from well-learned structures without compensating on rare ones.
3. **Training loss higher:** 1.71 vs 0.73 — expected since Dice loss adds a [0,1] component, but the combined loss landscape may be harder to optimize.
4. **Dice component may dominate:** At alpha=0.5, Dice gets equal weight to CE. For 1,328 classes where most batch samples contain only 4-10 classes, the Dice mean over present classes may produce noisy gradients.

### Possible improvements (not implemented)
- **Higher alpha (0.8-0.9):** Preserve CE signal, add only light Dice regularization
- **Focal loss instead of Dice:** Down-weight well-classified pixels rather than up-weighting rare classes
- **Move to augmentation (Step 11c):** Orthogonal improvement — more training data diversity may help rare classes more than loss reweighting
