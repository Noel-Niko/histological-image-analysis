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
| 10 | Deploy + run on Databricks | PENDING — user to run |
