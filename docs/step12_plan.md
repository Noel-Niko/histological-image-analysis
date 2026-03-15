# Plan: Step 12 — Data-Centric Improvements

## Context

The DINOv2-Large + UperNet model peaked at **68.8% mIoU** (Run 5, unfrozen backbone). Two subsequent experiments — weighted Dice+CE loss (Run 6, −1.6%) and extended augmentation (Run 7, −6.5%) — both regressed. The model rejects high-intensity training interventions.

**Root cause:** Data scarcity. 1,016 coronal slices for 1,328 classes (657 with zero training pixels) is a fundamentally underspecified problem. No loss trick or augmentation regime fixes this.

**Strategy:** Address data scarcity directly (multi-axis slicing), remove wasted model capacity (class pruning), and extract better measurements from existing models (TTA, sliding window eval).

---

## Revised Sequence

| Step | Action | Type | Rationale |
|------|--------|------|-----------|
| 0 | TTA on Run 5 model | Eval-only | Free mIoU improvement, zero compute, immediate baseline |
| 1 | Revert augmentation to Run 5 defaults | Code change | Prerequisite for clean experiments |
| 2 | Prune output head to ~671 classes | Code + architecture | Removes softmax dilution, speeds training |
| 3 | Add multi-axis slicing (train only) | Code + data | Addresses root cause — ~3x training data |
| 4 | Run 8: pruned + multi-axis + baseline aug + CE | Training run | Main experiment |
| 5 | Focal loss sweep (γ ∈ {2,3,5}) | Conditional | Only if Run 8 plateaus |
| 6 | Sliding window eval | Eval-only | Honest full-slice measurement |

---

## Step 0: Test-Time Augmentation (TTA) on Run 5

**What:** Average predictions across {original, horizontal_flip, 4×rot90} = 6 forward passes during evaluation. On dense segmentation with bilaterally symmetric structures (which brain anatomy has), TTA typically yields +1-3% mIoU.

**Why now:** Zero training cost. Uses existing Run 5 model. Establishes a stronger baseline before investing in new training runs. If TTA alone pushes Run 5 from 68.8% to 70-71%, that changes the calculus for subsequent steps.

**Implementation:**
- Standalone evaluation script or notebook cell
- Load Run 5 model from `/dbfs/FileStore/allen_brain_data/models/unfrozen` (or local copy)
- For each validation sample:
  1. Forward pass on original → logits_0
  2. Forward pass on horizontally flipped → flip logits back → logits_1
  3. Forward passes on 4 rot90 variants (k=0,1,2,3) → rotate logits back → logits_2..5
  4. Average: `mean_logits = (logits_0 + ... + logits_5) / 6`
  5. `preds = mean_logits.argmax(dim=1)`
- Report: TTA mIoU vs baseline 68.8%
- **No training, no model changes, no code changes to src/**

**Files to create/modify:**
| File | Action |
|------|--------|
| `notebooks/eval_tta.ipynb` or `scripts/eval_tta.py` | **Create** — TTA evaluation |

### Step 0 Results (2026-03-15)

**NEGATIVE RESULT — TTA with rotational variants catastrophically degrades performance.**

| Metric | Baseline | TTA (6 variants) | Delta |
|--------|----------|-------------------|-------|
| mIoU | 68.44% | 44.39% | **−24.05%** |
| Accuracy | 92.12% | 84.35% | −7.77% |
| Valid classes | 504 | 504 | — |

- **482/504 classes regressed**, only 8 improved, 14 unchanged
- Median per-class delta: −18.24%
- Worst regressions: structures with strong directional priors lost 70-80% IoU (e.g., ventral tegmental decussation: 79.5% → 0.0%)
- Baseline (68.44%) closely matches documented Run 5 mIoU (68.8%) — small difference from center-crop determinism

**Root cause:** The model was trained with flip + rot15° + jitter (Run 5 config). It has **never seen 90°/180°/270° rotated brain slices**. Brain coronal sections have strong directional priors (dorsal-ventral axis, medial-lateral organization) that are destroyed by rot90. The TTA effective budget: 33% correct (original ×2), 17% plausible (hflip), **50% noise** (rot90 k=1,2,3). The noise overwhelms the signal.

**Implications:**
1. **Rotational equivariance does not hold** for this model + data combination. Brain anatomy is bilaterally symmetric (left-right) but NOT rotationally symmetric.
2. **Reinforces Run 7 failure:** Extended augmentation (which included rot90) regressed by −6.5% during training. TTA confirms the model fundamentally cannot handle rotated brain slices.
3. **Hflip-only TTA** might yield a small gain (+0.1-0.5%) but was not isolated in this experiment. Not worth pursuing — the gain is marginal.
4. **Multi-axis slicing (Step 3) is different:** Sagittal/axial slices are genuinely different spatial patterns with their own directional priors, not rotations of coronal views. The model needs to learn these during training.
5. **The original estimate of +1-3% was wrong.** TTA only helps when the model is approximately equivariant to the augmentations used. Run 5 is not rot90-equivariant.

---

## Step 1: Revert Augmentation to Run 5 Defaults

**What:** Make augmentation configurable per-transform so future runs can use Run 5's baseline (flip + rot15° + jitter) without the Step 11c additions (rot90, elastic, blur).

**Why:** The Step 11c transforms are baked into `_apply_augmentation()` and activate whenever `augment=True`. Any future training run inherits the −6.5% regression. Need a way to control which transforms are active.

**Implementation:** Add `augmentation_preset` parameter to `BrainSegmentationDataset.__init__()`.

```python
def __init__(self, ..., augment: bool = True,
             augmentation_preset: str = "baseline"):
    # "baseline": flip + rot15 + jitter (Run 5 config)
    # "extended": all 6 transforms (Run 7 config)
    # "none": no augmentation (same as augment=False)
```

In `_apply_augmentation()`, check the preset to decide which transforms to run:
- `"baseline"`: skip rot90, elastic, blur blocks
- `"extended"`: run all 6 (current behavior)

This is backward-compatible: existing `augment=True` with default `"baseline"` restores Run 5 behavior.

**Files to modify:**
| File | Action |
|------|--------|
| `src/histological_image_analysis/dataset.py` | **Modify** — add `augmentation_preset` parameter |
| `tests/test_dataset.py` | **Modify** — add tests for preset behavior |

---

## Step 2: Prune Output Head to ~671 Classes

**What:** Remove the 657 zero-pixel classes from the model's output head. Train only on the ~671 classes that actually appear in training data.

**Why this is #2, not optional:**
- **Softmax dilution:** 1,328 output channels distributes probability mass across 657 phantom classes. The model wastes capacity learning to suppress them on every forward pass.
- **Gradient noise:** Backprop through 1,328-way softmax includes gradients from 657 classes that never provide positive signal. Halving the output space reduces noise.
- **Memory and speed:** Halving the decode head reduces VRAM and speeds training.
- **Low risk:** This is a data-driven filter, not a heuristic. If a class has zero training pixels, the model cannot learn it regardless.

**Implementation:**

1. **Scan training data** to identify which structure IDs have >0 pixels across all training slices.
   - Iterate `slicer.iter_slices("train", mapping, "interleaved")`
   - Count pixels per mapped class ID
   - Record set of present class IDs

2. **Build filtered mapping** — new method on `OntologyMapper`:
   ```python
   def build_present_mapping(
       self, full_mapping: dict[int, int],
       present_class_ids: set[int],
   ) -> dict[int, int]:
       """Remap only structure IDs whose full-mapping class is in present_class_ids.
       Returns new mapping with contiguous class IDs (0 = background)."""
   ```
   - Input: full mapping + set of class IDs that have training pixels
   - Output: new mapping with contiguous IDs for only present classes
   - Background (class 0) always included
   - Structures whose full-mapping class has zero pixels get mapped to 0 (background)

3. **Update notebook** to:
   - Build full mapping first (for scanning)
   - Scan training slices to find present classes
   - Build filtered mapping
   - Use filtered `NUM_LABELS` for `create_model()`
   - Store both mappings so we can map back to original structure names

**Key constraint:** The filtered mapping must be deterministic and reproducible — same training data always produces the same mapping. Using sorted class IDs ensures this (same pattern as `build_full_mapping()`).

**Files to modify:**
| File | Action |
|------|--------|
| `src/histological_image_analysis/ontology.py` | **Modify** — add `build_present_mapping()` |
| `tests/test_ontology.py` | **Modify** — add tests for filtered mapping |
| `src/histological_image_analysis/dataset.py` | May need utility to scan pixel counts |

---

## Step 3: Add Multi-Axis Slicing

**What:** Slice the CCFv3 volume along sagittal (axis 2) and axial (axis 1) planes in addition to coronal (axis 0). Add sagittal and axial slices to training only.

**Why:** The CCFv3 volume is 3D (1320 × 800 × 1140 at 10μm). Currently only coronal slices (axis 0, ~1016 valid) are used. Adding:
- **Sagittal** slices (axis 2, ML direction): up to ~1140 slices
- **Axial** slices (axis 1, DV direction): up to ~800 slices
- **Theoretical max:** ~3,260 total → ~2,600 training (80%)
- **Realistic estimate:** 2,000-2,600 — the 10% brain-pixel threshold will exclude sparse edge slices

This roughly **triples** training data with genuinely new viewing angles — not just augmented copies.

### Leakage-Aware Split Strategy (CRITICAL)

A coronal validation slice at AP=i shares voxels with every sagittal and axial training slice that passes through AP=i. While the 2D spatial context differs (so this isn't hard leakage), the safest approach:

- **Validation/test:** Coronal only, identical interleaved split as Run 5
- **Training:** Coronal + sagittal + axial slices

This gives:
- Direct comparability with Run 5 (identical validation set)
- No cross-orientation leakage concern
- Clean ablation: any mIoU improvement is attributable to additional training data

### Crop Size Consideration

Axial slices are 1320×1140 — a 518×518 crop covers ~18% of area vs ~29% for coronal (800×1140). Start with 518 for consistency. Only revisit (e.g., 768) if axial slices underperform.

### Implementation

1. **Parameterize `CCFv3Slicer`** to support different slicing axes:
   ```python
   def get_slice(self, index: int, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
       """Get 2D slice along specified axis."""
   ```
   - axis=0: coronal (current)
   - axis=1: axial (DV slice)
   - axis=2: sagittal (ML slice)

2. **Add `_get_valid_indices(axis)`** — generalized version of `_get_valid_ap_indices()`:
   - Same 10% brain-pixel threshold
   - Works for any axis
   - Verify counts empirically before committing

3. **Update `BrainSegmentationDataset`** to accept slices from multiple axes:
   - Option A: Accept multiple slicers (one per axis)
   - Option B: Single slicer with multi-axis `iter_slices()` that yields from all axes for training, coronal-only for val/test
   - **Prefer Option B** — single slicer, axis-aware iteration

4. **Split strategy:** `get_split_indices()` returns coronal indices for val/test. Training indices include coronal + all sagittal + all axial valid slices. Each slice carries its axis tag so `get_slice()` knows how to index.

**Files to modify:**
| File | Action |
|------|--------|
| `src/histological_image_analysis/ccfv3_slicer.py` | **Modify** — multi-axis slicing, `_get_valid_indices(axis)` |
| `src/histological_image_analysis/dataset.py` | **Modify** — accept multi-axis slices |
| `tests/test_ccfv3_slicer.py` | **Modify** — tests for sagittal/axial slicing |
| `tests/test_dataset.py` | **Modify** — tests for multi-axis dataset |

---

## Step 4: Run 8

**What:** Training run combining all improvements: pruned classes (~671), multi-axis training data (~2,000-2,600 samples), baseline augmentation (flip + rot15° + jitter), standard CE loss.

**Config:** Same as Run 5 except:
- `NUM_LABELS` ≈ 671 (from class pruning)
- Training samples ≈ 2,000-2,600 (from multi-axis)
- `augmentation_preset="baseline"` (reverted)

**Notebook:** `notebooks/finetune_pruned_multiaxis.ipynb` — new notebook. Cell 7 uses corrected MLflow save pattern.

**Expected outcome:** The combination of 2-3× more training data and a correctly-sized output head should be synergistic. Target: surpass Run 5's 68.8% mIoU.

**Files to create:**
| File | Action |
|------|--------|
| `notebooks/finetune_pruned_multiaxis.ipynb` | **Create** |
| `Makefile` | **Modify** — add deploy target |

---

## Step 5: Focal Loss Sweep (Conditional)

**What:** If Run 8 plateaus, sweep focal loss γ ∈ {1, 2, 3, 5}. Focal loss down-weights easy/well-classified examples, focusing gradient signal on hard boundary cases.

**Why γ search:** γ=2 is the default from the RetinaNet paper (object detection). For 671-class dense segmentation with extreme imbalance, γ=3-5 may work better.

**Prerequisite:** Run 8 results. Only pursue if mIoU improvement stalls.

**Implementation:** Uses existing `losses.py` infrastructure. Add `FocalLoss` class or modify `WeightedLossUperNet`.

---

## Step 6: Sliding Window Evaluation

**What:** Replace center_crop evaluation with sliding window inference. Current eval uses center_crop(518×518), evaluating only ~29% of each coronal slice (800×1140). Sliding window with overlap (stride 256, window 518) covers the full slice and averages predictions in overlapping regions.

**Why:** More honest measurement of model performance. Likely improves reported mIoU by capturing structures near slice edges. Not a training change — can be applied to any existing model.

**Implementation:**
- For each validation slice:
  1. Tile into overlapping 518×518 patches (stride 256)
  2. Forward pass each patch
  3. Average logits in overlapping regions (optional: Gaussian weighting, higher weight for center)
  4. Argmax on averaged logits
  5. Compute IoU against full-resolution ground truth

**Files to modify:**
| File | Action |
|------|--------|
| `src/histological_image_analysis/training.py` or new `inference.py` | **Modify/Create** — sliding window inference |
| `tests/` | **Modify** — tests for sliding window |

---

## Step 7: Write-Up and Documentation

**What:** Produce a final document (paper-ready or technical report) consolidating all experimental results, methodology, ablation analysis, and lessons learned.

**Source data:** `docs/experimental_results.md` contains all run metrics, per-class IoU tables, training curves, and hyperparameter comparisons. This was consolidated from notebook outputs on 2026-03-14.

**Structure:**
1. Introduction / Problem statement (brain structure segmentation from Nissl histology)
2. Dataset (CCFv3 10μm Nissl, ontology, class distribution, 657 zero-pixel classes)
3. Model architecture (DINOv2-Large + UperNet, frozen vs unfrozen)
4. Training methodology (split strategy, augmentation, loss functions)
5. Results — all 7 runs with ablation analysis
6. Analysis — data scarcity as binding constraint, negative results (augmentation, loss reweighting)
7. Future work (multi-axis slicing, class pruning, focal loss)

**Prerequisite:** Steps 0-4 results (Run 8) should be included if available. Can write a draft after Run 8, or write now with Runs 1-7 and append Run 8 later.

---

## Implementation Status

| Step | Task | Status |
|------|------|--------|
| — | Consolidate experimental results from notebooks | DONE — `docs/experimental_results.md` |
| — | Fix Run 5 accuracy error (89.4% → 92.5%) | DONE — corrected in all .md files |
| 0 | TTA on Run 5 model | DONE — **NEGATIVE RESULT**: 6-variant TTA regressed from 68.4% to 44.4% mIoU (−24%). rot90 variants are noise. |
| 1 | Revert augmentation (add `augmentation_preset`) | DONE — 6 tests, baseline/extended/none presets |
| 2 | Prune output head (~671 classes) | DONE — `build_present_mapping()`, 9 tests |
| 3 | Multi-axis slicing (train only, coronal val) | DONE — `get_slice(axis)`, `_get_valid_indices(axis)`, `multi_axis` flag, 14 tests |
| 4 | Run 8 notebook + deploy | DONE — Two notebooks: `finetune_pruned_multiaxis.ipynb` (combined) + `finetune_pruned_ablation.ipynb` (ablation with diagnostics, `ENABLE_MULTI_AXIS` flag) |
| 5 | Focal loss sweep (conditional) | PENDING |
| 6 | Sliding window eval | PENDING |
| 7 | Write-up / paper | PENDING |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Fewer valid sagittal/axial slices than expected | Verify counts empirically in Step 3 before committing to Run 8 |
| Cross-orientation leakage inflating val metrics | Val/test are coronal-only — identical to Run 5 |
| Class pruning introduces mapping bugs | TDD — test determinism, re-indexing, background handling |
| Pruned model can't predict structures absent from training | By definition it couldn't before either (657 classes at zero IoU) |
| Axial crops cover less context (18% vs 29%) | Start with 518; revisit crop size only if axial underperforms |
