# Step 14: Finishing the Human Brain Segmentation Project

**Date:** 2026-03-22
**Depends on:** `docs/step13_human_training_plan.md` (COMPLETE), `docs/human_paper_draft.md`
**Status:** IN PROGRESS — Step 14.1 running on Databricks. Steps 14.5 + 14.6 COMPLETE. Next: redeploy + wait for test results.

---

## Overview

Four deliverables to finish the project:
1. Test-set evaluation on depth-3 model (highest priority — paper has NO test numbers)
2. Paper figure generation notebooks
3. Paper draft improvements
4. Makefile + plan updates

---

## Step 14.1: Test-Set Evaluation Notebook

**File:** `notebooks/eval_human_depth3_test.ipynb` (NEW)

**Purpose:** Run CC + SW evaluation on the held-out test donor H0351.1015 (634 images) using the saved depth-3 model from `/dbfs/FileStore/allen_brain_data/models/human-allen-depth3`.

**Structure (7 cells):**

| Cell | Description |
|------|-------------|
| 0 (md) | Notebook header: test-set evaluation for depth-3 model |
| 1 | Install wheel + restart Python |
| 2 | Configuration: paths, test donor, model dir |
| 3 | Download DINOv2 weights (needed for image processor) |
| 4 | Build test dataset: load metadata → filter H0351.1015 → build depth-3 present_mapping → pre-cache test images → create AllenHumanDataset |
| 5 | Load saved model from DBFS (`UperNetForSemanticSegmentation.from_pretrained()`) + set up partial unfreezing state + forward pass validation |
| 6 | Evaluate: CC eval via Trainer + SW eval on ALL 634 test images (not 50 subset) + per-class IoU + save results JSON |

**Key differences from training notebook:**
- No training cell — evaluation only
- Loads pre-trained model from DBFS (not from DINOv2 scratch)
- SW eval on ALL test images (634), not 50-image subset — this gives more robust test numbers
- Saves per-class IoU results as `/dbfs/FileStore/allen_brain_data/models/human-allen-depth3/test_results.json` for figure generation
- Cache requires rebuild (test images only — ~634 files, ~70 min estimated)

**Databricks instructions:**
1. Deploy: `make deploy-wheel deploy-notebook-eval-depth3-test`
2. Cluster: single GPU (L40S 48 GB)
3. Run all cells sequentially
4. Expected time: ~70 min cache build + ~15 min CC eval + ~60 min SW eval (634 images) ≈ ~2.5 hrs total

---

## Step 14.2: Figure Generation Notebook

**File:** `notebooks/generate_human_paper_figures.ipynb` (NEW)

**Purpose:** Generate all 6 paper figures. Some figures need model inference (run on Databricks), others are pure data visualization (could run locally but simpler to do everything on Databricks).

**Structure (8 cells):**

| Cell | Description |
|------|-------------|
| 0 (md) | Notebook header |
| 1 | Install wheel + restart |
| 2 | Configuration: paths, model dirs, figure output dir |
| 3 | Load all 3 models + test data |
| 4 | **Figure 2**: Three-track comparison grid (Input / GT / Prediction for each track) — requires inference from all 3 models on selected validation images |
| 5 | **Figure 3**: Class granularity vs mIoU bar chart (hardcoded results data) |
| 6 | **Figure 4**: Depth-3 per-class IoU horizontal bars (from test_results.json or hardcoded from val eval) |
| 7 | **Figure 5**: Annotation density illustration (mouse dense / Allen sparse / BigBrain dense) — needs sample images from each dataset |
| 8 | **Figure 6**: Training convergence curves (from MLflow or hardcoded training logs) |

**Note:** Figure 1 (architecture diagram) reuses the mouse paper figure — no generation needed.

**Dependencies:** Step 14.1 should complete first (provides test_results.json for Figure 4). However, figures can be generated with val data if test results aren't ready yet.

**Databricks instructions:**
1. Deploy: `make deploy-notebook-human-figures`
2. Cluster: same GPU cluster as eval notebook
3. Run after test eval completes
4. Figures saved to `/dbfs/FileStore/allen_brain_data/figures/human/`

---

## Step 14.3: Paper Draft Improvements

**File:** `docs/human_paper_draft.md` (EDIT)

Changes:
1. **Table 1**: Add test-set numbers once available from Step 14.1
2. **Section 4.3 SW mIoU gap**: Flag 50-image subset limitation more prominently
3. **Section 5.2 expansion**: Add pixel-budget-per-class back-of-envelope detail (already partially there, expand with depth-3 vs 597 comparison)
4. **New subsection 6.x "Practical Deployment"**: Add PhD slide workflow (VSI → JPEG → rotate → depth-3 model → region prediction)
5. **Section 6 Limitations**: Move SW subset variance to limitations
6. **Minor**: Clarify that 597-class 200ep formal eval was intentionally skipped

---

## Step 14.4: Makefile + Plan Updates

**Files:** `Makefile` (EDIT), `docs/step13_human_training_plan.md` (EDIT)

- Add `deploy-notebook-eval-depth3-test` and `deploy-notebook-human-figures` targets to Makefile
- Update step13 plan with Steps 14.1-14.4 status

---

## Notebooks to Run on Databricks (Summary)

| Order | Notebook | Purpose | Est. Time | Dependencies |
|-------|----------|---------|-----------|--------------|
| 1 | `eval_human_depth3_test.ipynb` | Test-set CC + SW eval | ~2.5 hrs | Saved depth-3 model on DBFS |
| 2 | `generate_human_paper_figures.ipynb` | All 6 paper figures | ~30 min | Test results from #1 |

Both run on the same single-GPU cluster (L40S 48 GB).

---

## Deploy Commands

```bash
# Build wheel + deploy both new notebooks
make deploy-wheel deploy-notebook-eval-depth3-test deploy-notebook-human-figures
```

---

## Step 14.5: Figure Notebook Review Fixes

**File:** `notebooks/generate_human_paper_figures.ipynb` (EDIT)

**Status:** PENDING — fixes identified during code review.

### Issues Found

1. **Figure 2 title mismatch.** Markdown header says "Three-track comparison" but the figure only shows depth-3 on 3 val images. Not actually a three-track comparison. Fix: retitle to "Depth-3 Segmentation Examples" and increase from 3 to **15 sample images** (5x) for better visual coverage.

2. **Figure 3: 597-class SW mIoU (21.0%) is from epoch 50, not 200.** All other values are 200-epoch. Fix: add asterisk + footnote.

3. **Figure 5 suptitle hardcodes "~25% labeled"** but the cell computes `pct_labeled` dynamically. Fix: use the computed value in the suptitle via f-string.

4. **Figure 6 convergence data is interpolated.** Depth-3 intermediate mIoU values (epochs 50/100/150) are estimates, not formal checkpoints. Only epoch 200 was evaluated. Fix: add comment noting which points are approximate and label curves as "approximate" in the legend.

5. **No GPU cleanup after inference cells.** Fix: add `torch.cuda.empty_cache()` after Figure 2 inference loop.

6. **Figure 2: increase sample images from 3 to 15** for comprehensive visual coverage across the validation donor.

### Changes Required

| Cell | Change |
|------|--------|
| 0 (md) | Fix Figure 2 description: "Depth-3 segmentation examples (15 images)" |
| 3 (Fig 2) | Retitle suptitle. Change `n_images = 3` → `n_images = 15`. Adjust figsize. Add `torch.cuda.empty_cache()` at end. |
| 4 (Fig 3) | Add `*` to 597-class SW label: `"Human 597-class*\n(597 classes)"`. Add footnote text below chart. |
| 6 (Fig 5) | Use f-string in suptitle: `f"Allen Human (~{pct_labeled:.0f}% labeled)"` |
| 7 (Fig 6) | Add "(approx.)" to depth-3 legend. Add comment explaining which mIoU points are formal evals vs interpolated. |
| 8 (Summary) | Update fig2 description to match new title |

---

## Progress Tracker

| Step | Status | Notes |
|------|--------|-------|
| 14.1 Test eval notebook | RUNNING ON DATABRICKS | `eval_human_depth3_test.ipynb` — CC + SW on 634 test images |
| 14.2 Figure notebook | COMPLETE (needs review fixes) | `generate_human_paper_figures.ipynb` — 5 figures generated |
| 14.3 Paper draft improvements | COMPLETE | Practical Deployment subsection, expanded Limitations |
| 14.4 Makefile + plan updates | COMPLETE | 2 new deploy targets added |
| 14.5 Figure notebook review fixes | COMPLETE | All 6 issues fixed |
| 14.6 Three-track Figure 2 | COMPLETE | All 3 models (597/depth-3/BigBrain), same Allen images for A+B |

---

## Step 14.6: Three-Track Figure 2

**File:** `notebooks/generate_human_paper_figures.ipynb` (EDIT)

**Status:** COMPLETE

**Changes:**
- **Cell 1 (config):** Added `ALLEN_597_MODEL_DIR`, `BIGBRAIN_MODEL_DIR`, BigBrain volume paths
- **Cell 2 (model loading):** Loads all 3 models (depth-3, 597-class, BigBrain) + BigBrain NIfTI volumes. Builds separate LUTs for 597-class and depth-3. `load_image_and_mask()` now accepts `lut_array` parameter.
- **Cell 3 (Figure 2):** Three-track grid: 5 images × 3 tracks = 15 rows × 3 cols. Allen tracks A (597-class) and B (depth-3) show the SAME 5 val images with different class mappings. Track C (BigBrain) shows 5 val slices from NIfTI volume (grayscale → pseudo-RGB). Models loaded to GPU one at a time. Saved as JPEG. Color-coded track labels.
- **Cell 8 (Summary):** Updated to `fig2_three_track_comparison.jpg`

**Key design decisions:**
- Allen tracks share identical images → directly shows granularity effect on same tissue
- BigBrain uses pseudo-RGB (grayscale replicated to 3 channels) for shared `predict_center_crop()`
- Models moved to GPU one at a time (load → infer → .cpu() → empty_cache()) to conserve memory
- depth-3 model moved back to GPU at end of cell for Figure 5

---

## Implementation Order

1. ~~Create `eval_human_depth3_test.ipynb`~~ DONE
2. ~~Create `generate_human_paper_figures.ipynb`~~ DONE
3. ~~Update `Makefile` with new deploy targets~~ DONE
4. ~~Edit `docs/human_paper_draft.md` with improvements~~ DONE
5. ~~Update `docs/step13_human_training_plan.md` with Step 14 reference~~ DONE
6. ~~Apply figure notebook review fixes (Step 14.5)~~ DONE
7. ~~Three-track Figure 2 (Step 14.6)~~ DONE
8. Redeploy figure notebook to Databricks ← NEXT
9. After test eval completes: update paper draft Table 1 + Conclusion with test numbers
