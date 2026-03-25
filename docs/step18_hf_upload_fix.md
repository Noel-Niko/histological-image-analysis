# Step 18: HuggingFace Upload Fix — Missing BigBrain Model

## Problem

The project trains 3 models but only 2 were uploaded to HuggingFace:

| Model | HuggingFace Repo | Status |
|-------|-----------------|--------|
| Mouse (1,328 classes, 79.1% mIoU) | `Noel-Niko/dinov2-upernet-20260322-histology-annotation-mouse` | Uploaded |
| Human Allen depth-3 (44 classes, 65.5% mIoU) | `Noel-Niko/dinov2-upernet-20260322-histology-annotation-human` | Uploaded |
| Human BigBrain (10 tissue types, 60.8% mIoU) | `Noel-Niko/dinov2-upernet-20260322-histology-annotation-human-bigbrain` | **MISSING** |

The `make download-models` command tries to download all 3 and fails on BigBrain.

## Additional Bug

`download_models.py` line 140 uses `repo_id.rsplit("-", 1)[-1]` to determine the local
directory name. For `...-human-bigbrain`, this yields `bigbrain` but `annotate.py`
expects `models/human-bigbrain`. Downloads would go to the wrong directory.

## Changes

### 1. Fix download_models.py directory naming bug
- [x] Replace `rsplit("-", 1)` logic with a proper species-suffix mapping
- [x] Remove unused `SPECIES_MAP` dict

### 2. Add HF upload cell to all 3 training notebooks
- [x] `finetune_final_200ep.ipynb` — Cell 9 (after save)
- [x] `finetune_human_allen_depth3.ipynb` — Cell 8 (after save)
- [x] `finetune_human_bigbrain.ipynb` — Cell 8 (after save)

### 3. Create standalone upload notebook
- [x] `notebooks/upload_models_to_hf.ipynb` — For uploading the missing BigBrain model now
- Fetches model from DBFS if not local, then uploads to HuggingFace Hub

### 4. Add paper uploads to all HF upload cells
- [x] Each model card README now includes a paper summary section
- [x] Full paper uploaded as `paper.md` alongside model weights in each HF repo
- [x] Mouse model gets `mouse_paper_draft.md`
- [x] Both human models get `human_paper_draft.md`
- [x] Upload notebook updated with `PAPER_SUMMARIES` dict and `paper_path` in configs
- [x] All 3 training notebook upload cells updated with paper upload logic

## Model Locations

| Model | DBFS Path | Local Path |
|-------|-----------|------------|
| Mouse | `dbfs:/FileStore/allen_brain_data/models/final-200ep` | `./models/dinov2-upernet-final` |
| Human Allen | `dbfs:/FileStore/allen_brain_data/models/human-allen-depth3` | `./models/human-depth3` |
| Human BigBrain | `dbfs:/FileStore/allen_brain_data/models/human-bigbrain` | `./models/human-bigbrain` (not yet fetched) |

## Status: All code changes complete

All 4 tasks are done. To complete the upload:

1. Fetch the BigBrain model from DBFS:
   ```bash
   make fetch-human-bigbrain-from-dbfs
   ```
2. Open `notebooks/upload_models_to_hf.ipynb`
3. Run Cell 0 to verify the model is present locally
4. Run Cell 1 to set your HuggingFace token
5. Run Cell 2 to upload (defaults to `human-bigbrain` only)
6. Run Cell 3 to verify the upload

## Files Changed

- `scripts/download_models.py` — Fixed directory naming bug for `human-bigbrain`
- `notebooks/finetune_final_200ep.ipynb` — Added Cell 9: HF upload
- `notebooks/finetune_human_allen_depth3.ipynb` — Added Cell 8: HF upload
- `notebooks/finetune_human_bigbrain.ipynb` — Added Cell 8: HF upload
- `notebooks/upload_models_to_hf.ipynb` — **NEW** standalone upload notebook
- `docs/step18_hf_upload_fix.md` — This plan/progress doc
