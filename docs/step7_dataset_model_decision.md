# Step 7: Dataset + Model Combination — Decision & Status

## Status: DONE

**Last updated:** 2026-03-08
**Orchestrator:** Claude Code (this LLM session)
**Depends on:** Steps 5-6 (data download — separate LLM, see `docs/data_download_plan.md`)

---

## Critical Clarification from User

> "Our final goal is to provide Autofluorescence, and have the model be able to identify the brain sections."

The end-goal domain is **autofluorescence images**, not Nissl. The model must identify brain structures in autofluorescence data. Nissl is the easier starting domain (higher contrast, clearer boundaries), and the training plan must progress toward autofluorescence.

---

## Decisions Made

### Model: DINOv2-Large + UperNet segmentation head

| Property | Value |
|----------|-------|
| HuggingFace ID | `facebook/dinov2-large` |
| Architecture | ViT-L/14 backbone + UperNet dense prediction head |
| Parameters | 304M (backbone) + UperNet head |
| VRAM (fine-tuning) | ~12 GB |
| Headroom on L40S 48GB | ~36 GB free for batch size, mixed precision |
| License | Apache 2.0 |
| Why this model | Best general features, proven medical imaging transfer, simplest training loop, fits comfortably on hardware |

**Rejected alternatives:**
- SAM 2.1 — requires prompt engineering, more complex training loop
- Virchow2 — 632M params, ~20GB FT, less headroom, similar or marginal feature advantage for this task
- UNI2 — gated license, research-only restriction
- SegGPT — few-shot only, not suitable for full fine-tuning

### Datasets: Both CCFv3 resolutions + Mouse Atlas 2D + Human Atlas

| Dataset | Role | Resolution | Notes |
|---------|------|-----------|-------|
| CCFv3 `ara_nissl_10.nrrd` | Primary training (Nissl domain) | 10μm | Real Nissl histology, perfectly aligned with annotation |
| CCFv3 `annotation_10.nrrd` | Training labels | 10μm | Per-voxel structure IDs, 672 unique structures |
| CCFv3 template (25μm) | Training (autofluorescence domain) | 25μm | Autofluorescence — the **target domain** |
| CCFv3 `annotation_25.nrrd` | Training labels (autofluo) | 25μm | Paired with template volume |
| Mouse Atlas 2D sections | Validation/test | 1.047 μm/px at ds=4 | 509 sections, 132 with SVG annotations |
| Mouse Atlas SVGs | Validation labels | — | Rasterize to segmentation masks |
| Human Atlas Nissl | Supplementary (cross-species) | ds=4 | **RESOLVED:** 14,566 images, 6 donors, 1,067 datasets. Treatment filter is `'NISSL'` (UPPERCASE). ~2.9 GB at ds=4. |

### Structure Granularity: Progressive (Coarse → Fine)

| Phase | Classes | Ontology Depth | Structures |
|-------|---------|---------------|------------|
| **Phase 1** (first) | ~5 | Depth 1 | Cerebrum, Brain stem, Cerebellum, Fiber tracts, Ventricles |
| Phase 2 | ~20-50 | Depth 2-3 | Major subdivisions (cortex, hippocampus, thalamus, etc.) |
| **Phase 3** (final goal) | ~672 | Full | All annotated structures |

**User directive:** Do NOT forget to progress to Fine (~672 classes). The coarse phase is to prove the pipeline, not the final state.

### Domain Progression: Nissl → Autofluorescence

| Phase | Training Domain | Validation Domain | Rationale |
|-------|----------------|-------------------|-----------|
| **Phase A** | CCFv3 10μm Nissl slices | Mouse Atlas 2D Nissl sections | Nissl has high contrast, easier segmentation task, proves pipeline |
| **Phase B** | CCFv3 25μm autofluorescence slices | TBD (autofluorescence test data) | Target domain — autofluorescence has less contrast, harder task |
| **Phase C** | Mixed Nissl + autofluorescence | Both | Domain adaptation / joint training for robustness |

### Training Strategy

**Step 1 — Frozen backbone on Nissl (coarse, ~5 classes):**
- Freeze DINOv2 encoder, train only UperNet head
- Input: CCFv3 10μm Nissl coronal slices
- Labels: Annotation volume mapped to 5 coarse classes
- Purpose: Validate pipeline, data loading, loss convergence

**Step 2 — Full fine-tune on Nissl (coarse):**
- Unfreeze backbone with lower learning rate (e.g., 10× smaller than head)
- Same data as Step 1
- Purpose: Domain adaptation for Nissl texture

**Step 3 — Increase granularity:**
- Progress to medium (~20-50) then fine (~672) classes
- Same Nissl domain

**Step 4 — Autofluorescence domain:**
- Switch training data to CCFv3 25μm template volume slices
- May need domain adaptation techniques (e.g., gradual mixing, curriculum learning)
- Annotation volume at 25μm provides labels

**Step 5 — Joint training / evaluation:**
- Train on mixed Nissl + autofluorescence
- Evaluate per-structure IoU on held-out Mouse Atlas 2D sections

---

## Outstanding Tasks

### RESOLVED: Human Brain Atlas API Query

**Root cause:** Treatment name is `'NISSL'` (UPPERCASE), not `'Nissl'` (title case). The explorer notebook used title case → 0 results. The other LLM discovered this and updated the download script/plan.

**Correct query:**
```
GET https://api.brain-map.org/api/v2/data/query.json?criteria=model::SectionDataSet,rma::criteria,products[id$eq2],[failed$eqfalse],treatments[name$eq'NISSL'],rma::include,specimen(donor),plane_of_section,rma::options[num_rows$eqall]
```

**Result:** 1,067 datasets, 14,566 section images, 6 donors: H0351.1009, H0351.1012, H0351.1015, H0351.1016, H0351.2001, H0351.2002.

### Bug in Download Script: Missing Import

`scripts/download_allen_data.py` line 175 uses `ReferenceSpaceCache` but the import `from allensdk.core.reference_space_cache import ReferenceSpaceCache` is missing from the top of the file. This will cause a `NameError` when step 5B runs. However, step 5C downloads the same 25μm volumes via direct HTTP anyway (the `CCFV3_URLS` dict includes `annotation_25.nrrd` and `average_template_25.nrrd`), so step 5B may fail gracefully and the data is still obtained.

### Minor: Duplicate 25μm Download

The script downloads 25μm volumes twice:
- Step 5B via AllenSDK (`annotation_25.nrrd`, `template_25.nrrd`)
- Step 5C via direct HTTP (`annotation_25.nrrd`, `average_template_25.nrrd`)

The annotation file has the same name so the idempotency check would skip it. The template file has different names (`template_25.nrrd` vs `average_template_25.nrrd`) so both would be downloaded. Not critical — wastes ~85 MB of disk but data is correct.

---

## Data Characteristics (from explorer notebook)

### CCFv3 Volumes
- Annotation shape: (528, 320, 456) at 25μm — AP × DV × ML
- Template shape: (528, 320, 456) at 25μm — autofluorescence
- 672 unique structure IDs in annotation volume
- Annotation dtype: uint32, range: [0, 614454277]
- 671 of 672 IDs resolved from ontology (1 unmapped → filter None)

### Mouse Brain Atlas 2D
- 509 atlas sections, 132 annotated
- Full-res: 19,328×37,408 px at 1.047 μm/px
- At downsample=4: ~1,208×2,338 px per section
- SVG annotations: 154 path elements, 144 unique structure IDs per mid-brain section
- SVG size: ~115 KB per section

### Structure Ontology
- 1,327 total structures in full ontology tree
- Max depth: 10 levels
- Depth 1: grey matter, fiber tracts, ventricular systems, grooves, retina
- Depth 2: Cerebrum (567), Brain stem (343), Cerebellum (512), + fiber tract subdivisions

---

## Files Referenced

| File | Purpose |
|------|---------|
| `docs/progress.md` | Master project state |
| `docs/data_download_plan.md` | Steps 5-6 plan (other LLM) |
| `docs/databricks_connectivity.md` | Network test results |
| `docs/step7_dataset_model_decision.md` | **THIS FILE** — Step 7 decisions |
| `exploration/allen_brain_data_explorer.ipynb` | Data exploration notebook (all cells pass) |

---

## Resume Instructions

If context is lost, read these files in order:
1. `CLAUDE.md` — coding standards
2. `docs/progress.md` — master project state
3. `docs/step7_dataset_model_decision.md` — **this file** (Step 7 decisions — DONE)
4. `docs/step8_training_data_pipeline.md` — **Step 8 plan** (next step to implement)
5. `docs/data_download_plan.md` — download script plan (check if Steps 5-6 are complete)

**Step 7 is DONE.** All decisions finalized. Proceed to Step 8.

**Known issue (non-blocking):** Download script (`scripts/download_allen_data.py`) has a missing `ReferenceSpaceCache` import. Direct HTTP fallback covers it. Notify user if the download had errors.

**Next action:** Step 8 — Build training data pipeline. Full plan at `docs/step8_training_data_pipeline.md`.
