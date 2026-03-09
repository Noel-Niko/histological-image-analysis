# Step 7: Dataset + Model Combination — Decision & Status

## Status: IN PROGRESS

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

### Datasets: Both CCFv3 resolutions + Mouse Atlas 2D + Human Atlas (TBD)

| Dataset | Role | Resolution | Notes |
|---------|------|-----------|-------|
| CCFv3 `ara_nissl_10.nrrd` | Primary training (Nissl domain) | 10μm | Real Nissl histology, perfectly aligned with annotation |
| CCFv3 `annotation_10.nrrd` | Training labels | 10μm | Per-voxel structure IDs, 672 unique structures |
| CCFv3 template (25μm) | Training (autofluorescence domain) | 25μm | Autofluorescence — the **target domain** |
| CCFv3 `annotation_25.nrrd` | Training labels (autofluo) | 25μm | Paired with template volume |
| Mouse Atlas 2D sections | Validation/test | 1.047 μm/px at ds=4 | 509 sections, 132 with SVG annotations |
| Mouse Atlas SVGs | Validation labels | — | Rasterize to segmentation masks |
| Human Atlas Nissl | Supplementary (cross-species) | TBD | **API query returns 0 results — under investigation** |

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

### BLOCKING: Human Brain Atlas API Query — Returns 0 Results

**Problem:** The explorer notebook ran this query and got 0 datasets:
```
GET https://api.brain-map.org/api/v2/data/query.json?criteria=model::SectionDataSet,rma::criteria,products[id$eq2],[failed$eqfalse],treatments[name$eq'Nissl'],rma::include,specimen(donor),plane_of_section,rma::options[num_rows$eq50][order$eq'id']
```

**Contradicts:** `docs/progress.md` which says "Queried via `treatments[name$eq'Nissl']` — datasets found."

**Investigation needed — queries to try:**
1. Original query (confirm 0 results)
2. Without treatments filter: `model::SectionDataSet,rma::criteria,products[id$eq2],[failed$eqfalse],rma::options[num_rows$eq5]`
3. Different product IDs (1-5): `model::SectionDataSet,rma::criteria,products[id$eq{N}],rma::options[num_rows$eq3]`
4. List all products: `model::Product,rma::options[num_rows$eqall]`
5. Use `model::SectionImage` instead: `model::SectionImage,rma::criteria,data_set(products[id$eq2]),treatments[name$eq'Nissl'],rma::options[num_rows$eq5]`
6. Without treatment filter on SectionImage: `model::SectionImage,rma::criteria,data_set(products[id$eq2]),rma::options[num_rows$eq5]`

**Status:** NOT STARTED — was about to run when user requested this doc be written.

### Tell Other LLM: Human Atlas Download May Need Query Fix

The download plan (`docs/data_download_plan.md`) Step 5F assumes the human query works. It may need to be updated after the API investigation above.

### Tell Other LLM: AllenSDK Version Still Wrong

Line 30 of `docs/data_download_plan.md` still hedges on the AllenSDK version. Should just run `pip show allensdk` to confirm.

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
3. `docs/step7_dataset_model_decision.md` — **this file** (Step 7 decisions + outstanding tasks)
4. `docs/data_download_plan.md` — download script plan (being executed by another LLM)

**Next action when resuming:**
1. Run the Human Atlas API investigation queries listed in "Outstanding Tasks" above
2. Update this doc with findings
3. Communicate results to user (and to other LLM if download plan needs changes)
4. Once Step 7 is finalized, update `docs/progress.md` Step 7 to "Done"
5. Proceed to Step 8: Build training data pipeline
