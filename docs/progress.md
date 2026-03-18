# Histological Image Analysis — Research & Progress

## Project Goal
Fine-tune an open-source vision foundation model to identify brain structures in Nissl-stained mouse/human brain slides using Allen Brain Institute data as the structured training set.

## Compute Environment
- **Platform:** Databricks 17.3 LTS (Apache Spark 4.0.0, Scala 2.13)
- **Instance:** g6e.16xlarge (AWS)
- **GPU:** NVIDIA L40S (48 GB VRAM)
- **RAM:** 512 GB
- **Topology:** Single node
- **Constraint:** Cluster firewall blocks some external downloads — workaround is local download + upload to Workspace

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-06 | Use Allen Brain Institute data for training | Gold-standard annotated brain structure data with Nissl staining |
| 2026-03-06 | Target both mouse and human brains | Allen provides atlases for both species |
| 2026-03-06 | Use Workspace upload pattern for data | Databricks firewall blocks direct downloads |
| 2026-03-06 | FiftyOne/Voxel51 not needed for main pipeline | Our data is standard 2D images + segmentation masks, not DICOM. Matplotlib suffices for viz. |
| 2026-03-06 | Firewall blocks ALL Allen domains on Databricks | Tested via UI + CLI on dev cluster — `api.brain-map.org` and `download.alleninstitute.org` get `ConnectionResetError(104)`. PyPI and HuggingFace pass. See `docs/databricks_connectivity.md` |
| 2026-03-06 | Local download → Workspace upload for Allen data | Download ~6 GB locally, `databricks workspace import-dir` to `/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology`, read directly in notebooks |

---

## Completed Work

### Data Source Research — COMPLETE

7 Allen Brain Institute data sources evaluated. Key findings:

**PRIMARY — Allen Mouse Brain Atlas (Atlas ID=1, "Mouse, P56, Coronal")**
- 509 total atlas images, **132 annotated** with structure boundaries
- Full-res ~19,328×37,408 px at 1.047 μm/px
- SVG annotations with 144+ structure polygons per section (verified via API)
- API: `model::AtlasImage, rma::criteria,atlas_data_set(atlases[id$eq1])` — NOTE: must join through `atlas_data_set(atlases[...])`, NOT `atlas_data_set_id` directly
- Download: `atlas_image_download/{id}?downsample=N`
- SVG: `svg_download/{id}` returns SVG with `<path structure_id="...">` elements

**PRIMARY — CCFv3 3D Volumes (verified working via AllenSDK)**
- Annotation volume: (528, 320, 456) at 25μm — 672 unique structure IDs
- Template volume: same shape, grayscale autofluorescence
- `ara_nissl_10.nrrd` available at 10μm for actual Nissl histology
- AllenSDK `ReferenceSpaceCache` works — downloads from `http://download.alleninstitute.org/`
- Structure tree: `get_structures_by_id()` returns `None` for some annotation IDs — must filter

**SUPPLEMENTARY — Human Brain Atlas (Product ID=2)**
- Queried via `treatments[name$eq'Nissl']` — datasets found
- `section_image_download/{id}?downsample=N` works for human sections

**LOW RELEVANCE** — ABC Atlas, MERFISH S3 volumes (spatial transcriptomics, not histology)

### Model Research — COMPLETE

20 models evaluated across 3 categories. Top 5 recommendations:
1. **DINOv2-Large** (`facebook/dinov2-large`) — 304M params, ~12GB FT, Apache 2.0
2. **SAM 2.1** (`facebook/sam2.1-hiera-large`) — 224M params, ~16GB FT, Apache 2.0
3. **UNI2** (`MahmoodLab/UNI2-h`) — 632M params, ~20GB FT, gated/research-only
4. **Virchow2** (`paige-ai/Virchow2`) — 632M params, ~20GB FT, Apache 2.0
5. **SegGPT** (`BAAI/SegGPT`) — 370M params, ~14GB, Apache 2.0 (few-shot baseline)

Full comparison tables in sections below.

### Explorer Notebook — COMPLETE (all cells run)

`exploration/allen_brain_data_explorer.ipynb` — runs end-to-end locally. Sections:
1. Mouse Brain Atlas Nissl sections (509 images fetched, 8 displayed, hi-res detail view)
2. SVG annotations parsed (154 paths, 144 unique structure IDs for mid-brain section)
3. Nissl vs Annotated atlas plates comparison (4 sections, plain vs overlay)
4. CCFv3 3D volumes (annotation + template downloaded at 25μm, coronal/sagittal/axial slices)
5. Structure ontology (loaded via AllenSDK, sample structures displayed)
6. Human Brain Atlas Nissl sections
7. Side-by-side comparison of all sources
8. Summary with recommendations

**Bugs fixed during development:**
- `%pip install nrrd` → `pynrrd` (correct PyPI package name)
- `SectionImage` model with dataset ID `100960033` returns 0 results → switched to `AtlasImage` model with `atlas_data_set(atlases[id$eq1])` join
- `msg` field is a string (not list) when API returns 0 results → added `isinstance(msg, str)` guard
- `get_structures_by_id()` returns `None` for unmapped IDs → added filter

### FiftyOne Tutorial Notebook — COMPLETE (no changes)

`exploration/getting_started_medical_imaging_fiftyone (2).ipynb` — pre-existing, untouched.

---

## Data Source Research (Full Details)

### 1. Allen Mouse Brain Atlas (Nissl Reference) — **PRIMARY CANDIDATE**

- **URL:** `https://mouse.brain-map.org/`
- **API base:** `https://api.brain-map.org/api/v2/`
- **Data type:** 2D Nissl-stained coronal section images with structure annotations
- **Species:** Mouse (C57BL/6J)
- **What it contains:**
  - **Nissl reference atlas:** 509 atlas images (132 annotated) spanning the full mouse brain
  - **ISH data:** ~20,000+ gene experiments, each with ~60-70 sections
  - **Structure ontology:** ~700+ hierarchically organized brain structures
- **Image resolution:** Full-res ~19,328×37,408 px at 1.047 μm/px. Downloadable at downsample 0–8.
- **Key API endpoints (VERIFIED WORKING):**
  - List atlas images: `GET /api/v2/data/query.json?criteria=model::AtlasImage,rma::criteria,atlas_data_set(atlases[id$eq1]),rma::options[num_rows$eqall][order$eq'section_number']`
  - Download image: `GET /api/v2/atlas_image_download/{id}?downsample={0-8}`
  - Download with annotations: `GET /api/v2/atlas_image_download/{id}?downsample=4&annotation=true`
  - SVG overlay: `GET /api/v2/svg_download/{atlas_image_id}`
  - Structure ontology: `GET /api/v2/structure_graph_download/1.json`
- **Annotation format:** SVG with `<path structure_id="N">` polygons. 144 structures per mid-brain section.
- **Size estimate:** At downsample=4: ~509 × 500 KB ≈ 250 MB. Full-res: ~6-7 GB.
- **Training use:** Image + SVG → rasterize to per-pixel segmentation masks.
- **License:** Allen Institute Terms of Use — free for non-commercial research.

### 2. Allen Human Brain Atlas

- **URL:** `https://human.brain-map.org/`
- **API:** Same base, Product ID=2, filter by `treatments[name$eq'Nissl']`
- **Species:** Human (6 donors, ~900 structures)
- **Use:** Supplementary cross-species data
- **License:** Allen Institute Terms of Use

### 3. Common Coordinate Framework (CCFv3) — **HIGH VALUE FOR LABELS**

- **URL:** `https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/`
- **Verified shape:** (528, 320, 456) at 25μm — 672 unique structure IDs
- **Key files:** `annotation_10.nrrd` (~1.3GB), `annotation_25.nrrd` (~85MB), `ara_nissl_10.nrrd` (~1.3GB)
- **AllenSDK access (verified):**
  ```python
  from allensdk.core.reference_space_cache import ReferenceSpaceCache
  rsc = ReferenceSpaceCache(resolution=25, reference_space_key="annotation/ccf_2017",
                            manifest="/tmp/allen_ccf_cache/manifest.json")
  annotation, _ = rsc.get_annotation_volume()
  template, _ = rsc.get_template_volume()
  structure_tree = rsc.get_structure_tree()
  # NOTE: filter None from get_structures_by_id() results
  ```
- **Training use:** Slice 3D volumes at arbitrary planes → unlimited training pairs.

### 4–7. (Lower priority sources)

See previous entries — ABC Atlas, MERFISH, AllenSDK details unchanged. Key takeaway: ABC Atlas and MERFISH are spatial transcriptomics, NOT histological images. Low relevance for our task.

---

## Model Research (Full Details)

### Category 1: Pathology/Biomedical Foundation Models

| # | Model | HuggingFace ID | Params | Architecture | Pre-training Data | Tasks | Input Res | VRAM (FT) | License | Relevance |
|---|-------|---------------|--------|-------------|-------------------|-------|-----------|-----------|---------|-----------|
| 1 | **UNI** | `MahmoodLab/UNI` | 307M | ViT-L/16 (DINOv2) | 100K+ WSIs, 20+ tissue types | Feature extraction, classification | 224×224 | ~12 GB | Custom (research, gated) | **HIGH** |
| 2 | **UNI2** | `MahmoodLab/UNI2-h` | 632M | ViT-H/14 (DINOv2) | 350K+ WSIs, 20M+ tiles | Feature extraction, classification | 224×224 | ~20 GB | Custom (research, gated) | **HIGH** |
| 3 | **CONCH** | `MahmoodLab/CONCH` | 307M + text | CoCa (ViT-B + decoder) | 1.17M pathology image-text pairs | Zero-shot, retrieval, VQA | 448×448 | ~16 GB | Custom (research, gated) | MEDIUM |
| 4 | **Virchow2** | `paige-ai/Virchow2` | 632M | ViT-H/14 (DINOv2) | 3.1M WSIs | Feature extraction, classification | 224×224 | ~20 GB | Apache 2.0 | **HIGH** |
| 5 | **Prov-GigaPath** | `prov-gigapath/prov-gigapath` | 1.1B + 85M | ViT-g/14 + LongNet | 1.3B tiles, 171K WSIs | Feature extraction, slide-level | 256×256 | ~32 GB | Custom (gated) | MEDIUM |
| 6 | **BiomedCLIP** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 86M + text | ViT-B/16 + PubMedBERT | PMC-15M | Zero-shot, retrieval | 224×224 | ~6 GB | MIT | **HIGH** |
| 7 | **PLIP** | `vinid/plip` | 86M + text | ViT-B/16 (CLIP) | 200K pathology image-text pairs | Zero-shot, retrieval | 224×224 | ~6 GB | MIT | MEDIUM |
| 8 | **CTransPath** | `xiangjli/CTransPath` | 28M | Swin Transformer | 15M+ patches TCGA+PAIP | Feature extraction | 224×224 | ~4 GB | Apache 2.0 | MEDIUM |
| 9 | **Phikon** | `owkin/phikon` | 86M | ViT-B/16 (iBOT) | 40M tiles TCGA | Feature extraction | 224×224 | ~6 GB | Apache 2.0 | MEDIUM |

### Category 2: General Vision Models

| # | Model | HuggingFace ID | Params | Architecture | Tasks | Input Res | VRAM (FT) | License | Relevance |
|---|-------|---------------|--------|-------------|-------|-----------|-----------|---------|-----------|
| 10 | **SAM 2** | `facebook/sam2-hiera-large` | 224M | Hiera | Segmentation | 1024 longest | ~16 GB | Apache 2.0 | **HIGH** |
| 11 | **SAM 2.1** | `facebook/sam2.1-hiera-large` | 224M | Hiera | Segmentation | Variable | ~16 GB | Apache 2.0 | **HIGH** |
| 12 | **DINOv2** | `facebook/dinov2-large` | 304M | ViT-L/14 | Features → seg head | 518×518 | ~12 GB | Apache 2.0 | **HIGH** |
| 13 | **Florence-2** | `microsoft/Florence-2-large` | 770M | DaViT + seq2seq | Multi-task | 768×768 | ~24 GB | MIT | MEDIUM |
| 14 | **SegGPT** | `BAAI/SegGPT` | 370M | ViT-L/16 | Few-shot segmentation | 448×448 | ~14 GB | Apache 2.0 | **HIGH** |
| 15 | **Grounding DINO** | `IDEA-Research/grounding-dino-base` | 172M | Swin-B + BERT | Detection | 800×1333 | ~10 GB | Apache 2.0 | LOW |

### Category 3: Brain/Neuroscience-Specific Tools

| # | Tool/Model | URL/Package | Type | What it Does | Relevance |
|---|-----------|-------------|------|-------------|-----------|
| 16 | **DeepSlice** | `github.com/PolarBean/DeepSlice` | EfficientNet | Atlas coordinate prediction for sections | **HIGH** — complementary |
| 17 | **brainreg** | `brainglobe.info/brainreg` | ANTs registration | 3D brain → CCF registration | MEDIUM |
| 18 | **QUINT** | `quint-workflow.readthedocs.io` | Semi-auto workflow | 2D section → atlas registration + quantification | **HIGH** — our model automates this |
| 19 | **nnU-Net** | `github.com/MIC-DKFZ/nnUNet` | U-Net ensemble | Self-configuring medical segmentation | **HIGH** |
| 20 | **MedSAM** | `bowang-lab/MedSAM` | SAM fine-tuned | Medical image segmentation | **HIGH** |

### Model Recommendations (Top 5)

1. **DINOv2-Large + UperNet head** — `facebook/dinov2-large`, 304M, ~12GB, Apache 2.0. Best features, proven medical imaging, within budget.
2. **SAM 2.1 fine-tuned** — `facebook/sam2.1-hiera-large`, 224M, ~16GB, Apache 2.0. SOTA segmentation, interactive correction.
3. **UNI2 + seg head** — `MahmoodLab/UNI2-h`, 632M, ~20GB, gated. Best histology features, but restricted license.
4. **Virchow2 + seg head** — `paige-ai/Virchow2`, 632M, ~20GB, Apache 2.0. Largest pathology pretraining, open license.
5. **SegGPT** — `BAAI/SegGPT`, 370M, ~14GB, Apache 2.0. Few-shot baseline before committing to fine-tuning.

---

## Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `exploration/getting_started_medical_imaging_fiftyone (2).ipynb` | FiftyOne DICOM/CT tutorial (learning exercise) | Complete |
| `exploration/allen_brain_data_explorer.ipynb` | Browse Allen Brain data sources to choose dataset | Complete — all cells run |
| `exploration/databricks_connectivity_check.ipynb` | Verify Databricks can reach Allen data sources | Complete — 6 blocked, 2 pass |
| `notebooks/finetune_unfrozen.ipynb` | Run 5 — full 1,328-class, unfrozen backbone | Complete — 68.8% mIoU |
| `notebooks/finetune_augmented.ipynb` | Run 7 — extended augmentation (NEGATIVE) | Complete — 62.3% mIoU |
| `notebooks/eval_tta.ipynb` | Step 12 TTA eval on Run 5 (NEGATIVE) | Complete — 44.4% mIoU |
| `notebooks/finetune_pruned_multiaxis.ipynb` | Run 8 — pruned classes + multi-axis (combined) | Created — hit BatchNorm error, superseded by ablation |
| `notebooks/finetune_pruned_ablation.ipynb` | Run 8a — ablation with diagnostics | Complete — 69.0% mIoU |
| `notebooks/finetune_final_200ep.ipynb` | **Run 9 — FINAL mouse model, 200 epochs (BEST)** | **Complete — 74.8% mIoU (CC), 79.1% mIoU (SW)** |
| `notebooks/historical/finetune_coarse.ipynb` | Runs 1-2 — coarse 6-class | Complete (historical) |
| `notebooks/historical/finetune_depth2.ipynb` | Run 3 — depth-2 19-class | Complete (historical) |
| `notebooks/historical/finetune_full.ipynb` | Run 4 — full 1,328-class frozen | Complete (historical) |
| `notebooks/historical/finetune_weighted_loss.ipynb` | Run 6 — weighted Dice+CE (NEGATIVE) | Complete (historical) |

---

## Next Steps (resume here)
1. ~~Complete data source research~~ Done
2. ~~Complete model research~~ Done
3. ~~Build allen_brain_data_explorer.ipynb~~ Done — all cells pass
4. ~~Test Databricks connectivity~~ Done — Allen domains blocked, PyPI/HuggingFace pass. See `docs/databricks_connectivity.md`
5. ~~Download Allen data locally~~ Done — 20.87 GB total. 4 NRRD volumes, 509 mouse images, 509 SVGs, 14,565 human images, ontology JSON. Script: `scripts/download_allen_data.py`. Details: `docs/data_download_plan.md`.
6. ~~Upload to Databricks Workspace~~ Done — All files uploaded. `ara_nissl_10.nrrd` (2.17 GB) in DBFS at `dbfs:/FileStore/allen_brain_data/ccfv3/`. All other files in Workspace at `/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/`. Pending: cluster-side verification (Step 6B).
7. ~~Choose dataset + model combination~~ **DONE**. See `docs/step7_dataset_model_decision.md`.
   - DINOv2-Large + UperNet, CCFv3 (both 10μm Nissl + 25μm autofluorescence), Mouse Atlas 2D for validation
   - Final target domain is **autofluorescence** (not just Nissl). Training progresses Nissl → autofluorescence.
   - Granularity: coarse (~5 classes) → fine (~672 classes)
   - Human Atlas: 14,566 images, 6 donors. Treatment filter `'NISSL'` (UPPERCASE).
8. ~~Build training data pipeline~~ **DONE**. See `docs/step8_implementation_plan.md`.
   - 4 components built with TDD: `ontology.py`, `ccfv3_slicer.py`, `svg_rasterizer.py`, `dataset.py`
   - Code in `src/histological_image_analysis/`, tests in `tests/` — **61/61 tests pass**
   - Resolved: svgpathtools+PIL for SVG, random crop train + tiled eval, on-the-fly NRRD, spatial AP split
   - Ontology correction: root(997)→grey(8)→{CH(567),BS(343),CB(512)} at depth 2, not depth 1
   - Real ontology smoke test: 1,327 structures, 6 coarse classes verified
   - **JFrog Artifactory reminder:** Model weights must be manually added to JFrog before Step 9 on Databricks
   - **Security:** `.claude/settings.local.json` had Databricks token — RESOLVED: `.claude/` added to `.gitignore`, history rewritten with `filter-branch`, force-pushed both branches, token rotation still recommended
9. ~~Build fine-tuning infrastructure + training runs~~ **DONE**. See `docs/joyful-popping-planet.md` (full tracker).
   - `training.py`: 6 functions, **124/124 full suite passes** (39 training + 29 slicer + 38 ontology + 18 dataset+svg)
   - Deployment: `Makefile` (10 targets), `.env.example`, `README.md`, `make deploy` workflow
   - Notebook: `notebooks/step9_finetune_coarse.ipynb` — 8 cells, patterns from `ml-workflow-tool/training_templates/LESSONS_LEARNED.md`
   - **Run 1 (spatial split, 2026-03-10):** mIoU 50.7%, Cerebrum **NaN** — broken spatial split
   - **Cerebrum NaN ROOT CAUSE:** Contiguous spatial split along AP axis put ALL cerebrum pixels in train (0 in val/test)
   - **FIX:** `split_strategy="interleaved"` — every 10th slice → val, every 10th+1 → test
   - **Run 2 (interleaved split, 2026-03-10):** Overall accuracy **95.8%**, mIoU **88.0%**
     - Per-class IoU: BG 92.1%, Cerebrum **95.8%**, BS 94.3%, CB 92.9%, Fiber 73.0%, VS 80.1%
     - Eval loss: 0.175 (was 1.238). Training loss similar (0.240 vs 0.217) — model quality was fine, eval was broken
   - **Next:** Scale to finer granularity (Step 10)
10. ~~Move to fine granularity~~ **DONE**. See `docs/step10_plan.md`.
    - **Bug fix:** `get_class_names()` in `ontology.py` picked arbitrary structure per class for non-coarse mappings. Fixed: prefer shallowest-depth structure. 9 new tests (133 total).
    - **Notebooks:** `finetune_depth2.ipynb` (19 classes), `finetune_full.ipynb` (1,328 classes, batch_size=4). Renamed from step9/step10 prefixes.
    - **Run 3 (depth-2, 2026-03-10):** mIoU **69.6%**, accuracy 95.5%, 25 min on 1x L40S. Top: Cerebrum 95.8%, Brain stem 94.5%, Cerebellum 93.0%. 15/19 classes valid (4 absent from val).
    - **Run 4 (full, 2026-03-11):** mIoU **60.3%**, accuracy 88.9%, 5.5 hrs on 1x L40S. Top: Caudoputamen 95.2%, Background 94.3%, Main olfactory bulb 93.8%. 503/1,328 classes valid (825 absent from val).
    - **GPU memory lessons:** Multi-GPU DDP OOM'd at batch=4 (DDP overhead consumed headroom). Single GPU succeeded. UperNet doesn't support gradient checkpointing. See `docs/step10_gpu_memory_review.md`.
    - **DINOv2 research:** Giant (1.1B params) fits on L40S but unlikely to be highest-impact next step. Recommend unfreezing backbone first. See `docs/dinov2_model_research.md`.
11. ~~Step 11a — Unfreeze backbone~~ **DONE**. See `docs/step11_plan.md`.
    - **Run 5 (unfrozen, 2026-03-12):** mIoU **68.8%**, training loss 0.730, 11.3 hrs on 1x L40S.
    - **+8.5% mIoU gain** over frozen baseline (60.3%) — within expected 5-15% range.
    - Strategy: Last 4 DINOv2 blocks (20-23) unfrozen, differential LR (backbone 1e-5, head 1e-4), 100 epochs.
    - 3 deployment failures resolved: OOM at batch=4, BatchNorm ValueError at batch=1, DDP gradient checkpointing bug (`use_reentrant=False` required). Final: batch=2, grad_accum=2.
    - MLflow artifact gap fixed: all 4 notebooks now call `mlflow.log_artifacts()` before `mlflow.end_run()`.
    - LESSONS_LEARNED.md updated: gradient checkpointing + frozen boundary, PSP BatchNorm minimum batch, DDP on single-GPU Databricks, MLflow artifact logging.
12. **Step 11b — Class-weighted Dice + CE loss** — **DONE (NEGATIVE RESULT)**. See `docs/step11b_plan.md`.
    - **Run 6 (weighted loss, 2026-03-12):** mIoU **67.2%** (−1.6% vs unfrozen 68.8%), accuracy 89.7%, ~33 hrs on 1x L40S, 100 epochs.
    - Combined loss `L = 0.5 * weighted_CE + 0.5 * Dice` with inverse-frequency class weights clipped at 95th percentile.
    - 503 classes with valid IoU — identical to Run 5 (no new classes discovered). 657 classes have zero training pixels.
    - Regression likely caused by: (a) alpha=0.5 too aggressive — Dice diluted CE gradient signal on well-learned common classes, (b) Dice mean over only present-in-batch classes (4-10) produces noisy gradients for 1,328-class problem.
    - Infrastructure retained: `losses.py`, `WeightedLossUperNet`, `create_model(loss_fn=...)` remain available for future experiments with different alpha or focal loss.
    - Model saved: `/dbfs/FileStore/allen_brain_data/models/weighted-loss`.
13. **Step 11c — Extended augmentation** — **DONE (NEGATIVE RESULT)**. See `docs/step11c_plan.md`.
    - **Run 7 (augmented, 2026-03-15):** mIoU **62.3%** (−6.5% vs unfrozen 68.8%), accuracy 90.1%, eval loss 0.4894, 100 epochs on 1x L40S.
    - Extended `_apply_augmentation()` in `dataset.py` with 3 new transforms:
      - Random 90° rotation (50% prob) — `np.rot90`, exact, no interpolation artifacts
      - Gaussian blur (30% prob, σ=0.5-2.0) — `scipy.ndimage.gaussian_filter`, image only
      - Elastic deformation (30% prob, α=50, σ=5) — `scipy.ndimage.map_coordinates`, order=0 for mask
    - 503 classes valid (identical to Runs 5 and 6). 657 classes have zero training pixels.
    - Regression caused by: (a) augmentation too aggressive for 1,016 training samples — compounding 6 transforms heavily distorts every sample, (b) elastic deformation (α=50) warps anatomical boundaries, creating ambiguous labels, (c) both accuracy (−2.4%) and mIoU (−6.5%) regressed from Run 5.
    - MLflow artifacts saved with corrected pattern: `mlflow.transformers.log_model()` — confirmed working.
    - Model saved: `/dbfs/FileStore/allen_brain_data/models/augmented`.
    - Infrastructure retained: augmentation code and tests remain for possible future use with reduced intensity.
14. **Step 12 — Data-centric improvements** — **MOUSE MODEL COMPLETE**. See `docs/step12_plan.md`.
    - **Diagnosis:** Data scarcity is the binding constraint (1,016 samples / 1,328 classes / 657 with zero pixels). Model rejects high-intensity interventions (Runs 6-7 both regressed).
    - **Step 0 (TTA):** DONE — **NEGATIVE RESULT**. 6-variant TTA catastrophically regressed: 68.4% → 44.4% mIoU (−24.05%).
    - **Step 1 (augmentation preset):** DONE — `augmentation_preset` parameter added. 6 new tests.
    - **Step 2 (class pruning):** DONE — `build_present_mapping()` added. 9 new tests.
    - **Step 3 (multi-axis slicing):** DONE — Multi-axis support added. 17 new tests.
    - **Step 4 (Run 8a):** DONE — **NEUTRAL RESULT**. Pruned 673-class: 69.0% mIoU (+0.2% within noise). Cross-axis: coronal 68.9%, axial 3.2%, sagittal 0.5%.
    - **Step 5 (Run 9 — FINAL):** DONE — **74.8% mIoU (center-crop), 79.1% mIoU (sliding window), 96.9% accuracy (SW)**. 200 epochs, identical config to Run 5. +6.0% mIoU from extended training. Sliding window eval included (Step 6 partially addressed). Model saved to `/dbfs/FileStore/allen_brain_data/models/final-200ep`.
    - **Steps 6-7:** Sliding window eval done in Run 9 notebook. Write-up/paper PENDING.
    - **Steps 8-10 (human model):** NEXT — separate human brain segmentation model. Human data already downloaded and uploaded (4,463 SVGs, 14,565 images, 6 donors). Step 8 (ground truth investigation) is the immediate next step.
    - **Test suite:** 211/211 pass (was 181).
15. **Human data status — DOWNLOAD + UPLOAD COMPLETE (2026-03-16):**
    - See `docs/data_download_plan_human.md` for full details.
    - See `docs/human_data_search_results.md` for search findings.
    - **Allen Human SectionImage SVGs:** 4,463 files (541.8 MB), 524+ unique structure IDs mapping to Graph 10 ontology (1,839 structures). 96% of SVGs contain actual annotations. 6 donors.
    - **Allen Developing Human Atlas (21 pcw):** 169 images + 169 SVGs, atlas ID 3. 156/169 SVGs have structure annotations (92%). Uses Graph 16 ontology (3,317 structures).
    - **Human Ontologies:** Graph 10 (adult, 1,839 structures, 844 KB), Graph 16 (developing, 3,317 structures, 1.8 MB).
    - **BigBrain volumes:** 9-class tissue classification (16 MB, 200μm), histological intensity 8-bit (74 MB, 200μm), hippocampus L+R (3.5 MB, 400μm), layer segmentation (391 MB, 13 sections × 6 files each).
    - **siibra volumes (BigBrain space):** Julich-Brain v2.9 (122 regions, 2% voxel coverage, 1.9 MB), cortical layers (6 layers, 38.4% coverage, 8.1 MB), isocortex segmentation (binary cortex/subcortex, 25.8% coverage, 3.4 MB).
    - **All uploaded to Databricks Workspace** via `make deploy-human-annotations`.
    - **Download script:** `scripts/download_human_annotations.py` (idempotent, re-runnable).
    - **Allen adult AtlasImage SVGs are ALL EMPTY** — only SectionImage SVGs and developing atlas SVGs have actual annotations.
    - **Design decision:** Two clean, separate models — mouse-only and human-only. Human model free to use any mouse learnings (architecture, hyperparams, weight initialization) but trained exclusively on human data.
16. **Future steps** — after Step 12:
    - DINOv2-Giant: only after above exhausted (see `docs/dinov2_model_research.md`)

---

## Compact Context (for /compact)

**Project:** `histological-image-analysis` — Fine-tune a vision model to identify brain structures in Nissl-stained mouse/human brain slides using Allen Brain Institute data.

**Mouse model: COMPLETE.** Final model (Run 9): **74.8% mIoU** (center-crop), **79.1% mIoU** (sliding window), **96.9% accuracy** (SW), 200 epochs, DINOv2-Large + UperNet, 1,328 classes. Saved to `/dbfs/FileStore/allen_brain_data/models/final-200ep`. See `docs/mouse_model_findings.md` for full results and human model recommendations.

**Current focus: Human model pipeline (Step 8 — ground truth investigation).**

**Compute:** Databricks 17.3 LTS, g6e.16xlarge (L40S 48GB VRAM, 512GB RAM), single node. Firewall blocks Allen domains — use local download + Workspace upload.

**Completed:** (1) Data source research — 7 sources evaluated, Mouse Atlas + CCFv3 are primary. (2) Model research — 20 models, top pick is DINOv2-Large + UperNet seg head. (3) Explorer notebook runs end-to-end locally. (4) Connectivity tested — Allen domains BLOCKED (`ConnectionResetError 104`), PyPI + HuggingFace PASS. Details in `docs/databricks_connectivity.md`.

**Connectivity:** `api.brain-map.org` and `download.alleninstitute.org` blocked by corporate firewall on all Databricks clusters. `pypi.org` and `huggingface.co` work. Cluster `0306-215929-ai2l0t8w` is interactive-only (`jobs: false`), use Command Execution API (`/api/1.2/commands/execute`) for CLI runs. CLI v0.278.0, `--profile dev`.

**Key API findings:** Atlas images use `model::AtlasImage` with join `atlas_data_set(atlases[id$eq1])`, NOT `atlas_data_set_id`. Download via `atlas_image_download/{id}`. SVG annotations via `svg_download/{id}`. CCFv3 via AllenSDK `ReferenceSpaceCache`. Filter `None` from `get_structures_by_id()`.

**Steps 5-6 (download+upload):** DONE. 20.87 GB total. Details: `docs/step5_6_completion_report.md`. Key data: CCFv3 nissl 10μm (float32, 1320×800×1140, 2.17 GB on DBFS), CCFv3 template 25μm (uint16, 528×320×456), 509 mouse images + 509 SVGs, 14,565 human images (~18 GB), ontology JSON. Most files on Workspace at `/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/`. Exception: `ara_nissl_10.nrrd` on DBFS at `/dbfs/FileStore/allen_brain_data/ccfv3/` (exceeded 500 MB workspace limit). Step 6B (cluster verification) pending.

**Step 7 (dataset+model):** DONE. DINOv2-Large + UperNet. Final goal: autofluorescence. Training: Nissl → autofluorescence, coarse (6) → fine (1,327). Human Atlas RESOLVED (`'NISSL'` uppercase). Details: `docs/step7_dataset_model_decision.md`.

**Step 8 (training data pipeline):** DONE. 4 components in `src/histological_image_analysis/`: `ontology.py` (OntologyMapper — coarse/depth/full mappings, ancestor-chain algorithm), `ccfv3_slicer.py` (CCFv3Slicer — NRRD loading, float32/uint16 normalization, spatial AP split, on-the-fly slicing), `svg_rasterizer.py` (SVGRasterizer — svgpathtools bezier→polygon, PIL rasterization, nearest-neighbor resize), `dataset.py` (BrainSegmentationDataset — padding, random/center crop 518×518, augmentation, ImageNet normalization, outputs {"pixel_values": (3,518,518), "labels": (518,518)}). Tests in `tests/` — 61/61 pass. Deps added: torch, torchvision, transformers, Pillow, svgpathtools, numpy, pytest. Plan: `docs/step8_implementation_plan.md`.

**Ontology correction:** Root(997) depth-1 children are grey(8), fiber tracts(1009), VS(73), grooves(1024), retina(304325711). Cerebrum(567)/BS(343)/CB(512) are depth-2 under grey(8). Coarse mapping uses ancestor-chain walking, not depth. Real data: 1,327 structures → 6 coarse classes (637 Cerebrum, 375 BS, 87 CB, 191 fiber, 12 VS, 25 background).

**Security — RESOLVED:** `.claude/settings.local.json` contained Databricks API token. Fixed: `git filter-branch` rewrote all commits on both `master` and `preparation` branches to remove the file. Force-pushed both branches. `.claude/` in `.gitignore`. Token rotation still recommended as a precaution (data was transmitted to GitHub servers before rejection).

**JFrog Artifactory (Step 9):** Model weights (`facebook/dinov2-large`) must be manually added to JFrog by user before loading on Databricks. Code uses standard `from_pretrained()` so JFrog mirror swap is seamless.

**Deployment strategy:** src/ package installed as wheel on Databricks cluster. Training notebook (Step 9) imports from installed package — thin notebook for orchestration + visualization only. No UDF/Spark concern (single-node PyTorch).

**Step 9 (training + first run):** DONE. `training.py` with 6 functions: `create_model` (DINOv2 backbone + UperNet head, freeze support, `auxiliary_in_channels` must match backbone `hidden_size`), `compute_metrics` (manual mIoU), `make_compute_metrics` (closure factory), `preprocess_logits_for_metrics` (argmax to prevent eval OOM), `get_training_args` (critical: `remove_unused_columns=False`), `create_trainer`. 36 tests, 102/102 full suite. Notebook: `notebooks/step9_finetune_coarse.ipynb` (8 cells). Deployment: `Makefile` (10 targets), `.env.example`, `README.md`. `make deploy` builds wheel + uploads to DBFS + uploads notebook. Patterns from `ml-workflow-tool/training_templates/LESSONS_LEARNED.md`: retry on `snapshot_download`, `etag_timeout=86400`, JFrog Artifactory URL default, `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`, single `mlflow.start_run()`/`end_run()` across cells, `mlflow.log_params(HYPERPARAMS)`.

**Step 9 first training run (2026-03-10):** Coarse 6-class, frozen DINOv2-Large backbone, UperNet head-only, 50 epochs, batch_size=8, lr=1e-4, fp16. Runtime: 23 min on L40S 48GB. **Overall accuracy 78.8%, mIoU 50.7%.** Per-class IoU: BG 53.5%, Cerebrum **NaN** (critical), BS 74.1%, CB 70.1%, Fiber 41.1%, VS 14.7%. Model saved: `/dbfs/FileStore/allen_brain_data/models/coarse_6class`. MLflow experiment: `/Users/noel.nosse@grainger.com/histology-brain-segmentation` (ID: 1345391216675532).

**Cerebrum NaN ROOT CAUSE + FIX (2026-03-10):** Contiguous spatial split along AP axis put ALL cerebrum pixels (17.5M) in train, 0 in val/test. The mouse brain cerebrum is concentrated anteriorly; the posterior 20% (val+test) contains only brain stem, cerebellum, fiber tracts, and ventricular systems. Fix: `split_strategy="interleaved"` in `get_split_indices()` — every 10th slice → val, every 10th+1 → test, rest → train. All 6 classes present in all 3 splits. 124/124 tests pass.

**Step 9 second training run (2026-03-10, interleaved split):** Same config as Run 1 but with `split_strategy="interleaved"`. Runtime: 23 min on L40S. **Overall accuracy 95.8%, mIoU 88.0%.** Per-class IoU: BG 92.1%, Cerebrum **95.8%**, BS 94.3%, CB 92.9%, Fiber 73.0%, VS 80.1%. Eval loss: 0.175 (was 1.238). Training loss similar (0.240 vs 0.217) — model quality was fine in Run 1, eval was broken by missing classes. Cerebrum is now the best-performing class. Weakest: fiber tracts (73.0%, thin structures), VS (80.1%, small volume). Model saved: `/dbfs/FileStore/allen_brain_data/models/coarse_6class` (overwritten). Next: evaluate on test split, then decide direction (improve coarse vs move to fine granularity).

**Databricks deployment lessons:** (1) Workspace has 500 MB per-file limit — checkpoints to `/tmp/`, final model to `/dbfs/`. (2) `databricks workspace import` v0.278+ uses `--file` flag, not positional arg; parent dirs must exist. (3) Route all downloads through JFrog Artifactory `https://graingerreadonly.jfrog.io/artifactory/api/huggingfaceml/huggingfaceml-remote`. (4) HF Trainer `report_to="mlflow"` can leave runs open — always call `mlflow.end_run()`. (5) `find_unused_parameters=True` DDP warning is harmless on single GPU.

**Step 10 (fine granularity):** DONE. Bug fix: `get_class_names()` prefers shallowest-depth structure per class. 9 new tests (133 total). Notebooks renamed: `finetune_coarse.ipynb`, `finetune_depth2.ipynb`, `finetune_full.ipynb`. **Run 3 (depth-2):** mIoU 69.6%, accuracy 95.5%, 25 min 1x L40S. **Run 4 (full):** mIoU 60.3%, accuracy 88.9%, 5.5 hrs 1x L40S. 503/1,328 classes valid. Multi-GPU DDP OOM'd at batch=4 (per-GPU DDP overhead). UperNet doesn't support gradient checkpointing. See `docs/step10_gpu_memory_review.md`. DINOv2-Giant researched: fits on L40S but recommend unfreezing backbone first. See `docs/dinov2_model_research.md`. Plan: `docs/step10_plan.md`.

**Step 11a (backbone unfreezing):** DONE. Run 5: mIoU **68.8%** (+8.5% over frozen 60.3%), 11.3 hrs on 1x L40S, 100 epochs. Last 4 DINOv2 blocks (20-23) unfrozen, differential LR (backbone 1e-5, head 1e-4). 3 deployment failures resolved: (1) OOM at batch=4, (2) UperNet PSP BatchNorm ValueError at batch=1 (minimum batch=2), (3) gradient checkpointing `use_reentrant=True` broke gradient flow at frozen/unfrozen boundary (fix: `use_reentrant=False`). Final config: batch=2, grad_accum=2, `ddp_find_unused_parameters=True`. MLflow artifact gap fixed in all 4 notebooks (`mlflow.log_artifacts()` before `mlflow.end_run()`). LESSONS_LEARNED.md updated with 4 new sections. Plan: `docs/step11_plan.md`. Full roadmap: `docs/finetuning_recommendations.md` (15 recommendations in 4 phases).

**Step 11a-i (local inference tooling):** DONE. Model logged to MLflow (run ID `6cc49e1ccb0d4b30b371e9a071dcbe6f`). Created local inference infrastructure for PhD users: (1) `docs/model_download_guide.md` — comprehensive guide with 3 download methods (DBFS, MLflow, direct load), verification, troubleshooting. (2) `scripts/run_inference.py` — CLI tool for batch inference on histological images. (3) `verify_model.py` — quick verification script. (4) Models excluded from git via `.gitignore` (~1.2 GB each). (5) README updated with "Using Trained Models" section. Download: `databricks fs cp -r dbfs:/FileStore/allen_brain_data/models/unfrozen ./models/dinov2-upernet-unfrozen`. Inference example processes user-provided brain tissue images and outputs predicted segmentation masks with structure IDs. Script supports batch processing, GPU acceleration, and visualization. **Ready for PhD researcher testing.**

**Step 11b (weighted loss):** DONE — NEGATIVE RESULT. Run 6: mIoU **67.2%** (−1.6% vs unfrozen 68.8%), accuracy 89.7%, ~33 hrs, 100 epochs. Combined loss `0.5 * weighted_CE + 0.5 * Dice` with inverse-frequency weights. 503 classes valid (same as Run 5). Regression: alpha=0.5 too aggressive, Dice diluted CE signal on common classes without helping rare ones. 657 classes have zero training pixels — no loss function can help. Infrastructure retained for future use. Model: `/dbfs/FileStore/allen_brain_data/models/weighted-loss`.

**Step 11c (extended augmentation):** DONE — NEGATIVE RESULT. Run 7: mIoU **62.3%** (−6.5% vs unfrozen 68.8%), accuracy 90.1% (−2.4% vs Run 5's 92.5%), eval loss 0.4894, 100 epochs. Extended `_apply_augmentation()` with 3 new transforms (rot90, blur, elastic deformation). Regression: augmentation too aggressive for 1,016 samples — elastic deformation (α=50) warps anatomical boundaries, compounding 6 transforms heavily distorts every sample. Both mIoU and accuracy dropped. Infrastructure (code + 12 tests) retained. MLflow artifacts saved successfully with corrected pattern. Model: `/dbfs/FileStore/allen_brain_data/models/augmented`.

**Step 12 Run 8a (class pruning ablation):** DONE — NEUTRAL RESULT. Run 8a: mIoU **69.0%** (+0.2% vs Run 5 68.8%), accuracy 92.4% (−0.1%), 673 classes (655 pruned), 1,016 coronal slices, ~9.5 hrs. Softmax dilution hypothesis NOT confirmed. Cross-axis: coronal 68.9%, axial 3.2%, sagittal 0.5% — model is orientation-specific. Model: `/dbfs/FileStore/allen_brain_data/models/pruned-coronal-only`.

**Step 12 Run 9 (final mouse model, 200 epochs):** DONE — **BEST MODEL**. Run 9: mIoU **74.8%** (center-crop, +6.0% vs Run 5), **79.1%** (sliding window), accuracy **94.1%** (CC) / **96.9%** (SW), eval loss **0.300** (−17.3% vs Run 5's 0.363), training loss 0.557, 200 epochs, 23.0 hrs on 1x L40S. The +6.0% gain from doubled epochs was 3× larger than predicted (+1-2%), proving the model was substantially underfit at 100 epochs. Sliding window eval (518×518 tiles, stride 259, 50% overlap) revealed 671 valid classes (vs 503 center-crop) — 168 structures exist only at slice edges. Top classes: Caudoputamen 97.9%, Main olfactory bulb 97.5%, Background 97.3%. Model: `/dbfs/FileStore/allen_brain_data/models/final-200ep`. MLflow run: `final-200ep-1328class-20260316-1405`. All mouse findings documented in `docs/mouse_model_findings.md` for human team reference.

**MLflow artifact save pattern (CRITICAL — learned from Run 6 recovery + Run 7/8a failures):**
All future training notebooks MUST follow this pattern in the final save cell:
1. Set `model.config.id2label` and `model.config.label2id` from `class_names` BEFORE saving — `create_model()` does not set these.
2. `trainer.save_model(FINAL_MODEL_DIR)` — backup to DBFS.
3. `processor.save_pretrained(FINAL_MODEL_DIR)` — trainer.save_model() does NOT save it.
4. **`mlflow.log_artifacts(FINAL_MODEL_DIR, artifact_path="model")`** — log raw checkpoint files. Do NOT use `mlflow.transformers.log_model()` (UperNet is incompatible — see warning below).
5. Do NOT pass `registered_model_name` unless it's a 3-level Unity Catalog path (`catalog.schema.model`).

**NEVER use `mlflow.transformers.log_model()` for UperNet.** `UperNetForSemanticSegmentation` is NOT compatible with MLflow's transformers flavor — it is not a Pipeline, not in the AutoModel registry, and cannot be loaded by `mlflow.transformers.load_model()`. Passing the model object fails (`MlflowException: not a Pipeline`). Passing the saved directory path also fails (`KeyError: UperNetConfig` — not in AutoModel registry). This bug has occurred 4 times across 3 notebooks. **Use `mlflow.log_artifacts(FINAL_MODEL_DIR, artifact_path="model")` instead.** Load later with `UperNetForSemanticSegmentation.from_pretrained(path)`.

The older notebooks (`finetune_unfrozen.ipynb`, `finetune_weighted_loss.ipynb`) still use the old `mlflow.log_artifacts()` pattern. Run 6 artifacts were recovered under MLflow run `55e3e0c9c98a4c65ade76bd65f2245ee`.

**Training run history:**

| Run | Config | mIoU (CC) | mIoU (SW) | Delta vs prev best | Runtime | Notebook | Model Dir |
|-----|--------|-----------|-----------|---------------------|---------|----------|-----------|
| 1 | Coarse 6-class, frozen, spatial split | 50.7% | — | — | 23 min | `finetune_coarse.ipynb` | overwritten |
| 2 | Coarse 6-class, frozen, interleaved | 88.0% | — | +37.3% | 23 min | `finetune_coarse.ipynb` | `coarse_6class` |
| 3 | Depth-2 19-class, frozen | 69.6% | — | — | 25 min | `finetune_depth2.ipynb` | `depth2_19class` |
| 4 | Full 1,328-class, frozen | 60.3% | — | — | 5.5 hrs | `finetune_full.ipynb` | `full_1328class` |
| 5 | Full 1,328-class, unfrozen, 100ep | 68.8% | — | +8.5% | 11.3 hrs | `finetune_unfrozen.ipynb` | `unfrozen` |
| 6 | Full 1,328-class, weighted Dice+CE | 67.2% | — | −1.6% | ~33 hrs | `finetune_weighted_loss.ipynb` | `weighted-loss` |
| 7 | Full 1,328-class, extended augmentation | 62.3% | — | −6.5% | ~11-14 hrs | `finetune_augmented.ipynb` | `augmented` |
| TTA | Eval-only: 6-variant TTA on Run 5 | 44.4% | — | −24.0% | ~20 min | `eval_tta.ipynb` | — (eval only) |
| 8a | Pruned 673-class, unfrozen, coronal-only | 69.0% | — | +0.2% | ~9.5 hrs | `finetune_pruned_ablation.ipynb` | `pruned-coronal-only` |
| **9** | **Full 1,328-class, unfrozen, 200ep** | **74.8%** | **79.1%** | **+6.0%** | **23.0 hrs** | **`finetune_final_200ep.ipynb`** | **`final-200ep`** |

All model dirs are under `/dbfs/FileStore/allen_brain_data/models/`. Best model: **Run 9 (final-200ep) at 74.8% mIoU center-crop / 79.1% mIoU sliding window.** The two most impactful interventions were backbone unfreezing (+8.5%) and extended training to 200 epochs (+6.0%). Four other experiments (weighted loss, augmentation, TTA, class pruning) all failed.

**Current augmentation pipeline** (`dataset.py:_apply_augmentation`, controlled by `augmentation_preset`):
- **`"baseline"`** (default, Run 5 config): Horizontal flip (50%), rotation ±15°, color jitter (always)
- **`"extended"`** (Run 7 config): All of baseline PLUS random 90° rotation (50%), elastic deformation (30%, α=50, σ=5), Gaussian blur (30%, σ=0.5-2.0)
- **`"none"`**: No augmentation (equivalent to `augment=False`)

**Key constraints and lessons:**
- Batch=1 fails: UperNet PSP BatchNorm needs ≥2 values per channel. Minimum batch=2.
- `dataloader_drop_last=True` REQUIRED when training sample count is odd (multi-axis gives 2,649 samples; 2649%2=1 → last batch has 1 sample → same PSP BatchNorm crash). Confirmed on Databricks run.
- Gradient checkpointing: backbone only, `use_reentrant=False` REQUIRED at frozen/unfrozen boundary.
- `ddp_find_unused_parameters=True` needed: Databricks wraps in DDP even on single GPU.
- `dataloader_num_workers=4` is the default in `get_training_args()`.
- 657/1,328 classes have zero training pixels — no technique can learn absent classes.
- PhD researchers will submit smaller sub-sections of brain tissue, NOT full coronal/sagittal sections (noted for future Phase 3 inference pipeline).

**Data correction (2026-03-14):** Run 5 overall accuracy was documented as 89.4% in multiple .md files. Notebook output shows **92.48%**. Eval_loss is **0.3631**. All references corrected. The "accuracy paradox" analysis in step11c_plan.md (claiming Run 7 improved accuracy vs Run 5) was also wrong — Run 7's 90.1% is LOWER than Run 5's 92.5%.

**Experimental results:** All training run data consolidated in `docs/experimental_results.md` — summary table, hyperparameter comparison, per-class IoU tables (all runs), training curves (Runs 3-7). Runs 1-2 training curves LOST (notebook outputs not preserved; those notebooks were never run on Databricks — original runs used step9/step10 prefixed notebooks).

**Human data status:** 14,565 human Nissl images + 4,463 SVG annotations downloaded. Uploaded to Databricks Workspace. Allen Human SectionImage SVGs: 524+ unique structure IDs, 96% have annotations. Allen Developing Human Atlas: 169 images + 169 SVGs. BigBrain + siibra volumes also downloaded. See `docs/data_download_plan_human.md`. **Design decision: two separate models** — mouse model COMPLETE (Run 9), now pivot to human model using mouse learnings (architecture, hyperparams, optionally Run 9 weight initialization) but trained exclusively on human data. **Next step: Step 8 — investigate human ground truth quality and build training pipeline.**

**Repo files:** `docs/progress.md` (this file), `docs/experimental_results.md` (consolidated run data for paper), `docs/step12_plan.md` (Step 12 plan — now includes Steps 8-10 for human model), `docs/model_download_guide.md` (local inference guide), `docs/step11c_plan.md` (Step 11c plan + Run 7 results), `docs/step11b_plan.md` (Step 11b plan + Run 6 results), `docs/step11_plan.md` (Step 11 plan — complete), `docs/step10_plan.md` (Step 10 plan — complete), `docs/finetuning_recommendations.md` (comprehensive 4-phase roadmap with 15 recommendations), `docs/dinov2_model_research.md` (backbone research), `docs/step10_gpu_memory_review.md` (GPU memory lessons), `docs/joyful-popping-planet.md` (Step 9 tracker — complete), `docs/step8_implementation_plan.md`, `docs/step7_dataset_model_decision.md`, `docs/step5_6_completion_report.md`, `docs/references.md`, `docs/databricks_connectivity.md`. Source: `src/histological_image_analysis/{ontology,ccfv3_slicer,svg_rasterizer,dataset,training,losses}.py`. Tests: `tests/test_{ontology,ccfv3_slicer,svg_rasterizer,dataset,training,losses}.py` + `conftest.py` + `fixtures/` — **211/211 pass**. Active notebooks: `notebooks/finetune_final_200ep.ipynb` (**Run 9 — FINAL mouse model, 200ep, best**), `notebooks/finetune_unfrozen.ipynb` (Run 5), `notebooks/finetune_augmented.ipynb` (Run 7), `notebooks/eval_tta.ipynb` (Step 12 TTA eval), `notebooks/finetune_pruned_multiaxis.ipynb` (Run 8 combined — superseded), `notebooks/finetune_pruned_ablation.ipynb` (Run 8a ablation). Historical notebooks moved to `notebooks/historical/`. Scripts: `scripts/{download_allen_data,run_inference}.py`. Verification: `verify_model.py`. Infra: `Makefile` (20 targets), `.env.example`, `README.md`.
