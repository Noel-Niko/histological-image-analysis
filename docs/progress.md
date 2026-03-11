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
| `exploration/allen_brain_data_explorer.ipynb` | Browse Allen Brain data sources to choose dataset | **Complete — all cells run** |
| `exploration/databricks_connectivity_check.ipynb` | Verify Databricks can reach Allen data sources | **Complete — 6 blocked, 2 pass** |

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
10. ~~Move to fine granularity~~ **IN PROGRESS**. See `docs/step10_plan.md`.
    - **Decision:** Scale to depth-2 (19 classes), then full mapping (1,328 classes). Coarse model is strong (88% mIoU), diminishing returns to push higher.
    - **Bug fix:** `get_class_names()` in `ontology.py` picked arbitrary structure per class for non-coarse mappings. Fixed: prefer shallowest-depth structure. 9 new tests added (133 total).
    - **Notebooks created:** `step10_finetune_depth2.ipynb` (19 classes), `step10_finetune_full.ipynb` (1,328 classes, batch_size=4)
    - **Makefile:** Added `deploy-notebook-depth2`, `deploy-notebook-full`. `make deploy` now deploys all 3 notebooks.
    - **Multi-GPU strategy:** DDP (not FSDP). Model is 350M params (700 MB fp16) — fits on single GPU. DDP is automatic with HF Trainer on multi-GPU cluster. 8x L40S → ~6-7x speedup.
    - **Memory:** Full mapping at batch=4: ~5.7 GB per GPU (12% of L40S 48 GB). No OOM risk.
    - **Status:** All local code complete. Deployment + training pending.
11. **Evaluate** — test on held-out Mouse Atlas sections, measure per-structure IoU

---

## Compact Context (for /compact)

**Project:** `histological-image-analysis` — Fine-tune a vision model to identify brain structures in Nissl-stained mouse/human brain slides using Allen Brain Institute data.

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

**Step 10 (fine granularity):** IN PROGRESS. Local code complete, deployment pending. Bug fix: `get_class_names()` in `ontology.py` now prefers shallowest-depth structure per class (was arbitrary). 9 new tests (133 total). Notebooks: `step10_finetune_depth2.ipynb` (19 classes, `build_depth_mapping(target_depth=2)`), `step10_finetune_full.ipynb` (1,328 classes, `build_full_mapping()`, batch_size=4). Multi-GPU: DDP on 8x L40S (g6e.48xlarge), HF Trainer auto-detects. Model: 350M params, 700 MB fp16. Full mapping memory: ~5.7 GB/GPU (12% of 48 GB). Makefile: 12 targets incl. `deploy-notebook-depth2`, `deploy-notebook-full`. `make deploy` deploys all 3 notebooks. Plan: `docs/step10_plan.md`.

**Repo files:** `docs/progress.md` (this file), `docs/step10_plan.md` (Step 10 plan — active), `docs/joyful-popping-planet.md` (Step 9 tracker — complete), `docs/step8_implementation_plan.md`, `docs/step7_dataset_model_decision.md`, `docs/step5_6_completion_report.md`, `docs/references.md`, `docs/databricks_connectivity.md`. Source: `src/histological_image_analysis/{ontology,ccfv3_slicer,svg_rasterizer,dataset,training}.py`. Tests: `tests/test_{ontology,ccfv3_slicer,svg_rasterizer,dataset,training}.py` + `conftest.py` + `fixtures/`. Notebooks: `notebooks/step9_finetune_coarse.ipynb`, `notebooks/step10_finetune_depth2.ipynb`, `notebooks/step10_finetune_full.ipynb`. Infra: `Makefile` (12 targets), `.env.example`, `README.md`.
