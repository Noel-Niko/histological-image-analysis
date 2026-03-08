# Histological Image Analysis — Research & Progress

## Project Goal
Fine-tune an open-source vision foundation model to identify brain structures in Nissl-stained mouse/human brain slides using Allen Brain Institute data as the structured training set.

## Compute Environment
- **Platform:** Databricks 17.3 LTS (Apache Spark 4.0.0, Scala 2.13)
- **Instance:** g6e.16xlarge (AWS)
- **GPU:** NVIDIA L40S (48 GB VRAM)
- **RAM:** 512 GB
- **Topology:** Single node
- **Constraint:** Cluster firewall blocks some external downloads — workaround is local download + upload to DBFS

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-06 | Use Allen Brain Institute data for training | Gold-standard annotated brain structure data with Nissl staining |
| 2026-03-06 | Target both mouse and human brains | Allen provides atlases for both species |
| 2026-03-06 | Use DBFS upload pattern for data | Databricks firewall blocks direct downloads |
| 2026-03-06 | FiftyOne/Voxel51 not needed for main pipeline | Our data is standard 2D images + segmentation masks, not DICOM. Matplotlib suffices for viz. |
| 2026-03-06 | Firewall blocks ALL Allen domains on Databricks | Tested via UI + CLI on dev cluster — `api.brain-map.org` and `download.alleninstitute.org` get `ConnectionResetError(104)`. PyPI and HuggingFace pass. See `docs/databricks_connectivity.md` |
| 2026-03-06 | Local download → DBFS upload for Allen data | Download ~3 GB locally, `databricks fs cp` to `dbfs:/FileStore/allen_brain_data/`, read from `/dbfs/` in notebooks |

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
5. **Download Allen data locally** — CCFv3 volumes (10μm + 25μm), mouse atlas images + SVGs, ontology JSON (~3 GB total)
6. **Upload to DBFS** — `databricks fs cp ./local_allen_data/ dbfs:/FileStore/allen_brain_data/ --recursive --profile dev`
7. **Choose dataset + model combination** — review explorer notebook outputs, decide on primary dataset and model
8. **Build training data pipeline** — slice CCFv3 volumes into 2D pairs, rasterize SVG annotations
9. **Build fine-tuning notebook** — load chosen model, attach segmentation head, train on Allen data
10. **Evaluate** — test on held-out Mouse Atlas sections, measure per-structure IoU

---

## Compact Context (for /compact)

**Project:** `histological-image-analysis` — Fine-tune a vision model to identify brain structures in Nissl-stained mouse/human brain slides using Allen Brain Institute data.

**Compute:** Databricks 17.3 LTS, g6e.16xlarge (L40S 48GB VRAM, 512GB RAM), single node. Firewall blocks Allen domains — use local download + DBFS upload.

**Completed:** (1) Data source research — 7 sources evaluated, Mouse Atlas + CCFv3 are primary. (2) Model research — 20 models, top pick is DINOv2-Large + UperNet seg head. (3) Explorer notebook runs end-to-end locally. (4) Connectivity tested — Allen domains BLOCKED (`ConnectionResetError 104`), PyPI + HuggingFace PASS. Details in `docs/databricks_connectivity.md`.

**Connectivity:** `api.brain-map.org` and `download.alleninstitute.org` blocked by corporate firewall on all Databricks clusters. `pypi.org` and `huggingface.co` work. Cluster `0306-215929-ai2l0t8w` is interactive-only (`jobs: false`), use Command Execution API (`/api/1.2/commands/execute`) for CLI runs. CLI v0.278.0, `--profile dev`.

**Key API findings:** Atlas images use `model::AtlasImage` with join `atlas_data_set(atlases[id$eq1])`, NOT `atlas_data_set_id`. Download via `atlas_image_download/{id}`. SVG annotations via `svg_download/{id}`. CCFv3 via AllenSDK `ReferenceSpaceCache`. Filter `None` from `get_structures_by_id()`.

**Next:** Download Allen data locally (~3 GB), upload to `dbfs:/FileStore/allen_brain_data/` via `databricks fs cp --profile dev`, then choose dataset+model combo, build training pipeline, fine-tune, evaluate.

**Repo files:** `docs/progress.md` (this file), `docs/databricks_connectivity.md`, `exploration/allen_brain_data_explorer.ipynb`, `exploration/databricks_connectivity_check.ipynb`. READ `CLAUDE.md` before writing code — no global vars, TDD, SOLID, uv, don't commit.
