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

---

## Data Source Research

_Status: Complete_

### 1. Allen Mouse Brain Atlas (Nissl Reference) — **PRIMARY CANDIDATE**

- **URL:** `https://mouse.brain-map.org/`
- **API base:** `https://api.brain-map.org/api/v2/`
- **Data type:** 2D Nissl-stained coronal section images with structure annotations
- **Species:** Mouse (C57BL/6J)
- **What it contains:**
  - **Nissl reference atlas:** 132 coronal sections spanning the full mouse brain, Nissl-stained (cresyl violet), with expert-drawn structure boundaries
  - **ISH (in-situ hybridization) data:** ~20,000+ gene experiments, each with ~60-70 sagittal or coronal sections. Not Nissl but useful for cross-reference.
  - **Structure ontology:** ~700+ hierarchically organized brain structures with unique IDs, names, acronyms, colors, and parent-child relationships
- **Image resolution:** Full-resolution section images are ~15,000–30,000 px wide. Downloadable at multiple zoom levels (0–8). At zoom 0, images are ~200 px wide thumbnails. Full-res is ~0.35–1.07 μm/pixel.
- **Key API endpoints:**
  - List section datasets: `GET /api/v2/data/query.json?criteria=model::SectionDataSet,rma::criteria,[id$eq100960033]` (Nissl reference dataset ID: **100960033**)
  - List section images in a dataset: `GET /api/v2/data/query.json?criteria=model::SectionImage,rma::criteria,[data_set_id$eq100960033],rma::options[num_rows$eqall]`
  - Download image: `GET /api/v2/section_image_download/{section_image_id}?downsample={0-8}&quality={0-100}`
  - Image tile service: `GET /api/v2/image_download/{section_image_id}?zoom={level}&top={y}&left={x}&width={w}&height={h}`
  - SVG structure overlay: `GET /api/v2/svg_download/{section_image_id}?groups={structure_ids}`
  - Structure ontology: `GET /api/v2/structure_graph_download/{structure_graph_id}.json` (graph ID 1 = adult mouse)
  - Atlas plates (annotated drawings): `GET /api/v2/atlas_image_download/{atlas_image_id}`
- **Annotation format:** SVG overlays with polygons per structure. Each polygon maps to a structure ID in the ontology. Can be rasterized to pixel-level segmentation masks.
- **Size estimate:** ~132 Nissl sections × ~50 MB each at full-res ≈ 6–7 GB raw. At downsample=4 (~2K px wide), ~132 × 500 KB ≈ 66 MB.
- **Training use:** Image + SVG → rasterize SVGs to per-pixel segmentation masks. Each pixel gets a structure label. Ideal for semantic segmentation training.
- **License:** Allen Institute Terms of Use — free for non-commercial research. Citation required.
- **Relevance:** **HIGH** — This is the most directly relevant dataset. Nissl-stained sections with expert structure annotations are exactly what we need.

### 2. Allen Human Brain Atlas

- **URL:** `https://human.brain-map.org/`
- **API base:** Same `api.brain-map.org` with human product IDs
- **Data type:** Nissl and ISH section images of human brain tissue
- **Species:** Human (6 donors, ~900 anatomical structures)
- **What it contains:**
  - Nissl-stained sections from 6 adult human brain donors
  - MRI data registered to histological sections
  - ISH data for ~1,000 genes across multiple brain regions
  - Structure annotations with ~900 named regions
- **Image resolution:** Variable; full-resolution histological sections can exceed 50,000 px
- **Key differences from mouse:**
  - Far fewer complete serial sections (brains were sub-sampled, not serially sectioned)
  - Annotations are per-region blocks rather than full-brain coronal plates
  - More heterogeneous staining quality across donors
- **Size estimate:** Several GB for Nissl sections across all donors
- **Training use:** Supplementary training data for human brain structure identification. Fewer annotations than mouse atlas.
- **License:** Allen Institute Terms of Use
- **Relevance:** **MEDIUM-HIGH** — Useful for human brain generalization but less complete annotations than the mouse atlas

### 3. Common Coordinate Framework (CCFv3) — **HIGH VALUE FOR LABELS**

- **URL:** `https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/`
- **Access via:** AllenSDK `ReferenceSpace` class, or direct NRRD download
- **Data type:** 3D annotated reference volume
- **Species:** Mouse (average template built from 1,675 specimens)
- **What it contains:**
  - **Template volume:** Average 3D brain volume at 10/25/50 μm isotropic resolution. Grayscale (autofluorescence-based, not Nissl).
  - **Annotation volume:** Same resolution, each voxel labeled with a structure ID from the ontology (~700 structures)
  - **Nissl volume:** A 3D reconstructed Nissl-stained volume registered to the CCF template (available at 10 μm resolution)
- **Key files:**
  - `annotation_10.nrrd` — 10 μm annotation volume (~1.3 GB)
  - `annotation_25.nrrd` — 25 μm annotation volume (~85 MB)
  - `template_10.nrrd` — 10 μm average template
  - `ara_nissl_10.nrrd` — 10 μm Nissl-stained volume (~1.3 GB)
- **AllenSDK access:**
  ```python
  from allensdk.core.reference_space_cache import ReferenceSpaceCache
  rsc = ReferenceSpaceCache(resolution=25, reference_space_key="annotation/ccf_2017")
  annotation, meta = rsc.get_annotation_volume()  # (528, 320, 456) array at 25μm
  template, meta = rsc.get_template_volume()
  tree = rsc.get_structure_tree()
  ```
- **Training use:** Slice the 3D volumes at arbitrary planes to generate unlimited 2D training pairs (Nissl slice + annotation mask). The Nissl volume is actual histological data reconstructed in 3D. Can also generate augmented training data by slicing at non-standard angles.
- **Size estimate:** ~4 GB total for all volumes at 10 μm
- **License:** Allen Institute Terms of Use
- **Relevance:** **HIGH** — The `ara_nissl_10.nrrd` paired with `annotation_10.nrrd` gives us perfectly aligned Nissl images + structure labels in 3D. We can generate thousands of 2D training slices.

### 4. Allen Brain Atlas API (`api.brain-map.org`)

- **URL:** `https://api.brain-map.org/api/v2/`
- **Documentation:** `https://help.brain-map.org/display/api/`
- **Type:** REST API (JSON responses)
- **Key capabilities:**
  - **RMA (RESTful Model Access):** General query syntax for all data models
  - **Section image download:** Full-resolution or downsampled, with quality control
  - **SVG annotation download:** Vector structure boundaries for any atlas section
  - **Structure graph:** Full ontology tree with colors, parent relationships, hierarchy levels
  - **Informatics search:** Find images by structure, gene, coordinate
- **Rate limits:** No published rate limit, but courtesy applies. Batch downloads should use reasonable parallelism.
- **Auth:** No authentication required for public data
- **Relevance:** **HIGH** — This is the primary mechanism for programmatically building our training dataset

### 5. ABC Atlas Access (`abc_atlas_access`)

- **URL:** `https://alleninstitute.github.io/abc_atlas_access/`
- **GitHub:** `https://github.com/AllenInstitute/abc_atlas_access`
- **PyPI:** `pip install abc_atlas_access`
- **Data type:** Cell type taxonomy, spatial transcriptomics (MERFISH), gene expression matrices, cluster annotations
- **Species:** Mouse (C57BL/6J-638850 specimen primarily)
- **What it provides:**
  - Programmatic access to ABC Atlas data hosted on S3
  - Cell metadata, cluster assignments, UMAP coordinates
  - MERFISH spatial coordinates mapped to CCF
  - Image volumes (reconstructed from MERFISH, not Nissl histology)
- **Image data:** The "image volumes" under `s3://allen-brain-cell-atlas/image_volumes/` are reconstructed boundary/confidence maps from MERFISH, NOT raw histological images. Formats: OME-Zarr, NIfTI.
- **Training use:** Limited for our purpose. The spatial coordinates could supplement training by providing additional structure context, but the image volumes are not Nissl stains.
- **License:** Allen Institute Terms of Use, CC-BY-4.0 for some datasets
- **Relevance:** **LOW** — Not histological images. Useful as supplementary spatial context only.

### 6. MERFISH Image Volumes (S3)

- **S3 path:** `s3://allen-brain-cell-atlas/image_volumes/MERFISH-C57BL6J-638850-CCF/20230630/`
- **Data type:** Reconstructed spatial transcriptomics images, boundary maps, confidence maps
- **Format:** OME-Zarr (multi-resolution pyramidal), NIfTI
- **Species:** Mouse
- **What it contains:**
  - Reconstructed average template registered to CCF
  - Boundary maps showing cluster boundaries
  - NOT Nissl histological images — these are computational reconstructions from MERFISH spot data
- **Training use:** Not directly applicable. These are gene-expression-derived spatial maps, not histological staining.
- **Relevance:** **LOW**

### 7. AllenSDK (`allensdk`)

- **PyPI:** `pip install allensdk`
- **GitHub:** `https://github.com/AllenInstitute/AllenSDK`
- **Key modules for our use:**
  - `allensdk.core.reference_space_cache.ReferenceSpaceCache` — Download CCF annotation/template/Nissl volumes
  - `allensdk.api.queries.ontologies_api.OntologiesApi` — Query structure ontology
  - `allensdk.api.queries.image_download_api.ImageDownloadApi` — Download section images
  - `allensdk.api.queries.svg_api.SvgApi` — Download SVG annotations
  - `allensdk.core.mouse_connectivity_cache.MouseConnectivityCache` — Access connectivity data + registered images
- **Dependencies:** numpy, scipy, h5py, pandas, requests, nrrd, SimpleITK
- **Relevance:** **HIGH** — Primary tool for downloading all Allen data programmatically

### Data Source Recommendation

**For training data, use these sources in priority order:**

1. **CCFv3 Nissl + Annotation volumes** (`ara_nissl_10.nrrd` + `annotation_10.nrrd`) — Best option. Perfectly aligned Nissl histology and structure labels in 3D. Can generate unlimited 2D slices at any plane. Download via AllenSDK.

2. **Mouse Brain Atlas Nissl sections + SVG annotations** (via API) — 132 high-res 2D sections with expert-drawn structure boundaries. Use as validation/test set and additional training data.

3. **Human Brain Atlas Nissl sections** — Supplementary data for cross-species generalization.

---

## Model Research

_Status: Complete_

### Category 1: Pathology/Biomedical Foundation Models

| # | Model | HuggingFace ID | Params | Architecture | Pre-training Data | Tasks | Input Res | VRAM (FT) | License | Relevance |
|---|-------|---------------|--------|-------------|-------------------|-------|-----------|-----------|---------|-----------|
| 1 | **UNI** | `MahmoodLab/UNI` | 307M | ViT-L/16 (DINOv2) | 100K+ WSIs from Mass General Brigham, 20+ tissue types | Feature extraction, classification | 224×224 | ~12 GB | Custom (research only, gated) | **HIGH** — Pathology-pretrained, excellent tissue features |
| 2 | **UNI2** | `MahmoodLab/UNI2-h` | 632M | ViT-H/14 (DINOv2) | 350K+ WSIs, 20M+ tiles | Feature extraction, classification | 224×224 | ~20 GB | Custom (research, gated) | **HIGH** — Larger UNI, better features |
| 3 | **CONCH** | `MahmoodLab/CONCH` | 307M (vision) + text enc | CoCa (ViT-B + text decoder) | 1.17M pathology image-caption pairs from PubMed/textbooks | Zero-shot classification, retrieval, VQA | 448×448 | ~16 GB | Custom (research, gated) | MEDIUM — Vision-language, good for zero-shot but not segmentation |
| 4 | **Virchow2** | `paige-ai/Virchow2` | 632M | ViT-H/14 (DINOv2) | 3.1M WSIs from clinical archives | Feature extraction, classification | 224×224 | ~20 GB | Apache 2.0 | **HIGH** — Largest pathology training set, permissive license |
| 5 | **Prov-GigaPath** | `prov-gigapath/prov-gigapath` | 1.1B (tile) + 85M (slide) | ViT-g/14 (DINOv2) + LongNet | 1.3B tiles from 171K WSIs (Providence Health) | Feature extraction, slide-level classification | 256×256 | ~32 GB | Custom (research, gated) | MEDIUM — Very large, close to VRAM limit |
| 6 | **BiomedCLIP** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 86M (vision) + text enc | ViT-B/16 + PubMedBERT | PMC-15M (15M biomedical image-text pairs from PubMed Central) | Zero-shot classification, retrieval | 224×224 | ~6 GB | MIT | **HIGH** — Open license, biomedical domain, lightweight |
| 7 | **PLIP** | `vinid/plip` | 86M (vision) + text enc | ViT-B/16 (CLIP) | 200K pathology image-text pairs from Twitter/social media | Zero-shot classification, retrieval | 224×224 | ~6 GB | MIT | MEDIUM — Smaller training set, social media sourced |
| 8 | **CTransPath** | `xiangjli/CTransPath` (unofficial) | 28M | Swin Transformer (modified) | 15M+ patches from TCGA + PAIP | Feature extraction | 224×224 | ~4 GB | Apache 2.0 | MEDIUM — Small, efficient, but older |
| 9 | **Phikon** | `owkin/phikon` | 86M | ViT-B/16 (DINOv2-like iBOT) | 40M tiles from TCGA (6K WSIs) | Feature extraction, classification | 224×224 | ~6 GB | Apache 2.0 | MEDIUM — Good baseline, open license |

### Category 2: General Vision Models (Fine-tunable for Segmentation)

| # | Model | HuggingFace ID | Params | Architecture | Pre-training | Tasks | Input Res | VRAM (FT) | License | Relevance |
|---|-------|---------------|--------|-------------|-------------|-------|-----------|-----------|---------|-----------|
| 10 | **SAM 2** | `facebook/sam2-hiera-large` | 224M | Hiera (hierarchical ViT) | SA-V (50K videos, 600K masks) + SA-1B (11M images) | Instance/interactive segmentation | Variable (1024 longest side) | ~16 GB | Apache 2.0 | **HIGH** — State-of-art segmentation, fine-tunable with adapters |
| 11 | **SAM 2.1** | `facebook/sam2.1-hiera-large` | 224M | Hiera | SA-V + SA-1B + refinement | Instance/interactive segmentation | Variable | ~16 GB | Apache 2.0 | **HIGH** — Improved SAM 2 |
| 12 | **DINOv2** | `facebook/dinov2-large` | 304M | ViT-L/14 | LVD-142M (142M curated images) | Feature extraction → linear segmentation | 518×518 | ~12 GB | Apache 2.0 | **HIGH** — Excellent features, add segmentation head |
| 13 | **Florence-2** | `microsoft/Florence-2-large` | 770M | DaViT + seq2seq decoder | 900M image-text pairs (FLD-5B) | Caption, detection, segmentation, OCR | 768×768 | ~24 GB | MIT | MEDIUM — Multi-task but not ideal for dense segmentation |
| 14 | **SegGPT** | `BAAI/SegGPT` | 370M | ViT-L/16 (Painter) | In-context learning on diverse segmentation tasks | In-context segmentation (few-shot) | 448×448 | ~14 GB | Apache 2.0 | **HIGH** — Few-shot segmentation with example images |
| 15 | **Grounding DINO** | `IDEA-Research/grounding-dino-base` | 172M | Swin-B + BERT text encoder | COCO, O365, GoldG, Cap4M | Open-vocabulary detection | 800×1333 | ~10 GB | Apache 2.0 | LOW — Detection not segmentation, text-guided |

### Category 3: Brain/Neuroscience-Specific Tools

| # | Tool/Model | URL/Package | Type | What it Does | Relevance |
|---|-----------|-------------|------|-------------|-----------|
| 16 | **DeepSlice** | `github.com/PolarBean/DeepSlice` | CNN (EfficientNet) | Predicts atlas coordinates (AP, DV, ML angle) for histological brain sections | **HIGH** — Complementary: registers sections to atlas, doesn't segment structures but aligns to CCF |
| 17 | **brainreg** | `brainglobe.info/brainreg` | ANTs-based registration | Registers 3D brain images to Allen CCF atlas | MEDIUM — 3D registration, not 2D section segmentation |
| 18 | **QUINT** | `quint-workflow.readthedocs.io` | Workflow (QuickNII + VisuAlign + Nutil) | Manual/semi-auto registration of 2D sections to 3D atlas, then quantify features per region | **HIGH** — Existing workflow for exactly our problem, but semi-manual. Our model would automate this. |
| 19 | **nnU-Net** | `github.com/MIC-DKFZ/nnUNet` | U-Net ensemble | Self-configuring medical image segmentation | **HIGH** — Gold standard for medical segmentation, auto-configures for any dataset |
| 20 | **Segment Anything in Medical Images (MedSAM)** | `bowang-lab/MedSAM` | SAM fine-tuned | SAM fine-tuned on 1.5M medical image-mask pairs | **HIGH** — Already adapted for medical imaging |

### Model Recommendations (Top 5 for Our Use Case)

**Rank 1: DINOv2-Large + Segmentation Head (or nnU-Net with DINOv2 backbone)**
- `facebook/dinov2-large` (304M params, ~12 GB FT)
- **Why:** Best general-purpose visual features. Add a linear/UperNet segmentation head and fine-tune on Allen Nissl slices. Proven to work on medical imaging. Apache 2.0 license. Well within L40S VRAM budget.
- **Approach:** Full fine-tune or LoRA on encoder + train segmentation head from scratch.

**Rank 2: SAM 2.1 fine-tuned for semantic segmentation**
- `facebook/sam2.1-hiera-large` (224M params, ~16 GB FT)
- **Why:** State-of-the-art segmentation architecture. Can be adapted from instance to semantic segmentation with a class-specific decoder. Interactive mode useful for correction. Apache 2.0.
- **Approach:** Fine-tune mask decoder, freeze/LoRA the image encoder. Or use MedSAM as starting point.

**Rank 3: UNI2 + Segmentation Head**
- `MahmoodLab/UNI2-h` (632M params, ~20 GB FT)
- **Why:** Pre-trained specifically on histopathology. Nissl staining shares visual characteristics with other histological stains. Best feature quality for tissue.
- **Caveat:** Gated model, research-only license. Must apply for access.
- **Approach:** LoRA fine-tune encoder + segmentation head.

**Rank 4: Virchow2 + Segmentation Head**
- `paige-ai/Virchow2` (632M params, ~20 GB FT)
- **Why:** Largest pathology training set (3.1M WSIs). Apache 2.0 license (fully permissive). Histology-aware features.
- **Approach:** Same as UNI2 — add segmentation decoder, LoRA fine-tune.

**Rank 5: SegGPT (few-shot baseline)**
- `BAAI/SegGPT` (370M params, ~14 GB inference)
- **Why:** In-context learning — provide a few labeled Nissl sections as examples, segment new sections without fine-tuning. Excellent for rapid prototyping and evaluation before committing to full fine-tuning.
- **Approach:** Zero/few-shot inference first. If results are promising, fine-tune.

---

## Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `exploration/getting_started_medical_imaging_fiftyone (2).ipynb` | FiftyOne DICOM/CT tutorial | Complete |
| `exploration/allen_brain_data_explorer.ipynb` | Browse Allen Brain data sources to choose dataset | In Progress |

---

## Next Steps
1. ~~Complete data source research~~ Done
2. ~~Complete model research~~ Done
3. Build `allen_brain_data_explorer.ipynb` — **IN PROGRESS**
4. Run explorer notebook on Databricks to visually evaluate data sources
5. Choose dataset + model combination
6. Build fine-tuning pipeline
