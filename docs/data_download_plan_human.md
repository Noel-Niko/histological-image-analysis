# Human Brain Annotation Data — Download & Upload Plan

**Date:** 2026-03-15
**Depends on:** `docs/human_data_search_results.md` (search findings)
**Pattern:** Follows same approach as `docs/data_download_plan.md` (mouse/human image download)

---

## Context

### What We Found

The search (`docs/human_data_search_results.md`) identified 6 usable annotation sources for human brain histological segmentation. This document plans the download, local storage, and Databricks upload of each.

### How We Previously Downloaded Data

From `docs/data_download_plan.md` and `scripts/download_allen_data.py`:

1. **Local download** using Python `requests` library (Allen API) or `wget` (direct HTTP/FTP)
2. **Local storage** under `data/allen_brain_data/` (gitignored, ~20 GB)
3. **Databricks upload:**
   - Files <500 MB → `databricks workspace import-dir` to `/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/`
   - Files >500 MB → `databricks fs cp` to `dbfs:/FileStore/allen_brain_data/`
4. **CLI profile:** `databricks --profile dev`
5. **Firewall:** All Allen domains blocked on Databricks → must download locally first

### PhD Cover-Slip Context

The target use case is human tissue on standard cover slips provided by a PhD collaborator. Annotations should support:
- Brain region identification from partial tissue sections
- Cortical layer identification
- Tissue-type classification

---

## Data Items to Download

| # | Dataset | Source | Size | Priority |
|---|---------|--------|------|----------|
| 1 | Human SectionImage SVGs (4,463 annotated) | Allen API | ~100 MB | HIGH |
| 2 | Human Structure Ontology (Graph 10) | Allen API | ~2 MB | HIGH |
| 3 | Developing Human 21 pcw Atlas SVGs (169 images) | Allen API | ~50 MB | HIGH |
| 4 | Developing Human 21 pcw Atlas Images (169) | Allen API | ~200 MB | HIGH |
| 5 | Developing Human Structure Ontology (Graph 16) | Allen API | ~5 MB | HIGH |
| 6 | BigBrain 9-class Classified Volume (200μm) | BigBrain FTP | 16 MB | HIGH |
| 7 | BigBrain Histological Volume 8-bit (200μm) | BigBrain FTP | 74 MB | HIGH |
| 8 | BigBrain Layer Segmentation (13 sections) | BigBrain FTP | ~50 MB | MEDIUM |
| 9 | BigBrain Hippocampus ROI (bilateral, 400μm) | BigBrain FTP | ~4 MB | MEDIUM |
| 10 | Julich-Brain v2.9 in BigBrain space (122 regions) | siibra API | ~50 MB | HIGH |
| 11 | Cortical Layers in BigBrain space (6 layers) | siibra API | ~30 MB | HIGH |
| 12 | Isocortex Segmentation in BigBrain space | siibra API | ~20 MB | MEDIUM |
| 13 | BigBrain License | BigBrain FTP | <1 KB | HIGH |
| | **Total** | | **~600 MB** | |

---

## Directory Structure (local)

```
data/
├── allen_brain_data/                  # EXISTING
│   ├── ccfv3/                         # Mouse CCFv3 (existing)
│   ├── mouse_atlas/                   # Mouse atlas (existing)
│   ├── human_atlas/
│   │   ├── images/                    # EXISTING — 14,565 Nissl JPEGs
│   │   └── svgs/                      # NEW — 4,463 SVG annotations
│   ├── developing_human_atlas/        # NEW
│   │   ├── images/                    # 169 atlas plate JPEGs (downsample=4)
│   │   └── svgs/                      # 169 SVG annotations
│   ├── ontology/
│   │   ├── structure_graph_1.json     # EXISTING — mouse
│   │   ├── structure_graph_10.json    # NEW — human adult
│   │   └── structure_graph_16.json    # NEW — developing human
│   └── metadata/
│       ├── atlas_images_metadata.json         # EXISTING — mouse
│       ├── human_atlas_images_metadata.json   # EXISTING — human sections
│       └── developing_human_atlas_metadata.json  # NEW — 21 pcw atlas
├── bigbrain/                          # NEW
│   ├── classified_volume/
│   │   └── full_cls_200um_9classes.nii.gz     # 9-class tissue classification (16 MB)
│   ├── histological_volume/
│   │   └── full8_200um_optbal.nii.gz          # Intensity volume, 8-bit (74 MB)
│   ├── layer_segmentation/
│   │   ├── s0301/                     # 13 section directories
│   │   ├── s1066/                     # Each: .mnc + .png files
│   │   └── ...
│   ├── siibra/                        # Fetched via siibra Python API
│   │   ├── julich_brain_v29_bigbrain.nii.gz   # 122 cytoarchitectonic regions (2% coverage)
│   │   ├── cortical_layers_bigbrain.nii.gz    # 6 cortical layers (38.4% coverage)
│   │   └── isocortex_bigbrain.nii.gz          # Cortex vs subcortical (25.8% coverage)
│   ├── hippocampus/
│   │   ├── hippocampus_left_400um.nii.gz
│   │   └── hippocampus_right_400um.nii.gz
│   └── LICENSE.txt
```

### Databricks Target Paths

```python
WS_ROOT = "/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology"

# New human annotation data
HUMAN_SVGS_DIR     = f"{WS_ROOT}/human_atlas/svgs/"
HUMAN_ONTOLOGY     = f"{WS_ROOT}/ontology/structure_graph_10.json"
DEV_HUMAN_ONTOLOGY = f"{WS_ROOT}/ontology/structure_graph_16.json"
DEV_HUMAN_IMAGES   = f"{WS_ROOT}/developing_human_atlas/images/"
DEV_HUMAN_SVGS     = f"{WS_ROOT}/developing_human_atlas/svgs/"
DEV_HUMAN_METADATA = f"{WS_ROOT}/metadata/developing_human_atlas_metadata.json"

# BigBrain FTP data
BIGBRAIN_CLASSIFIED = f"{WS_ROOT}/bigbrain/classified_volume/full_cls_200um_9classes.nii.gz"
BIGBRAIN_HISTOLOGY  = f"{WS_ROOT}/bigbrain/histological_volume/full8_200um_optbal.nii.gz"
BIGBRAIN_LAYERS_DIR = f"{WS_ROOT}/bigbrain/layer_segmentation/"
BIGBRAIN_HIPPO_L    = f"{WS_ROOT}/bigbrain/hippocampus/hippocampus_left_400um.nii.gz"
BIGBRAIN_HIPPO_R    = f"{WS_ROOT}/bigbrain/hippocampus/hippocampus_right_400um.nii.gz"

# siibra-fetched volumes (BigBrain space)
JULICH_V29          = f"{WS_ROOT}/bigbrain/siibra/julich_brain_v29_bigbrain.nii.gz"
CORTICAL_LAYERS     = f"{WS_ROOT}/bigbrain/siibra/cortical_layers_bigbrain.nii.gz"
ISOCORTEX           = f"{WS_ROOT}/bigbrain/siibra/isocortex_bigbrain.nii.gz"
```

---

## Execution Steps

### Step 1: Extend download script for human SVGs

- **Status:** DONE — 4,463/4,463 SVGs (541.8 MB), 524+ unique structure IDs, 96% have annotations
- **File:** Created `scripts/download_human_annotations.py`
- **Details:**
  1. Load existing `human_atlas_images_metadata.json`
  2. Filter to `annotated=True` (4,463 images)
  3. Download SVGs via `svg_download/{id}`
  4. Save as `human_atlas/svgs/{donor}_{section_number}_{id}.svg`
  5. Rate limit: 0.1s delay between requests
  6. Idempotent: skip existing files
  7. Expected: ~100 MB, ~4,463 files, ~7.5 min at 10 req/sec

### Step 2: Download human structure ontologies

- **Status:** DONE — Graph 10: 1,839 structures (844 KB), Graph 16: 3,317 structures (1.8 MB)
- **Details:**
  - `GET structure_graph_download/10.json` → `ontology/structure_graph_10.json`
  - `GET structure_graph_download/16.json` → `ontology/structure_graph_16.json`
  - Both are single API calls, <5 MB each

### Step 3: Download developing human 21 pcw atlas

- **Status:** DONE — 169 images + 169 SVGs, 0 errors
- **Details:**
  1. Query AtlasImage for atlas ID 3: `model::AtlasImage,rma::criteria,atlas_data_set(atlases[id$eq3]),rma::options[num_rows$eqall]`
  2. Save metadata as `metadata/developing_human_atlas_metadata.json`
  3. Download atlas images: `atlas_image_download/{id}?downsample=4` → `developing_human_atlas/images/{section}_{id}.jpg`
  4. Download SVGs: `svg_download/{id}` → `developing_human_atlas/svgs/{section}_{id}.svg`
  5. 169 images + 169 SVGs, rate-limited

### Step 4: Download BigBrain data

- **Status:** DONE — classified 16 MB, histological 74 MB, hippocampus 3.5 MB, layer seg 390.9 MB (78 files), license
- **Details:**
  FTP base: `https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/`
  1. Classified volume: `3D_Classified_Volumes/Histological_Space/nii/full_cls_200um_9classes.nii.gz` (16 MB)
  2. Histological intensity: `3D_Volumes/Histological_Space/nii/full8_200um_optbal.nii.gz` (74 MB)
  3. Hippocampus: `3D_ROIs/Hippocampus/nii/hippocampus_{left,right}_400um.nii.gz` (4 MB total)
  4. Layer segmentation: `Layer_Segmentation/Manual_Annotations/` — 13 section dirs (~50 MB)
  5. License: `https://ftp.bigbrainproject.org/bigbrain-ftp/License.txt`

### Step 4b: Save siibra volumes as NIfTI

- **Status:** DONE — julich_brain_v29 (1.9 MB), cortical_layers (8.1 MB), isocortex (3.4 MB)
- **Details:**
  Fetch via `siibra` Python API and save with `nibabel`:
  1. Julich-Brain v2.9 → `data/bigbrain/siibra/julich_brain_v29_bigbrain.nii.gz`
  2. Cortical layers → `data/bigbrain/siibra/cortical_layers_bigbrain.nii.gz`
  3. Isocortex → `data/bigbrain/siibra/isocortex_bigbrain.nii.gz`
  ```python
  import siibra, nibabel as nib
  bigbrain = siibra.spaces['BigBrain microscopic template (histology)']
  for name, parc_name in [
      ('julich_brain_v29_bigbrain', 'Julich-Brain Cytoarchitectonic Atlas (v2.9)'),
      ('cortical_layers_bigbrain', 'Cortical layer segmentation of the BigBrain model'),
      ('isocortex_bigbrain', 'Isocortex Segmentation'),
  ]:
      mp = siibra.parcellations[parc_name].get_map(space=bigbrain, maptype='labelled')
      nib.save(mp.fetch(), f'data/bigbrain/siibra/{name}.nii.gz')
  ```

### Step 5: Verify all downloads

- **Status:** DONE — All NIfTI volumes verified (nibabel), all SVGs contain structure_id, all JSON parseable
- **Details:**
  - Validate SVG files contain `structure_id` attributes
  - Validate NIfTI files load with `nibabel`
  - Validate JSON files parse correctly
  - Print summary: file counts, sizes, structure counts

### Step 6: Extend OntologyMapper for human

- **Status:** NOT STARTED
- **File:** `src/histological_image_analysis/ontology.py`
- **Details:**
  - Add `build_human_mapping()` method (load graph 10)
  - Add `build_developing_human_mapping()` method (load graph 16)
  - Provide `remap_mask()` compatible with human structure IDs
  - Tests: extend `tests/test_ontology.py`

### Step 7: Create human SVG rasterizer

- **Status:** NOT STARTED
- **File:** Extend or adapt `src/histological_image_analysis/svg_rasterizer.py`
- **Details:**
  - Existing rasterizer handles mouse atlas SVGs (`<path structure_id="...">`)
  - Human SectionImage SVGs likely use same format — verify and adapt if needed
  - Human images are much larger (~27K×34K native vs ~8K mouse) — handle downsample alignment
  - Tests: extend `tests/test_svg_rasterizer.py`

### Step 8: Upload to Databricks

- **Status:** DONE — `make deploy-human-annotations` uploaded all data to workspace
- **Details:**
  - All files <500 MB → workspace upload:
    ```bash
    databricks workspace import-dir data/allen_brain_data/human_atlas/svgs/ \
      /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/human_atlas/svgs/ \
      --profile dev
    databricks workspace import-dir data/allen_brain_data/developing_human_atlas/ \
      /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/developing_human_atlas/ \
      --profile dev
    databricks workspace import-dir data/bigbrain/ \
      /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/bigbrain/ \
      --profile dev
    ```
  - Upload new ontology files:
    ```bash
    databricks workspace import \
      /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/ontology/structure_graph_10.json \
      --file data/allen_brain_data/ontology/structure_graph_10.json \
      --format AUTO --overwrite --profile dev
    ```

### Step 9: Create human BrainSegmentationDataset

- **Status:** NOT STARTED
- **Details:**
  - New dataset class (or mode of existing) that:
    1. Loads human SectionImage JPEG + corresponding SVG
    2. Rasterizes SVG to pixel mask (using human ontology)
    3. Returns `(image, mask)` pairs at 518×518
  - Unlike mouse (which uses 3D volume slicing), human data is 2D image + 2D SVG pairs
  - Augmentation: reuse existing pipeline (flip, rotation, jitter — no elastic deformation)
  - Train/val split: by donor (e.g., 4 donors train, 1 val, 1 test)

### Step 10: Add Makefile targets

- **Status:** DONE — `deploy-human-annotations` target added to Makefile
- **Details:**
  - Added `deploy-human-annotations` target for uploading human annotation data
  - Add notebook deployment target for human training notebook (pending — no human notebook yet)

---

## Execution Log

_Updated as steps are completed._

| Timestamp | Step | Status | Notes |
|-----------|------|--------|-------|
| 2026-03-16 08:05 | Step 0: Create directories | DONE | All dirs created under data/bigbrain/ and data/allen_brain_data/ |
| 2026-03-16 08:05 | Step 2: Human ontologies | DONE | Graph 10 (1,839 structures, 844 KB), Graph 16 (3,317 structures, 1.8 MB) |
| 2026-03-16 08:06 | Step 4: BigBrain FTP (main volumes) | DONE | classified_volume (16 MB), histological_volume (74 MB), hippocampus L+R (3.5 MB), LICENSE |
| 2026-03-16 08:12 | Step 4: BigBrain FTP (layer seg) | DONE | 13 sections, 78 files, 390.9 MB (includes raw + geo + classified layers) |
| 2026-03-16 08:12 | Step 4b: siibra volumes | DONE | julich_brain_v29 (1.9 MB), cortical_layers (8.1 MB), isocortex (3.4 MB) |
| 2026-03-16 10:31 | Step 1: Human SVGs | DONE | 4,463/4,463 downloaded (1 retry), 541.8 MB, 524+ unique structure IDs |
| 2026-03-16 08:52 | Step 3: Developing atlas | DONE | 169 images + 169 SVGs, 0 errors |
| 2026-03-16 10:32 | Step 5: Verification | DONE | All NIfTI volumes verified, SVGs contain structure_id, JSONs parseable |
| 2026-03-16 10:35 | Step 8: Databricks upload | DONE | `make deploy-human-annotations` — all data uploaded to workspace |

---

## Resume Instructions

### Quick Status Check

1. Check SVG download progress: `ls data/allen_brain_data/human_atlas/svgs/ | wc -l` — should be ~4,463
2. Check developing atlas: `ls data/allen_brain_data/developing_human_atlas/images/ | wc -l` — should be 169
3. Check BigBrain: `ls data/bigbrain/` — should have `classified_volume/`, `layer_segmentation/`, `hippocampus/`

### Key Files

| File | Purpose |
|------|---------|
| `docs/human_data_search_results.md` | Search findings (what data exists, where) |
| `docs/data_download_plan_human.md` | This file — download/upload plan |
| `docs/data_download_plan.md` | Previous download plan (mouse + human images) |
| `scripts/download_allen_data.py` | Existing download script (extend for human SVGs) |
| `src/histological_image_analysis/ontology.py` | Extend for human ontology (graph 10) |
| `src/histological_image_analysis/svg_rasterizer.py` | Extend for human SVGs |

### siibra — Critical Gate (PASSED)

**Julich-Brain v2.9 in BigBrain space:** Verified working. 122 labelled regions, 357×463×411 voxels, ~320μm resolution. Only 2% voxel coverage (cytoarchitectonic mapping is incomplete). Fetch takes ~4 minutes (merges 122 individual region volumes).

**Cortical Layer Segmentation in BigBrain space:** FETCHED and verified.
- Shape: (269, 463, 384), dtype uint32, ~340μm voxels
- 7 labels: background + 6 cortical layers (L1–L6), left/right hemispheres merged
- **38.4% voxel coverage** — much denser than Julich-Brain v2.9's 2%
- 23 region entries (6 layers × 2 hemispheres + parent nodes)

**Isocortex Segmentation in BigBrain space:** FETCHED and verified.
- Shape: (303, 385, 348), dtype uint32, 400μm voxels
- 3 labels: background (0), Isocortex (100), Non-isocortical structures (200)
- **25.8% voxel coverage**
- Binary mask distinguishing cortex from subcortical tissue

**v3.1 (772 regions, includes substantia nigra):** Only in MNI space. Would need MNI→BigBrain registration.

**BigBrain FTP — Verified Paths (2026-03-15):**

| File | Path | Size |
|------|------|------|
| 9-class tissue classification (200μm) | `3D_Classified_Volumes/Histological_Space/nii/full_cls_200um_9classes.nii.gz` | 16.0 MB |
| 3-class tissue classification (200μm) | `3D_Classified_Volumes/Histological_Space/nii/full_cls_200um.nii.gz` | 11.6 MB |
| Tissue mask (200μm) | `3D_Classified_Volumes/Histological_Space/nii/full_cls_200um_mask.nii.gz` | 3.0 MB |
| Histological volume 8-bit (200μm) | `3D_Volumes/Histological_Space/nii/full8_200um_optbal.nii.gz` | 74.0 MB |
| Histological volume 8-bit (300μm) | `3D_Volumes/Histological_Space/nii/full8_300um_optbal.nii.gz` | 22.6 MB |
| Histological volume 8-bit (100μm) | `3D_Volumes/Histological_Space/nii/full8_100um_optbal.nii.gz` | 574.3 MB |
| Layer seg. manual annotations | `Layer_Segmentation/Manual_Annotations/` | 13 section dirs |
| License | `License.txt` | <1 KB |

**Recommended approach:** Use v2.9 (122 regions) in BigBrain space as primary parcellation + cortical layers (38.4% coverage, 6 layers) + BigBrain 9-class tissue classification (dense, every voxel) for full coverage. No mouse data mixing.

### API Quick Reference

```
# Human SectionImage SVGs
GET https://api.brain-map.org/api/v2/svg_download/{section_image_id}

# 21 pcw Atlas Images
GET https://api.brain-map.org/api/v2/data/query.json?criteria=model::AtlasImage,rma::criteria,atlas_data_set(atlases[id$eq3]),rma::options[num_rows$eqall]
GET https://api.brain-map.org/api/v2/atlas_image_download/{atlas_image_id}?downsample=4
GET https://api.brain-map.org/api/v2/svg_download/{atlas_image_id}

# Structure Ontologies
GET https://api.brain-map.org/api/v2/structure_graph_download/10.json  # Human adult
GET https://api.brain-map.org/api/v2/structure_graph_download/16.json  # Developing human

# BigBrain FTP
https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/
```