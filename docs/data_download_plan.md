# Allen Brain Data — Local Download & DBFS Upload Plan

## Context

This document tracks the execution of **steps 5–6** from `docs/progress.md`:
- **Step 5:** Download Allen Brain Institute data locally (~3 GB)
- **Step 6:** Upload to DBFS via `databricks fs cp`

### Why This Is Needed

The Databricks corporate firewall blocks all Allen Brain Institute domains (`api.brain-map.org`, `download.alleninstitute.org`) with `ConnectionResetError(104)`. PyPI and HuggingFace are accessible. See `docs/databricks_connectivity.md` for full test results.

### How This Fits the Larger Task

```
Step 5  ← Download Allen data locally        ← THIS DOCUMENT
Step 6  ← Upload to DBFS                     ← THIS DOCUMENT
Step 7  ← Choose dataset + model combination
Step 8  ← Build training data pipeline (CCFv3 slicing + SVG rasterization)
Step 9  ← Build fine-tuning notebook
Step 10 ← Evaluate (per-structure IoU on held-out sections)
```

Steps 7–10 depend on the data being available on DBFS. The download script and DBFS paths established here will be referenced by all subsequent notebooks.

---

## Local Environment

- **Python venv:** `.venv/` with `allensdk`, `requests`, `pynrrd` (run `pip show allensdk` to confirm version — `__version__` reports `0.16.3` but this may be a metadata issue; the SDK functions used here work regardless)
- **Databricks CLI:** v0.278.0 (`databricks --profile dev`)
- **Package manager:** `uv`
- **Local machine:** macOS (Darwin 24.6.0), no firewall restrictions on Allen domains (verified: API returns HTTP 200)
- **NOTE:** `pynrrd` was missing from `pyproject.toml` — added in Step 5-pre-A. `allensdk` and `requests` were already declared.

---

## Data Inventory

| # | Dataset | Source | Estimated Size | Download Method |
|---|---------|--------|----------------|-----------------|
| 1 | CCFv3 annotation volume (25μm) | `download.alleninstitute.org` | ~85 MB | AllenSDK `ReferenceSpaceCache` |
| 2 | CCFv3 template volume (25μm) | `download.alleninstitute.org` | ~85 MB | AllenSDK `ReferenceSpaceCache` |
| 3 | CCFv3 annotation volume (10μm) | `download.alleninstitute.org` | ~1.3 GB | Direct HTTP download |
| 4 | CCFv3 Nissl volume (10μm) | `download.alleninstitute.org` | ~1.3 GB | Direct HTTP download |
| 5 | Mouse atlas images (509 sections) | `api.brain-map.org` | ~250 MB | API loop: `atlas_image_download/{id}?downsample=4` |
| 6 | Mouse atlas SVG annotations (132 sections) | `api.brain-map.org` | ~10 MB | API loop: `svg_download/{id}` |
| 7 | Structure ontology JSON | `api.brain-map.org` | ~300 KB | Single API call: `structure_graph_download/1.json` |
| 8 | Human Brain Atlas Nissl sections (14,566 images, 6 donors, 1,067 datasets) | `api.brain-map.org` | ~2.9 GB (est. ~200 KB/img at ds=4) | API: query Product ID=2 with `treatments[name$eq'NISSL']` (UPPERCASE), download via `section_image_download/{id}?downsample=4` |
| | **Total** | | **~6 GB+** | |

**IMPORTANT — Human API discovery (2026-03-08):**
- Treatment name is `'NISSL'` (UPPERCASE), NOT `'Nissl'` (title case). Queries with `'Nissl'` return 0 rows.
- Must query via `model::SectionDataSet` first to get dataset IDs, then `model::SectionImage` with `data_set_id` to get images.
- 6 donors: `H0351.1009`, `H0351.1012`, `H0351.1015`, `H0351.1016`, `H0351.2001`, `H0351.2002`
- 1,067 NISSL datasets, 14,566 total section images
- Image dimensions: ~27K×34K px native, ~1.7K×2.1K at downsample=4
- At ~200 KB per image at ds=4, total estimate: ~2.9 GB for human sections alone
- **Decision needed:** Download all 14,566 or subsample (e.g., 1 donor, or coronal only)?

---

## Directory Structure (local)

```
data/
├── allen_brain_data/
│   ├── ccfv3/
│   │   ├── annotation_25.nrrd        # Item 1
│   │   ├── template_25.nrrd          # Item 2
│   │   ├── annotation_10.nrrd        # Item 3
│   │   └── ara_nissl_10.nrrd         # Item 4
│   ├── mouse_atlas/
│   │   ├── images/                   # Item 5 — {section_number}_{id}.jpg (509 files)
│   │   └── svgs/                     # Item 6 — {section_number}_{id}.svg (132 files)
│   ├── human_atlas/
│   │   └── images/                   # Item 8 — {section_number}_{id}.jpg (Nissl sections, count TBD)
│   ├── ontology/
│   │   └── structure_graph_1.json    # Item 7
│   └── metadata/
│       ├── atlas_images_metadata.json       # Mouse atlas API response
│       └── human_atlas_images_metadata.json # Human atlas API response
```

### DBFS Target Paths (for Databricks notebooks)

```python
DBFS_ROOT = "/dbfs/FileStore/allen_brain_data"

ANNOTATION_25 = f"{DBFS_ROOT}/ccfv3/annotation_25.nrrd"
TEMPLATE_25   = f"{DBFS_ROOT}/ccfv3/template_25.nrrd"
ANNOTATION_10 = f"{DBFS_ROOT}/ccfv3/annotation_10.nrrd"
NISSL_10      = f"{DBFS_ROOT}/ccfv3/ara_nissl_10.nrrd"

ATLAS_IMAGES_DIR    = f"{DBFS_ROOT}/mouse_atlas/images/"
ATLAS_SVGS_DIR      = f"{DBFS_ROOT}/mouse_atlas/svgs/"
HUMAN_IMAGES_DIR    = f"{DBFS_ROOT}/human_atlas/images/"
STRUCTURE_ONTOLOGY  = f"{DBFS_ROOT}/ontology/structure_graph_1.json"
ATLAS_METADATA      = f"{DBFS_ROOT}/metadata/atlas_images_metadata.json"
HUMAN_METADATA      = f"{DBFS_ROOT}/metadata/human_atlas_images_metadata.json"
```

---

## Execution Steps

### Step 5-pre-A: Add missing dependencies to pyproject.toml
- **Status:** DONE
- **Details:**
  - Add `allensdk`, `pynrrd`, and `requests` to `pyproject.toml` `[project].dependencies`
  - Per CLAUDE.md / 12-factor principle #2: all dependencies must be explicitly declared
  - Run `uv lock` after updating to regenerate lockfile

### Step 5-pre-B: Add `data/` to .gitignore
- **Status:** DONE
- **Details:**
  - Create or update `.gitignore` to exclude `data/` directory
  - The `data/allen_brain_data/` directory will hold ~3 GB+ of binary downloads — must not be committed

### Step 5A: Create download script
- **Status:** DONE
- **File:** `scripts/download_allen_data.py`
- **Details:**
  - Single Python script that downloads all 8 data items (7 mouse + 1 human)
  - Uses `allensdk` for CCFv3 25μm volumes
  - Uses `requests` for direct HTTP downloads (10μm volumes, images, SVGs, ontology, human sections)
  - Saves metadata JSON for reproducibility
  - Idempotent: skips files that already exist with correct size
  - Progress reporting via stdout (12-factor: logs as event streams)
  - No global variables; uses dataclass/function-based design per CLAUDE.md
  - No widgets, no hardcoded secrets

### Step 5B: Download CCFv3 volumes (25μm) via AllenSDK
- **Status:** NOT STARTED
- **Details:**
  - `ReferenceSpaceCache(resolution=25, reference_space_key="annotation/ccf_2017")`
  - `get_annotation_volume()` → saves `annotation_25.nrrd`
  - `get_template_volume()` → saves `template_25.nrrd`
  - AllenSDK downloads to its own cache dir, script copies to our structure
  - Expected: (528, 320, 456) shape for both volumes

### Step 5C: Download CCFv3 volumes (10μm) via direct HTTP
- **Status:** NOT STARTED
- **Details:**
  - `https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_10.nrrd`
  - `https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd`
  - These are ~1.3 GB each — stream download with progress
  - Verify file size after download

### Step 5D: Download mouse atlas images (509 sections at downsample=4)
- **Status:** NOT STARTED
- **Details:**
  - First: query API for atlas image list
    ```
    GET https://api.brain-map.org/api/v2/data/query.json?criteria=model::AtlasImage,rma::criteria,atlas_data_set(atlases[id$eq1]),rma::options[num_rows$eqall][order$eq'section_number']
    ```
  - Save full API response as `metadata/atlas_images_metadata.json`
  - Loop through all image IDs: `GET https://api.brain-map.org/api/v2/atlas_image_download/{id}?downsample=4`
  - Save as `mouse_atlas/images/{section_number}_{id}.jpg`
  - Rate limit: 0.1s delay between requests to be polite to Allen API

### Step 5E: Download SVG annotations (132 annotated sections)
- **Status:** NOT STARTED
- **Details:**
  - From the metadata JSON, filter to images where `annotations_count > 0` (or `sub_images` has SVG data)
  - Actually: all 509 images may have SVGs, but only 132 have structure annotations
  - Try all 509, save those that return valid SVG content
  - `GET https://api.brain-map.org/api/v2/svg_download/{id}`
  - Save as `mouse_atlas/svgs/{section_number}_{id}.svg`

### Step 5F: Download Human Brain Atlas Nissl sections
- **Status:** NOT STARTED
- **Details:**
  - **Step 1:** Query datasets: `model::SectionDataSet,rma::criteria,products[id$eq2],[failed$eqfalse],treatments[name$eq'NISSL'],rma::include,specimen(donor),plane_of_section`
  - **Step 2:** For each dataset, query images: `model::SectionImage,rma::criteria,[data_set_id$eq{ds_id}]`
  - Save combined metadata as `metadata/human_atlas_images_metadata.json`
  - Loop through section image IDs: `GET https://api.brain-map.org/api/v2/section_image_download/{id}?downsample=4`
  - Save as `human_atlas/images/{donor}_{section_number}_{id}.jpg`
  - Rate limit: 0.1s delay between requests
  - **Scope:** 14,566 total images across 6 donors, 1,067 datasets. ~2.9 GB at downsample=4.
  - **CRITICAL:** Treatment name is `'NISSL'` (UPPERCASE). `'Nissl'` returns 0 rows.
  - **Decision (2026-03-08):** Download ALL 14,566 images. Most complete dataset for cross-species training.

### Step 5G: Download structure ontology
- **Status:** NOT STARTED
- **Details:**
  - `GET https://api.brain-map.org/api/v2/structure_graph_download/1.json`
  - Save as `ontology/structure_graph_1.json`
  - This contains the full hierarchical structure tree (~700+ structures)

### Step 5H: Verify all downloads
- **Status:** NOT STARTED
- **Details:**
  - Check all files exist and are non-empty
  - Validate NRRD files can be read with `pynrrd`
  - Validate JSON files parse correctly
  - Validate image files are non-zero and openable
  - Print summary table: file, size, status

### Step 6A: Upload to DBFS
- **Status:** NOT STARTED
- **Details:**
  ```bash
  databricks fs cp data/allen_brain_data/ dbfs:/FileStore/allen_brain_data/ --recursive --profile dev
  ```
  - Verify upload: `databricks fs ls dbfs:/FileStore/allen_brain_data/ --profile dev`

### Step 6B: Verify DBFS access from cluster
- **Status:** NOT STARTED
- **Details:**
  - Run a quick check via Command Execution API (or notebook) that files are readable
  - Verify NRRD files load, images display, JSON parses

---

## API Reference (for resuming context)

### Allen Brain Atlas REST API

**Base URL:** `https://api.brain-map.org/api/v2/`

**List atlas images (Mouse, P56, Coronal — Atlas ID 1):**
```
GET /data/query.json?criteria=model::AtlasImage,rma::criteria,atlas_data_set(atlases[id$eq1]),rma::options[num_rows$eqall][order$eq'section_number']
```
Returns JSON with `msg` array of objects: `{id, section_number, sub_image: {section_image_id}, ...}`

**IMPORTANT:** Must use `atlas_data_set(atlases[id$eq1])` join, NOT `atlas_data_set_id` directly — the latter returns 0 results.

**Download atlas image:**
```
GET /atlas_image_download/{id}?downsample=4
```
Returns JPEG binary. `downsample=0` is full-res (~19K×37K px), `downsample=4` is 16× smaller.

**Download SVG annotation:**
```
GET /svg_download/{id}
```
Returns SVG with `<path structure_id="N">` polygons. 144+ structures per mid-brain section.

**Download structure ontology:**
```
GET /structure_graph_download/1.json
```
Returns JSON with full hierarchical structure tree.

**List Human Brain Atlas NISSL datasets (Product ID=2):**
```
GET /data/query.json?criteria=model::SectionDataSet,rma::criteria,products[id$eq2],[failed$eqfalse],treatments[name$eq'NISSL'],rma::include,specimen(donor),plane_of_section,rma::options[num_rows$eqall]
```
Returns JSON with `msg` array of dataset objects. **CRITICAL:** Treatment name is `'NISSL'` (UPPERCASE), NOT `'Nissl'`.

**List section images for a dataset:**
```
GET /data/query.json?criteria=model::SectionImage,rma::criteria,[data_set_id$eq{DATASET_ID}],rma::options[num_rows$eqall]
```
Returns JSON with `msg` array of section image objects (id, section_number, width, height, etc.).

**Download human section image:**
```
GET /section_image_download/{id}?downsample=4
```
Returns JPEG binary. Same downsample parameter as mouse atlas images.

### CCFv3 Direct Downloads

```
https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_10.nrrd
https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd
```

### AllenSDK (for 25μm volumes)

```python
from allensdk.core.reference_space_cache import ReferenceSpaceCache

rsc = ReferenceSpaceCache(
    resolution=25,
    reference_space_key="annotation/ccf_2017",
    manifest="path/to/manifest.json"
)
annotation, annotation_meta = rsc.get_annotation_volume()  # shape (528, 320, 456)
template, template_meta = rsc.get_template_volume()          # shape (528, 320, 456)
structure_tree = rsc.get_structure_tree()
# NOTE: filter None from get_structures_by_id() results
```

---

## Execution Log

_Updated as steps are completed._

| Timestamp | Step | Status | Notes |
|-----------|------|--------|-------|
| 2026-03-08 | 5-pre-A | DONE | Added `pynrrd>=1.1.0` to `pyproject.toml`. `allensdk` and `requests` were already present. Ran `uv lock` — resolved 213 packages. |
| 2026-03-08 | 5-pre-B | DONE | Created `.gitignore` with `data/`, `.venv/`, `python/`, `__pycache__/`, `.env`, IDE files. |
| 2026-03-08 | 5F research | IN PROGRESS | Discovered Human API treatment name is `'NISSL'` (UPPERCASE), not `'Nissl'`. Found 14,566 images across 6 donors, 1,067 datasets. ~2.9 GB at ds=4. **Decision needed: download all or subsample?** |
| 2026-03-08 | 5F decision | DONE | User chose: download ALL 14,566 human images. Total estimate now ~6 GB. |
| 2026-03-08 | 5A | DONE | Created `scripts/download_allen_data.py`. Handles all 8 data items. Idempotent (skips existing files). Uses `DownloadContext` dataclass, no globals. |
| 2026-03-08 | 5B-5H | RUNNING | Executing download script... |

---

## Resume Instructions

If context is lost, read this file and `docs/progress.md` to understand the current state. Key points:

1. Check the **Execution Log** table above for what's been completed
2. The download script is at `scripts/download_allen_data.py`
3. Downloaded data goes to `data/allen_brain_data/` (see directory structure above)
4. DBFS upload target is `dbfs:/FileStore/allen_brain_data/`
5. After steps 5–6 are complete, update `docs/progress.md` steps 5 and 6 to "Done"
6. Then proceed to step 7: choose dataset + model combination