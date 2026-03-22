# Allen Brain Data — Local Download & Workspace Upload Plan

## Context

This document tracks the execution of **steps 5–6** from `docs/progress.md`:
- **Step 5:** Download Allen Brain Institute data locally (~3 GB)
- **Step 6:** Upload to Databricks Workspace via `databricks workspace import-dir`

### Why This Is Needed

The Databricks corporate firewall blocks all Allen Brain Institute domains (`api.brain-map.org`, `download.alleninstitute.org`) with `ConnectionResetError(104)`. PyPI and HuggingFace are accessible. See `docs/databricks_connectivity.md` for full test results.

### How This Fits the Larger Task

```
Step 5  ← Download Allen data locally        ← THIS DOCUMENT
Step 6  ← Upload to Workspace                 ← THIS DOCUMENT
Step 7  ← Choose dataset + model combination
Step 8  ← Build training data pipeline (CCFv3 slicing + SVG rasterization)
Step 9  ← Build fine-tuning notebook
Step 10 ← Evaluate (per-structure IoU on held-out sections)
```
Steps 7–10 depend on the data being available on Databricks. The download script and Workspace paths established here will be referenced by all subsequent notebooks.

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
| 1 | CCFv3 annotation volume (25μm) | `download.alleninstitute.org` | ~4 MB | Direct HTTP: `annotation/ccf_2017/annotation_25.nrrd` |
| 2 | CCFv3 template volume (25μm) | `download.alleninstitute.org` | ~33 MB | Direct HTTP: `average_template/average_template_25.nrrd` |
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
- **Decision (2026-03-08):** Download ALL 14,566 images. User chose most complete dataset for cross-species training.

---

## Directory Structure (local)

```
data/
├── allen_brain_data/
│   ├── ccfv3/
│   │   ├── annotation_25.nrrd           # Item 1
│   │   ├── average_template_25.nrrd    # Item 2 (Allen naming convention)
│   │   ├── annotation_10.nrrd        # Item 3
│   │   └── ara_nissl_10.nrrd         # Item 4
│   ├── mouse_atlas/
│   │   ├── images/                   # Item 5 — {section_number}_{id}.jpg (509 files)
│   │   └── svgs/                     # Item 6 — {section_number}_{id}.svg (509 files — all sections have SVGs)
│   ├── human_atlas/
│   │   └── images/                   # Item 8 — {section_number}_{id}.jpg (Nissl sections, count TBD)
│   ├── ontology/
│   │   └── structure_graph_1.json    # Item 7
│   └── metadata/
│       ├── atlas_images_metadata.json       # Mouse atlas API response
│       └── human_atlas_images_metadata.json # Human atlas API response
```

### Databricks Target Paths (Workspace files, for notebooks)

```python
WS_ROOT = "/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology"

ANNOTATION_25 = f"{WS_ROOT}/ccfv3/annotation_25.nrrd"
TEMPLATE_25   = f"{WS_ROOT}/ccfv3/average_template_25.nrrd"
ANNOTATION_10 = f"{WS_ROOT}/ccfv3/annotation_10.nrrd"
NISSL_10      = "/dbfs/FileStore/allen_brain_data/ccfv3/ara_nissl_10.nrrd"  # DBFS — too large for Workspace (2.17 GB > 500 MB limit)

ATLAS_IMAGES_DIR    = f"{WS_ROOT}/mouse_atlas/images/"
ATLAS_SVGS_DIR      = f"{WS_ROOT}/mouse_atlas/svgs/"
HUMAN_IMAGES_DIR    = f"{WS_ROOT}/human_atlas/images/"
STRUCTURE_ONTOLOGY  = f"{WS_ROOT}/ontology/structure_graph_1.json"
ATLAS_METADATA      = f"{WS_ROOT}/metadata/atlas_images_metadata.json"
HUMAN_METADATA      = f"{WS_ROOT}/metadata/human_atlas_images_metadata.json"
```

> **Note:** Workspace files may have per-file size limits (~500 MB on some workspaces). The 10μm NRRD volumes are ~1.3 GB each. If an upload fails due to size limits, fall back to a Unity Catalog Volume for those files.

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
  - Uses `requests` for all downloads (direct HTTP for NRRD volumes, API for images/SVGs/ontology)
  - Does NOT use AllenSDK (numpy 2.0 incompatibility)
  - Saves metadata JSON for reproducibility
  - Idempotent: skips files that already exist with correct size
  - Progress reporting via stdout (12-factor: logs as event streams)
  - No global variables; uses dataclass/function-based design per CLAUDE.md
  - No widgets, no hardcoded secrets

### Steps 5B+5C: Download all CCFv3 volumes via direct HTTP
- **Status:** DONE
- **Details:**
  - **Switched from AllenSDK to direct HTTP** — AllenSDK has numpy 2.0 incompatibility (`VisibleDeprecationWarning` import error). Direct download avoids the issue entirely.
  - 25μm annotation: `https://download.alleninstitute.org/.../annotation/ccf_2017/annotation_25.nrrd` (~4 MB)
  - 25μm template: `https://download.alleninstitute.org/.../average_template/average_template_25.nrrd` (~33 MB)
  - 10μm Nissl: `https://download.alleninstitute.org/.../ara_nissl/ara_nissl_10.nrrd` (~1.3 GB)
  - 10μm annotation: `https://download.alleninstitute.org/.../annotation/ccf_2017/annotation_10.nrrd` (~1.3 GB)
  - Note: 25μm files are compressed NRRD, hence much smaller than the ~85 MB uncompressed estimate
  - Stream download with progress for large files
  - **Template file saved as `average_template_25.nrrd`** (matches Allen naming)

### Step 5D: Download mouse atlas images (509 sections at downsample=4)
- **Status:** DONE — 509/509, 0 errors
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
- **Status:** DONE — 509/509 (all sections had SVGs, not just 132)
- **Details:**
  - From the metadata JSON, filter to images where `annotations_count > 0` (or `sub_images` has SVG data)
  - Actually: all 509 images may have SVGs, but only 132 have structure annotations
  - Try all 509, save those that return valid SVG content
  - `GET https://api.brain-map.org/api/v2/svg_download/{id}`
  - Save as `mouse_atlas/svgs/{section_number}_{id}.svg`

### Step 5F: Download Human Brain Atlas Nissl sections
- **Status:** DONE — 14,565/14,566 (1 error). Total: ~18 GB human images on disk.
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
- **Status:** DONE — 880 KB
- **Details:**
  - `GET https://api.brain-map.org/api/v2/structure_graph_download/1.json`
  - Save as `ontology/structure_graph_1.json`
  - This contains the full hierarchical structure tree (~700+ structures)

### Step 5H: Verify all downloads
- **Status:** DONE
- **Details:**
  - Check all files exist and are non-empty
  - Validate NRRD files can be read with `pynrrd`
  - Validate JSON files parse correctly
  - Validate image files are non-zero and openable
  - Print summary table: file, size, status

### Step 6A: Upload to Databricks Workspace
- **Status:** DONE
- **Details:**
  ```bash
  databricks workspace import-dir data/allen_brain_data/ /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology --profile dev
  ```
  - Verify upload: `databricks workspace ls /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology --profile dev`
  - Browse in UI: https://grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com/browse/folders/3112028122207057?o=2648165776397546
  - If any file exceeds the workspace size limit, use `databricks fs cp <local-file> dbfs:/FileStore/allen_brain_data/<file> --profile dev` as a fallback for that specific file and note the alternate path.

### Step 6B: Verify Workspace access from cluster
- **Status:** NOT STARTED
- **Details:**
  - Run a quick check via Command Execution API (or notebook) that files are readable at `/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology/`
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

### CCFv3 25μm Direct Downloads (used instead of AllenSDK)

```
https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd
https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/average_template_25.nrrd
```

**NOTE:** AllenSDK `ReferenceSpaceCache` was originally planned but has a numpy 2.0 incompatibility (`cannot import name 'VisibleDeprecationWarning' from 'numpy'`). Direct HTTP downloads work fine and avoid the dependency issue entirely. AllenSDK is still declared in `pyproject.toml` in case it's needed later for structure tree queries (if numpy is downgraded or allensdk is updated).

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
| 2026-03-08 | 5A fix | DONE | Removed AllenSDK dependency from script — numpy 2.0 incompatibility. Now uses direct HTTP for all volumes. Template file is `average_template_25.nrrd`. |
| 2026-03-08 | 5B+5C | DONE | CCFv3 volumes: annotation_25 (4 MB), average_template_25 (33 MB), ara_nissl_10 (2,170 MB), annotation_10 (33 MB). Total: ~2.24 GB. Note: 10μm annotation is compressed NRRD, much smaller than 1.3 GB estimate. |
| 2026-03-08 | 5D | DONE | 509/509 mouse atlas images, 0 errors. |
| 2026-03-08 | 5E | DONE | 509/509 SVGs — ALL sections had SVGs (not just the expected 132). |
| 2026-03-08 | 5F metadata | DONE | Queried 1,067 datasets → found 14,566 section images. Saved metadata JSON. |
| 2026-03-08 | 5F images | DONE | 14,565/14,566 downloaded (1 error). Completed at 01:55. Total human images: ~18 GB on disk (avg ~1.2 MB/img, not ~200 KB as estimated). |
| 2026-03-09 | 5G | DONE | Structure ontology: 880 KB. |
| 2026-03-09 | 5H | DONE | All NRRD files verified: annotation_25 (528,320,456 uint32), template_25 (528,320,456 uint16), annotation_10 (1320,800,1140 uint32), nissl_10 (1320,800,1140 float32). All JSON parseable. 509 mouse images, 509 SVGs, 14,565 human images. Total: **20.87 GB**. |
| 2026-03-09 | STEP 5 | **COMPLETE** | All downloads finished. |
| 2026-03-09 | 6A metadata | DONE | Uploaded `metadata/` to Workspace (2 JSON files). |
| 2026-03-09 | 6A mouse | DONE | Uploaded `mouse_atlas/` to Workspace (509 images + 509 SVGs). |
| 2026-03-09 | 6A ontology | DONE | Uploaded `ontology/structure_graph_1.json` to Workspace. |
| 2026-03-09 | 6A ccfv3 | PARTIAL | annotation_25.nrrd (4 MB), annotation_10.nrrd (33 MB), average_template_25.nrrd (33 MB) → Workspace. **ara_nissl_10.nrrd (2.17 GB) → DBFS** (`dbfs:/FileStore/allen_brain_data/ccfv3/ara_nissl_10.nrrd`) — exceeds 500 MB workspace limit. DBFS upload in progress. |
| 2026-03-09 | 6A human | DONE | Uploaded 14,565 human images to Workspace via `import-dir`. Completed ~02:00. |
| 2026-03-09 | 6A nissl DBFS | DONE | Uploaded `ara_nissl_10.nrrd` (2.17 GB) to `dbfs:/FileStore/allen_brain_data/ccfv3/ara_nissl_10.nrrd` — too large for Workspace. |
| 2026-03-09 | STEP 6A | **COMPLETE** | All data uploaded to Databricks. |

---

## Resume Instructions (for next LLM session)

**READ THIS FIRST** if you are picking up this work.

### Quick Status Check

1. **Check if the download script finished:** Run `ls -la data/allen_brain_data/human_atlas/images/ | wc -l` — if result is ~14,566, the download completed. If fewer, re-run the script (it's idempotent and skips existing files).
2. **Re-run if needed:** `.venv/bin/python scripts/download_allen_data.py --output-dir data/allen_brain_data`
3. **Check the Execution Log table above** for the last completed step.

### What Was Done (2026-03-08 to 2026-03-09)

- Created `scripts/download_allen_data.py` — downloads all 8 data items from Allen Brain Institute
- Script uses direct HTTP (NOT AllenSDK) due to numpy 2.0 incompatibility
- Script uses `DownloadContext` dataclass, no global variables, per `CLAUDE.md` rules
- Added `pynrrd` to `pyproject.toml`, ran `uv lock`
- Created `.gitignore` with `data/` excluded
- **All downloads completed** — 20.87 GB total on disk:
  - 4 NRRD volumes (annotation_25, average_template_25, annotation_10, ara_nissl_10)
  - 509 mouse atlas images + 509 SVGs
  - 14,565 human NISSL images (1 download error out of 14,566)
  - Structure ontology JSON (880 KB)
  - All verified: NRRDs readable, JSONs parseable, correct shapes
- **All uploads to Databricks completed:**
  - Workspace: metadata, mouse_atlas (images+SVGs), ontology, human_atlas (14,565 images), ccfv3 (annotation_25, average_template_25, annotation_10)
  - DBFS fallback: `ara_nissl_10.nrrd` (2.17 GB) at `dbfs:/FileStore/allen_brain_data/ccfv3/` — exceeds 500 MB workspace limit

### What Still Needs Doing

1. **Step 6B: Verify files accessible from Databricks cluster** — run a quick notebook or CLI command to confirm reads work
2. **Update `docs/progress.md`** — mark steps 5 and 6 as done
3. **Proceed to step 7** — choose dataset + model combination (see `docs/progress.md` for model research results)

### Key Files

| File | Purpose |
|------|---------|
| `docs/data_download_plan.md` | This file — tracks download/upload progress |
| `docs/progress.md` | Master project tracker — data research, model research, next steps |
| `docs/databricks_connectivity.md` | Firewall test results and CLI commands |
| `scripts/download_allen_data.py` | Download script (idempotent, re-runnable) |
| `data/allen_brain_data/` | Downloaded data (~6 GB, gitignored) |
| `exploration/allen_brain_data_explorer.ipynb` | Allen data exploration notebook (runs locally) |
| `exploration/databricks_connectivity_check.ipynb` | Network connectivity test notebook |
| `.claude/CLAUDE.md` | Codebase rules — no globals, TDD, SOLID, uv, no widgets, 12-factor, don't commit |

### Known Issues

- **AllenSDK has numpy 2.0 incompatibility** — `cannot import name 'VisibleDeprecationWarning' from 'numpy'`. Workaround: use direct HTTP instead of `ReferenceSpaceCache`. AllenSDK is still in `pyproject.toml` in case it's needed later.
- **Workspace file size limit** — confirmed 500 MB cap. `ara_nissl_10.nrrd` (2.17 GB) uploaded to DBFS instead: `dbfs:/FileStore/allen_brain_data/ccfv3/ara_nissl_10.nrrd`. Notebooks must use `/dbfs/FileStore/...` path for this file only; all other files are on Workspace path.
- **Human API treatment name** — must be `'NISSL'` (UPPERCASE), not `'Nissl'`. The latter returns 0 rows.

### Databricks Config

- **CLI:** `databricks` v0.278.0, profiles in `~/.databrickscfg`: `[dev]`, `[stage]`, `[prod]`
- **Workspace:** `grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com`
- **Cluster:** `0306-215929-ai2l0t8w` (interactive-only, `jobs: false`)
- **For programmatic runs:** use Command Execution API (`/api/1.2/commands/execute`), NOT `databricks jobs submit`