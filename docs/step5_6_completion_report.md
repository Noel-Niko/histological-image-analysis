# Steps 5-6 Completion Report: Allen Brain Data Download & Upload

**Date:** 2026-03-08 to 2026-03-09
**Duration:** ~6 hours (download) + ~2 hours (upload)

---

## Objective

Download all Allen Brain Institute data blocked by the Databricks corporate firewall, then upload it to Databricks Workspace so subsequent notebooks can access it for model training.

This corresponds to steps 5 and 6 of the project plan in `docs/progress.md`.

---

## What Was Accomplished

### Step 5: Local Download (20.87 GB)

Built and ran `scripts/download_allen_data.py` to download 8 data items from Allen Brain Institute servers to `data/allen_brain_data/`.

#### Final Download Results

| Dataset | Files | Size on Disk | Verified |
|---------|-------|-------------|----------|
| CCFv3 annotation volume (25um) | 1 | 4.0 MB | NRRD read OK, shape (528, 320, 456) uint32 |
| CCFv3 template volume (25um) | 1 | 33.0 MB | NRRD read OK, shape (528, 320, 456) uint16 |
| CCFv3 Nissl volume (10um) | 1 | 2,169.9 MB | NRRD read OK, shape (1320, 800, 1140) float32 |
| CCFv3 annotation volume (10um) | 1 | 32.8 MB | NRRD read OK, shape (1320, 800, 1140) uint32 |
| Mouse atlas images (downsample=4) | 509 | ~30 MB | 509/509 OK |
| Mouse atlas SVG annotations | 509 | ~17 MB | 509/509 OK |
| Human NISSL images (downsample=4) | 14,565 | ~18 GB | 14,565/14,566 (1 error) |
| Structure ontology JSON | 1 | 880 KB | JSON parse OK |
| Metadata JSONs | 2 | 11 MB | JSON parse OK |
| **Total** | **15,589** | **20.87 GB** | |

### Step 6: Databricks Upload

All data uploaded to Databricks dev workspace.

| Dataset | Destination | Method |
|---------|-------------|--------|
| Mouse atlas (images + SVGs) | Workspace: `.../histology/mouse_atlas/` | `databricks workspace import-dir` |
| Human atlas (14,565 images) | Workspace: `.../histology/human_atlas/` | `databricks workspace import-dir` |
| CCFv3 annotation_25.nrrd (4 MB) | Workspace: `.../histology/ccfv3/` | `databricks workspace import-dir` |
| CCFv3 annotation_10.nrrd (33 MB) | Workspace: `.../histology/ccfv3/` | `databricks workspace import-dir` |
| CCFv3 average_template_25.nrrd (33 MB) | Workspace: `.../histology/ccfv3/` | `databricks workspace import-dir` (separate run) |
| CCFv3 ara_nissl_10.nrrd (2.17 GB) | **DBFS**: `dbfs:/FileStore/allen_brain_data/ccfv3/` | `databricks fs cp` (exceeded 500 MB workspace limit) |
| Metadata JSONs | Workspace: `.../histology/metadata/` | `databricks workspace import-dir` |
| Ontology JSON | Workspace: `.../histology/ontology/` | `databricks workspace import-dir` |

**Workspace root:** `/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology`

### Step 6B: Cluster Verification — NOT YET DONE

Files have been uploaded but not yet verified as readable from a running Databricks cluster. This is the one remaining sub-step.

---

## How It Was Done

### 1. Prerequisites

- Added `pynrrd>=1.1.0` to `pyproject.toml` (was missing). Ran `uv lock`.
- Created `.gitignore` with `data/` to keep 20.87 GB of binaries out of git.

### 2. Download Script Design

Created `scripts/download_allen_data.py` with these design choices:

- **No global variables** — all state in a `DownloadContext` dataclass (per `CLAUDE.md` rules)
- **Idempotent** — checks if each file exists before downloading; safe to re-run
- **Direct HTTP only** — AllenSDK was originally planned for CCFv3 25um volumes, but it has a numpy 2.0 incompatibility (`cannot import name 'VisibleDeprecationWarning'`). Replaced with direct HTTP downloads from `download.alleninstitute.org`.
- **Rate limited** — 0.1s delay between Allen API image requests
- **Progress logging to stdout** — follows 12-factor principle (logs as event streams)
- **Streaming for large files** — NRRD volumes downloaded in 8KB chunks with progress percentage

### 3. Human Brain Atlas API Discovery

The Human Brain Atlas query required investigation. Key findings:

- Treatment name is `'NISSL'` (UPPERCASE), NOT `'Nissl'`. Title case returns 0 rows.
- Must query in two steps: first `model::SectionDataSet` to get dataset IDs, then `model::SectionImage` per dataset.
- 6 donors, 1,067 datasets, 14,566 total section images.
- Images average ~1.2 MB each at downsample=4 (not ~200 KB as initially estimated), making the total ~18 GB rather than ~2.9 GB.

### 4. Upload Strategy

- Used `databricks workspace import-dir` for bulk uploads (handles many small files well).
- Discovered workspace has a **500 MB per-file limit**. The 2.17 GB `ara_nissl_10.nrrd` failed on workspace upload.
- Used `databricks fs cp` to upload that single file to DBFS as fallback.
- Workspace `import` (single file) has an even stricter **10 MB limit** — only `import-dir` supports files up to 500 MB.
- Uploaded directories in parallel to maximize throughput.

---

## Problems Encountered and Solutions

| Problem | Solution |
|---------|----------|
| AllenSDK `VisibleDeprecationWarning` import error with numpy 2.0 | Replaced AllenSDK usage with direct HTTP downloads to `download.alleninstitute.org` |
| Human Brain Atlas `treatments[name$eq'Nissl']` returns 0 rows | Treatment name is `'NISSL'` (UPPERCASE). Discovered by querying datasets without filter and inspecting `treatments` field. |
| `databricks workspace import-dir` fails at 500 MB file | Used `databricks fs cp` to upload the 2.17 GB file to DBFS instead |
| `databricks workspace import` single-file limit is 10 MB | Used `import-dir` (pointing at a temp dir with one file) as workaround for 33 MB files |
| `databricks fs cp` fails with "no such directory" | Must run `databricks fs mkdirs` first to create the DBFS directory |
| Human image size estimate was 200 KB/img, actual was 1.2 MB/img | Total grew from ~6 GB estimate to 20.87 GB actual. No action needed — disk space sufficient. |
| 10um annotation NRRD was estimated at 1.3 GB, actual was 33 MB | Compressed NRRD format. No issue — just a documentation correction. |

---

## Files Created or Modified

| File | Action | Purpose |
|------|--------|---------|
| `scripts/download_allen_data.py` | Created | Download script for all Allen data |
| `.gitignore` | Created | Exclude `data/`, `.venv/`, IDE files, `.env` |
| `pyproject.toml` | Modified | Added `pynrrd>=1.1.0` |
| `uv.lock` | Regenerated | After pyproject.toml change |
| `data/allen_brain_data/` | Created | 20.87 GB downloaded data (gitignored) |
| `docs/data_download_plan.md` | Modified | Updated with execution log and completion status |
| `docs/progress.md` | Modified | Marked steps 5 and 6 as done |

---

## Databricks Path Reference (for notebooks)

```python
WS_ROOT = "/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology"

# CCFv3 3D volumes
ANNOTATION_25 = f"{WS_ROOT}/ccfv3/annotation_25.nrrd"         # (528, 320, 456) uint32
TEMPLATE_25   = f"{WS_ROOT}/ccfv3/average_template_25.nrrd"   # (528, 320, 456) uint16
ANNOTATION_10 = f"{WS_ROOT}/ccfv3/annotation_10.nrrd"         # (1320, 800, 1140) uint32
NISSL_10      = "/dbfs/FileStore/allen_brain_data/ccfv3/ara_nissl_10.nrrd"  # DBFS fallback (2.17 GB)

# 2D atlas images
ATLAS_IMAGES_DIR  = f"{WS_ROOT}/mouse_atlas/images/"    # 509 JPEGs
ATLAS_SVGS_DIR    = f"{WS_ROOT}/mouse_atlas/svgs/"      # 509 SVGs
HUMAN_IMAGES_DIR  = f"{WS_ROOT}/human_atlas/images/"    # 14,565 JPEGs

# Reference data
STRUCTURE_ONTOLOGY = f"{WS_ROOT}/ontology/structure_graph_1.json"
ATLAS_METADATA     = f"{WS_ROOT}/metadata/atlas_images_metadata.json"
HUMAN_METADATA     = f"{WS_ROOT}/metadata/human_atlas_images_metadata.json"
```

---

## What Comes Next

Per `docs/progress.md`, the remaining steps are:

1. **Step 6B** — Verify files are readable from Databricks cluster
2. **Step 7** — Choose dataset + model combination (already done in a parallel session)
3. **Step 8** — Build training data pipeline (CCFv3 slicing + SVG rasterization)
4. **Step 9** — Build fine-tuning notebook
5. **Step 10** — Evaluate on held-out Mouse Atlas sections
