# Databricks Connectivity Testing — Results & Plan

## Summary

All Allen Brain Institute endpoints are blocked by the Databricks corporate firewall. PyPI and HuggingFace are accessible. The workaround is to download Allen data locally and upload to the Databricks Workspace.

---

## Test Environment

- **Workspace:** dev (`grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com`)
- **Cluster:** `0306-215929-ai2l0t8w` — "Noel Nosse's insight-search-all-singlegpu-singlenode-singleuser Cluster"
- **Instance:** g6e.16xlarge (L40S 48GB VRAM, 512GB RAM)
- **Runtime:** Databricks 17.3 LTS (Spark 4.0.0, Scala 2.13), ML Runtime
- **Security mode:** SINGLE_USER
- **Workload type:** `notebooks: true, jobs: false` (interactive only, no job submissions)
- **Network:** AWS VPC with corporate firewall (IAM profile: `gtg-mlops-dev-use1-dbx-shared-iprofile-iamrole`)

## Test Method

1. Created `exploration/databricks_connectivity_check.ipynb` — 10 HTTP checks against all data sources
2. First ran notebook interactively in Databricks UI — confirmed blocks
3. Re-ran via Databricks CLI (`databricks` v0.278.0) using the Command Execution API (`/api/1.2/commands/execute`) to verify results are cluster-level, not UI-level

### CLI Execution Steps Used

```bash
# Import notebook
databricks workspace import /Users/noel.nosse@grainger.com/connectivity_check \
  --file exploration/databricks_connectivity_check.ipynb \
  --format JUPYTER --language PYTHON --overwrite --profile dev

# Start cluster (interactive cluster can't use `jobs submit`)
databricks clusters start 0306-215929-ai2l0t8w --profile dev

# Create execution context
curl -X POST "${HOST}/api/1.2/contexts/create" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"clusterId": "...", "language": "python"}'

# Execute code
curl -X POST "${HOST}/api/1.2/commands/execute" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"clusterId": "...", "contextId": "...", "language": "python", "command": "..."}'

# Poll for results
curl "${HOST}/api/1.2/commands/status?clusterId=...&contextId=...&commandId=..."
```

**Note:** `databricks jobs submit` fails on this cluster because `workload_type.clients.jobs = false`. Must use the Command Execution API for programmatic runs on interactive clusters.

---

## Results

| # | Check | Domain | Status | HTTP | Latency | Error |
|---|-------|--------|--------|------|---------|-------|
| 1 | Allen API Base | `api.brain-map.org` | **BLOCKED** | -- | -- | `ConnectionResetError(104, 'Connection reset by peer')` |
| 2 | Nissl Dataset Metadata | `api.brain-map.org` | **BLOCKED** | -- | -- | `ConnectionResetError(104)` |
| 3 | Section Image List | `api.brain-map.org` | **BLOCKED** | -- | -- | `ConnectionResetError(104)` |
| 4 | Image Download | `api.brain-map.org` | SKIPPED | -- | -- | Depends on #3 |
| 5 | SVG Annotation | `api.brain-map.org` | SKIPPED | -- | -- | Depends on #3 |
| 6 | Structure Ontology | `api.brain-map.org` | **BLOCKED** | -- | -- | `ConnectionResetError(104)` |
| 7 | CCFv3 Download Server | `download.alleninstitute.org` | **BLOCKED** | -- | -- | `ConnectionResetError(104)` |
| 8 | Human Brain Atlas API | `api.brain-map.org` | **BLOCKED** | -- | -- | `ConnectionResetError(104)` |
| 9 | PyPI (allensdk) | `pypi.org` | **PASS** | 200 | 24.1 ms | -- |
| 10 | HuggingFace Hub | `huggingface.co` | **PASS** | 200 | 45.9 ms | -- |

### Blocked Domains

- `api.brain-map.org` — All Allen Brain Atlas REST API calls (queries, image downloads, SVG annotations, ontology)
- `download.alleninstitute.org` — CCFv3 NRRD volume downloads

### Accessible Domains

- `pypi.org` — `pip install allensdk` works, but the SDK's download functions fail (they call the blocked Allen API/download servers internally)
- `huggingface.co` — Model downloads for DINOv2, SAM 2, Virchow2, etc. all work

### Root Cause

Corporate firewall actively resets TCP connections (`errno 104`) to Allen Institute domains. This is a network-level block, not DNS or timeout — connections are initiated then immediately torn down. Consistent across both interactive UI execution and CLI API execution on the same cluster. Likely applies to all clusters in this VPC.

---

## Implications for Project

### What WORKS on Databricks
- `pip install allensdk` (and any PyPI package)
- `pip install` any HuggingFace dependency
- Downloading models from HuggingFace Hub (DINOv2, SAM2, etc.)
- All training/fine-tuning compute
- Reading data already on Workspace files or Unity Catalog

### What DOES NOT work on Databricks
- Any HTTP call to `api.brain-map.org` (API queries, image downloads, SVG downloads)
- Any HTTP call to `download.alleninstitute.org` (NRRD volumes)
- AllenSDK `ReferenceSpaceCache` downloads (uses `download.alleninstitute.org` internally)
- Any interactive exploration of Allen data

---

## Chosen Approach: Local Download → Workspace Upload

Download all Allen data on local machine (no firewall), then upload to Databricks Workspace for notebook consumption.

### Data to Download Locally

| Dataset | Source | Size Estimate | Local Command |
|---------|--------|---------------|---------------|
| CCFv3 annotation volume (25μm) | AllenSDK | ~85 MB | `ReferenceSpaceCache(resolution=25).get_annotation_volume()` |
| CCFv3 template volume (25μm) | AllenSDK | ~85 MB | `ReferenceSpaceCache(resolution=25).get_template_volume()` |
| CCFv3 Nissl volume (10μm) | Direct download | ~1.3 GB | `wget https://download.alleninstitute.org/.../ara_nissl_10.nrrd` |
| CCFv3 annotation volume (10μm) | Direct download | ~1.3 GB | `wget https://download.alleninstitute.org/.../annotation_10.nrrd` |
| Mouse Atlas images (509 sections, downsample=4) | Allen API | ~250 MB | Script: loop `atlas_image_download/{id}?downsample=4` |
| Mouse Atlas SVG annotations (132 annotated sections) | Allen API | ~10 MB | Script: loop `svg_download/{id}` |
| Structure ontology (graph ID 1) | Allen API | ~300 KB | `structure_graph_download/1.json` |
| **Total** | | **~3 GB** | |

### Workspace Upload

```bash
# Upload to Workspace via CLI
databricks workspace import-dir data/allen_brain_data/ /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology --profile dev
```

### Workspace Paths (for Databricks notebooks)

```python
WS_ROOT = "/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology"

# CCFv3 volumes
ANNOTATION_25 = f"{WS_ROOT}/ccfv3/annotation_25.nrrd"
TEMPLATE_25 = f"{WS_ROOT}/ccfv3/template_25.nrrd"
NISSL_10 = f"{WS_ROOT}/ccfv3/ara_nissl_10.nrrd"
ANNOTATION_10 = f"{WS_ROOT}/ccfv3/annotation_10.nrrd"

# Mouse atlas images
ATLAS_IMAGES_DIR = f"{WS_ROOT}/mouse_atlas/images/"       # {id}.jpg
ATLAS_SVGS_DIR = f"{WS_ROOT}/mouse_atlas/svgs/"           # {id}.svg
STRUCTURE_ONTOLOGY = f"{WS_ROOT}/ontology/structure_graph_1.json"
```

---

## Alternative Options Considered

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Local download → Workspace upload** | One-time effort, then all data on Databricks | Manual step, ~6 GB transfer | **CHOSEN** |
| **Google Colab for exploration** | No firewall, quick iteration | Separate platform, session timeouts, data transfer still needed | Viable backup |
| **Firewall whitelist request** | Permanent fix | Depends on org process, may not be approved for personal project | Worth trying in parallel |

---

## Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `exploration/databricks_connectivity_check.ipynb` | Verify network access to Allen data sources | **Complete — 6 blocked, 2 pass** |
| `exploration/allen_brain_data_explorer.ipynb` | Browse Allen data interactively | Complete (runs locally, blocked on Databricks) |

---

## Compact Context (for /compact)

**Connectivity:** All Allen Brain Institute domains (`api.brain-map.org`, `download.alleninstitute.org`) are BLOCKED by Databricks corporate firewall (`ConnectionResetError 104`). PyPI and HuggingFace PASS. Tested on dev workspace cluster `0306-215929-ai2l0t8w` via both UI and CLI Command Execution API. The cluster is interactive-only (`jobs: false`) — must use `/api/1.2/commands/execute` for programmatic runs, not `jobs submit`.

**Plan:** Download ~6 GB of Allen data locally (CCFv3 volumes + mouse/human atlas images/SVGs + ontology), upload to `/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology` via `databricks workspace import-dir --profile dev`. Notebooks read directly from the Workspace path. Model downloads from HuggingFace work directly on cluster.

**CLI config:** `~/.databrickscfg` has profiles `[dev]`, `[stage]`, `[prod]`. Using `--profile dev`. CLI version 0.278.0.
