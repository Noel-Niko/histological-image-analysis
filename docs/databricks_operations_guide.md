# Databricks Operations Guide

Lessons learned from working with Databricks in this project. This guide covers CLI usage, file storage, networking, and workarounds discovered during the histological image analysis project.

---

## CLI Setup

### Configuration

Profiles are stored in `~/.databrickscfg`:

```ini
[dev]
host = https://<workspace-url>.cloud.databricks.com
token = dapi...

[stage]
host = https://...
token = dapi...

[prod]
host = https://...
token = dapi...
```

All CLI commands use `--profile <name>` to target a specific workspace.

### CLI Version

This project used Databricks CLI **v0.278.0**. Command syntax varies significantly between CLI versions. Always check `databricks <command> --help` for available subcommands.

Key differences from older documentation:
- `databricks runs submit` does not exist — it's `databricks jobs submit`
- `databricks workspace ls` does not exist — it's `databricks workspace list`
- `databricks fs` subcommands use `cp`, `ls`, `mkdirs` (not `list`)

---

## File Storage Options

Databricks has three main storage locations for files. Choose based on file size and access pattern.

### 1. Workspace Files

**Path pattern:** `/Workspace/Users/<email>/<path>`

**Best for:** Files under 500 MB, notebooks, small datasets, metadata JSONs, images.

**Upload:**
```bash
# Bulk upload (up to 500 MB per file)
databricks workspace import-dir ./local_dir/ /Workspace/Users/user@company.com/target/ --profile dev

# Single file (up to 10 MB only!)
databricks workspace import /Workspace/.../file.json --file ./local/file.json --format AUTO --overwrite --profile dev
```

**Access from notebooks:**
```python
# Direct path — just use the /Workspace path
import json
with open("/Workspace/Users/user@company.com/target/file.json") as f:
    data = json.load(f)
```

**Limits:**
- `import-dir`: **500 MB per file** — larger files get `Error: Maximum file size of 524288000 exceeded`
- `import` (single file): **10 MB per file** — larger files get `Error: File size imported is (N bytes), exceeded max size (10485760 bytes)`
- Use `import-dir` pointed at a temp directory containing the single file as a workaround for files between 10-500 MB

**Gotchas:**
- `import-dir` stops on first error — if file 3 of 10 fails, files 4-10 are NOT uploaded
- Upload large files separately or split uploads into size-safe batches
- Workspace files are visible in the Databricks UI file browser

### 2. DBFS (Databricks File System)

**Path pattern:** `dbfs:/FileStore/<path>` (CLI) or `/dbfs/FileStore/<path>` (notebooks)

**Best for:** Large binary files that exceed workspace limits (>500 MB).

**Upload:**
```bash
# Must create directory first!
databricks fs mkdirs dbfs:/FileStore/my_data/subdir --profile dev

# Upload file
databricks fs cp ./local_file.nrrd dbfs:/FileStore/my_data/subdir/file.nrrd --profile dev

# Bulk upload
databricks fs cp ./local_dir/ dbfs:/FileStore/my_data/ --recursive --profile dev

# Verify
databricks fs ls dbfs:/FileStore/my_data/ --profile dev
```

**Access from notebooks:**
```python
# Use /dbfs/ prefix in notebooks
import nrrd
data, header = nrrd.read("/dbfs/FileStore/my_data/file.nrrd")
```

**Gotchas:**
- `databricks fs cp` fails with `no such directory` if the target directory doesn't exist — run `databricks fs mkdirs` first
- No per-file size limit (tested with 2.17 GB)
- DBFS paths in CLI use `dbfs:/` prefix; in notebooks use `/dbfs/` prefix (different!)

### 3. Unity Catalog Volumes

**Path pattern:** `/Volumes/<catalog>/<schema>/<volume>/<path>`

**Best for:** Production data, governed datasets, data shared across teams.

Not used in this project but recommended for production deployments where data governance is required.

---

## Mixed Storage Strategy

When a dataset contains files of varying sizes, use a mixed strategy:

```python
# Small files on Workspace (fast, visible in UI)
WS_ROOT = "/Workspace/Users/user@company.com/project/data"
SMALL_FILE = f"{WS_ROOT}/metadata.json"
IMAGES_DIR = f"{WS_ROOT}/images/"

# Large files on DBFS (no size limit)
LARGE_FILE = "/dbfs/FileStore/project_data/big_volume.nrrd"
```

Document which files are where — notebooks need the correct path prefix for each file.

---

## Networking and Firewalls

### Corporate Firewall Behavior

Databricks clusters in corporate VPCs may have network restrictions:

- **Symptom:** `ConnectionResetError(104, 'Connection reset by peer')` — active TCP reset, NOT a timeout or DNS failure
- **Diagnosis:** Create a connectivity check notebook that tests each domain with `requests.get()` and captures status codes / errors
- **Pattern:** Firewall blocks are per-domain, not per-port. All paths on a blocked domain fail.

### What Typically Works

- **PyPI** (`pypi.org`) — `pip install` works on most clusters
- **HuggingFace** (`huggingface.co`) — model downloads work
- **Internal services** — depends on VPC configuration

### What May Be Blocked

- External data APIs (Allen Brain Institute, etc.)
- Non-standard download servers
- Anything not on the corporate allow-list

### Workaround: Local Download + Upload

When external data sources are blocked:

1. Download data on a local machine (no firewall)
2. Upload to Databricks via CLI (`workspace import-dir` or `fs cp`)
3. Notebooks read from Workspace/DBFS paths instead of making HTTP calls

Build an **idempotent download script** that:
- Checks if each file exists before downloading
- Saves metadata (API responses) alongside the data for reproducibility
- Logs progress to stdout
- Can be safely re-run after interruption

---

## Cluster Types and Job Execution

### Interactive vs Job Clusters

Clusters have a `workload_type.clients` setting:
- `notebooks: true, jobs: true` — supports both interactive and job workloads
- `notebooks: true, jobs: false` — **interactive only**

Check with: `databricks clusters get <cluster-id> --profile dev`

### Running Code on Interactive-Only Clusters

If `jobs: false`, `databricks jobs submit` fails with:
```
The cluster <id> does not support jobs workload
```

**Workaround: Command Execution API**

```bash
# 1. Create execution context
curl -X POST "${HOST}/api/1.2/contexts/create" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"clusterId": "<cluster-id>", "language": "python"}'
# Returns: {"id": "<context-id>"}

# 2. Execute code
curl -X POST "${HOST}/api/1.2/commands/execute" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"clusterId": "<cluster-id>", "contextId": "<context-id>", "language": "python", "command": "print(1+1)"}'
# Returns: {"id": "<command-id>"}

# 3. Poll for results
curl "${HOST}/api/1.2/commands/status?clusterId=<cluster-id>&contextId=<context-id>&commandId=<command-id>" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Cluster Lifecycle

Clusters auto-terminate after idle timeout. Before running commands:

```bash
# Check state
databricks clusters get <cluster-id> --profile dev | grep state

# Start if terminated
databricks clusters start <cluster-id> --profile dev

# Poll until RUNNING (takes 3-10 minutes)
while true; do
  state=$(databricks clusters get <cluster-id> --profile dev -o json | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
  echo "State: $state"
  [ "$state" = "RUNNING" ] && break
  sleep 10
done
```

---

## CLI Command Reference

### Workspace Operations

```bash
# List contents
databricks workspace list /Workspace/Users/user@company.com/ --profile dev

# Create directory
databricks workspace mkdirs /Workspace/Users/user@company.com/new_dir --profile dev

# Bulk upload (up to 500 MB/file)
databricks workspace import-dir ./local/ /Workspace/.../target/ --profile dev

# Single file upload (up to 10 MB)
databricks workspace import /Workspace/.../file --file ./file --format AUTO --overwrite --profile dev

# Import notebook
databricks workspace import /Workspace/.../notebook \
  --file ./notebook.ipynb --format JUPYTER --language PYTHON --overwrite --profile dev

# Download
databricks workspace export /Workspace/.../file --filename ./local_file --profile dev
databricks workspace export-dir /Workspace/.../dir/ ./local_dir/ --profile dev

# Delete
databricks workspace delete /Workspace/.../file --profile dev
```

### DBFS Operations

```bash
# Create directory (required before upload!)
databricks fs mkdirs dbfs:/FileStore/my_data/ --profile dev

# Upload
databricks fs cp ./file dbfs:/FileStore/my_data/file --profile dev
databricks fs cp ./dir/ dbfs:/FileStore/my_data/dir/ --recursive --profile dev

# List
databricks fs ls dbfs:/FileStore/my_data/ --profile dev

# Download
databricks fs cp dbfs:/FileStore/my_data/file ./local_file --profile dev
```

### Cluster Operations

```bash
# List clusters
databricks clusters list --profile dev

# Get cluster details
databricks clusters get <cluster-id> --profile dev

# Start cluster
databricks clusters start <cluster-id> --profile dev

# Get cluster state only
databricks clusters get <cluster-id> --profile dev -o json | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])"
```

---

## Common Pitfalls

1. **`import-dir` stops on first error.** If uploading a directory with mixed file sizes, a single file exceeding 500 MB kills the entire upload. Upload large files separately first.

2. **`fs cp` requires `mkdirs` first.** Unlike local `cp`, DBFS won't auto-create parent directories. Always `mkdirs` before `cp`.

3. **Path prefixes differ.** CLI uses `dbfs:/FileStore/...`, notebooks use `/dbfs/FileStore/...`. Workspace paths are the same in both contexts.

4. **`workspace import` vs `workspace import-dir` have different size limits.** Single file import: 10 MB. Directory import: 500 MB per file. For files between 10-500 MB, use `import-dir` with a temp directory containing just that file.

5. **Cluster `jobs: false` is common on shared interactive clusters.** Don't assume `databricks jobs submit` will work. Check `workload_type.clients.jobs` first. Use Command Execution API as fallback.

6. **JSON escaping in Command Execution API.** When embedding Python code in JSON payloads for the Command Execution API, avoid shell heredocs with Python f-strings. Instead, write a Python script that handles the API calls directly using `urllib.request` or `requests`.

7. **Compressed files are much smaller than uncompressed.** NRRD files with gzip compression can be 10-50x smaller than their uncompressed in-memory representation. Don't estimate download sizes from numpy array sizes.

8. **Allen API treatment names are case-sensitive.** `'NISSL'` works, `'Nissl'` returns 0 rows. Always verify with an unfiltered query first.
