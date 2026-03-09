# Step 8: Training Data Pipeline — Plan & Status

## Status: NOT STARTED (planning research complete)

**Last updated:** 2026-03-09
**Depends on:** Steps 5-6 DONE (20.87 GB downloaded + uploaded). Step 6B (cluster verification) still pending. Pipeline code can be written and tested locally against `data/allen_brain_data/` now.

---

## Goal

Build a training data pipeline that converts raw Allen Brain data into `(image_tile, segmentation_mask)` pairs suitable for DINOv2-Large + UperNet semantic segmentation fine-tuning.

**Two data sources → two sub-pipelines:**
1. **CCFv3 3D volume slicer** — slice NRRD volumes into 2D coronal slices, tile to model input size
2. **Mouse Atlas SVG rasterizer** — convert SVG annotations into pixel-level segmentation masks for 2D atlas sections

Both pipelines must support the coarse (5-class) → fine (672-class) structure grouping via the ontology tree.

---

## Research Findings (from explorer notebook analysis)

### CCFv3 Volume Details

**Files (downloaded to `data/allen_brain_data/ccfv3/`):**

| File | Content | Shape (at 25μm) | Shape (at 10μm) | dtype | Disk Size |
|------|---------|-----------------|-----------------|-------|-----------|
| `ara_nissl_10.nrrd` | Nissl histology (training domain 1) | N/A | (1320, 800, 1140) | **float32** | 2,169.9 MB |
| `annotation_10.nrrd` | Structure ID labels | N/A | (1320, 800, 1140) | uint32 | 32.8 MB (compressed) |
| `average_template_25.nrrd` | Autofluorescence (target domain) | (528, 320, 456) | N/A | uint16 | 33.0 MB |
| `annotation_25.nrrd` | Structure ID labels | (528, 320, 456) | N/A | uint32 | 4.0 MB |

**IMPORTANT — Verified dtypes from Step 5 completion report (`docs/step5_6_completion_report.md`):**
- `ara_nissl_10.nrrd` is **float32** (not uint8) — will need normalization to [0, 255] or [0, 1] before use
- `average_template_25.nrrd` is **uint16** — may need rescaling depending on value range
- Annotation volumes are uint32 with structure IDs (some IDs up to 614454277)

**Volume axes:** AP (anterior-posterior) × DV (dorsal-ventral) × ML (medial-lateral)

**Slicing (from explorer notebook):**
```python
# Coronal slice at position `ap`:
image_slice = nissl_volume[ap, :, :]   # shape: (DV, ML)
label_slice = annotation_volume[ap, :, :]  # shape: (DV, ML), values are structure IDs

# For autofluorescence domain:
image_slice = template_volume[ap, :, :]
label_slice = annotation_volume_25[ap, :, :]
```

**25μm volumes (verified):**
- 528 coronal slices available (AP axis)
- Each slice is 320 × 456 pixels (DV × ML)
- 672 unique structure IDs in annotation volume
- Annotation value range: [0, 614454277] (0 = background)

**10μm volumes (verified shapes from download):**
- 1320 coronal slices (AP axis)
- Each slice is 800 × 1140 pixels (DV × ML)
- Same structure IDs but at higher resolution
- **Nissl volume is float32** — need to determine value range and normalize (e.g., percentile clip → uint8)

### SVG Annotation Details

**SVG parsing code (from explorer notebook):**
```python
import xml.etree.ElementTree as ET

root = ET.fromstring(svg_text)
ns = {"svg": "http://www.w3.org/2000/svg"}

# Find all path elements (structure boundaries)
paths = root.findall(".//svg:path", ns)
if not paths:
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
if not paths:
    paths = root.findall(".//path")

# Extract structure IDs
structure_ids_in_svg = set()
for path in paths:
    sid = path.get("structure_id") or path.get("data-structure-id", "")
    if sid:
        structure_ids_in_svg.add(sid)
```

**SVG format:**
- SVG dimensions: `width="10496" height="8064"` (varies per section)
- Each `<path>` has a `structure_id` attribute (integer)
- Path `d` attribute contains bezier curve definitions (polygon boundaries)
- 154 path elements, 144 unique structure IDs per mid-brain section
- **509 SVGs downloaded** (all atlas sections, not just the 132 originally noted as "annotated"). Need to check which have meaningful structure annotations vs sparse/empty content.

**Rasterization approach:**
- Parse SVG path `d` attribute → convert bezier curves to polygon points
- Use `matplotlib.path.Path` or `svgpathtools` + `PIL.ImageDraw` to fill polygons
- Output: 2D array same size as the image where each pixel = structure_id
- Must handle overlapping paths (z-order: later paths on top)

### Structure Ontology

**Loading (from explorer notebook):**
```python
# Via direct API (preferred — avoids AllenSDK numpy issues):
ontology_url = f"{API_BASE}/structure_graph_download/1.json"
ontology_data = fetch_json(ontology_url)

# Flatten hierarchy:
def flatten_ontology(node, depth=0, result=None):
    if result is None:
        result = []
    node["depth"] = depth
    result.append(node)
    for child in node.get("children", []):
        flatten_ontology(child, depth + 1, result)
    return result

flat_structures = flatten_ontology(ontology_data["msg"][0])
```

**Structure object fields:**
```python
{
    "id": 8,
    "acronym": "TMv",
    "name": "Tuberomammillary nucleus, ventral part",
    "rgb_triplet": [255, 76, 62],
    "color_hex_triplet": "ff4c3e",
    "depth": 5,
    "parent_id": 1234,
    "children": [...]
}
```

**Coarse grouping (Phase 1 — 5 classes):**

The ontology root (`id=997`, "Brain") has these depth-1 children:
| Class ID | Name | Ontology ID | Notes |
|----------|------|-------------|-------|
| 0 | Background | 0 | Pixels outside brain |
| 1 | Cerebrum | 567 | Cortex, hippocampus, basal ganglia, etc. |
| 2 | Brain stem | 343 | Midbrain, pons, medulla |
| 3 | Cerebellum | 512 | Cerebellar cortex + nuclei |
| 4 | Fiber tracts | 1009 | White matter bundles |
| 5 | Ventricular system | 73 | Ventricles + choroid plexus |

**Mapping fine → coarse:** Walk each structure's `parent_id` chain up to depth 1. Build a lookup dict: `{fine_structure_id: coarse_class_id}`. Apply to annotation volume via vectorized numpy indexing.

### DINOv2 + UperNet Input Requirements

**Model:** HuggingFace `transformers` likely provides `Dinov2ForSemanticSegmentation` or similar. If not available as a pre-built class, attach UperNet head manually via `mmsegmentation` or custom code.

**Input requirements:**
- DINOv2-Large native resolution: **518×518** (patch size 14, 37×37 patches)
- Input: RGB 3-channel (repeat grayscale 3× for single-channel brain slices)
- Normalization: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Labels: 2D integer tensor same spatial size as input, values 0..num_classes-1
- Loss: Cross-entropy per pixel (standard for semantic segmentation)

**Tiling strategy for CCFv3 slices:**
- 25μm slices are 320×456 — can pad/resize to 518×518 directly (single tile per slice)
- 10μm slices are ~800×1140 — need to tile into overlapping 518×518 patches, or resize
- Mouse Atlas at ds=4: ~1208×2338 — must tile into multiple 518×518 patches

---

## Pipeline Components to Build

### Component 1: Structure Ontology Loader + Grouper

**Input:** `ontology/structure_graph_1.json`
**Output:** Mapping dict `{structure_id: coarse_class_id}` for any granularity level

**Logic:**
1. Load JSON, flatten tree recursively
2. For a given target depth (1=coarse/5 classes, 2-3=medium, full=fine):
   - Walk each structure's parent chain up to target depth
   - Map to the ancestor at that depth
3. Return lookup dict + class names list

### Component 2: CCFv3 Volume Slicer

**Input:** NRRD volume files (nissl, template, annotation at 10μm and 25μm)
**Output:** Iterator of `(image_tile, mask_tile)` pairs

**Logic:**
1. Load NRRD files with `pynrrd`
2. **Normalize image volumes:**
   - `ara_nissl_10.nrrd` is **float32** — inspect value range, apply percentile clipping (e.g., 1st-99th), scale to [0, 255] uint8
   - `average_template_25.nrrd` is **uint16** — inspect max value, scale to [0, 255] uint8
3. Iterate over AP axis (coronal slices)
4. For each slice:
   - Extract 2D image (nissl or template) and 2D annotation
   - Map annotation structure IDs → class IDs using ontology grouper
   - Skip slices that are mostly background (< 10% brain pixels)
   - Tile/resize to 518×518
   - Convert grayscale → 3-channel RGB
5. Apply train/val/test split (e.g., 80/10/10 by AP position)

### Component 3: SVG Rasterizer (for Mouse Atlas 2D validation)

**Input:** SVG file + corresponding atlas image
**Output:** Segmentation mask (2D array, same size as image)

**Logic:**
1. Parse SVG with `xml.etree.ElementTree`
2. Extract all `<path>` elements with `structure_id`
3. Parse SVG path `d` attribute (bezier curves) → polygon vertices
4. Rasterize each polygon onto a blank canvas:
   - Use `PIL.ImageDraw.polygon()` or `matplotlib.path.Path.contains_points()`
   - Fill with structure_id value
   - Handle overlaps by drawing in SVG document order (later paths overwrite)
5. Map structure IDs → class IDs using ontology grouper
6. Resize/tile to match model input

### Component 4: PyTorch Dataset + DataLoader

**Purpose:** Wrap the above into a `torch.utils.data.Dataset` for training

**Logic:**
1. Pre-compute all tiles from CCFv3 slicer (or lazy-load)
2. Apply data augmentation (random crop, flip, rotation, color jitter)
3. Normalize with DINOv2 ImageNet stats
4. Return `{"pixel_values": tensor, "labels": tensor}`

---

## Repo Structure Plan

```
src/
├── histological_image_analysis/
│   ├── __init__.py
│   ├── ontology.py          # Component 1: structure grouper
│   ├── ccfv3_slicer.py      # Component 2: volume slicing + tiling
│   ├── svg_rasterizer.py    # Component 3: SVG → segmentation mask
│   └── dataset.py           # Component 4: PyTorch dataset
tests/
├── test_ontology.py
├── test_ccfv3_slicer.py
├── test_svg_rasterizer.py
└── test_dataset.py
```

**Per CLAUDE.md:** TDD — write tests first. No global variables. Use dependency injection.

**Per CLAUDE.md Databricks exception:** The eventual training notebook (Step 9) will use inline functions, NOT imports from `src/`. But `src/` modules will be tested locally and can be uploaded to Workspace alongside data if needed.

---

## Dependencies to Add to pyproject.toml

```toml
# Already present:
# pynrrd, matplotlib, requests

# Needed for Step 8:
torch >= 2.0
torchvision >= 0.15
transformers >= 4.40
Pillow >= 10.0
svgpathtools >= 1.6    # For parsing SVG bezier curves to polygons
numpy >= 1.24
```

**Note:** `torch` and `transformers` are large. On Databricks, they're pre-installed on ML runtimes. Locally, install via `uv add torch torchvision transformers Pillow svgpathtools`.

---

## Data Flow Diagram

```
CCFv3 NRRD volumes                    Mouse Atlas 2D
─────────────────                     ──────────────
ara_nissl_10.nrrd ──┐                 atlas_image.jpg ──┐
annotation_10.nrrd ─┤                 annotation.svg  ──┤
template_25.nrrd ───┤                                   │
annotation_25.nrrd ─┘                                   │
        │                                               │
        ▼                                               ▼
  CCFv3 Slicer                               SVG Rasterizer
  (coronal 2D slices)                        (polygon → pixel mask)
        │                                               │
        ▼                                               ▼
  (image, annotation)                         (image, mask)
        │                                               │
        ├───── Ontology Grouper ──────────────────┤
        │      (structure_id → class_id)                │
        ▼                                               ▼
  (image_tile, class_mask)                    (image_tile, class_mask)
        │                                               │
        └──────────── PyTorch Dataset ──────────────────┘
                           │
                           ▼
                    DataLoader (batches)
                           │
                           ▼
                DINOv2 + UperNet (Step 9)
```

---

## Open Questions for User

1. **SVG rasterization library:** `svgpathtools` parses bezier curves cleanly but doesn't rasterize directly. Alternative: use `cairosvg` to render SVG → PNG → parse colors. Which approach is preferred?
2. **10μm volume tiling:** The 10μm slices are ~800×1140. Options: (a) resize to 518×518 (loses detail), (b) tile into overlapping 518×518 patches (more training samples, keeps detail), (c) random crop 518×518 during training. Recommendation: (c) random crop for training, (b) tiled inference for evaluation.
3. **Data storage format:** Save pre-processed tiles as individual files (PNG + NPY), or load from NRRD on-the-fly? On-the-fly is simpler but slower. Pre-processed is faster but takes disk space.
4. **Train/val/test split:** Split by AP position (spatial), or random? Spatial split prevents data leakage from adjacent slices being too similar.

---

## Files Referenced

| File | Purpose |
|------|---------|
| `docs/progress.md` | Master project state |
| `docs/step7_dataset_model_decision.md` | Model + dataset decisions (DONE) |
| `docs/step8_training_data_pipeline.md` | **THIS FILE** — pipeline plan |
| `docs/step5_6_completion_report.md` | Download/upload results — verified shapes, dtypes, paths |
| `docs/data_download_plan.md` | Original download plan (historical) |
| `exploration/allen_brain_data_explorer.ipynb` | Source of CCFv3/SVG parsing code |
| `scripts/download_allen_data.py` | Downloads raw data |
| `data/allen_brain_data/` | Local data directory (20.87 GB, `.gitignore`d) |

---

## Resume Instructions

If context is lost, read these files in order:
1. `CLAUDE.md` — coding standards (no globals, TDD, SOLID, uv, don't commit)
2. `docs/progress.md` — master project state
3. `docs/step8_training_data_pipeline.md` — **THIS FILE** (the active work item)
4. `docs/step5_6_completion_report.md` — verified data shapes, dtypes, Databricks paths
5. `docs/step7_dataset_model_decision.md` — model + dataset decisions (DONE)

**Next actions when resuming:**
1. Steps 5-6 are DONE (20.87 GB downloaded + uploaded). Step 6B (cluster verification) still pending — not blocking for local development.
2. Resolve open questions above with user (SVG lib, tiling strategy, storage format, split strategy).
3. Add `torch`, `transformers`, `svgpathtools`, `Pillow` to `pyproject.toml` and `uv lock`.
4. Create `src/histological_image_analysis/` package structure.
5. Build Component 1 (ontology loader/grouper) first — it's a dependency for all other components and has no external data dependency.
6. Build Components 2-4 following TDD.
7. Verify pipeline produces valid `(image, mask)` pairs with a quick visualization notebook.
8. Run Step 6B verification on Databricks cluster when ready.

**Databricks paths (from `docs/step5_6_completion_report.md`):**
```python
WS_ROOT = "/Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology"

# CCFv3 3D volumes
ANNOTATION_25 = f"{WS_ROOT}/ccfv3/annotation_25.nrrd"         # (528, 320, 456) uint32, 4 MB
TEMPLATE_25   = f"{WS_ROOT}/ccfv3/average_template_25.nrrd"   # (528, 320, 456) uint16, 33 MB
ANNOTATION_10 = f"{WS_ROOT}/ccfv3/annotation_10.nrrd"         # (1320, 800, 1140) uint32, 33 MB
NISSL_10      = "/dbfs/FileStore/allen_brain_data/ccfv3/ara_nissl_10.nrrd"  # DBFS! (2.17 GB, exceeded 500 MB workspace limit)

# 2D atlas data
ATLAS_IMAGES_DIR  = f"{WS_ROOT}/mouse_atlas/images/"    # 509 JPEGs
ATLAS_SVGS_DIR    = f"{WS_ROOT}/mouse_atlas/svgs/"      # 509 SVGs
HUMAN_IMAGES_DIR  = f"{WS_ROOT}/human_atlas/images/"    # 14,565 JPEGs (~18 GB)

# Reference data
ONTOLOGY           = f"{WS_ROOT}/ontology/structure_graph_1.json"
ATLAS_METADATA     = f"{WS_ROOT}/metadata/atlas_images_metadata.json"
HUMAN_METADATA     = f"{WS_ROOT}/metadata/human_atlas_images_metadata.json"
```

**CRITICAL PATH NOTE:** `NISSL_10` is on DBFS (`/dbfs/...`), NOT Workspace. This is because the 2.17 GB file exceeded the 500 MB workspace per-file limit. All other files are on Workspace. Code must handle both path styles.

**Local paths (for testing):**
```python
LOCAL_DATA = "data/allen_brain_data"
LOCAL_NISSL_10      = f"{LOCAL_DATA}/ccfv3/ara_nissl_10.nrrd"
LOCAL_ANNOTATION_10 = f"{LOCAL_DATA}/ccfv3/annotation_10.nrrd"
LOCAL_TEMPLATE_25   = f"{LOCAL_DATA}/ccfv3/average_template_25.nrrd"
LOCAL_ANNOTATION_25 = f"{LOCAL_DATA}/ccfv3/annotation_25.nrrd"
LOCAL_ONTOLOGY      = f"{LOCAL_DATA}/ontology/structure_graph_1.json"
```