# Step 8: Training Data Pipeline — Plan & Status

## Status: COMPLETE

**Last updated:** 2026-03-09
**Depends on:** Steps 5-6 DONE (20.87 GB downloaded + uploaded). Step 6B (cluster verification) still pending.
**Implementation:** 4 components in `src/histological_image_analysis/`, 65 tests in `tests/`. See `docs/step8_implementation_plan.md` for detailed tracker.

---

## Goal

Build a training data pipeline that converts raw Allen Brain data into `(image_tile, segmentation_mask)` pairs suitable for DINOv2-Large + UperNet semantic segmentation fine-tuning.

**Two data sources → two sub-pipelines:**
1. **CCFv3 3D volume slicer** — slice NRRD volumes into 2D coronal slices, tile to model input size
2. **Mouse Atlas SVG rasterizer** — convert SVG annotations into pixel-level segmentation masks for 2D atlas sections

Both pipelines must support the coarse (6-class) → fine (1,327-class) structure grouping via the ontology tree.

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

**Corrected coarse grouping (Phase 1 — 6 classes including background):**

> **Correction:** The original plan incorrectly described Cerebrum/BS/CB as depth-1 children of root. They are actually depth-2 under "Basic cell groups and regions" (grey, id=8). The hierarchy is:
> ```
> Root (997)
> ├── Basic cell groups and regions [grey] (8)      ← depth 1
> │   ├── Cerebrum [CH] (567)                       ← depth 2
> │   ├── Brain stem [BS] (343)                     ← depth 2
> │   └── Cerebellum [CB] (512)                     ← depth 2
> ├── fiber tracts (1009)                           ← depth 1
> ├── ventricular systems [VS] (73)                 ← depth 1
> ├── grooves [grv] (1024)                          ← depth 1 (→ background)
> └── retina (304325711)                            ← depth 1 (→ background)
> ```

| Class ID | Name | Ontology ID | Notes |
|----------|------|-------------|-------|
| 0 | Background | 0 | Pixels outside brain + grooves + retina |
| 1 | Cerebrum | 567 | Cortex, hippocampus, basal ganglia, etc. |
| 2 | Brain stem | 343 | Midbrain, pons, medulla |
| 3 | Cerebellum | 512 | Cerebellar cortex + nuclei |
| 4 | Fiber tracts | 1009 | White matter bundles |
| 5 | Ventricular system | 73 | Ventricles + choroid plexus |

**Mapping fine → coarse (ancestor-chain algorithm):** Walk each structure's `parent_structure_id` chain upward until hitting a known coarse ancestor (567, 343, 512, 1009, 73) or an excluded ancestor (1024, 304325711 → background). This is NOT depth-based — it's ancestor-chain-based. See `ontology.py:_find_coarse_class()`.

**num_labels for UperNet:** `get_num_labels(mapping)` returns `max(mapping.values()) + 1`. For coarse: 6. For full: 1,328.

### DINOv2 + UperNet Input Requirements

**Model:** No `Dinov2ForSemanticSegmentation` class exists in `transformers`. Use `UperNetForSemanticSegmentation` with `Dinov2Backbone`. Config: `UperNetConfig(backbone_config=Dinov2Config(), num_labels=N)`.

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

**Deployment (resolved):** Install `src/` package as a wheel on Databricks cluster. Training notebook (Step 9) imports from installed package — thin notebook for orchestration + visualization only. No UDF/Spark concern (single-node PyTorch).

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

## Open Questions — RESOLVED

1. **SVG rasterization library:** → `svgpathtools` + PIL. Parse bezier → sample points → `ImageDraw.polygon()`.
2. **10μm volume tiling:** → Random crop 518×518 (train) + tiled inference (eval).
3. **Data storage format:** → On-the-fly from NRRD (volumes loaded into memory).
4. **Train/val/test split:** → Spatial by AP position (80/10/10). Slices with <10% brain pixels skipped.

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

**Step 8 is COMPLETE.** All 4 components built with TDD. See `docs/step8_implementation_plan.md` for the full implementation tracker with change log.

**Next:** Step 9 (fine-tuning notebook). JFrog Artifactory: model weights must be manually added before loading on Databricks.

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