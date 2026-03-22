# Step 8 Implementation Plan: Training Data Pipeline

**Status:** COMPLETE
**Last updated:** 2026-03-09

---

## Context

Build 4 pipeline components that convert raw Allen Brain data into `(image_tile, segmentation_mask)` pairs for DINOv2 + UperNet semantic segmentation fine-tuning. Steps 1-7 are done. Code goes in `src/histological_image_analysis/` with TDD tests in `tests/`. Per CLAUDE.md: no global variables, dependency injection, TDD, `uv` for package management.

**Resolved decisions:**
- SVG rasterization: `svgpathtools` + PIL (parse bezier → polygon → fill)
- 10um tiling: random crop (train) + tiled inference (eval)
- Storage: on-the-fly from NRRD (volumes loaded into memory)
- Train/val/test split: spatial by AP position (80/10/10)
- Deployment: src/ wheel installed on Databricks cluster + thin training notebook
- **JFrog Artifactory (Step 9):** Model weights must be manually added to JFrog by user before loading on Databricks

## Ontology Structure Correction

The Step 8 planning doc had an error in the coarse grouping. The **actual** hierarchy:

```
Root (997)
├── Basic cell groups and regions [grey] (8)      ← depth 1
│   ├── Cerebrum [CH] (567)                       ← depth 2
│   ├── Brain stem [BS] (343)                     ← depth 2
│   └── Cerebellum [CB] (512)                     ← depth 2
├── fiber tracts (1009)                           ← depth 1
├── ventricular systems [VS] (73)                 ← depth 1
├── grooves [grv] (1024)                          ← depth 1 (sulci — map to background)
└── retina (304325711)                            ← depth 1 (not in brain volume — map to background)
```

**Corrected coarse grouping (6 classes):**

| Class | Name | Ontology ID |
|-------|------|-------------|
| 0 | Background | 0 |
| 1 | Cerebrum | 567 |
| 2 | Brain stem | 343 |
| 3 | Cerebellum | 512 |
| 4 | Fiber tracts | 1009 |
| 5 | Ventricular systems | 73 |

## DINOv2 + UperNet (for Step 9)

- No `Dinov2ForSemanticSegmentation` class in `transformers`
- Use: `UperNetForSemanticSegmentation` + `Dinov2Backbone`
- Config: `UperNetConfig(backbone_config=Dinov2Config(), num_labels=N)`
- Input: `pixel_values` `(B, 3, H, W)`, `labels` `(B, H, W)`
- Loss: `CrossEntropyLoss` (built-in)

## SVG Data Note

Local SVGs have ~14 paths per file (not 154 — that was a specific mid-brain section at full res via API). Each path has `structure_id`, `d` (bezier), `style` attributes. SVG dimensions (e.g., 4400×4064) are in SVG coordinate space and may not match the downloaded image dimensions at ds=4 (~1208×2338). Rasterizer must render at SVG native dimensions then resize to target.

## Known Limitations (to address in Step 9)

- **Class imbalance:** Cerebrum will dominate pixel counts vs ventricular systems. May need weighted `CrossEntropyLoss` or class-balanced sampling in training.
- **Crop size 518×518:** Matches DINOv2-Large pretrained resolution (ViT-L/14, 37×37 patches × 14px = 518).

---

## Implementation Checklist

### Step 0: Project Scaffolding
- [x] Create `src/histological_image_analysis/__init__.py`
- [x] Create `tests/__init__.py`
- [x] Create `tests/conftest.py` (shared fixtures)
- [x] Create `tests/fixtures/` directory with minimal test data
- [x] Add dependencies to `pyproject.toml`
- [x] Run `uv lock && uv sync`

### Step 1: Ontology Mapper (`ontology.py`)
- [x] Write `tests/test_ontology.py` (TDD — tests first)
- [x] Implement `src/histological_image_analysis/ontology.py`
- [x] All 22 tests pass

**Class: `OntologyMapper`**
- `__init__(ontology_path: str | Path)` — load JSON, flatten tree recursively, build `{id: node}` lookup and `{id: parent_id}` chain
- `build_coarse_mapping() -> dict[int, int]` — ancestor-chain algorithm (see below)
- `build_depth_mapping(target_depth: int) -> dict[int, int]` — maps to ancestor at given depth
- `build_full_mapping() -> dict[int, int]` — sorted structure IDs → contiguous class IDs (deterministic)
- `get_class_names(mapping: dict[int, int]) -> list[str]` — class names for a given mapping
- `remap_mask(mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray` — vectorized remap

**Coarse mapping algorithm (ancestor-chain based, NOT depth-based):**
```python
COARSE_ANCESTORS = {
    567: 1,   # Cerebrum → class 1
    343: 2,   # Brain stem → class 2
    512: 3,   # Cerebellum → class 3
    1009: 4,  # Fiber tracts → class 4
    73: 5,    # Ventricular systems → class 5
}
EXCLUDED = {1024, 304325711}  # grooves, retina → background (0)

def build_coarse_mapping(self) -> dict[int, int]:
    mapping = {}
    for structure_id in self._all_structure_ids:
        # Walk up ancestor chain from structure_id to root
        current = structure_id
        coarse_class = 0  # default: background
        while current is not None:
            if current in COARSE_ANCESTORS:
                coarse_class = COARSE_ANCESTORS[current]
                break
            if current in EXCLUDED:
                coarse_class = 0
                break
            current = self._parent_lookup.get(current)
        mapping[structure_id] = coarse_class
    return mapping
```

**Fine-grained mapping determinism:** Structure IDs are **sorted** before assigning contiguous class IDs. Class 0 is always background.

### Step 2: CCFv3 Volume Slicer (`ccfv3_slicer.py`)
- [x] Write `tests/test_ccfv3_slicer.py`
- [x] Implement `src/histological_image_analysis/ccfv3_slicer.py`
- [x] All 15 tests pass

**Class: `CCFv3Slicer`**
- `__init__(image_path, annotation_path, ontology_mapper)`
- `load_volumes()` — read NRRD, validate shapes match, normalize image
- `get_slice(ap_index) -> (image_2d, annotation_2d)`
- `get_split_indices(train_frac=0.8, val_frac=0.1) -> dict[str, list[int]]`
- `iter_slices(split, mapping) -> Iterator[(image, class_mask)]`

**NRRD axis specification:**
- Axis 0 = AP (anterior-posterior): 0 = most anterior, max = most posterior
- Axis 1 = DV (dorsal-ventral)
- Axis 2 = ML (medial-lateral)
- 25μm: shape (528, 320, 456) → AP indices 0-527, coronal slice shape (320, 456)
- 10μm: shape (1320, 800, 1140) → AP indices 0-1319, coronal slice shape (800, 1140)

**Both volume types supported** (nissl for Phase A, template for Phase B):
- `ara_nissl_10.nrrd` (float32): percentile clip (1st-99th percentile), scale to [0, 255], cast uint8
- `average_template_25.nrrd` (uint16): scale by `255.0 / volume.max()`, cast uint8

**Validation in `load_volumes()`:**
- Assert image and annotation volumes have identical shapes
- Log volume dtype, shape, value range after loading

**Spatial split:**
- Valid AP indices = those with ≥10% non-background pixels in annotation
- Split valid indices by position: first 80% → train, next 10% → val, last 10% → test
- No overlap between splits

### Step 3: SVG Rasterizer (`svg_rasterizer.py`)
- [x] Write `tests/test_svg_rasterizer.py`
- [x] Implement `src/histological_image_analysis/svg_rasterizer.py`
- [x] All 10 tests pass

**Class: `SVGRasterizer`**
- `__init__(ontology_mapper: OntologyMapper)`
- `rasterize(svg_path, target_width, target_height) -> np.ndarray` — returns mask at target dimensions
- `_parse_paths(svg_path) -> list[(structure_id, polygon_points)]`
- `_bezier_to_points(d_string, num_samples=20) -> list[(x, y)]`

**SVG coordinate system handling:**
1. Parse SVG root `width`/`height` attributes (these define the SVG coordinate space)
2. Create canvas at SVG native dimensions (e.g., 4400×4064), initialized to **0 (background)**
3. Draw all paths in SVG document order (later paths overwrite — correct z-order)
4. Resize resulting mask to `target_width × target_height` using **nearest-neighbor** interpolation (preserves integer class IDs)

**Error handling:**
- Paths without `structure_id` attribute: skip, log warning
- Invalid/malformed bezier `d` attribute: try/except around `svgpathtools.parse_path()`, skip on error, log warning
- Unknown structure_ids (not in ontology): keep the raw ID in the mask (the ontology mapper will map unknowns to background when `remap_mask` is called)
- **Namespace fallback chain** (from explorer notebook):
  1. Try `.//svg:path` with `ns={"svg": "http://www.w3.org/2000/svg"}`
  2. Try `.//{http://www.w3.org/2000/svg}path`
  3. Try `.//path` (no namespace)

### Step 4: PyTorch Dataset (`dataset.py`)
- [x] Write `tests/test_dataset.py`
- [x] Implement `src/histological_image_analysis/dataset.py`
- [x] All 14 tests pass

**Class: `BrainSegmentationDataset(Dataset)`**
- `__init__(slicer, split, mapping, crop_size=518, augment=True)`
- `__len__()`, `__getitem__(idx) -> {"pixel_values": tensor, "labels": tensor}`

**Augmentation details (training only):**
- **Padding:** If slice < 518 in any dimension, pad with **0** (both image and labels)
- **Random crop:** 518×518 from padded slice. If slice == 518, no crop needed (use as-is)
- **Horizontal flip:** 50% probability
- **Rotation:** ±15°, fill value **0** for out-of-bounds pixels (both image and labels)
- **Color jitter:** brightness ±0.2, contrast ±0.2 (image only, not labels)

**Normalization:**
- Grayscale uint8 → float32 [0, 1] → repeat to 3 channels
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Output: float32 tensor `(3, 518, 518)`
- Labels: long tensor `(518, 518)`, values 0..num_classes-1

### Step 5: Integration & Documentation
- [x] Full test suite passes: `uv run pytest tests/ -v` — **61/61 passed**
- [x] Smoke test with real ontology data — 1,327 structures, 6 coarse classes, 1,327 fine classes
- [x] Update `docs/step8_training_data_pipeline.md` with corrections
- [x] Update `docs/progress.md`

---

## Test Fixtures

```
tests/
├── __init__.py
├── conftest.py              # shared pytest fixtures
├── fixtures/
│   ├── minimal_ontology.json  # Subset: root(997) → grey(8) → CH(567), BS(343), CB(512);
│   │                          #   fiber tracts(1009); VS(73); grooves(1024); retina(304325711)
│   │                          #   Plus 2-3 leaf nodes under Cerebrum for depth testing
│   └── sample.svg            # 2-3 simple rectangular paths with known structure_ids
│                              #   (use M/L path commands, not beziers, for simplicity)
├── test_ontology.py
├── test_ccfv3_slicer.py      # Uses synthetic numpy arrays, no fixture files
├── test_svg_rasterizer.py    # Uses fixtures/sample.svg
└── test_dataset.py           # Mocks CCFv3Slicer
```

---

## Change Log

| Date | Change |
|------|--------|
| 2026-03-09 | Plan created. Ontology depth correction discovered. Open questions resolved. |
| 2026-03-09 | Plan review: 14 issues addressed. Added: ancestor-chain algorithm, axis specs, SVG coord handling, background fill, data validation, deterministic fine mapping, augmentation fill values, error handling, fixtures structure, class imbalance note. |
| 2026-03-09 | All 4 components implemented with TDD. 61/61 tests pass. Smoke test with real ontology: 1,327 structures, coarse mapping verified (637 Cerebrum, 375 BS, 87 CB, 191 fiber, 12 VS, 25 background). |
| 2026-03-09 | Security fix: `.claude/settings.local.json` (Databricks token) removed from all git history via `filter-branch`. Force-pushed master + preparation. `.claude/` in `.gitignore`. |
| 2026-03-09 | Post-review fixes: (1) Added ±15° rotation augmentation to dataset.py. (2) Fixed get_class_names coarse-detection heuristic (anchor-point check). (3) Added get_num_labels() helper. (4) Separated exploration deps from core in pyproject.toml. (5) Fixed progress.md compact context ("coarse (6) → fine (1,327)"). (6) Updated step8_training_data_pipeline.md with all corrections. Test count: 61→65. |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `pyproject.toml` | Modify: add torch, torchvision, transformers, Pillow, svgpathtools, numpy, pytest |
| `src/histological_image_analysis/__init__.py` | Create |
| `src/histological_image_analysis/ontology.py` | Create |
| `src/histological_image_analysis/ccfv3_slicer.py` | Create |
| `src/histological_image_analysis/svg_rasterizer.py` | Create |
| `src/histological_image_analysis/dataset.py` | Create |
| `tests/__init__.py` | Create |
| `tests/conftest.py` | Create |
| `tests/fixtures/minimal_ontology.json` | Create |
| `tests/fixtures/sample.svg` | Create |
| `tests/test_ontology.py` | Create |
| `tests/test_ccfv3_slicer.py` | Create |
| `tests/test_svg_rasterizer.py` | Create |
| `tests/test_dataset.py` | Create |
| `docs/step8_training_data_pipeline.md` | Update with corrections |
| `docs/progress.md` | Update Step 8 status |
