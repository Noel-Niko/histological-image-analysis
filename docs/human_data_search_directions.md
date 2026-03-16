# Search Directions: Human Brain Ground Truth Annotation Data

**Purpose:** This document provides directions for an LLM agent to search for and evaluate sources of pixel-level human brain structure annotations suitable for training a semantic segmentation model. The agent should search each source, report findings in the specified format, and evaluate against the acceptance criteria.

---

## What We Have

- **14,565 human Nissl-stained brain section images** from the Allen Human Brain Atlas (6 donors, ~18 GB, downsample=4)
- Donor IDs: `H0351.1009`, `H0351.1012`, `H0351.1015`, `H0351.1016`, `H0351.2001`, `H0351.2002`
- These were downloaded as `SectionImage` (raw donor tissue) via `section_image_download/{id}?downsample=4`
- Metadata file: `data/allen_brain_data/metadata/human_atlas_images_metadata.json` — 14,566 records with fields: `id`, `section_number`, `_donor`, `_plane`, `_dataset_id`

## What We Do NOT Have

- **No pixel-level structure annotations** for any human brain image
- No segmentation masks, no SVG boundary files, no 3D annotation volume
- Without annotations, we cannot train a supervised segmentation model for human brain structures

## What We Need

Pixel-level annotations that map brain structure identities to spatial locations in histological (or histology-registerable) images of the human brain. These annotations will serve as ground truth labels for training a DINOv2-Large + UperNet semantic segmentation model.

---

## Search Targets (in priority order)

### Target 1: Allen Human Brain Atlas — Reference Plates with SVG Annotations

**Priority: HIGHEST — If this works, it is the fastest path to a trained human model.**

**Background:** For the mouse brain, Allen provides two data types:
- `AtlasImage` — Reference atlas plates (hand-drawn diagrams with structure boundaries). These have SVG annotations downloadable via `svg_download/{id}`.
- `SectionImage` — Raw donor tissue sections (no annotations).

We downloaded only `SectionImage` for human. **Nobody has tested whether `AtlasImage` exists for the human atlas.**

**Specific API calls to make** (base URL: `https://api.brain-map.org/api/v2`):

1. **Query for human AtlasImage records:**
   ```
   GET /data/query.json?criteria=model::AtlasImage,rma::criteria,atlas_data_set(atlases[id$eq2]),rma::options[num_rows$eqall]
   ```
   - Atlas ID 2 is the Human Brain Atlas (Atlas ID 1 = Mouse).
   - If this returns 0 results, try alternative atlas IDs. Discover available atlases:
     ```
     GET /data/query.json?criteria=model::Atlas,rma::options[num_rows$eqall]
     ```
   - Also try querying by product ID instead of atlas ID:
     ```
     GET /data/query.json?criteria=model::AtlasImage,rma::criteria,atlas_data_set[id$eq2],rma::options[num_rows$eq10]
     ```

2. **If AtlasImage records exist, test SVG download:**
   ```
   GET /svg_download/{human_atlas_image_id}
   ```
   - A successful SVG response starts with `<svg` or `<?xml`.
   - Check: does it contain `<path structure_id="...">` elements (same format as mouse)?
   - Try at least 5 different image IDs to confirm consistency.

3. **Query human structure ontology:**
   ```
   GET /structure_graph_download/10.json
   ```
   - Structure graph ID 10 is listed in Allen documentation for human. Also try IDs: 2, 3, 4, 7, 11, 12, 16, 17.
   - Discover all available structure graphs:
     ```
     GET /data/query.json?criteria=model::StructureGraph,rma::options[num_rows$eqall]
     ```
   - Report: how many structures? What depth? Sample names at depth 1-3.

4. **Check for human 3D reference volume (equivalent to CCFv3):**
   - Search `https://download.alleninstitute.org/informatics-archive/current-release/` for any `human_*` directories
   - Check: `https://download.alleninstitute.org/informatics-archive/current-release/human_brain/` (may not exist)
   - Search Allen documentation for "human brain reference space" or "human brain annotation volume"

**Report format for Target 1:**
```
AtlasImage query result: [N records found / 0 records / error]
Atlas IDs tried: [list]
SVG download test: [success with N paths / empty response / 404 / error]
Sample SVG structure_ids: [list of first 10]
Structure ontology: [graph ID, N structures, max depth]
3D volume: [found at URL / not found]
```

### Target 2: Allen Human Brain Atlas Web Viewer Annotations

**Priority: HIGH — The web viewer shows structure boundaries; those annotations must come from somewhere.**

**Background:** The Allen Human Brain Atlas web viewer at `https://atlas.brain-map.org/` displays interactive structure delineations over human brain sections. These boundaries are rendered client-side, meaning the annotation data is fetched from an API or bundled in the viewer.

**Search directions:**

1. **Examine the Allen Human Brain Atlas interactive viewer:**
   - URL: `https://human.brain-map.org/`
   - URL: `https://atlas.brain-map.org/atlas?atlas=265297126` (Human atlas)
   - Look at the network requests in the browser dev tools documentation/API docs. What endpoints serve the structure boundary overlays?

2. **Search Allen API documentation:**
   - `https://help.brain-map.org/display/api/` — search for "structure mask", "annotation", "svg", "human"
   - `https://help.brain-map.org/display/api/Image-to-Image+Synchronization` — can structure boundaries be generated via the synchronization service?
   - `https://help.brain-map.org/display/api/Downloading+an+Atlas+Image` — does this document work for human atlas IDs?
   - `https://community.brain-map.org/` — search forum for "human SVG", "human annotation download", "human atlas segmentation"

3. **Test the SectionImage-based annotation approach:**
   - For each human SectionImage, Allen may provide an "annotated" version or an associated atlas image via the `image_sync` API:
     ```
     GET /image_to_atlas/{section_image_id}.json
     ```
   - This maps a section image to the corresponding atlas plate and provides coordinate transforms. If atlas plates have structure annotations, we could project them onto section images.

4. **Check structure mask download:**
   ```
   GET /structure_mask_download/{structure_id}?section_image_id={id}
   ```
   - This endpoint may exist for generating per-structure binary masks on any section image

**Report format for Target 2:**
```
Web viewer annotation source: [API endpoint / bundled data / not determined]
Image sync API: [works for human / doesn't exist / error]
Structure mask download: [works / doesn't exist / error]
Documentation findings: [relevant URLs and summaries]
Forum findings: [relevant threads and summaries]
```

### Target 3: BigBrain — 20μm 3D Histological Model

**Priority: MEDIUM — Real histological data with cytoarchitectonic annotations, but requires registration.**

**Background:** BigBrain (Amunts et al., 2013, Science 340:1472-1475) is a 3D reconstruction of a single human brain at 20μm isotropic resolution from 7,400 histological sections (Merker-stained, similar to Nissl). The Jülich Research Centre provides cytoarchitectonic annotations mapped to BigBrain space.

**Search directions:**

1. **BigBrain data portal:**
   - `https://bigbrain.loris.ca/` — primary data access point
   - `https://bigbrainproject.org/` — project home page
   - Look for: 3D annotation volumes, cytoarchitectonic maps, segmentation masks
   - Check: are downloads programmatic (HTTP/FTP) or interactive-only?

2. **BigBrain on EBRAINS/HBP:**
   - `https://ebrains.eu/` — search for "BigBrain" datasets
   - `https://search.kg.ebrains.eu/` — knowledge graph, search for BigBrain annotation volumes
   - EBRAINS often provides DOI-linked datasets with direct download

3. **Jülich Cytoarchitectonic Atlas:**
   - `https://julich-brain-atlas.de/` — Jülich Brain Atlas portal
   - `https://siibra-explorer.apps.hbp.eu/` — interactive viewer with cytoarchitectonic regions
   - Search for: "JuBrain", "probabilistic cytoarchitectonic maps", "Jülich brain parcellation"
   - Key question: are the probability maps available as downloadable NIfTI/NRRD volumes in BigBrain space?

4. **Specific data products to find:**
   - `BigBrain 3D volume` — the histological intensity volume (for image data)
   - `Jülich cytoarchitectonic maps in BigBrain space` — probability maps per brain region
   - `BigBrain cortical layer segmentation` — if available, provides layer-level annotations of cortex
   - Any pre-computed segmentation masks in BigBrain space

5. **Registration feasibility:**
   - Can Allen human Nissl sections (our 14,565 images) be registered to BigBrain space?
   - Tools: ANTs (`antsRegistration`), `brainreg` (Python), FreeSurfer
   - Is there published work registering Allen human sections to BigBrain? Search Google Scholar for: `"Allen Human Brain Atlas" AND "BigBrain" AND registration`

**Report format for Target 3:**
```
BigBrain volume: [available at URL, format, size / not found]
Cytoarchitectonic annotations: [N regions, format, resolution, download URL / not found]
Cortical layer segmentation: [available / not found]
Download method: [direct HTTP / FTP / EBRAINS login required / interactive only]
License: [CC-BY / other / unclear]
Registration feasibility: [published methods exist / no prior work / tools available]
```

### Target 4: Other Human Brain Histological Atlases

**Priority: MEDIUM-LOW — Cast a wider net if Targets 1-3 don't yield usable annotations.**

**Search directions:**

1. **BrainSpan (Allen Institute):**
   - `https://www.brainspan.org/` — developmental human brain atlas
   - Check: does it provide structure annotations for histological sections?
   - Note: developmental atlas (prenatal/postnatal) — may not match adult brain ontology

2. **Human Connectome Project (HCP):**
   - `https://www.humanconnectome.org/`
   - MRI-based parcellations (Glasser et al., 2016 — 360 cortical areas)
   - Only useful if registerable to histological space (MRI → histology registration is hard)
   - Lower priority — this is MRI, not histology

3. **AMBA (Allen Mouse Brain Atlas) to human mappings:**
   - Has anyone built a cross-species ontology mapping? Search for: `"Allen Brain Atlas" AND "cross-species" AND "homology" AND human`
   - If a mouse-to-human structure mapping exists, we could potentially use the mouse model to generate pseudo-labels at human-mapped regions

4. **NeuroMorpho.Org, Scalable Brain Atlas, or other open neuroanatomy databases:**
   - `https://scalablebrainatlas.incf.org/` — multi-species brain atlas viewer
   - Check if they provide downloadable human brain parcellation volumes
   - `https://neuromorpho.org/` — primarily neuron morphology, likely not useful

5. **Published papers with human brain segmentation datasets:**
   - Google Scholar search: `"human brain" AND "histological segmentation" AND ("ground truth" OR "annotation" OR "labeled dataset")`
   - Google Scholar search: `"Nissl stain" AND "human brain" AND "semantic segmentation"`
   - Any paper that trained a model on human brain histology must have had annotations — where did they come from?

**Report format for Target 4:**
```
Source: [name]
Data type: [histology / MRI / other]
Annotations: [N structures, type (mask/SVG/volume), resolution]
Download: [URL, method, authentication]
License: [name]
Relevance: [HIGH/MEDIUM/LOW + reason]
```

### Target 5: Allen API — Exhaustive Endpoint Discovery

**Priority: LOW — Systematic sweep if targeted queries from Target 1 fail.**

Only pursue this if Target 1 returns nothing and Target 2 doesn't reveal the annotation source.

1. **List all available data models:**
   ```
   GET https://api.brain-map.org/api/v2/data/enumerate.json
   ```
   - Look for any model related to annotations, masks, or structure boundaries

2. **List all products:**
   ```
   GET https://api.brain-map.org/api/v2/data/query.json?criteria=model::Product,rma::options[num_rows$eqall]
   ```
   - Identify all human-related products beyond Product ID 2

3. **List all atlases:**
   ```
   GET https://api.brain-map.org/api/v2/data/query.json?criteria=model::Atlas,rma::options[num_rows$eqall]
   ```
   - Identify human atlas IDs for targeted AtlasImage queries

4. **Test svg_download on known human SectionImage IDs:**
   - Pick 5 human SectionImage IDs from our metadata (e.g., from the JSON file)
   - Try: `GET /svg_download/{section_image_id}` — does it return anything?
   - Even if SectionImages don't normally have SVGs, Allen may have added them

---

## Acceptance Criteria

Any data source must meet ALL of the following to be usable for training:

### Hard Requirements (must have)

| Criterion | Requirement | Rationale |
|-----------|-------------|-----------|
| **Pixel-level annotations** | Must provide spatial boundaries of brain structures as SVG paths, segmentation masks (PNG/TIFF), or 3D annotation volumes (NIfTI/NRRD) | Cannot train a supervised segmentation model without per-pixel labels |
| **Structure identity** | Each annotated region must have a named structure identity (e.g., "hippocampus", "caudate nucleus"), not just unlabeled contours | Need semantic labels, not instance boundaries |
| **Minimum structure count** | At least 20 distinct brain structures annotated | Fewer than 20 is too coarse for a useful brain parcellation model |
| **Minimum image count** | At least 50 annotated images (or 50 slices extractable from a 3D volume) | Need sufficient training data to avoid catastrophic overfitting |
| **Histological basis** | Data must be from histological tissue sections (Nissl, Merker, cresyl violet, or similar stain) OR must be registerable to histological space with published methods | MRI parcellations alone are not sufficient — our input images are histological |
| **Programmatic download** | Data must be downloadable via HTTP, FTP, or authenticated API (not interactive-only web portals) | Need automated ingestion for reproducible pipeline |
| **Research license** | License must permit academic/research use | Non-negotiable for a research project |

### Soft Requirements (strongly preferred)

| Criterion | Preference | Rationale |
|-----------|------------|-----------|
| **Adult human brain** | Prefer adult over developmental/pediatric | Matches our existing 14,565 adult human Nissl images |
| **Nissl or similar stain** | Prefer Nissl-compatible stain (Nissl, Merker, cresyl violet) | Domain match with our input images; avoids stain transfer problem |
| **Resolution ≤ 100μm** | Prefer ≤ 100μm per pixel (our data is ~10μm native at downsample=4) | Higher resolution provides more spatial detail for small structures |
| **Allen-compatible ontology** | Prefer annotations using Allen Human Brain Atlas ontology | Direct compatibility with our existing ontology mapper infrastructure |
| **Multiple subjects** | Prefer multi-subject data (>1 brain) | Reduces bias from single-subject anatomical variation |
| **Coronal plane available** | Prefer coronal sections (matches our mouse training paradigm) | Architectural decisions (crop size, augmentation) were optimized for coronal |

### Comparability Notes

For context when evaluating annotation quality:

- Our **mouse model** (Run 5, best) achieves 68.8% mIoU on 503/1,328 valid classes (657 classes have zero training pixels)
- Mouse annotations come from the **CCFv3 3D annotation volume** (10μm isotropic, 1,327 structures) — this is the gold standard
- For human, even a much coarser annotation (e.g., 50-100 major regions) would be valuable as a starting point
- The model architecture supports any number of output classes — we resize the output head to match the label count

---

## Reporting Instructions

For each target investigated, report:

1. **What was found** — specific data products, URLs, formats, sizes
2. **What was NOT found** — explicitly state if a source was queried and returned nothing
3. **API responses** — include the actual JSON response (or first 50 lines) for any Allen API queries. This lets us verify the query was correct.
4. **Acceptance criteria evaluation** — score each source against the hard and soft requirements table
5. **Recommended next action** — for each source: "use directly", "investigate further", "requires registration pipeline", or "not viable"
6. **Download commands** — if usable data is found, provide the exact `curl`/`wget`/Python commands to download it

---

## Anti-Patterns (what NOT to do)

- **Do not download large files** — only query APIs and read documentation. Do not download multi-GB volumes.
- **Do not assume unavailability** — if an API returns 0 results, try alternative query parameters before concluding the data doesn't exist. The Allen API is sensitive to exact parameter formatting (e.g., `NISSL` vs `Nissl` returns different results).
- **Do not stop at the first source** — investigate ALL targets even if Target 1 succeeds. We need a complete picture of available human annotation data.
- **Do not confuse MRI parcellations with histological annotations** — MRI-based atlases (HCP, FreeSurfer) are not directly usable unless a histology-to-MRI registration pathway exists.
- **Do not report a source as "available" without verifying the download mechanism** — confirm that the data can actually be programmatically accessed, not just viewed in a web interface.
