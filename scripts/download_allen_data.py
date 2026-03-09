"""Download Allen Brain Institute data locally for Workspace upload.

This script downloads all Allen data items that are blocked by the
Databricks corporate firewall. See docs/data_download_plan.md for
full context.

Usage:
    python scripts/download_allen_data.py [--output-dir data/allen_brain_data]

Data items downloaded:
    1. CCFv3 annotation volume (25um) via direct HTTP
    2. CCFv3 template volume (25um) via direct HTTP
    3. CCFv3 annotation volume (10um) via direct HTTP
    4. CCFv3 Nissl volume (10um) via direct HTTP
    5. Mouse atlas images (509 sections, downsample=4)
    6. Mouse atlas SVG annotations
    7. Structure ontology JSON
    8. Human Brain Atlas Nissl sections (14,566 images, downsample=4)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import nrrd
import requests


# --- Constants ---

ALLEN_API_BASE = "https://api.brain-map.org/api/v2"
ALLEN_DOWNLOAD_BASE = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf"

CCFV3_URLS = {
    "annotation_25.nrrd": f"{ALLEN_DOWNLOAD_BASE}/annotation/ccf_2017/annotation_25.nrrd",
    "average_template_25.nrrd": f"{ALLEN_DOWNLOAD_BASE}/average_template/average_template_25.nrrd",
    "ara_nissl_10.nrrd": f"{ALLEN_DOWNLOAD_BASE}/ara_nissl/ara_nissl_10.nrrd",
    "annotation_10.nrrd": f"{ALLEN_DOWNLOAD_BASE}/annotation/ccf_2017/annotation_10.nrrd",
}

REQUEST_TIMEOUT = 60
DOWNLOAD_CHUNK_SIZE = 8192
IMAGE_DOWNLOAD_DELAY = 0.1  # seconds between API image requests
DEFAULT_OUTPUT_DIR = "data/allen_brain_data"


# --- Data classes ---


@dataclass
class DownloadResult:
    name: str
    status: str  # "ok", "skipped", "error"
    path: str = ""
    size_bytes: int = 0
    error: str = ""


@dataclass
class DownloadContext:
    """Holds all paths and state for the download session."""

    output_dir: Path
    ccfv3_dir: Path = field(init=False)
    mouse_images_dir: Path = field(init=False)
    mouse_svgs_dir: Path = field(init=False)
    human_images_dir: Path = field(init=False)
    ontology_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    results: list[DownloadResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ccfv3_dir = self.output_dir / "ccfv3"
        self.mouse_images_dir = self.output_dir / "mouse_atlas" / "images"
        self.mouse_svgs_dir = self.output_dir / "mouse_atlas" / "svgs"
        self.human_images_dir = self.output_dir / "human_atlas" / "images"
        self.ontology_dir = self.output_dir / "ontology"
        self.metadata_dir = self.output_dir / "metadata"

    def create_dirs(self) -> None:
        for d in [
            self.ccfv3_dir,
            self.mouse_images_dir,
            self.mouse_svgs_dir,
            self.human_images_dir,
            self.ontology_dir,
            self.metadata_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# --- Helper functions ---


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_json(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def stream_download(url: str, dest: Path, description: str) -> DownloadResult:
    """Download a large file with streaming and progress."""
    if dest.exists() and dest.stat().st_size > 0:
        size = dest.stat().st_size
        log(f"  SKIP (exists, {size / 1e6:.1f} MB): {dest.name}")
        return DownloadResult(description, "skipped", str(dest), size)

    log(f"  Downloading {description}...")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) < DOWNLOAD_CHUNK_SIZE:
                    pct = downloaded / total * 100
                    log(f"    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)")

        size = dest.stat().st_size
        log(f"  OK ({size / 1e6:.1f} MB): {dest.name}")
        return DownloadResult(description, "ok", str(dest), size)
    except Exception as e:
        log(f"  ERROR: {e}")
        return DownloadResult(description, "error", str(dest), error=str(e))


def download_image(url: str, dest: Path) -> bool:
    """Download a single image file. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception:
        return False


# --- Download step functions ---


def download_ccfv3_volumes(ctx: DownloadContext) -> None:
    """Steps 5B+5C: All CCFv3 volumes (25μm + 10μm) via direct HTTP."""
    log("=== Steps 5B+5C: CCFv3 volumes (direct HTTP) ===")

    for filename, url in CCFV3_URLS.items():
        dest = ctx.ccfv3_dir / filename
        result = stream_download(url, dest, f"CCFv3 {filename}")
        ctx.results.append(result)


def download_mouse_atlas_images(ctx: DownloadContext) -> list[dict]:
    """Step 5D: Mouse atlas images at downsample=4. Returns metadata list."""
    log("=== Step 5D: Mouse atlas images (downsample=4) ===")

    # Query API for atlas image list
    metadata_path = ctx.metadata_dir / "atlas_images_metadata.json"
    if metadata_path.exists():
        log("  Loading cached metadata...")
        with open(metadata_path) as f:
            images = json.load(f)
    else:
        log("  Querying Allen API for mouse atlas images...")
        url = f"{ALLEN_API_BASE}/data/query.json"
        params = {
            "criteria": "model::AtlasImage,rma::criteria,atlas_data_set(atlases[id$eq1]),rma::options[num_rows$eqall][order$eq'section_number']"
        }
        data = fetch_json(url, params)
        images = data["msg"]
        with open(metadata_path, "w") as f:
            json.dump(images, f, indent=2)
        log(f"  Found {len(images)} atlas images, saved metadata.")

    # Download each image
    total = len(images)
    downloaded = 0
    skipped = 0
    errors = 0

    for i, img in enumerate(images):
        img_id = img["id"]
        section_num = img.get("section_number", 0)
        dest = ctx.mouse_images_dir / f"{section_num:04d}_{img_id}.jpg"

        url = f"{ALLEN_API_BASE}/atlas_image_download/{img_id}?downsample=4"
        if download_image(url, dest):
            if dest.stat().st_size > 0:
                downloaded += 1
            else:
                skipped += 1
        else:
            errors += 1

        if (i + 1) % 50 == 0:
            log(f"  Progress: {i + 1}/{total} (downloaded={downloaded}, skipped={skipped}, errors={errors})")

        time.sleep(IMAGE_DOWNLOAD_DELAY)

    log(f"  Done: {downloaded} downloaded, {skipped} skipped, {errors} errors out of {total}")
    ctx.results.append(DownloadResult(
        f"Mouse atlas images ({total} sections)",
        "ok" if errors == 0 else "partial",
        str(ctx.mouse_images_dir),
    ))
    return images


def download_mouse_svgs(ctx: DownloadContext, images: list[dict]) -> None:
    """Step 5E: SVG annotations for mouse atlas sections."""
    log("=== Step 5E: Mouse atlas SVG annotations ===")

    total = len(images)
    downloaded = 0
    skipped = 0
    empty = 0

    for i, img in enumerate(images):
        img_id = img["id"]
        section_num = img.get("section_number", 0)
        dest = ctx.mouse_svgs_dir / f"{section_num:04d}_{img_id}.svg"

        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue

        url = f"{ALLEN_API_BASE}/svg_download/{img_id}"
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.text.strip().startswith("<"):
                dest.write_text(resp.text)
                downloaded += 1
            else:
                empty += 1
        except Exception:
            empty += 1

        if (i + 1) % 50 == 0:
            log(f"  Progress: {i + 1}/{total} (downloaded={downloaded}, skipped={skipped}, empty={empty})")

        time.sleep(IMAGE_DOWNLOAD_DELAY)

    log(f"  Done: {downloaded} downloaded, {skipped} skipped, {empty} empty/missing out of {total}")
    ctx.results.append(DownloadResult(
        f"Mouse SVG annotations ({downloaded + skipped} files)",
        "ok",
        str(ctx.mouse_svgs_dir),
    ))


def download_human_atlas(ctx: DownloadContext) -> None:
    """Step 5F: Human Brain Atlas Nissl sections (all 14,566 images)."""
    log("=== Step 5F: Human Brain Atlas Nissl sections ===")

    metadata_path = ctx.metadata_dir / "human_atlas_images_metadata.json"

    if metadata_path.exists():
        log("  Loading cached metadata...")
        with open(metadata_path) as f:
            all_images = json.load(f)
    else:
        log("  Querying Allen API for human NISSL datasets...")
        url = f"{ALLEN_API_BASE}/data/query.json"

        # Step 1: Get all NISSL datasets with donor info
        datasets_params = {
            "criteria": (
                "model::SectionDataSet,"
                "rma::criteria,products[id$eq2],[failed$eqfalse],"
                "treatments[name$eq'NISSL'],"
                "rma::include,specimen(donor),plane_of_section,"
                "rma::options[num_rows$eqall][order$eq'id']"
            )
        }
        ds_data = fetch_json(url, datasets_params)
        datasets = ds_data["msg"]
        log(f"  Found {len(datasets)} NISSL datasets")

        # Step 2: Get section images for each dataset
        all_images = []
        for i, ds in enumerate(datasets):
            ds_id = ds["id"]
            donor_name = ds.get("specimen", {}).get("donor", {}).get("name", "unknown")
            plane = ds.get("plane_of_section", {}).get("name", "unknown")

            img_params = {
                "criteria": f"model::SectionImage,rma::criteria,[data_set_id$eq{ds_id}],rma::options[num_rows$eqall]"
            }
            img_data = fetch_json(url, img_params)
            section_images = img_data.get("msg", [])

            for img in section_images:
                img["_donor"] = donor_name
                img["_plane"] = plane
                img["_dataset_id"] = ds_id
            all_images.extend(section_images)

            if (i + 1) % 100 == 0:
                log(f"  Queried {i + 1}/{len(datasets)} datasets ({len(all_images)} images so far)")

            time.sleep(IMAGE_DOWNLOAD_DELAY)

        with open(metadata_path, "w") as f:
            json.dump(all_images, f, indent=2)
        log(f"  Found {len(all_images)} total section images, saved metadata.")

    # Download each image
    total = len(all_images)
    downloaded = 0
    skipped = 0
    errors = 0

    for i, img in enumerate(all_images):
        img_id = img["id"]
        section_num = img.get("section_number", 0)
        donor = img.get("_donor", "unknown")
        dest = ctx.human_images_dir / f"{donor}_{section_num:04d}_{img_id}.jpg"

        url = f"{ALLEN_API_BASE}/section_image_download/{img_id}?downsample=4"
        if download_image(url, dest):
            if dest.stat().st_size > 0:
                downloaded += 1
            else:
                skipped += 1
        else:
            errors += 1

        if (i + 1) % 200 == 0:
            log(f"  Progress: {i + 1}/{total} (downloaded={downloaded}, skipped={skipped}, errors={errors})")

        time.sleep(IMAGE_DOWNLOAD_DELAY)

    log(f"  Done: {downloaded} downloaded, {skipped} skipped, {errors} errors out of {total}")
    ctx.results.append(DownloadResult(
        f"Human NISSL images ({total} sections)",
        "ok" if errors == 0 else "partial",
        str(ctx.human_images_dir),
    ))


def download_ontology(ctx: DownloadContext) -> None:
    """Step 5G: Structure ontology JSON."""
    log("=== Step 5G: Structure ontology ===")

    dest = ctx.ontology_dir / "structure_graph_1.json"
    if dest.exists() and dest.stat().st_size > 0:
        size = dest.stat().st_size
        log(f"  SKIP (exists, {size / 1e3:.1f} KB)")
        ctx.results.append(DownloadResult("Structure ontology", "skipped", str(dest), size))
        return

    try:
        url = f"{ALLEN_API_BASE}/structure_graph_download/1.json"
        data = fetch_json(url)
        with open(dest, "w") as f:
            json.dump(data, f, indent=2)
        size = dest.stat().st_size
        log(f"  OK ({size / 1e3:.1f} KB)")
        ctx.results.append(DownloadResult("Structure ontology", "ok", str(dest), size))
    except Exception as e:
        log(f"  ERROR: {e}")
        ctx.results.append(DownloadResult("Structure ontology", "error", error=str(e)))


def verify_downloads(ctx: DownloadContext) -> None:
    """Step 5H: Verify all downloads."""
    log("=== Step 5H: Verification ===")

    # Verify NRRD files
    for nrrd_file in ctx.ccfv3_dir.glob("*.nrrd"):
        try:
            data, header = nrrd.read(str(nrrd_file))
            log(f"  NRRD OK: {nrrd_file.name} — shape={data.shape}, dtype={data.dtype}")
        except Exception as e:
            log(f"  NRRD ERROR: {nrrd_file.name} — {e}")

    # Verify JSON files
    for json_file in [
        ctx.ontology_dir / "structure_graph_1.json",
        ctx.metadata_dir / "atlas_images_metadata.json",
        ctx.metadata_dir / "human_atlas_images_metadata.json",
    ]:
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)
                count = len(data) if isinstance(data, list) else "N/A"
                log(f"  JSON OK: {json_file.name} — entries={count}, size={json_file.stat().st_size / 1e3:.1f} KB")
            except Exception as e:
                log(f"  JSON ERROR: {json_file.name} — {e}")

    # Count image files
    mouse_images = list(ctx.mouse_images_dir.glob("*.jpg"))
    mouse_svgs = list(ctx.mouse_svgs_dir.glob("*.svg"))
    human_images = list(ctx.human_images_dir.glob("*.jpg"))
    log(f"  Mouse images: {len(mouse_images)} files")
    log(f"  Mouse SVGs: {len(mouse_svgs)} files")
    log(f"  Human images: {len(human_images)} files")

    # Total size
    total_size = sum(f.stat().st_size for f in ctx.output_dir.rglob("*") if f.is_file())
    log(f"  Total download size: {total_size / 1e9:.2f} GB")


def print_summary(ctx: DownloadContext) -> None:
    """Print final summary table."""
    log("")
    log("=" * 70)
    log("DOWNLOAD SUMMARY")
    log("=" * 70)
    log(f"{'Item':<45} {'Status':<10} {'Size':>12}")
    log("-" * 70)
    for r in ctx.results:
        size_str = f"{r.size_bytes / 1e6:.1f} MB" if r.size_bytes > 0 else ""
        log(f"{r.name:<45} {r.status:<10} {size_str:>12}")
        if r.error:
            log(f"  Error: {r.error}")
    log("-" * 70)


# --- Main ---


def main(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    ctx = DownloadContext(output_dir=Path(output_dir))
    ctx.create_dirs()

    log(f"Output directory: {ctx.output_dir.resolve()}")
    log("")

    # Steps 5B-5G
    download_ccfv3_volumes(ctx)
    mouse_images = download_mouse_atlas_images(ctx)
    download_mouse_svgs(ctx, mouse_images)
    download_human_atlas(ctx)
    download_ontology(ctx)

    # Step 5H
    verify_downloads(ctx)
    print_summary(ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Allen Brain Institute data locally.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
