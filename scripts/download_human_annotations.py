"""Download human brain annotation data locally for Workspace upload.

This script downloads human-specific annotation data that complements
the existing human Nissl images. See docs/data_download_plan_human.md.

Usage:
    python scripts/download_human_annotations.py [--output-dir data/allen_brain_data]
    python scripts/download_human_annotations.py --step svgs
    python scripts/download_human_annotations.py --step developing-atlas
    python scripts/download_human_annotations.py --step all

Data items downloaded:
    1. Human SectionImage SVGs (4,463 annotated sections)
    2. Developing Human 21 pcw Atlas (169 images + 169 SVGs + metadata)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


# --- Constants ---

ALLEN_API_BASE = "https://api.brain-map.org/api/v2"
REQUEST_TIMEOUT = 60
IMAGE_DOWNLOAD_DELAY = 0.1  # seconds between API requests
DEFAULT_OUTPUT_DIR = "data/allen_brain_data"


# --- Helpers ---


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_json(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success. Skips if exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        if resp.status_code == 200 and len(resp.content) > 0:
            dest.write_bytes(resp.content)
            return True
        return False
    except Exception:
        return False


def download_svg(url: str, dest: Path) -> bool:
    """Download an SVG file. Returns True if valid SVG content received."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200 and resp.text.strip().startswith("<"):
            dest.write_text(resp.text)
            return True
        return False
    except Exception:
        return False


# --- Download steps ---


def download_human_svgs(output_dir: Path) -> None:
    """Download SVG annotations for annotated human SectionImages."""
    log("=== Downloading human SectionImage SVGs ===")

    metadata_path = output_dir / "metadata" / "human_atlas_images_metadata.json"
    svgs_dir = output_dir / "human_atlas" / "svgs"
    svgs_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        metadata = json.load(f)

    annotated = [m for m in metadata if m.get("annotated") is True]
    total = len(annotated)
    log(f"  Found {total} annotated images out of {len(metadata)} total")

    downloaded = 0
    skipped = 0
    errors = 0

    for i, img in enumerate(annotated):
        img_id = img["id"]
        section_num = img.get("section_number", 0)
        donor = img.get("_donor", "unknown")
        dest = svgs_dir / f"{donor}_{section_num:04d}_{img_id}.svg"

        url = f"{ALLEN_API_BASE}/svg_download/{img_id}"

        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
        elif download_svg(url, dest):
            downloaded += 1
        else:
            errors += 1

        if (i + 1) % 100 == 0:
            log(
                f"  Progress: {i + 1}/{total} "
                f"(downloaded={downloaded}, skipped={skipped}, errors={errors})"
            )

        time.sleep(IMAGE_DOWNLOAD_DELAY)

    log(
        f"  Done: {downloaded} downloaded, {skipped} skipped, "
        f"{errors} errors out of {total}"
    )

    # Quick verification
    svg_files = list(svgs_dir.glob("*.svg"))
    total_size = sum(f.stat().st_size for f in svg_files)
    log(f"  SVG directory: {len(svg_files)} files, {total_size / 1e6:.1f} MB")


def download_developing_atlas(output_dir: Path) -> None:
    """Download 21 pcw developing human atlas images + SVGs."""
    log("=== Downloading 21 pcw developing human atlas ===")

    images_dir = output_dir / "developing_human_atlas" / "images"
    svgs_dir = output_dir / "developing_human_atlas" / "svgs"
    metadata_path = output_dir / "metadata" / "developing_human_atlas_metadata.json"
    images_dir.mkdir(parents=True, exist_ok=True)
    svgs_dir.mkdir(parents=True, exist_ok=True)

    # Query for atlas images (Atlas ID 3 = 21 pcw developing human)
    if metadata_path.exists():
        log("  Loading cached metadata...")
        with open(metadata_path) as f:
            images = json.load(f)
    else:
        log("  Querying Allen API for developing human atlas images...")
        url = f"{ALLEN_API_BASE}/data/query.json"
        params = {
            "criteria": (
                "model::AtlasImage,"
                "rma::criteria,atlas_data_set(atlases[id$eq3]),"
                "rma::options[num_rows$eqall][order$eq'section_number']"
            )
        }
        data = fetch_json(url, params)
        images = data["msg"]
        with open(metadata_path, "w") as f:
            json.dump(images, f, indent=2)
        log(f"  Found {len(images)} atlas images, saved metadata.")

    total = len(images)
    log(f"  Downloading {total} atlas images + SVGs...")

    img_downloaded = 0
    img_skipped = 0
    img_errors = 0
    svg_downloaded = 0
    svg_skipped = 0
    svg_errors = 0

    for i, img in enumerate(images):
        img_id = img["id"]
        section_num = img.get("section_number", 0)

        # Download atlas image
        img_dest = images_dir / f"{section_num:04d}_{img_id}.jpg"
        img_url = f"{ALLEN_API_BASE}/atlas_image_download/{img_id}?downsample=4"

        if img_dest.exists() and img_dest.stat().st_size > 0:
            img_skipped += 1
        elif download_file(img_url, img_dest):
            img_downloaded += 1
        else:
            img_errors += 1

        # Download SVG annotation
        svg_dest = svgs_dir / f"{section_num:04d}_{img_id}.svg"
        svg_url = f"{ALLEN_API_BASE}/svg_download/{img_id}"

        if svg_dest.exists() and svg_dest.stat().st_size > 0:
            svg_skipped += 1
        elif download_svg(svg_url, svg_dest):
            svg_downloaded += 1
        else:
            svg_errors += 1

        if (i + 1) % 25 == 0:
            log(
                f"  Progress: {i + 1}/{total} — "
                f"images({img_downloaded}+{img_skipped}s+{img_errors}e) "
                f"svgs({svg_downloaded}+{svg_skipped}s+{svg_errors}e)"
            )

        time.sleep(IMAGE_DOWNLOAD_DELAY)

    log(
        f"  Images: {img_downloaded} downloaded, {img_skipped} skipped, "
        f"{img_errors} errors"
    )
    log(
        f"  SVGs: {svg_downloaded} downloaded, {svg_skipped} skipped, "
        f"{svg_errors} errors"
    )


def verify_human_data(output_dir: Path) -> None:
    """Verify all human annotation downloads."""
    log("=== Verification ===")

    # SVGs
    svgs_dir = output_dir / "human_atlas" / "svgs"
    svg_files = list(svgs_dir.glob("*.svg"))
    svg_size = sum(f.stat().st_size for f in svg_files) if svg_files else 0
    log(f"  Human SVGs: {len(svg_files)} files, {svg_size / 1e6:.1f} MB")

    # Check a sample SVG for structure_id attributes
    if svg_files:
        sample = svg_files[0].read_text()[:2000]
        has_structure = "structure_id" in sample
        log(f"  SVG contains structure_id: {has_structure}")

    # Developing atlas
    dev_imgs = list((output_dir / "developing_human_atlas" / "images").glob("*.jpg"))
    dev_svgs = list((output_dir / "developing_human_atlas" / "svgs").glob("*.svg"))
    log(f"  Developing atlas images: {len(dev_imgs)} files")
    log(f"  Developing atlas SVGs: {len(dev_svgs)} files")

    # Ontologies
    for graph_id in [10, 16]:
        ont_path = output_dir / "ontology" / f"structure_graph_{graph_id}.json"
        if ont_path.exists():
            log(f"  Ontology graph {graph_id}: {ont_path.stat().st_size / 1e3:.0f} KB")
        else:
            log(f"  Ontology graph {graph_id}: MISSING")

    # Total size
    total_size = sum(
        f.stat().st_size
        for f in output_dir.rglob("*")
        if f.is_file() and "ccfv3" not in str(f) and "mouse_atlas" not in str(f)
    )
    log(f"  Total human annotation data: {total_size / 1e6:.1f} MB")


# --- Main ---


def main(output_dir: str = DEFAULT_OUTPUT_DIR, step: str = "all") -> None:
    output_path = Path(output_dir)
    log(f"Output directory: {output_path.resolve()}")
    log(f"Step: {step}")
    log("")

    if step in ("all", "svgs"):
        download_human_svgs(output_path)

    if step in ("all", "developing-atlas"):
        download_developing_atlas(output_path)

    if step in ("all", "verify"):
        verify_human_data(output_path)

    log("")
    log("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download human brain annotation data."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "svgs", "developing-atlas", "verify"],
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, step=args.step)
