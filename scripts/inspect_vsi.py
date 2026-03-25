"""Inspect Olympus VSI files and display series/resolution metadata.

Usage:
    python scripts/inspect_vsi.py /path/to/slides/
    make inspect-vsi IMAGES=/path/to/slides/
"""

import argparse
import sys

from histological_image_analysis.vsi import (
    find_vsi_files,
    inspect_vsi_file,
    validate_bftools,
    validate_java,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Olympus VSI files — show series, dimensions, and pixel sizes.",
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing .vsi files to inspect.",
    )
    parser.add_argument(
        "--bftools-dir",
        default="tools/bftools",
        help="Path to bftools directory (default: tools/bftools).",
    )
    args = parser.parse_args()

    if not validate_java():
        print("ERROR: Java not found. Bio-Formats requires Java 11+.")
        print()
        print("Install Java:")
        print("  macOS:  brew install openjdk@11")
        print("  Ubuntu: sudo apt install openjdk-11-jre")
        print("  Windows: https://adoptium.net/")
        sys.exit(1)

    try:
        from pathlib import Path

        showinf_path, _ = validate_bftools(Path(args.bftools_dir))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    vsi_files = find_vsi_files(args.image_dir)
    if not vsi_files:
        print(f"No .vsi files found in {args.image_dir}")
        sys.exit(0)

    print(f"Found {len(vsi_files)} VSI file(s) in {args.image_dir}")
    print()

    for vsi_path in vsi_files:
        print(f"{'=' * 70}")
        print(f"File: {vsi_path.name}")
        print(f"{'=' * 70}")

        try:
            series = inspect_vsi_file(vsi_path, showinf_path)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print()
            continue
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            print()
            continue

        if not series:
            print("  No series found.")
            print()
            continue

        # Print header
        print(
            f"  {'Series':>6}  {'Width':>8}  {'Height':>8}  "
            f"{'µm/pixel':>10}  {'Channels':>8}  {'Notes'}"
        )
        print(f"  {'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 8}  {'-' * 15}")

        for s in series:
            pixel_size = f"{s.pixel_size_um:.4f}" if s.pixel_size_um else "N/A"
            notes = []
            if s.is_thumbnail:
                notes.append("thumbnail")
            if s.width < 1000 and s.height < 1000:
                notes.append("small (label/macro?)")
            if s.long_side > 10000:
                notes.append("high-res pyramid")
            notes_str = ", ".join(notes) if notes else ""

            print(
                f"  {s.index:>6}  {s.width:>8}  {s.height:>8}  "
                f"{pixel_size:>10}  {s.channels:>8}  {notes_str}"
            )

        print()

    print("To convert, run:")
    print(f"  make convert-vsi IMAGES={args.image_dir} RESOLUTION=10")


if __name__ == "__main__":
    main()
