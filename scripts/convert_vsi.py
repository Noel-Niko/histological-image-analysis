"""Convert Olympus VSI files to TIFF at a target resolution.

Usage:
    python scripts/convert_vsi.py /path/to/slides/ --resolution 10
    make convert-vsi IMAGES=/path/to/slides/ RESOLUTION=10
"""

import argparse
import sys

from pathlib import Path

from histological_image_analysis.vsi import (
    DEFAULT_TARGET_UM_PER_PIXEL,
    build_output_path,
    convert_vsi_file,
    find_vsi_files,
    validate_bftools,
    validate_java,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Olympus VSI files to TIFF at a target resolution. "
            "Automatically selects the pyramid level closest to the target µm/pixel."
        ),
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing .vsi files to convert.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_TARGET_UM_PER_PIXEL,
        help=(
            f"Target resolution in µm/pixel (default: {DEFAULT_TARGET_UM_PER_PIXEL}). "
            "Lower values = higher resolution, larger files. "
            "Training data uses ~10 µm/pixel (mouse) to ~200 µm/pixel (BigBrain)."
        ),
    )
    parser.add_argument(
        "--bftools-dir",
        default="tools/bftools",
        help="Path to bftools directory (default: tools/bftools).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing converted files.",
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
        showinf_path, bfconvert_path = validate_bftools(Path(args.bftools_dir))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    vsi_files = find_vsi_files(args.image_dir)
    if not vsi_files:
        print(f"No .vsi files found in {args.image_dir}")
        sys.exit(0)

    print(f"Found {len(vsi_files)} VSI file(s) in {args.image_dir}")
    print(f"Target resolution: {args.resolution} µm/pixel")
    print()

    converted = 0
    skipped = 0
    errors = 0

    for vsi_path in vsi_files:
        output_path = build_output_path(vsi_path)
        print(f"  {vsi_path.name} → {output_path.name} ... ", end="", flush=True)

        if output_path.exists() and not args.force:
            print("SKIPPED (already exists)")
            skipped += 1
            continue

        # Remove existing file if --force
        if output_path.exists() and args.force:
            output_path.unlink()

        try:
            result = convert_vsi_file(
                vsi_path=vsi_path,
                bfconvert_path=bfconvert_path,
                showinf_path=showinf_path,
                output_path=output_path,
                target_um_per_pixel=args.resolution,
            )

            if result.skipped:
                print("SKIPPED (already exists)")
                skipped += 1
            else:
                pixel_info = (
                    f"{result.pixel_size_um:.2f} µm/px"
                    if result.pixel_size_um
                    else "unknown µm/px"
                )
                print(
                    f"OK (series {result.series_index}, "
                    f"{result.input_width}x{result.input_height}, "
                    f"{pixel_info})"
                )
                converted += 1

        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            errors += 1
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {e}")
            errors += 1

    print()
    print(f"Results: {converted} converted, {skipped} skipped, {errors} errors")

    if converted > 0:
        print()
        print("Next: annotate the converted TIFFs:")
        print(f"  make annotate-mouse IMAGES={args.image_dir}")


if __name__ == "__main__":
    main()
