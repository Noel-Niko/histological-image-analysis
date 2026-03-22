#!/usr/bin/env python3
"""
Annotate histological brain tissue images with detected brain regions.

Produces a single annotated image per input file, showing the original image
with a color overlay of segmented brain regions and a legend of detected structures.
Output files are placed alongside the originals with the naming convention:
    {stem}-annotated-{YYYYMMDDTHHMMSS}.png

Usage:
    # Guided mode (interactive prompts)
    python scripts/annotate.py

    # Direct mode (skip prompts)
    python scripts/annotate.py /path/to/slides/

    # Annotate a single file
    python scripts/annotate.py /path/to/slide.jpg

    # Use human brain model
    python scripts/annotate.py /path/to/slides/ --species human

    # Higher accuracy with sliding window (slower)
    python scripts/annotate.py /path/to/slides/ --sliding-window

    # Force CPU
    python scripts/annotate.py /path/to/slides/ --cpu

    # Custom model path
    python scripts/annotate.py /path/to/slides/ --model ./models/custom
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from histological_image_analysis.annotation import (
    build_annotated_filename,
    create_annotated_overlay,
    resolve_model_path,
)
from histological_image_analysis.inference import (
    SUPPORTED_EXTENSIONS,
    create_device,
    get_image_files,
    load_model,
    run_inference,
    run_sliding_window_inference,
)


SORTED_EXTENSIONS = sorted(SUPPORTED_EXTENSIONS)


def _get_id2label(model) -> dict:
    """Extract id2label mapping from model config, with fallback."""
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        # Normalize keys to int for consistent lookup
        return {int(k): v for k, v in id2label.items()}
    return {}


def _print_guided_header():
    """Print the guided-mode intro with supported file types and preparation info."""
    print("=" * 60)
    print("Brain Region Annotation Tool")
    print("=" * 60)
    print()
    print("This tool annotates histological brain tissue images with")
    print("color-coded brain region overlays. Your original files are")
    print("never modified — new annotated files are created alongside them.")
    print()
    print("BEFORE YOU BEGIN")
    print("-" * 40)
    print("Place your brain tissue images in a single folder.")
    print()
    print("Supported image formats:")
    print(f"  {', '.join(SORTED_EXTENSIONS)}")
    print()
    print("Image requirements:")
    print("  - RGB or grayscale (auto-converted to RGB)")
    print("  - Any resolution (resized internally by the model)")
    print("  - Nissl-stained histological sections work best")
    print()
    print("NOT supported (convert these first):")
    print("  - DICOM (.dcm) — convert to .png or .tiff")
    print("  - NRRD (.nrrd) / NIfTI (.nii.gz) — use volume slicing tools")
    print("  - PDF or SVG — export as raster images first")
    print()
    print("OUTPUT")
    print("-" * 40)
    print("For each input image, an annotated version is saved in the")
    print("SAME folder with the naming convention:")
    print("  {name}-annotated-{timestamp}.png")
    print()
    print("Example:")
    print("  slide_001.jpg  -->  slide_001-annotated-20260322T143052.png")
    print()


def _prompt_for_path() -> str:
    """Prompt the user to enter the path to their images folder."""
    print("Enter the path to your images folder or a single image file.")
    print()
    print("Examples:")
    print("  /Users/yourname/Desktop/brain_slides/")
    print("  /Users/yourname/Documents/research/histology/sample_001.jpg")
    print("  ~/Downloads/slides/")
    print()

    while True:
        try:
            raw = input("Path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)

        if not raw:
            print("  No path entered. Please try again.\n")
            continue

        # Expand ~ to home directory
        expanded = os.path.expanduser(raw)
        path = Path(expanded)

        if not path.exists():
            print(f"  Path not found: {expanded}")
            print("  Please check the path and try again.\n")
            continue

        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(f"  Unsupported file type: {path.suffix}")
                print(f"  Supported: {', '.join(SORTED_EXTENSIONS)}")
                print()
                continue
            return str(path)

        # It's a directory — check it has images
        image_count = sum(
            1 for p in path.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if image_count == 0:
            print(f"  No supported images found in: {expanded}")
            print(f"  Supported: {', '.join(SORTED_EXTENSIONS)}")
            print("  Please check the folder and try again.\n")
            continue

        print(f"  Found {image_count} image(s) in {expanded}")
        return str(path)


def _print_shortcut_reminder(input_path: str, species: str, sliding_window: bool):
    """Print a reminder for direct-mode usage next time."""
    print()
    print("-" * 60)
    print("TIP: Next time, skip the prompts by running:")
    cmd = f"  make annotate IMAGES={input_path}"
    if species == "human":
        cmd = f"  make annotate-human IMAGES={input_path}"
    if sliding_window:
        cmd = f"  make annotate-sliding IMAGES={input_path}"
    print(cmd)
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate brain tissue images with detected brain regions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/annotate.py                              # Guided mode
    python scripts/annotate.py /path/to/slides/             # Direct mode
    python scripts/annotate.py /path/to/slide.jpg --species human
    python scripts/annotate.py /path/to/slides/ --sliding-window

Output:
    Each input image gets a sibling file:
    slide_001.jpg -> slide_001-annotated-20260322T143052.png
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Path to an image file or directory of images (omit for guided mode)",
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["mouse", "human"],
        default="mouse",
        help="Species model to use (default: mouse)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model directory path (overrides --species)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU inference (no GPU)"
    )
    parser.add_argument(
        "--sliding-window", action="store_true",
        help="Use sliding window inference for full-resolution tiled prediction. "
             "Slower but more accurate at image edges.",
    )
    parser.add_argument(
        "--stride", type=int, default=259,
        help="Stride for sliding window inference (default: 259, i.e. 50%% overlap)",
    )

    args = parser.parse_args()

    # Guided mode: no input path provided
    guided = args.input is None
    if guided:
        _print_guided_header()
        input_str = _prompt_for_path()
    else:
        input_str = args.input

    # Resolve input path(s)
    input_path = Path(input_str)
    if not input_path.exists():
        print(f"ERROR: Path not found: {input_str}")
        sys.exit(1)

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"ERROR: Unsupported image format: {input_path.suffix}")
            print(f"Supported: {', '.join(SORTED_EXTENSIONS)}")
            sys.exit(1)
        image_files = [str(input_path)]
    else:
        image_files = get_image_files(str(input_path))

        # Warn about files that will be skipped
        all_files = [
            p for p in input_path.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ]
        skipped = [
            p for p in all_files
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS
        ]
        if skipped:
            print()
            print(f"WARNING: {len(skipped)} file(s) will NOT be processed (unsupported format):")
            for p in sorted(skipped):
                print(f"  SKIP  {p.name}  ({p.suffix})")
            print(f"\nSupported formats: {', '.join(SORTED_EXTENSIONS)}")
            print()

    # Resolve model path
    model_dir = resolve_model_path(args.species, model_path=args.model)

    # Set up device
    device = create_device(force_cpu=args.cpu)
    if device.type == "cuda":
        import torch
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for inference")

    # Load model
    print(f"Loading {args.species} model from {model_dir}...")
    model, processor = load_model(model_dir, device)
    id2label = _get_id2label(model)
    num_labels = model.config.num_labels
    print(f"Loaded model ({num_labels} classes, {len(id2label)} named regions)")

    # Process images
    mode = "sliding window" if args.sliding_window else "center-crop"
    print(f"\nAnnotating {len(image_files)} image(s) [{mode}]...")
    print("=" * 60)

    for image_path in tqdm(image_files, desc="Annotating"):
        print(f"\n{Path(image_path).name}")

        # Run inference
        if args.sliding_window:
            prediction, resized_prediction = run_sliding_window_inference(
                image_path, model, device, stride=args.stride,
            )
        else:
            prediction, resized_prediction = run_inference(
                image_path, model, processor, device,
            )

        if prediction is None:
            print(f"  Skipped (failed to load)")
            continue

        # Load original image
        original = np.array(Image.open(image_path).convert("RGB"))

        # Use resized prediction (matches original image dimensions)
        overlay = create_annotated_overlay(original, resized_prediction, id2label)

        # Save annotated image alongside original
        output_path = build_annotated_filename(image_path)
        overlay.save(output_path)

        unique_classes = len(np.unique(resized_prediction)) - (1 if 0 in resized_prediction else 0)
        print(f"  Saved: {output_path}")
        print(f"  Detected {unique_classes} brain regions")

    print("\n" + "=" * 60)
    print("Annotation complete!")
    print(f"  Processed {len(image_files)} image(s)")
    print("  Annotated files saved alongside originals")

    # Show shortcut reminder only in guided mode
    if guided:
        _print_shortcut_reminder(input_str, args.species, args.sliding_window)


if __name__ == "__main__":
    main()
