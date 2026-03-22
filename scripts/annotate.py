#!/usr/bin/env python3
"""
Annotate histological brain tissue images with detected brain regions.

Produces a single annotated image per input file, showing the original image
with a color overlay of segmented brain regions and a legend of detected structures.
Output files are placed alongside the originals with the naming convention:
    {stem}-annotated-{YYYYMMDDTHHMMSS}.png

Usage:
    # Annotate a directory of mouse brain slides (default)
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


def _get_id2label(model) -> dict:
    """Extract id2label mapping from model config, with fallback."""
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        # Normalize keys to int for consistent lookup
        return {int(k): v for k, v in id2label.items()}
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Annotate brain tissue images with detected brain regions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/annotate.py /path/to/slides/
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
        help="Path to an image file or directory of images",
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

    # Resolve input path(s)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Path not found: {args.input}")
        sys.exit(1)

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"ERROR: Unsupported image format: {input_path.suffix}")
            print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            sys.exit(1)
        image_files = [str(input_path)]
    else:
        image_files = get_image_files(str(input_path))

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


if __name__ == "__main__":
    main()
