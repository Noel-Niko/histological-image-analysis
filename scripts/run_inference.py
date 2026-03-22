#!/usr/bin/env python3
"""
Run inference on histological brain images using trained DINOv2-UperNet model.

This script allows PhD researchers to test the model on their own brain tissue images.
Supports batch processing, GPU acceleration, and saves both segmentation masks and
visualization overlays.

Usage:
    # Single image
    python scripts/run_inference.py --image path/to/brain_slice.jpg --output results/

    # Batch processing (directory)
    python scripts/run_inference.py --image-dir images/ --output results/

    # Force CPU (no GPU)
    python scripts/run_inference.py --image path/to/image.png --output results/ --cpu

    # Custom model path
    python scripts/run_inference.py --image image.jpg --model ./models/custom-model --output results/

Requirements:
    - Model downloaded to ./models/dinov2-upernet-final/
    - See docs/model_download_guide.md for download instructions

    # Sliding window (full-resolution tiled inference)
    python scripts/run_inference.py --image image.jpg --output results/ --sliding-window

Compute Requirements:
    CAN RUN ON LAPTOP (CPU or GPU)

    Minimum:
    - RAM: 8 GB (16 GB recommended)
    - Disk: 2 GB free (for model + results)
    - CPU: Any modern processor (2+ cores)

    Performance:
    - CPU only: ~10-30 seconds per image (depends on CPU)
    - Laptop GPU (NVIDIA MX/GTX series): ~2-5 seconds per image
    - Desktop GPU (RTX 3060+): ~1-2 seconds per image

    The model (342M parameters, ~1.2 GB) fits comfortably in laptop RAM.
    GPU acceleration is optional but recommended for batch processing.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from histological_image_analysis.inference import (
    CROP_SIZE,
    create_device,
    get_image_files,
    load_model,
    run_inference,
    run_sliding_window_inference,
)


def save_results(
    image_path: str,
    prediction: np.ndarray,
    resized_prediction: np.ndarray,
    output_dir: str,
    num_labels: int,
):
    """Save segmentation results: masks + visualization overlay."""
    os.makedirs(output_dir, exist_ok=True)

    base_name = Path(image_path).stem
    original_image = Image.open(image_path).convert("RGB")

    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    Image.fromarray(prediction.astype(np.uint16)).save(mask_path)

    resized_mask_path = os.path.join(output_dir, f"{base_name}_mask_resized.png")
    Image.fromarray(resized_prediction.astype(np.uint16)).save(resized_mask_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original_image)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")

    im1 = axes[1].imshow(prediction, cmap="nipy_spectral", vmin=0, vmax=num_labels)
    axes[1].set_title(
        f"Segmentation (518x518)\n{len(np.unique(prediction))} classes detected",
        fontsize=14,
    )
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(resized_prediction, cmap="nipy_spectral", vmin=0, vmax=num_labels)
    axes[2].set_title(
        f"Segmentation (original size)\n{original_image.size[0]}x{original_image.size[1]}",
        fontsize=14,
    )
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Brain Region Segmentation: {Path(image_path).name}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    viz_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()

    unique_classes = len(np.unique(resized_prediction))
    print(f"  Saved: {mask_path}")
    print(f"  Saved: {resized_mask_path}")
    print(f"  Saved: {viz_path}")
    print(f"  Detected {unique_classes} / {num_labels} possible classes")


def main():
    parser = argparse.ArgumentParser(
        description="Run brain region segmentation inference on histological images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image
    python scripts/run_inference.py --image path/to/brain_slice.jpg --output results/

    # Batch processing
    python scripts/run_inference.py --image-dir images/ --output results/

    # Force CPU
    python scripts/run_inference.py --image image.png --output results/ --cpu

For more information, see: docs/model_download_guide.md
        """,
    )

    parser.add_argument(
        "--image", type=str, help="Path to a single input image"
    )
    parser.add_argument(
        "--image-dir", type=str, help="Path to directory containing images"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./inference_results",
        help="Output directory for results (default: ./inference_results)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="./models/dinov2-upernet-final",
        help="Path to model directory (default: ./models/dinov2-upernet-final)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU inference (no GPU)"
    )
    parser.add_argument(
        "--sliding-window", action="store_true",
        help="Use sliding window inference for full-resolution tiled prediction "
             "(518x518 tiles, stride 259, 50%% overlap). Slower but captures "
             "structures at image edges that center-crop misses.",
    )
    parser.add_argument(
        "--stride", type=int, default=259,
        help="Stride for sliding window inference (default: 259, i.e. 50%% overlap)",
    )

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Must specify either --image or --image-dir")

    if args.image and args.image_dir:
        parser.error("Cannot specify both --image and --image-dir")

    device = create_device(force_cpu=args.cpu)
    if device.type == "cuda":
        import torch
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        reason = "--cpu flag specified" if args.cpu else "GPU not available"
        print(f"Using CPU for inference ({reason})")

    print(f"Loading model from {args.model}...")
    model, processor = load_model(args.model, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({num_params:,} parameters, {model.config.num_labels} classes)")

    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            sys.exit(1)
        image_files = [args.image]
    else:
        image_files = get_image_files(args.image_dir)

    mode = "sliding window" if args.sliding_window else "center-crop"
    print(f"\nProcessing {len(image_files)} image(s) [{mode}]...")
    print(f"Output directory: {args.output}")
    if args.sliding_window:
        print(f"Sliding window: {CROP_SIZE}x{CROP_SIZE} tiles, stride {args.stride}")
    print("=" * 60)

    for image_path in tqdm(image_files, desc="Processing images"):
        print(f"\n{Path(image_path).name}")

        if args.sliding_window:
            prediction, resized_prediction = run_sliding_window_inference(
                image_path, model, device, stride=args.stride,
            )
        else:
            prediction, resized_prediction = run_inference(
                image_path, model, processor, device
            )

        if prediction is None:
            print(f"  Skipped (failed to load)")
            continue

        save_results(
            image_path,
            prediction,
            resized_prediction,
            args.output,
            model.config.num_labels,
        )

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"  Results saved to: {args.output}")
    print("\nOutput files for each image:")
    print("  - <name>_mask.png              : Segmentation at model resolution (518x518)")
    print("  - <name>_mask_resized.png      : Segmentation at original image size")
    print("  - <name>_visualization.png     : Side-by-side comparison")


if __name__ == "__main__":
    main()
