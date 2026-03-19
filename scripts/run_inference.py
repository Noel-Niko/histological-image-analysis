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
    ✓ CAN RUN ON LAPTOP (CPU or GPU)

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
from typing import Optional, List, Tuple

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import UperNetForSemanticSegmentation, AutoImageProcessor
from tqdm import tqdm


def load_model(
    model_path: str, device: torch.device
) -> Tuple[UperNetForSemanticSegmentation, AutoImageProcessor]:
    """Load the trained model and image processor."""
    print(f"Loading model from {model_path}...")

    # Check model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print(f"\nPlease download the model first:")
        print(f"  databricks fs cp -r dbfs:/FileStore/allen_brain_data/models/final-200ep {model_path}")
        print(f"\nOr see: docs/model_download_guide.md")
        sys.exit(1)

    # Load processor
    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
        print(f"✓ Loaded image processor (size: {processor.size})")
    except Exception as e:
        print(f"ERROR: Failed to load image processor: {e}")
        print(f"\nThe model directory may be incomplete.")
        print(f"See docs/model_download_guide.md for troubleshooting.")
        sys.exit(1)

    # Load model
    try:
        model = UperNetForSemanticSegmentation.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Loaded model ({num_params:,} parameters, {model.config.num_labels} classes)")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    return model, processor


def run_inference(
    image_path: str,
    model: UperNetForSemanticSegmentation,
    processor: AutoImageProcessor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single image.

    Args:
        image_path: Path to input image
        model: Loaded model
        processor: Loaded image processor
        device: Device to run on

    Returns:
        Tuple of (prediction, resized_prediction)
        - prediction: (H, W) array of class IDs at model resolution (518x518)
        - resized_prediction: (H, W) array of class IDs at original image size
    """
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
    except Exception as e:
        print(f"ERROR: Failed to load image {image_path}: {e}")
        return None, None

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [1, num_labels, H, W]

    # Get predicted class for each pixel
    prediction = logits.argmax(dim=1)[0]  # Shape: [H, W]
    prediction_np = prediction.cpu().numpy()

    # Resize prediction to original image size
    prediction_pil = Image.fromarray(prediction_np.astype(np.uint16))
    resized_prediction_pil = prediction_pil.resize(
        original_size, resample=Image.NEAREST
    )
    resized_prediction = np.array(resized_prediction_pil)

    return prediction_np, resized_prediction


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 518


def _normalize_tile(tile: np.ndarray) -> torch.Tensor:
    """uint8 grayscale or RGB (H, W, 3) -> float32 tensor (1, 3, H, W)."""
    if tile.ndim == 2:
        img = tile.astype(np.float32) / 255.0
        img_3ch = np.stack([img, img, img], axis=0)
    else:
        img_3ch = tile.astype(np.float32).transpose(2, 0, 1) / 255.0
    for c in range(3):
        img_3ch[c] = (img_3ch[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return torch.from_numpy(img_3ch).unsqueeze(0)


def run_sliding_window_inference(
    image_path: str,
    model: UperNetForSemanticSegmentation,
    device: torch.device,
    stride: int = 259,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run sliding window inference for full-resolution tiled prediction.

    Tiles the image with 518x518 windows at the given stride, averages logits
    in overlapping regions, then argmax. Returns full-resolution prediction.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
    except Exception as e:
        print(f"ERROR: Failed to load image {image_path}: {e}")
        return None, None

    img_np = np.array(image)  # (H, W, 3)
    h, w = img_np.shape[:2]
    num_labels = model.config.num_labels
    crop_size = CROP_SIZE

    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant", constant_values=0)
    ph, pw = img_np.shape[:2]

    logit_sum = np.zeros((num_labels, ph, pw), dtype=np.float32)
    count_map = np.zeros((ph, pw), dtype=np.float32)

    y_starts = list(range(0, ph - crop_size + 1, stride))
    x_starts = list(range(0, pw - crop_size + 1, stride))
    if y_starts[-1] + crop_size < ph:
        y_starts.append(ph - crop_size)
    if x_starts[-1] + crop_size < pw:
        x_starts.append(pw - crop_size)

    model.eval()
    use_cuda = device.type == "cuda"
    n_tiles = len(y_starts) * len(x_starts)
    tile_idx = 0
    for y in y_starts:
        for x in x_starts:
            tile = img_np[y:y + crop_size, x:x + crop_size]
            pixel_values = _normalize_tile(tile).to(device)
            with torch.no_grad():
                if use_cuda:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(pixel_values=pixel_values).logits
                else:
                    logits = model(pixel_values=pixel_values).logits
            tile_logits = logits.squeeze(0).float().cpu().numpy()
            logit_sum[:, y:y + crop_size, x:x + crop_size] += tile_logits
            count_map[y:y + crop_size, x:x + crop_size] += 1.0
            tile_idx += 1

    count_map = np.maximum(count_map, 1.0)
    avg_logits = logit_sum / count_map[np.newaxis, :, :]
    prediction = avg_logits.argmax(axis=0).astype(np.int64)[:h, :w]

    return prediction, prediction


def save_results(
    image_path: str,
    prediction: np.ndarray,
    resized_prediction: np.ndarray,
    output_dir: str,
    num_labels: int,
):
    """
    Save segmentation results.

    Saves:
    1. Raw segmentation mask (PNG, uint16) at model resolution
    2. Resized segmentation mask (PNG, uint16) at original image size
    3. Visualization overlay (PNG) with input and segmentation side-by-side
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename
    base_name = Path(image_path).stem

    # Load original image for visualization
    original_image = Image.open(image_path).convert("RGB")

    # 1. Save raw mask (model resolution 518x518)
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    Image.fromarray(prediction.astype(np.uint16)).save(mask_path)

    # 2. Save resized mask (original resolution)
    resized_mask_path = os.path.join(output_dir, f"{base_name}_mask_resized.png")
    Image.fromarray(resized_prediction.astype(np.uint16)).save(resized_mask_path)

    # 3. Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")

    # Segmentation at model resolution
    im1 = axes[1].imshow(prediction, cmap="nipy_spectral", vmin=0, vmax=num_labels)
    axes[1].set_title(
        f"Segmentation (518×518)\n{len(np.unique(prediction))} classes detected",
        fontsize=14,
    )
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Segmentation at original resolution
    im2 = axes[2].imshow(resized_prediction, cmap="nipy_spectral", vmin=0, vmax=num_labels)
    axes[2].set_title(
        f"Segmentation (original size)\n{original_image.size[0]}×{original_image.size[1]}",
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

    # Print statistics
    unique_classes = len(np.unique(resized_prediction))
    print(f"  ✓ Saved: {mask_path}")
    print(f"  ✓ Saved: {resized_mask_path}")
    print(f"  ✓ Saved: {viz_path}")
    print(f"  → Detected {unique_classes} / {num_labels} possible classes")


def get_image_files(image_dir: str) -> List[str]:
    """Get all image files from a directory."""
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        print(f"ERROR: Directory not found: {image_dir}")
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_files = [
        str(p) for p in image_dir_path.iterdir() if p.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"ERROR: No image files found in {image_dir}")
        print(f"Supported formats: {', '.join(extensions)}")
        sys.exit(1)

    return sorted(image_files)


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

    # Validate arguments
    if not args.image and not args.image_dir:
        parser.error("Must specify either --image or --image-dir")

    if args.image and args.image_dir:
        parser.error("Cannot specify both --image and --image-dir")

    # Set device
    if args.cpu:
        device = torch.device("cpu")
        print("Using CPU for inference (--cpu flag specified)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    # Load model
    model, processor = load_model(args.model, device)

    # Get list of images to process
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

    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        print(f"\n{Path(image_path).name}")

        # Run inference
        if args.sliding_window:
            prediction, resized_prediction = run_sliding_window_inference(
                image_path, model, device, stride=args.stride,
            )
        else:
            prediction, resized_prediction = run_inference(
                image_path, model, processor, device
            )

        if prediction is None:
            print(f"  ✗ Skipped (failed to load)")
            continue

        # Save results
        save_results(
            image_path,
            prediction,
            resized_prediction,
            args.output,
            model.config.num_labels,
        )

    print("\n" + "=" * 60)
    print(f"✓ Processing complete!")
    print(f"  Results saved to: {args.output}")
    print(f"\nOutput files for each image:")
    print(f"  - <name>_mask.png              : Segmentation at model resolution (518×518)")
    print(f"  - <name>_mask_resized.png      : Segmentation at original image size")
    print(f"  - <name>_visualization.png     : Side-by-side comparison")


if __name__ == "__main__":
    main()
