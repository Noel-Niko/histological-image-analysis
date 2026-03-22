"""Shared inference functions for brain region segmentation.

Provides model loading, single-image inference, sliding-window inference,
and utility functions shared by scripts/run_inference.py and scripts/annotate.py.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 518

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def create_device(force_cpu: bool = False) -> torch.device:
    """Select computation device (CPU or CUDA)."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_path: str, device: torch.device
) -> Tuple[UperNetForSemanticSegmentation, AutoImageProcessor]:
    """Load the trained model and image processor.

    Parameters
    ----------
    model_path : str
        Path to model directory containing config.json, preprocessor_config.json,
        and model weights (model.safetensors or pytorch_model.bin).
    device : torch.device
        Device to move the model to.

    Returns
    -------
    tuple of (model, processor)

    Raises
    ------
    SystemExit
        If the model directory is missing or incomplete.
    """
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("\nPlease download the model first:")
        print("  make download-models")
        print("\nOr see: docs/model_download_guide.md")
        sys.exit(1)

    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load image processor: {e}")
        print("\nThe model directory may be incomplete.")
        print("See docs/model_download_guide.md for troubleshooting.")
        sys.exit(1)

    try:
        model = UperNetForSemanticSegmentation.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    return model, processor


def run_inference(
    image_path: str,
    model: UperNetForSemanticSegmentation,
    processor: AutoImageProcessor,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run inference on a single image.

    Parameters
    ----------
    image_path : str
        Path to input image.
    model : UperNetForSemanticSegmentation
        Loaded model.
    processor : AutoImageProcessor
        Loaded image processor.
    device : torch.device
        Device to run on.

    Returns
    -------
    tuple of (prediction, resized_prediction)
        prediction: (H, W) array of class IDs at model resolution.
        resized_prediction: (H, W) array of class IDs at original image size.
        Both are None if image loading fails.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
    except Exception as e:
        print(f"ERROR: Failed to load image {image_path}: {e}")
        return None, None

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    prediction = logits.argmax(dim=1)[0]
    prediction_np = prediction.cpu().numpy()

    prediction_pil = Image.fromarray(prediction_np.astype(np.uint16))
    resized_prediction_pil = prediction_pil.resize(
        original_size, resample=Image.NEAREST
    )
    resized_prediction = np.array(resized_prediction_pil)

    return prediction_np, resized_prediction


def _normalize_tile(tile: np.ndarray) -> torch.Tensor:
    """Convert uint8 grayscale or RGB (H, W, 3) to float32 tensor (1, 3, H, W)."""
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

    Parameters
    ----------
    image_path : str
        Path to input image.
    model : UperNetForSemanticSegmentation
        Loaded model.
    device : torch.device
        Device to run on.
    stride : int
        Stride between tiles (default 259 = 50% overlap).

    Returns
    -------
    tuple of (prediction, prediction)
        Both are the same full-resolution (H, W) array of class IDs,
        or (None, None) if image loading fails.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ERROR: Failed to load image {image_path}: {e}")
        return None, None

    img_np = np.array(image)
    h, w = img_np.shape[:2]
    num_labels = model.config.num_labels
    crop_size = CROP_SIZE

    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        img_np = np.pad(
            img_np, ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant", constant_values=0,
        )
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

    count_map = np.maximum(count_map, 1.0)
    avg_logits = logit_sum / count_map[np.newaxis, :, :]
    prediction = avg_logits.argmax(axis=0).astype(np.int64)[:h, :w]

    return prediction, prediction


def get_image_files(image_dir: str) -> List[str]:
    """Get all image files from a directory, sorted by name.

    Parameters
    ----------
    image_dir : str
        Path to directory containing images.

    Returns
    -------
    list of str
        Sorted list of absolute image file paths.

    Raises
    ------
    SystemExit
        If directory doesn't exist or contains no images.
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        print(f"ERROR: Directory not found: {image_dir}")
        sys.exit(1)

    image_files = [
        str(p) for p in image_dir_path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"ERROR: No image files found in {image_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    return sorted(image_files)
