"""PyTorch Dataset for brain structure segmentation.

Wraps CCFv3Slicer to provide (pixel_values, labels) pairs suitable
for DINOv2 + UperNet semantic segmentation training.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from histological_image_analysis.ccfv3_slicer import CCFv3Slicer

logger = logging.getLogger(__name__)

# DINOv2 pretrained with ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BrainSegmentationDataset(Dataset):
    """PyTorch Dataset for brain structure segmentation.

    Pre-loads all slices from the slicer into memory, then applies
    padding, cropping, augmentation, and normalization on the fly.

    Parameters
    ----------
    slicer : CCFv3Slicer
        Volume slicer providing (image, mask) pairs.
    split : str
        One of "train", "val", "test".
    mapping : dict
        Structure ID → class ID mapping from OntologyMapper.
    crop_size : int
        Output spatial size (default 518 for DINOv2).
    augment : bool
        Whether to apply random augmentation (training only).
    """

    def __init__(
        self,
        slicer: CCFv3Slicer,
        split: str,
        mapping: dict[int, int],
        crop_size: int = 518,
        augment: bool = True,
    ) -> None:
        self._crop_size = crop_size
        self._augment = augment

        # Pre-load all slices into memory
        self._slices: list[tuple[np.ndarray, np.ndarray]] = list(
            slicer.iter_slices(split, mapping)
        )
        logger.info(
            "Loaded %d %s slices (crop_size=%d, augment=%s)",
            len(self._slices),
            split,
            crop_size,
            augment,
        )

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, mask = self._slices[idx]

        # Pad if needed (fill with 0)
        image, mask = self._pad_if_needed(image, mask)

        # Crop
        if self._augment:
            image, mask = self._random_crop(image, mask)
        else:
            image, mask = self._center_crop(image, mask)

        # Augmentation (training only)
        if self._augment:
            image, mask = self._apply_augmentation(image, mask)

        # Convert grayscale → 3-channel, normalize
        pixel_values = self._normalize(image)

        return {
            "pixel_values": pixel_values,
            "labels": torch.from_numpy(mask.copy()).long(),
        }

    def _pad_if_needed(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad image and mask to at least crop_size in both dimensions."""
        h, w = image.shape[:2]
        pad_h = max(0, self._crop_size - h)
        pad_w = max(0, self._crop_size - w)

        if pad_h == 0 and pad_w == 0:
            return image, mask

        # Pad symmetrically, fill with 0
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        mask = np.pad(
            mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        return image, mask

    def _random_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random crop to crop_size × crop_size."""
        h, w = image.shape[:2]
        top = np.random.randint(0, h - self._crop_size + 1)
        left = np.random.randint(0, w - self._crop_size + 1)
        return (
            image[top : top + self._crop_size, left : left + self._crop_size],
            mask[top : top + self._crop_size, left : left + self._crop_size],
        )

    def _center_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center crop to crop_size × crop_size."""
        h, w = image.shape[:2]
        top = (h - self._crop_size) // 2
        left = (w - self._crop_size) // 2
        return (
            image[top : top + self._crop_size, left : left + self._crop_size],
            mask[top : top + self._crop_size, left : left + self._crop_size],
        )

    def _apply_augmentation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to image and mask.

        - Horizontal flip (50% probability)
        - Color jitter on image only (brightness ±0.2, contrast ±0.2)
        """
        # Horizontal flip
        if np.random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Color jitter (image only)
        image = image.astype(np.float32)
        brightness = 1.0 + np.random.uniform(-0.2, 0.2)
        contrast = 1.0 + np.random.uniform(-0.2, 0.2)
        image = image * contrast + (brightness - 1.0) * 128
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image, mask

    def _normalize(self, image: np.ndarray) -> torch.Tensor:
        """Convert uint8 grayscale to normalized 3-channel float tensor.

        1. uint8 [0, 255] → float32 [0, 1]
        2. Replicate to 3 channels
        3. Apply ImageNet normalization
        """
        # To float [0, 1]
        img_float = image.astype(np.float32) / 255.0

        # Stack to 3 channels: (H, W) → (3, H, W)
        img_3ch = np.stack([img_float, img_float, img_float], axis=0)

        # ImageNet normalize each channel
        for c in range(3):
            img_3ch[c] = (img_3ch[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        return torch.from_numpy(img_3ch)
