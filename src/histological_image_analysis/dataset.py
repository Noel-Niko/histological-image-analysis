"""PyTorch Dataset for brain structure segmentation.

Wraps CCFv3Slicer to provide (pixel_values, labels) pairs suitable
for DINOv2 + UperNet semantic segmentation training.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
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
    split_strategy : str
        "spatial" (contiguous AP blocks) or "interleaved" (every Kth slice).
        Use "interleaved" to ensure all brain regions appear in all splits.
    """

    _VALID_PRESETS = ("baseline", "extended", "none")

    def __init__(
        self,
        slicer: CCFv3Slicer,
        split: str,
        mapping: dict[int, int],
        crop_size: int = 518,
        augment: bool = True,
        split_strategy: str = "spatial",
        augmentation_preset: str = "baseline",
        multi_axis: bool = False,
    ) -> None:
        if augmentation_preset not in self._VALID_PRESETS:
            msg = (
                f"augmentation_preset must be one of {self._VALID_PRESETS}, "
                f"got '{augmentation_preset}'"
            )
            raise ValueError(msg)

        self._crop_size = crop_size
        self._augmentation_preset = augmentation_preset
        # "none" preset disables augmentation entirely (same as augment=False)
        self._augment = augment and augmentation_preset != "none"

        # Pre-load all slices into memory
        self._slices: list[tuple[np.ndarray, np.ndarray]] = list(
            slicer.iter_slices(
                split, mapping,
                split_strategy=split_strategy,
                multi_axis=multi_axis,
            )
        )
        logger.info(
            "Loaded %d %s slices (crop_size=%d, augment=%s, multi_axis=%s)",
            len(self._slices),
            split,
            crop_size,
            augment,
            multi_axis,
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

    def _random_rot90(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random 90-degree rotation (0°, 90°, 180°, or 270°).

        Brain sections have no canonical mounting orientation, so all
        four orientations are equally valid.  Uses ``np.rot90`` which
        is exact — no interpolation artifacts, no fill pixels.
        """
        k = np.random.randint(0, 4)  # 0=0°, 1=90°, 2=180°, 3=270°
        if k == 0:
            return image, mask
        return np.rot90(image, k=k).copy(), np.rot90(mask, k=k).copy()

    def _gaussian_blur(
        self,
        image: np.ndarray,
        sigma_range: tuple[float, float] = (0.5, 2.0),
    ) -> np.ndarray:
        """Apply Gaussian blur to simulate microscopy focus variation.

        Applied to image only — mask is never blurred.

        Parameters
        ----------
        image : np.ndarray
            Grayscale image (H, W), uint8.
        sigma_range : tuple[float, float]
            Range for random sigma selection.
        """
        sigma = np.random.uniform(*sigma_range)
        blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)
        return np.clip(blurred, 0, 255).astype(np.uint8)

    def _elastic_deform(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 50.0,
        sigma: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation to image and mask.

        Generates random displacement fields, smooths them with a
        Gaussian filter, then applies the same deformation to both
        image (bilinear, order=1) and mask (nearest-neighbor, order=0).

        Parameters
        ----------
        image : np.ndarray
            Grayscale image (H, W), uint8.
        mask : np.ndarray
            Segmentation mask (H, W), int64.
        alpha : float
            Displacement magnitude (pixels).
        sigma : float
            Gaussian smoothing sigma for displacement field.
        """
        shape = image.shape[:2]
        dx = gaussian_filter(np.random.standard_normal(shape), sigma) * alpha
        dy = gaussian_filter(np.random.standard_normal(shape), sigma) * alpha
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        coords = [y + dy, x + dx]

        image_out = map_coordinates(
            image, coords, order=1, mode="constant", cval=0,
        ).astype(np.uint8)
        mask_out = map_coordinates(
            mask, coords, order=0, mode="constant", cval=0,
        ).astype(np.int64)

        return image_out, mask_out

    def _apply_augmentation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to image and mask.

        Which transforms are applied depends on ``augmentation_preset``:

        - ``"baseline"``: flip, rotation ±15°, color jitter (Run 5 config)
        - ``"extended"``: all 6 transforms including rot90, elastic, blur (Run 7)

        Full pipeline (extended preset):

        1. Horizontal flip (50% probability)
        2. Random 90° rotation (50% probability) — extended only
        3. Rotation ±15° (skip if |angle| ≤ 0.5°)
        4. Elastic deformation (30% probability) — extended only
        5. Gaussian blur on image only (30% probability) — extended only
        6. Color jitter on image only (always: brightness ±0.2, contrast ±0.2)

        Spatial transforms (1-4) use nearest-neighbor for mask, fill=0
        (Background class).
        """
        extended = self._augmentation_preset == "extended"

        # 1. Horizontal flip
        if np.random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # 2. Random 90° rotation (extended only)
        if extended and np.random.random() < 0.5:
            image, mask = self._random_rot90(image, mask)

        # 3. Rotation ±15°
        angle = np.random.uniform(-15, 15)
        if abs(angle) > 0.5:
            img_pil = Image.fromarray(image)
            image = np.array(
                img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            )
            mask_pil = Image.fromarray(mask.astype(np.int32), mode="I")
            mask = np.array(
                mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
            ).astype(np.int64)

        # 4. Elastic deformation (extended only)
        if extended and np.random.random() < 0.3:
            image, mask = self._elastic_deform(image, mask)

        # 5. Gaussian blur (image only, extended only)
        if extended and np.random.random() < 0.3:
            image = self._gaussian_blur(image)

        # 6. Color jitter (image only, always applied)
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
