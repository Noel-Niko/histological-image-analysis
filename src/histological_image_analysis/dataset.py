"""PyTorch Datasets for brain structure segmentation.

Wraps volume slicers and image+SVG pairs to provide (pixel_values, labels)
pairs suitable for DINOv2 + UperNet semantic segmentation training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset

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
    slicer : object
        Volume slicer providing (image, mask) pairs via ``iter_slices()``.
        Must implement ``iter_slices(split, mapping, split_strategy=...,
        multi_axis=...) → Iterator[tuple[ndarray, ndarray]]``.
        Compatible with both ``CCFv3Slicer`` and ``BigBrainSlicer``.
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
        slicer: Any,
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


def split_by_donor(
    pairs: list[tuple[Path, Path, str]],
    train_donors: list[str],
    val_donors: list[str],
    test_donors: list[str],
) -> dict[str, list[tuple[Path, Path]]]:
    """Split (image, svg, donor) triples by donor into train/val/test.

    Parameters
    ----------
    pairs : list of (image_path, svg_path, donor_id) triples
        All annotated image-SVG pairs with their donor ID.
    train_donors : list of str
        Donor IDs for training.
    val_donors : list of str
        Donor IDs for validation.
    test_donors : list of str
        Donor IDs for testing.

    Returns
    -------
    dict with keys "train", "val", "test", each a list of (image, svg) tuples.
    """
    train_set = set(train_donors)
    val_set = set(val_donors)
    test_set = set(test_donors)

    splits: dict[str, list[tuple[Path, Path]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for image_path, svg_path, donor_id in pairs:
        if donor_id in train_set:
            splits["train"].append((image_path, svg_path))
        elif donor_id in val_set:
            splits["val"].append((image_path, svg_path))
        elif donor_id in test_set:
            splits["test"].append((image_path, svg_path))

    logger.info(
        "Donor split: train=%d (%s), val=%d (%s), test=%d (%s)",
        len(splits["train"]),
        train_donors,
        len(splits["val"]),
        val_donors,
        len(splits["test"]),
        test_donors,
    )
    return splits


class AllenHumanDataset(Dataset):
    """PyTorch Dataset for Allen Human Brain SVG-annotated sections.

    Lazy-loads RGB images from disk and rasterizes SVG annotations on each
    ``__getitem__()`` call. Uses ``sparse=True`` so unlabeled pixels are
    255 (ignored by ``CrossEntropyLoss(ignore_index=255)``).

    When ``cache_dir`` is provided, loads pre-resized image+mask ``.npz``
    files from the cache directory instead of raw JPEGs + SVG rasterization.
    Use ``build_cache()`` to create the cache before training.

    Parameters
    ----------
    image_svg_pairs : list of (image_path, svg_path)
        Pairs of paths to JPEG images and SVG annotation files.
    rasterizer : SVGRasterizer
        SVG rasterizer instance. Not used when ``cache_dir`` is set.
    mapping : dict
        Structure ID → class ID mapping. Structure IDs not in the mapping
        are mapped to 0. Pixels with value 255 (unlabeled) are preserved.
    crop_size : int
        Output spatial size (default 518 for DINOv2).
    augment : bool
        Whether to apply random augmentation (training only).
    cache_dir : str or Path or None
        Directory containing pre-cached ``.npz`` files. When set,
        ``__getitem__`` loads from ``{cache_dir}/{image_stem}.npz``
        instead of reading JPEG + rasterizing SVG. Default None (lazy load).
    """

    def __init__(
        self,
        image_svg_pairs: list[tuple[Path, Path]],
        rasterizer: Any,
        mapping: dict[int, int],
        crop_size: int = 518,
        augment: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._pairs = image_svg_pairs
        self._rasterizer = rasterizer
        self._mapping = mapping
        self._crop_size = crop_size
        self._augment = augment
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

        # Build LUT for structure ID → class ID remapping
        # Preserves 255 as ignore_index
        if mapping:
            max_sid = max(max(mapping.keys()), 255)
        else:
            max_sid = 255
        self._lut = np.zeros(max_sid + 1, dtype=np.int64)
        for sid, cid in mapping.items():
            if sid <= max_sid:
                self._lut[sid] = cid
        self._lut[255] = 255  # preserve ignore_index

        logger.info(
            "AllenHumanDataset: %d pairs, crop_size=%d, augment=%s, cache=%s",
            len(self._pairs),
            crop_size,
            augment,
            "enabled" if self._cache_dir else "disabled",
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path, svg_path = self._pairs[idx]

        if self._cache_dir is not None:
            # Load pre-cached resized image + remapped mask
            stem = Path(image_path).stem
            npz = np.load(self._cache_dir / f"{stem}.npz")
            image = npz["image"]  # (H, W, 3) uint8
            mask = npz["mask"]    # (H, W) int64
        else:
            # Lazy load: read JPEG + rasterize SVG
            img = Image.open(image_path).convert("RGB")
            image = np.array(img)  # (H, W, 3) uint8

            h, w = image.shape[:2]

            # Rasterize SVG to mask (sparse: unlabeled=255, annotated=structure_id)
            raw_mask = self._rasterizer.rasterize(
                svg_path, target_width=w, target_height=h, sparse=True,
            )

            # Remap structure IDs to class IDs, preserving 255
            mask = self._lut[raw_mask]

        # Pad if needed
        image, mask = self._pad_if_needed(image, mask)

        # Crop
        if self._augment:
            image, mask = self._random_crop(image, mask)
        else:
            image, mask = self._center_crop(image, mask)

        # Augmentation (training only)
        if self._augment:
            image, mask = self._apply_augmentation(image, mask)

        # Normalize RGB → (3, H, W) float tensor
        pixel_values = self._normalize_rgb(image)

        return {
            "pixel_values": pixel_values,
            "labels": torch.from_numpy(mask.copy()).long(),
        }

    def _pad_if_needed(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad image and mask to at least crop_size in both dimensions.

        Image is padded with 0 (black). Mask is padded with 255 (ignore).
        """
        h, w = image.shape[:2]
        pad_h = max(0, self._crop_size - h)
        pad_w = max(0, self._crop_size - w)

        if pad_h == 0 and pad_w == 0:
            return image, mask

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        mask = np.pad(
            mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=255,
        )
        return image, mask

    def _random_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random crop to crop_size x crop_size."""
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
        """Center crop to crop_size x crop_size."""
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
        """Apply baseline augmentation to RGB image and mask.

        Uses ignore_index=255 as fill value for mask (not 0) to avoid
        teaching the model that rotated borders are annotated background.

        1. Horizontal flip (50%)
        2. Rotation ±15° (mask filled with 255)
        3. Color jitter (brightness ±0.2, contrast ±0.2)
        """
        # 1. Horizontal flip
        if np.random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # 2. Rotation ±15°
        angle = np.random.uniform(-15, 15)
        if abs(angle) > 0.5:
            img_pil = Image.fromarray(image)
            image = np.array(
                img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            )
            mask_pil = Image.fromarray(mask.astype(np.int32), mode="I")
            mask = np.array(
                mask_pil.rotate(
                    angle, resample=Image.NEAREST, fillcolor=255,
                )
            ).astype(np.int64)

        # 3. Color jitter (image only)
        image = image.astype(np.float32)
        brightness = 1.0 + np.random.uniform(-0.2, 0.2)
        contrast = 1.0 + np.random.uniform(-0.2, 0.2)
        image = image * contrast + (brightness - 1.0) * 128
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image, mask

    @staticmethod
    def _normalize_rgb(image: np.ndarray) -> torch.Tensor:
        """Convert uint8 RGB to normalized 3-channel float tensor.

        1. uint8 [0, 255] → float32 [0, 1]
        2. Transpose (H, W, 3) → (3, H, W)
        3. Apply ImageNet normalization per channel
        """
        img_float = image.astype(np.float32) / 255.0
        img_3ch = np.transpose(img_float, (2, 0, 1))  # (3, H, W)

        for c in range(3):
            img_3ch[c] = (img_3ch[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        return torch.from_numpy(img_3ch)

    @staticmethod
    def build_cache(
        image_svg_pairs: list[tuple[Path, Path]],
        rasterizer: Any,
        mapping: dict[int, int],
        cache_dir: str | Path,
        max_dim: int = 1024,
    ) -> int:
        """Pre-cache resized images + remapped masks as ``.npz`` files.

        For each (image, SVG) pair:
        1. Load RGB image, resize so longest side = ``max_dim``
        2. Rasterize SVG at original size, remap via LUT, resize (nearest)
        3. Save as ``{cache_dir}/{image_stem}.npz``

        Parameters
        ----------
        image_svg_pairs : list of (image_path, svg_path)
            All pairs to cache.
        rasterizer : SVGRasterizer
            SVG rasterizer instance.
        mapping : dict
            Structure ID → class ID mapping.
        cache_dir : str or Path
            Output directory for ``.npz`` files.
        max_dim : int
            Maximum dimension (longest side) for resized images.

        Returns
        -------
        int
            Number of files cached.
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Build LUT
        if mapping:
            max_sid = max(max(mapping.keys()), 255)
        else:
            max_sid = 255
        lut = np.zeros(max_sid + 1, dtype=np.int64)
        for sid, cid in mapping.items():
            if sid <= max_sid:
                lut[sid] = cid
        lut[255] = 255

        cached = 0
        for image_path, svg_path in image_svg_pairs:
            stem = Path(image_path).stem
            out_path = cache_path / f"{stem}.npz"

            if out_path.exists():
                cached += 1
                continue

            # Load image
            img_pil = Image.open(image_path).convert("RGB")
            orig_w, orig_h = img_pil.size

            # Rasterize SVG at original dimensions
            raw_mask = rasterizer.rasterize(
                svg_path,
                target_width=orig_w,
                target_height=orig_h,
                sparse=True,
            )
            mask = lut[raw_mask]

            # Resize if needed
            longest = max(orig_h, orig_w)
            if longest > max_dim:
                scale = max_dim / longest
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                img_pil = img_pil.resize(
                    (new_w, new_h), resample=Image.BILINEAR,
                )
                mask_pil = Image.fromarray(mask.astype(np.int32)).resize(
                    (new_w, new_h), resample=Image.NEAREST,
                )
                mask = np.array(mask_pil).astype(np.int64)

            image = np.array(img_pil)
            np.savez_compressed(out_path, image=image, mask=mask)
            cached += 1

        logger.info(
            "Cache built: %d files in %s (max_dim=%d)",
            cached,
            cache_path,
            max_dim,
        )
        return cached
