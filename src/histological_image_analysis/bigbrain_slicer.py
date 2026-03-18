"""BigBrain volume slicer for extracting 2D slices from 3D NIfTI volumes.

Loads NIfTI volumes (or numpy arrays), normalizes image data, and provides
iteration over coronal slices with interleaved train/val/test splitting
and gap-based exclusion for spatial leakage mitigation.

Volume convention (BigBrain 200μm):
    - Shape: (696, 770, 605) — axes are spatial (X, Y, Z)
    - Axis 0 = coronal slicing dimension
    - Axis 1, 2 = in-plane dimensions
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Minimum fraction of non-background pixels for a slice to be "valid"
MIN_BRAIN_FRACTION = 0.10


class BigBrainSlicer:
    """Extract and normalize 2D slices from BigBrain 3D NIfTI volumes.

    Parameters
    ----------
    histology_path : str or Path
        Path to histological volume NIfTI file (uint8 or uint16).
    annotation_path : str or Path
        Path to annotation volume NIfTI file (uint8 tissue class labels).
    """

    def __init__(
        self,
        histology_path: str | Path,
        annotation_path: str | Path,
    ) -> None:
        self._histology_path = Path(histology_path)
        self._annotation_path = Path(annotation_path)
        self._image_volume: np.ndarray | None = None
        self._annotation_volume: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        image: np.ndarray,
        annotation: np.ndarray,
    ) -> BigBrainSlicer:
        """Create a slicer from pre-loaded numpy arrays (for testing).

        Parameters
        ----------
        image : np.ndarray
            Image volume (uint8, uint16, or float32).
        annotation : np.ndarray
            Annotation volume (uint8 tissue class labels).

        Raises
        ------
        ValueError
            If image and annotation shapes don't match.
        """
        if image.shape != annotation.shape:
            msg = (
                f"Image shape {image.shape} does not match "
                f"annotation shape {annotation.shape}"
            )
            raise ValueError(msg)

        instance = cls.__new__(cls)
        instance._histology_path = None
        instance._annotation_path = None
        instance._image_volume = cls._normalize_image(image)
        instance._annotation_volume = annotation

        logger.info(
            "Loaded volumes from arrays: shape=%s, image dtype=%s→uint8, "
            "annotation dtype=%s",
            image.shape,
            image.dtype,
            annotation.dtype,
        )
        return instance

    def load_volumes(self) -> None:
        """Load NIfTI files from disk, normalize image to uint8.

        Raises
        ------
        ValueError
            If image and annotation volumes have different shapes.
        """
        import nibabel as nib

        logger.info("Loading histological volume: %s", self._histology_path)
        img_nii = nib.load(str(self._histology_path))
        image_raw = np.asarray(img_nii.dataobj)

        logger.info("Loading annotation volume: %s", self._annotation_path)
        annot_nii = nib.load(str(self._annotation_path))
        annotation_raw = np.asarray(annot_nii.dataobj)

        if image_raw.shape != annotation_raw.shape:
            msg = (
                f"Image shape {image_raw.shape} does not match "
                f"annotation shape {annotation_raw.shape}"
            )
            raise ValueError(msg)

        logger.info(
            "Raw image: shape=%s, dtype=%s, range=[%.2f, %.2f]",
            image_raw.shape,
            image_raw.dtype,
            float(image_raw.min()),
            float(image_raw.max()),
        )

        self._image_volume = self._normalize_image(image_raw)
        self._annotation_volume = annotation_raw

        logger.info(
            "Volumes loaded: shape=%s, %d unique labels",
            self._image_volume.shape,
            len(np.unique(self._annotation_volume)),
        )

    @staticmethod
    def _normalize_image(volume: np.ndarray) -> np.ndarray:
        """Normalize an image volume to uint8 [0, 255].

        - uint8: pass through
        - float32: percentile clip (1st-99th), then scale to [0, 255]
        - uint16 / other integer: scale by 255 / max_value
        """
        if volume.dtype == np.uint8:
            return volume

        vol_float = volume.astype(np.float64)

        if np.issubdtype(volume.dtype, np.floating):
            p_low = np.percentile(vol_float, 1)
            p_high = np.percentile(vol_float, 99)
            vol_float = np.clip(vol_float, p_low, p_high)

        v_min = vol_float.min()
        v_max = vol_float.max()
        if v_max > v_min:
            vol_float = (vol_float - v_min) / (v_max - v_min) * 255.0
        else:
            vol_float = np.zeros_like(vol_float)

        return vol_float.astype(np.uint8)

    @property
    def image_volume(self) -> np.ndarray:
        if self._image_volume is None:
            raise RuntimeError("Volumes not loaded. Call load_volumes() first.")
        return self._image_volume

    @property
    def annotation_volume(self) -> np.ndarray:
        if self._annotation_volume is None:
            raise RuntimeError("Volumes not loaded. Call load_volumes() first.")
        return self._annotation_volume

    @property
    def num_slices(self) -> int:
        """Number of slices along axis 0."""
        return self.image_volume.shape[0]

    def get_slice(
        self, index: int, axis: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get a single 2D slice along the specified axis.

        Parameters
        ----------
        index : int
            Position along the slicing axis.
        axis : int
            Slicing axis: 0=coronal, 1=axial, 2=sagittal.

        Returns
        -------
        tuple of (image_2d, annotation_2d)
            Image is uint8, annotation has raw tissue labels.
        """
        if axis not in (0, 1, 2):
            msg = f"axis must be 0, 1, or 2, got {axis}"
            raise ValueError(msg)

        dim_size = self.image_volume.shape[axis]
        if index < 0 or index >= dim_size:
            raise IndexError(
                f"Index {index} out of range [0, {dim_size}) for axis {axis}"
            )

        slicing: list[int | slice] = [slice(None)] * 3
        slicing[axis] = index
        return (
            self.image_volume[tuple(slicing)],
            self.annotation_volume[tuple(slicing)],
        )

    def _get_valid_indices(self, axis: int = 0) -> list[int]:
        """Find indices along axis where at least MIN_BRAIN_FRACTION pixels are non-zero.

        Parameters
        ----------
        axis : int
            Slicing axis: 0, 1, or 2.
        """
        vol = self.annotation_volume
        dim_size = vol.shape[axis]

        other_dims = [s for i, s in enumerate(vol.shape) if i != axis]
        total_pixels = other_dims[0] * other_dims[1]
        threshold = total_pixels * MIN_BRAIN_FRACTION

        valid = []
        slicing: list[int | slice] = [slice(None)] * 3
        for idx in range(dim_size):
            slicing[axis] = idx
            brain_pixels = np.count_nonzero(vol[tuple(slicing)])
            if brain_pixels >= threshold:
                valid.append(idx)
            slicing[axis] = slice(None)

        axis_names = {0: "coronal", 1: "axial", 2: "sagittal"}
        logger.info(
            "Valid %s slices (axis %d): %d / %d (%.1f%% brain threshold)",
            axis_names.get(axis, str(axis)),
            axis,
            len(valid),
            dim_size,
            MIN_BRAIN_FRACTION * 100,
        )
        return valid

    def get_split_indices(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        split_strategy: str = "interleaved",
        gap: int = 0,
    ) -> dict[str, list[int]]:
        """Split valid indices into train/val/test.

        Parameters
        ----------
        train_frac : float
            Fraction for training (default 0.8).
        val_frac : float
            Fraction for validation (default 0.1).
        split_strategy : str
            "interleaved" (default) or "spatial".
        gap : int
            Number of slices to exclude from training on each side of
            val/test slices. Creates a buffer zone to mitigate spatial
            leakage between adjacent coronal slices. Default 0 (no gap).
            Example: gap=2 at 200μm resolution creates a 1mm buffer.
        """
        if split_strategy not in ("spatial", "interleaved"):
            msg = (
                f"split_strategy must be 'spatial' or 'interleaved', "
                f"got '{split_strategy}'"
            )
            raise ValueError(msg)

        valid = self._get_valid_indices(axis=0)
        n = len(valid)

        if split_strategy == "spatial":
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            result = {
                "train": valid[:n_train],
                "val": valid[n_train : n_train + n_val],
                "test": valid[n_train + n_val :],
            }
        else:
            # Interleaved: every stride-th slice for val, offset by 1 for test
            stride = round(1.0 / val_frac) if val_frac > 0 else n
            train_indices: list[int] = []
            val_indices: list[int] = []
            test_indices: list[int] = []

            for i, idx in enumerate(valid):
                if i % stride == 0:
                    val_indices.append(idx)
                elif i % stride == 1:
                    test_indices.append(idx)
                else:
                    train_indices.append(idx)

            result = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }

        # Apply gap exclusion: remove train slices within ±gap of val/test
        if gap > 0:
            val_test_set = set(result["val"]) | set(result["test"])
            excluded = set()
            for vt_idx in val_test_set:
                for offset in range(-gap, gap + 1):
                    excluded.add(vt_idx + offset)
            # Remove excluded indices from train (keep val/test untouched)
            result["train"] = [
                idx for idx in result["train"] if idx not in excluded
            ]

        logger.info(
            "%s split (gap=%d): train=%d, val=%d, test=%d",
            split_strategy,
            gap,
            len(result["train"]),
            len(result["val"]),
            len(result["test"]),
        )

        return result

    def iter_slices(
        self,
        split: str,
        mapping: dict[int, int],
        split_strategy: str = "interleaved",
        multi_axis: bool = False,
        gap: int = 0,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over slices in a split, with remapped class masks.

        Parameters
        ----------
        split : str
            One of "train", "val", "test".
        mapping : dict
            Label → class ID mapping (e.g., identity for 9-class).
        split_strategy : str
            Split strategy passed to get_split_indices().
        multi_axis : bool
            If True and split is "train", additionally yield axial and
            sagittal slices. Val/test remain coronal-only.
        gap : int
            Gap exclusion passed to get_split_indices().

        Yields
        ------
        tuple of (image_2d, class_mask_2d)
            Image is uint8, class mask is int64.
        """
        splits = self.get_split_indices(
            split_strategy=split_strategy, gap=gap
        )
        indices = splits[split]

        # Coronal slices (all splits)
        for idx in indices:
            img, annot = self.get_slice(idx, axis=0)
            class_mask = self._remap_mask(annot, mapping)
            yield img, class_mask

        # Multi-axis: add axial and sagittal slices for training only
        if multi_axis and split == "train":
            for axis in (1, 2):
                axis_names = {1: "axial", 2: "sagittal"}
                valid = self._get_valid_indices(axis=axis)
                logger.info(
                    "Adding %d %s slices to training",
                    len(valid),
                    axis_names[axis],
                )
                for idx in valid:
                    img, annot = self.get_slice(idx, axis=axis)
                    class_mask = self._remap_mask(annot, mapping)
                    yield img, class_mask

    @staticmethod
    def _remap_mask(
        mask: np.ndarray, mapping: dict[int, int]
    ) -> np.ndarray:
        """Remap annotation labels using the given mapping.

        Labels not in the mapping are mapped to 0 (background).
        Uses a lookup array for vectorized operation.
        """
        max_val = int(mask.max())
        lut = np.zeros(max_val + 1, dtype=np.int64)
        for label, class_id in mapping.items():
            if label <= max_val:
                lut[label] = class_id
        return lut[mask]
