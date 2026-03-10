"""CCFv3 volume slicer for extracting 2D coronal slices from 3D brain volumes.

Loads NRRD volumes (or numpy arrays), normalizes image data, and provides
iteration over coronal slices with spatial train/val/test splitting.

Axis convention:
    - Axis 0 = AP (anterior-posterior): 0 = most anterior
    - Axis 1 = DV (dorsal-ventral)
    - Axis 2 = ML (medial-lateral)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from histological_image_analysis.ontology import OntologyMapper

logger = logging.getLogger(__name__)

# Minimum fraction of non-background pixels for a slice to be "valid"
MIN_BRAIN_FRACTION = 0.10


class CCFv3Slicer:
    """Extract and normalize 2D coronal slices from CCFv3 3D volumes.

    Parameters
    ----------
    image_path : str or Path
        Path to image NRRD file (nissl float32 or template uint16).
    annotation_path : str or Path
        Path to annotation NRRD file (uint32 structure IDs).
    ontology_mapper : OntologyMapper
        Ontology mapper for structure remapping.
    """

    def __init__(
        self,
        image_path: str | Path,
        annotation_path: str | Path,
        ontology_mapper: OntologyMapper,
    ) -> None:
        self._image_path = Path(image_path)
        self._annotation_path = Path(annotation_path)
        self._ontology_mapper = ontology_mapper
        self._image_volume: np.ndarray | None = None
        self._annotation_volume: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        image: np.ndarray,
        annotation: np.ndarray,
        ontology_mapper: OntologyMapper,
    ) -> CCFv3Slicer:
        """Create a slicer from pre-loaded numpy arrays (for testing).

        Parameters
        ----------
        image : np.ndarray
            Image volume (float32 or uint16).
        annotation : np.ndarray
            Annotation volume (uint32 structure IDs).
        ontology_mapper : OntologyMapper
            Ontology mapper instance.

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
        instance._ontology_mapper = ontology_mapper
        instance._image_path = None
        instance._annotation_path = None
        instance._image_volume = cls._normalize_image(instance, image)
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
        """Load NRRD files from disk, normalize image to uint8.

        Raises
        ------
        ValueError
            If image and annotation volumes have different shapes.
        """
        import nrrd

        logger.info("Loading image volume: %s", self._image_path)
        image_raw, _ = nrrd.read(str(self._image_path))

        logger.info("Loading annotation volume: %s", self._annotation_path)
        annotation_raw, _ = nrrd.read(str(self._annotation_path))

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
        self._annotation_volume = annotation_raw.astype(np.uint32)

        logger.info(
            "Volumes loaded: shape=%s, %d unique structure IDs",
            self._image_volume.shape,
            len(np.unique(self._annotation_volume)),
        )

    def _normalize_image(self, volume: np.ndarray) -> np.ndarray:
        """Normalize an image volume to uint8 [0, 255].

        - float32 (nissl): percentile clip (1st-99th), then scale to [0, 255]
        - uint16 (template): scale by 255 / max_value
        - uint8: pass through
        """
        if volume.dtype == np.uint8:
            return volume

        vol_float = volume.astype(np.float64)

        if np.issubdtype(volume.dtype, np.floating):
            # Float volume (nissl): percentile clip
            p_low = np.percentile(vol_float, 1)
            p_high = np.percentile(vol_float, 99)
            vol_float = np.clip(vol_float, p_low, p_high)
        # For integer types (uint16, etc.), no clipping needed

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
        """Number of coronal slices (AP axis length)."""
        return self.image_volume.shape[0]

    def get_slice(self, ap_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a single coronal slice at the given AP index.

        Parameters
        ----------
        ap_index : int
            Position along the anterior-posterior axis.

        Returns
        -------
        tuple of (image_2d, annotation_2d)
            Image is uint8, annotation has raw structure IDs.

        Raises
        ------
        IndexError
            If ap_index is out of range.
        """
        if ap_index < 0 or ap_index >= self.num_slices:
            raise IndexError(
                f"AP index {ap_index} out of range [0, {self.num_slices})"
            )
        return (
            self.image_volume[ap_index, :, :],
            self.annotation_volume[ap_index, :, :],
        )

    def _get_valid_ap_indices(self) -> list[int]:
        """Find AP indices where at least MIN_BRAIN_FRACTION pixels are brain."""
        valid = []
        total_pixels = (
            self.annotation_volume.shape[1] * self.annotation_volume.shape[2]
        )
        threshold = total_pixels * MIN_BRAIN_FRACTION

        for ap in range(self.num_slices):
            brain_pixels = np.count_nonzero(self.annotation_volume[ap, :, :])
            if brain_pixels >= threshold:
                valid.append(ap)

        logger.info(
            "Valid slices: %d / %d (%.1f%% brain threshold)",
            len(valid),
            self.num_slices,
            MIN_BRAIN_FRACTION * 100,
        )
        return valid

    def get_split_indices(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        split_strategy: str = "spatial",
    ) -> dict[str, list[int]]:
        """Split valid AP indices into train/val/test.

        Parameters
        ----------
        train_frac : float
            Fraction of valid slices for training (default 0.8).
        val_frac : float
            Fraction of valid slices for validation (default 0.1).
            Test fraction is 1 - train_frac - val_frac.
        split_strategy : str
            "spatial" — contiguous blocks along AP axis (original behavior).
            "interleaved" — every Kth slice for val/test, rest for train.
            Interleaved ensures all brain regions are represented in every
            split, avoiding class absence in val/test for structures that
            are spatially concentrated (e.g., Cerebrum in anterior brain).

        Returns
        -------
        dict with keys "train", "val", "test", each a list of AP indices.

        Raises
        ------
        ValueError
            If split_strategy is not "spatial" or "interleaved".
        """
        if split_strategy not in ("spatial", "interleaved"):
            msg = (
                f"split_strategy must be 'spatial' or 'interleaved', "
                f"got '{split_strategy}'"
            )
            raise ValueError(msg)

        valid = self._get_valid_ap_indices()
        n = len(valid)

        if split_strategy == "spatial":
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            return {
                "train": valid[:n_train],
                "val": valid[n_train : n_train + n_val],
                "test": valid[n_train + n_val :],
            }

        # Interleaved: every stride-th slice for val, offset by 1 for test
        stride = round(1.0 / val_frac) if val_frac > 0 else n
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for i, ap in enumerate(valid):
            if i % stride == 0:
                val_indices.append(ap)
            elif i % stride == 1:
                test_indices.append(ap)
            else:
                train_indices.append(ap)

        logger.info(
            "Interleaved split (stride=%d): train=%d, val=%d, test=%d",
            stride,
            len(train_indices),
            len(val_indices),
            len(test_indices),
        )

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

    def iter_slices(
        self,
        split: str,
        mapping: dict[int, int],
        split_strategy: str = "spatial",
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over slices in a split, with remapped class masks.

        Parameters
        ----------
        split : str
            One of "train", "val", "test".
        mapping : dict
            Structure ID → class ID mapping from OntologyMapper.
        split_strategy : str
            Split strategy passed to get_split_indices().

        Yields
        ------
        tuple of (image_2d, class_mask_2d)
            Image is uint8 (DV, ML), class mask is int64 (DV, ML).
        """
        splits = self.get_split_indices(split_strategy=split_strategy)
        indices = splits[split]

        for ap in indices:
            img, annot = self.get_slice(ap)
            class_mask = self._ontology_mapper.remap_mask(annot, mapping)
            yield img, class_mask
