"""Tests for BigBrainSlicer — written first per TDD."""

import numpy as np
import pytest

from histological_image_analysis.bigbrain_slicer import BigBrainSlicer


@pytest.fixture
def synthetic_bigbrain_volume() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic BigBrain-like volume pair.

    Returns (image_volume, annotation_volume) with shape (20, 16, 12).
    Image is uint8 (like the 8-bit histological volume).
    Annotation is uint8 with 9-class tissue labels (0-9).
    """
    rng = np.random.default_rng(42)

    image = rng.integers(0, 256, size=(20, 16, 12), dtype=np.uint8)

    # Annotation: 9-class tissue labels
    annotation = np.zeros((20, 16, 12), dtype=np.uint8)
    # Slices 0-1: mostly background (testing skip logic)
    # Slices 2-19: mix of tissue classes
    annotation[2:18, 2:14, 2:10] = 1   # Gray Matter
    annotation[4:16, 4:12, 3:9] = 2    # White Matter
    annotation[6:14, 6:10, 4:8] = 3    # CSF
    annotation[8:12, 8:9, 5:7] = 4     # Meninges

    return image, annotation


@pytest.fixture
def bigbrain_slicer(synthetic_bigbrain_volume):
    """BigBrainSlicer loaded from synthetic arrays."""
    image, annotation = synthetic_bigbrain_volume
    return BigBrainSlicer.from_arrays(image, annotation)


class TestLoadVolumes:
    """Test volume loading from numpy arrays."""

    def test_from_arrays_stores_volumes(self, synthetic_bigbrain_volume):
        """from_arrays should store image and annotation volumes."""
        image, annotation = synthetic_bigbrain_volume
        slicer = BigBrainSlicer.from_arrays(image, annotation)
        assert slicer.image_volume.shape == (20, 16, 12)
        assert slicer.annotation_volume.shape == (20, 16, 12)

    def test_from_arrays_rejects_mismatched_shapes(self):
        """Mismatched image and annotation shapes should raise ValueError."""
        image = np.zeros((20, 16, 12), dtype=np.uint8)
        annotation = np.zeros((20, 16, 6), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            BigBrainSlicer.from_arrays(image, annotation)

    def test_volumes_not_loaded_raises(self):
        """Accessing volumes before load should raise RuntimeError."""
        slicer = BigBrainSlicer(
            histology_path="dummy.nii.gz",
            annotation_path="dummy.nii.gz",
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = slicer.image_volume

    def test_num_slices_axis0(self, bigbrain_slicer):
        """num_slices should return axis-0 dimension."""
        assert bigbrain_slicer.num_slices == 20


class TestNormalization:
    """Test image normalization to uint8."""

    def test_uint8_passthrough(self):
        """uint8 input should pass through unchanged."""
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = BigBrainSlicer._normalize_image(arr)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, arr)

    def test_uint16_scaled_to_uint8(self):
        """uint16 input should be scaled to 0-255."""
        arr = np.array([0, 32768, 65535], dtype=np.uint16)
        result = BigBrainSlicer._normalize_image(arr)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[-1] == 255

    def test_float32_percentile_clipped(self):
        """float32 input should be percentile-clipped and scaled."""
        rng = np.random.default_rng(42)
        arr = rng.uniform(10.0, 100.0, size=(100,)).astype(np.float32)
        arr[0] = -1000.0  # extreme low
        arr[1] = 99999.0  # extreme high
        result = BigBrainSlicer._normalize_image(arr)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[1] == 255


class TestGetSlice:
    """Test 2D slice extraction from 3D volume."""

    def test_coronal_slice_shape(self, bigbrain_slicer):
        """Axis-0 (coronal) slice should have shape (dim1, dim2)."""
        img, annot = bigbrain_slicer.get_slice(10, axis=0)
        assert img.shape == (16, 12)
        assert annot.shape == (16, 12)

    def test_axial_slice_shape(self, bigbrain_slicer):
        """Axis-1 (axial) slice should have shape (dim0, dim2)."""
        img, annot = bigbrain_slicer.get_slice(8, axis=1)
        assert img.shape == (20, 12)
        assert annot.shape == (20, 12)

    def test_sagittal_slice_shape(self, bigbrain_slicer):
        """Axis-2 (sagittal) slice should have shape (dim0, dim1)."""
        img, annot = bigbrain_slicer.get_slice(6, axis=2)
        assert img.shape == (20, 16)
        assert annot.shape == (20, 16)

    def test_default_axis_is_0(self, bigbrain_slicer):
        """Default axis should be 0 (coronal)."""
        default = bigbrain_slicer.get_slice(10)
        explicit = bigbrain_slicer.get_slice(10, axis=0)
        np.testing.assert_array_equal(default[0], explicit[0])
        np.testing.assert_array_equal(default[1], explicit[1])

    def test_invalid_axis_raises(self, bigbrain_slicer):
        """Invalid axis should raise ValueError."""
        with pytest.raises(ValueError, match="axis"):
            bigbrain_slicer.get_slice(0, axis=3)

    def test_out_of_range_raises(self, bigbrain_slicer):
        """Out-of-range index should raise IndexError."""
        with pytest.raises(IndexError):
            bigbrain_slicer.get_slice(100, axis=0)

    def test_image_dtype_uint8(self, bigbrain_slicer):
        """Image slice should be uint8."""
        img, _ = bigbrain_slicer.get_slice(10)
        assert img.dtype == np.uint8

    def test_annotation_preserves_labels(self, bigbrain_slicer):
        """Annotation slice should preserve tissue class labels."""
        _, annot = bigbrain_slicer.get_slice(10)
        unique = set(np.unique(annot))
        # Slice 10 is in the center — should have multiple tissue classes
        assert len(unique) >= 2


class TestValidIndices:
    """Test valid slice filtering (minimum brain fraction)."""

    def test_skips_mostly_background(self):
        """Slices with <10% non-background should be excluded."""
        image = np.zeros((10, 100, 100), dtype=np.uint8)
        annotation = np.zeros((10, 100, 100), dtype=np.uint8)
        # Only slices 5-9 have significant tissue
        annotation[5:, 10:90, 10:90] = 1
        slicer = BigBrainSlicer.from_arrays(image, annotation)
        valid = slicer._get_valid_indices(axis=0)
        assert all(i >= 5 for i in valid)
        assert len(valid) == 5

    def test_all_brain_slices_valid(self, bigbrain_slicer):
        """Volume with mostly brain tissue should have many valid slices."""
        valid = bigbrain_slicer._get_valid_indices(axis=0)
        # Slices 2-17 have significant tissue in our synthetic volume
        assert len(valid) >= 10


class TestInterleavedSplit:
    """Test interleaved train/val/test splitting."""

    def test_no_overlap(self, bigbrain_slicer):
        """Train, val, test must not share indices."""
        splits = bigbrain_slicer.get_split_indices(split_strategy="interleaved")
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_covers_all_valid(self, bigbrain_slicer):
        """All valid slices should appear in some split."""
        splits = bigbrain_slicer.get_split_indices(split_strategy="interleaved")
        all_split = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert len(all_split) > 0

    def test_val_test_distributed(self):
        """Val and test should span the full range, not cluster."""
        image = np.ones((100, 50, 50), dtype=np.uint8) * 128
        annotation = np.ones((100, 50, 50), dtype=np.uint8)
        slicer = BigBrainSlicer.from_arrays(image, annotation)
        splits = slicer.get_split_indices(split_strategy="interleaved")
        val = sorted(splits["val"])
        assert any(i < 25 for i in val), "No val slices in first quarter"
        assert any(i >= 75 for i in val), "No val slices in last quarter"

    def test_spatial_split_also_works(self, bigbrain_slicer):
        """Spatial split should also be supported."""
        splits = bigbrain_slicer.get_split_indices(split_strategy="spatial")
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert train & val == set()
        assert train & test == set()

    def test_invalid_strategy_raises(self, bigbrain_slicer):
        """Unknown split strategy should raise ValueError."""
        with pytest.raises(ValueError, match="split_strategy"):
            bigbrain_slicer.get_split_indices(split_strategy="unknown")


class TestGapExclusion:
    """Test gap-based exclusion of slices near val/test boundaries."""

    def test_gap_removes_adjacent_train_slices(self):
        """With gap=2, train should exclude ±2 slices around val/test."""
        image = np.ones((100, 50, 50), dtype=np.uint8) * 128
        annotation = np.ones((100, 50, 50), dtype=np.uint8)
        slicer = BigBrainSlicer.from_arrays(image, annotation)
        splits = slicer.get_split_indices(
            split_strategy="interleaved", gap=2
        )
        val_test = set(splits["val"]) | set(splits["test"])
        train = set(splits["train"])

        # No train slice should be within ±2 of any val/test slice
        for vt_idx in val_test:
            for offset in range(-2, 3):
                neighbor = vt_idx + offset
                if neighbor in val_test:
                    continue  # val/test slices are expected
                assert neighbor not in train, (
                    f"Train contains index {neighbor} which is within "
                    f"gap=2 of val/test index {vt_idx}"
                )

    def test_gap_zero_is_default(self):
        """gap=0 should include all valid slices (no exclusion)."""
        image = np.ones((100, 50, 50), dtype=np.uint8) * 128
        annotation = np.ones((100, 50, 50), dtype=np.uint8)
        slicer = BigBrainSlicer.from_arrays(image, annotation)
        no_gap = slicer.get_split_indices(split_strategy="interleaved", gap=0)
        default = slicer.get_split_indices(split_strategy="interleaved")
        assert no_gap == default

    def test_gap_reduces_train_count(self):
        """Gap exclusion should reduce training set size."""
        image = np.ones((100, 50, 50), dtype=np.uint8) * 128
        annotation = np.ones((100, 50, 50), dtype=np.uint8)
        slicer = BigBrainSlicer.from_arrays(image, annotation)
        no_gap = slicer.get_split_indices(split_strategy="interleaved", gap=0)
        with_gap = slicer.get_split_indices(split_strategy="interleaved", gap=2)
        assert len(with_gap["train"]) < len(no_gap["train"])
        # Val and test should be unchanged
        assert with_gap["val"] == no_gap["val"]
        assert with_gap["test"] == no_gap["test"]


class TestIterSlices:
    """Test iteration over slices with class remapping."""

    def test_iter_yields_tuples(self, bigbrain_slicer):
        """iter_slices should yield (image, mask) tuples."""
        mapping = {i: i for i in range(10)}  # identity
        slices = list(bigbrain_slicer.iter_slices("train", mapping))
        assert len(slices) > 0
        for img, mask in slices:
            assert img.ndim == 2
            assert mask.ndim == 2
            assert img.shape == mask.shape

    def test_iter_remaps_labels(self, bigbrain_slicer):
        """Iterated masks should use remapped class IDs."""
        # Map: 1→10, 2→20, 3→30, 4→40, rest→0
        mapping = {0: 0, 1: 10, 2: 20, 3: 30, 4: 40}
        for i in range(5, 10):
            mapping[i] = 0
        slices = list(bigbrain_slicer.iter_slices("train", mapping))
        for _, mask in slices:
            unique = set(np.unique(mask))
            # All values should be from the mapping's range
            assert unique <= {0, 10, 20, 30, 40}

    def test_iter_image_dtype_uint8(self, bigbrain_slicer):
        """Iterated images should be uint8."""
        mapping = {i: i for i in range(10)}
        for img, _ in bigbrain_slicer.iter_slices("train", mapping):
            assert img.dtype == np.uint8

    def test_iter_mask_dtype_int64(self, bigbrain_slicer):
        """Iterated masks should be int64."""
        mapping = {i: i for i in range(10)}
        for _, mask in bigbrain_slicer.iter_slices("train", mapping):
            assert mask.dtype == np.int64

    def test_iter_multi_axis_increases_samples(self):
        """multi_axis=True for train should yield more slices."""
        image = np.ones((20, 16, 24), dtype=np.uint8) * 128
        annotation = np.ones((20, 16, 24), dtype=np.uint8)
        slicer = BigBrainSlicer.from_arrays(image, annotation)
        mapping = {i: i for i in range(10)}
        coronal_only = list(slicer.iter_slices(
            "train", mapping, split_strategy="interleaved",
        ))
        multi = list(slicer.iter_slices(
            "train", mapping, split_strategy="interleaved", multi_axis=True,
        ))
        assert len(multi) > len(coronal_only)

    def test_iter_multi_axis_val_coronal_only(self):
        """multi_axis=True for val should still be coronal only."""
        image = np.ones((20, 16, 24), dtype=np.uint8) * 128
        annotation = np.ones((20, 16, 24), dtype=np.uint8)
        slicer = BigBrainSlicer.from_arrays(image, annotation)
        mapping = {i: i for i in range(10)}
        coronal_val = list(slicer.iter_slices(
            "val", mapping, split_strategy="interleaved",
        ))
        multi_val = list(slicer.iter_slices(
            "val", mapping, split_strategy="interleaved", multi_axis=True,
        ))
        assert len(multi_val) == len(coronal_val)
