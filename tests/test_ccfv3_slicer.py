"""Tests for CCFv3Slicer — written first per TDD."""

import numpy as np
import pytest

from histological_image_analysis.ccfv3_slicer import CCFv3Slicer
from histological_image_analysis.ontology import OntologyMapper


@pytest.fixture
def mapper(minimal_ontology_path):
    return OntologyMapper(minimal_ontology_path)


class TestNormalization:
    """Test volume normalization for different dtypes."""

    def test_normalize_float32_to_uint8(self, mapper):
        """float32 nissl volumes should be percentile-clipped and scaled to uint8."""
        slicer = CCFv3Slicer.__new__(CCFv3Slicer)
        # Simulate float32 with outliers
        arr = np.array([0.0, 1.0, 50.0, 100.0, 5000.0], dtype=np.float32)
        result = slicer._normalize_image(arr)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_normalize_float32_clips_outliers(self, mapper):
        """Extreme values should be clipped at 1st/99th percentile."""
        slicer = CCFv3Slicer.__new__(CCFv3Slicer)
        rng = np.random.default_rng(42)
        arr = rng.uniform(10.0, 100.0, size=(100,)).astype(np.float32)
        arr[0] = -1000.0  # extreme low
        arr[1] = 99999.0  # extreme high
        result = slicer._normalize_image(arr)
        # After clipping, min should be 0 and max should be 255
        assert result[0] == 0
        assert result[1] == 255

    def test_normalize_uint16_to_uint8(self, mapper):
        """uint16 template volumes should be scaled to uint8."""
        slicer = CCFv3Slicer.__new__(CCFv3Slicer)
        arr = np.array([0, 100, 32768, 65535], dtype=np.uint16)
        result = slicer._normalize_image(arr)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[-1] == 255


class TestLoadVolumes:
    """Test volume loading from numpy arrays (avoiding NRRD dependency in tests)."""

    def test_load_validates_matching_shapes(
        self, mapper, synthetic_volume, tmp_path
    ):
        """Image and annotation must have identical shapes."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        assert slicer.image_volume.shape == slicer.annotation_volume.shape

    def test_load_rejects_mismatched_shapes(self, mapper):
        """Mismatched shapes should raise ValueError."""
        image = np.zeros((10, 8, 12), dtype=np.float32)
        annotation = np.zeros((10, 8, 6), dtype=np.uint32)
        with pytest.raises(ValueError, match="shape"):
            CCFv3Slicer.from_arrays(image, annotation, mapper)


class TestGetSlice:
    """Test coronal slice extraction."""

    def test_slice_shape_matches_dv_ml(self, mapper, synthetic_volume):
        """Coronal slice should have shape (DV, ML)."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        img_slice, annot_slice = slicer.get_slice(5)
        assert img_slice.shape == (8, 12)  # DV × ML
        assert annot_slice.shape == (8, 12)

    def test_slice_image_is_uint8(self, mapper, synthetic_volume):
        """Image slice should be normalized to uint8."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        img_slice, _ = slicer.get_slice(5)
        assert img_slice.dtype == np.uint8

    def test_slice_annotation_preserves_ids(self, mapper, synthetic_volume):
        """Annotation slice should keep raw structure IDs."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        _, annot_slice = slicer.get_slice(5)
        # Slice 5 should have structure IDs from the synthetic volume
        assert 567 in annot_slice or 688 in annot_slice or 343 in annot_slice

    def test_slice_out_of_range_raises(self, mapper, synthetic_volume):
        """AP index out of range should raise IndexError."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        with pytest.raises(IndexError):
            slicer.get_slice(100)


class TestSpatialSplit:
    """Test train/val/test splitting by AP position."""

    def test_split_no_overlap(self, mapper, synthetic_volume):
        """Train, val, and test sets must not share AP indices."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices()
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_split_covers_valid_slices(self, mapper, synthetic_volume):
        """All valid (non-background) slices should appear in some split."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices()
        all_split = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        # At least some slices should be valid
        assert len(all_split) > 0

    def test_split_skips_mostly_background(self, mapper):
        """Slices with <10% brain tissue should be excluded."""
        # Create volume where slices 0-4 are mostly background
        image = np.random.default_rng(42).uniform(
            0, 100, size=(10, 100, 100)
        ).astype(np.float32)
        annotation = np.zeros((10, 100, 100), dtype=np.uint32)
        # Only slices 5-9 have significant brain tissue
        annotation[5:, 10:90, 10:90] = 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices()
        all_indices = (
            set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        )
        # Slices 0-4 should be excluded (mostly background)
        for i in range(5):
            assert i not in all_indices

    def test_split_fractions_approximately_correct(self, mapper):
        """Split proportions should roughly match 80/10/10."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(100, 50, 50)
        ).astype(np.float32)
        annotation = np.ones((100, 50, 50), dtype=np.uint32) * 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices()
        total = (
            len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        )
        assert total == 100
        assert 70 <= len(splits["train"]) <= 90
        assert 5 <= len(splits["val"]) <= 15
        assert 5 <= len(splits["test"]) <= 15


class TestInterleavedSplit:
    """Test interleaved train/val/test splitting for class coverage."""

    def test_interleaved_no_overlap(self, mapper, synthetic_volume):
        """Train, val, and test sets must not share AP indices."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices(split_strategy="interleaved")
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_interleaved_covers_valid_slices(self, mapper, synthetic_volume):
        """All valid slices should appear in some split."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices(split_strategy="interleaved")
        all_split = (
            set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        )
        # Should cover same valid indices as spatial split
        spatial = slicer.get_split_indices(split_strategy="spatial")
        spatial_all = (
            set(spatial["train"]) | set(spatial["val"]) | set(spatial["test"])
        )
        assert all_split == spatial_all

    def test_interleaved_fractions_approximately_correct(self, mapper):
        """Split proportions should roughly match 80/10/10."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(100, 50, 50)
        ).astype(np.float32)
        annotation = np.ones((100, 50, 50), dtype=np.uint32) * 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices(split_strategy="interleaved")
        total = (
            len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        )
        assert total == 100
        assert 70 <= len(splits["train"]) <= 90
        assert 5 <= len(splits["val"]) <= 15
        assert 5 <= len(splits["test"]) <= 15

    def test_interleaved_distributes_across_ap_range(self, mapper):
        """Val/test indices should span the full AP range, not cluster."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(100, 50, 50)
        ).astype(np.float32)
        annotation = np.ones((100, 50, 50), dtype=np.uint32) * 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices(split_strategy="interleaved")

        val_indices = sorted(splits["val"])
        test_indices = sorted(splits["test"])

        # Val should have indices in both first and last quarter
        assert any(i < 25 for i in val_indices), "No val slices in first quarter"
        assert any(i >= 75 for i in val_indices), "No val slices in last quarter"

        # Same for test
        assert any(i < 25 for i in test_indices), "No test slices in first quarter"
        assert any(i >= 75 for i in test_indices), "No test slices in last quarter"

    def test_interleaved_val_test_are_regular(self, mapper):
        """Val and test indices should be approximately evenly spaced."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(100, 50, 50)
        ).astype(np.float32)
        annotation = np.ones((100, 50, 50), dtype=np.uint32) * 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        splits = slicer.get_split_indices(split_strategy="interleaved")

        val_indices = sorted(splits["val"])
        # Check that gaps between consecutive val indices are uniform
        gaps = [val_indices[i+1] - val_indices[i] for i in range(len(val_indices)-1)]
        # All gaps should be the same (stride)
        assert len(set(gaps)) == 1, f"Val indices not evenly spaced: gaps={gaps}"

    def test_invalid_strategy_raises(self, mapper, synthetic_volume):
        """Unknown split strategy should raise ValueError."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        with pytest.raises(ValueError, match="split_strategy"):
            slicer.get_split_indices(split_strategy="unknown")

    def test_spatial_strategy_unchanged(self, mapper, synthetic_volume):
        """Explicitly passing 'spatial' should produce same result as default."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        default = slicer.get_split_indices()
        explicit = slicer.get_split_indices(split_strategy="spatial")
        assert default == explicit


class TestMultiAxisSlicing:
    """Test multi-axis slicing (Step 12, Step 3)."""

    def test_get_slice_axis0_coronal(self, mapper, synthetic_volume):
        """Axis 0 (coronal) slice shape should be (DV, ML)."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        img, annot = slicer.get_slice(5, axis=0)
        assert img.shape == (8, 12)
        assert annot.shape == (8, 12)

    def test_get_slice_axis1_axial(self, mapper, synthetic_volume):
        """Axis 1 (axial) slice shape should be (AP, ML)."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        img, annot = slicer.get_slice(3, axis=1)
        assert img.shape == (10, 12)
        assert annot.shape == (10, 12)

    def test_get_slice_axis2_sagittal(self, mapper, synthetic_volume):
        """Axis 2 (sagittal) slice shape should be (AP, DV)."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        img, annot = slicer.get_slice(5, axis=2)
        assert img.shape == (10, 8)
        assert annot.shape == (10, 8)

    def test_get_slice_default_axis_is_coronal(self, mapper, synthetic_volume):
        """Default axis should be 0 (coronal), matching existing behavior."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        default = slicer.get_slice(5)
        explicit = slicer.get_slice(5, axis=0)
        assert np.array_equal(default[0], explicit[0])
        assert np.array_equal(default[1], explicit[1])

    def test_get_slice_invalid_axis_raises(self, mapper, synthetic_volume):
        """Invalid axis should raise ValueError."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        with pytest.raises(ValueError, match="axis"):
            slicer.get_slice(0, axis=3)

    def test_get_slice_axis1_out_of_range_raises(self, mapper, synthetic_volume):
        """Axial index out of range should raise IndexError."""
        image, annotation = synthetic_volume  # shape (10, 8, 12)
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        with pytest.raises(IndexError):
            slicer.get_slice(8, axis=1)  # DV dim is 8

    def test_get_valid_indices_axis0(self, mapper):
        """Valid coronal indices should work as before."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(10, 100, 100)
        ).astype(np.float32)
        annotation = np.zeros((10, 100, 100), dtype=np.uint32)
        annotation[5:, 10:90, 10:90] = 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        valid = slicer._get_valid_indices(axis=0)
        # Only slices 5-9 have significant brain tissue
        assert all(i >= 5 for i in valid)
        assert len(valid) == 5

    def test_get_valid_indices_axis1(self, mapper):
        """Valid axial indices should respect 10% threshold."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(10, 8, 12)
        ).astype(np.float32)
        annotation = np.zeros((10, 8, 12), dtype=np.uint32)
        # Fill DV rows 3-7 with brain tissue across full AP and ML range
        annotation[:, 3:8, :] = 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        valid = slicer._get_valid_indices(axis=1)
        # DV indices 3-7 should have brain tissue
        assert all(i >= 3 for i in valid)
        assert len(valid) >= 5

    def test_get_valid_indices_axis2(self, mapper):
        """Valid sagittal indices should respect 10% threshold."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(10, 8, 12)
        ).astype(np.float32)
        annotation = np.zeros((10, 8, 12), dtype=np.uint32)
        # Fill ML columns 2-10 with brain tissue
        annotation[:, :, 2:11] = 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        valid = slicer._get_valid_indices(axis=2)
        assert all(2 <= i <= 10 for i in valid)
        assert len(valid) >= 9

    def test_get_valid_indices_backward_compat(self, mapper, synthetic_volume):
        """_get_valid_indices(axis=0) should match _get_valid_ap_indices()."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        ap_valid = slicer._get_valid_ap_indices()
        axis0_valid = slicer._get_valid_indices(axis=0)
        assert ap_valid == axis0_valid

    def test_iter_slices_multi_axis_train_more_samples(self, mapper):
        """Multi-axis training should yield more slices than coronal only."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(20, 16, 24)
        ).astype(np.float32)
        annotation = np.ones((20, 16, 24), dtype=np.uint32) * 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        mapping = mapper.build_coarse_mapping()

        coronal_only = list(slicer.iter_slices(
            "train", mapping, split_strategy="interleaved",
        ))
        multi_axis = list(slicer.iter_slices(
            "train", mapping, split_strategy="interleaved", multi_axis=True,
        ))
        assert len(multi_axis) > len(coronal_only)

    def test_iter_slices_multi_axis_val_coronal_only(self, mapper):
        """Multi-axis val should be identical to coronal-only val."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(20, 16, 24)
        ).astype(np.float32)
        annotation = np.ones((20, 16, 24), dtype=np.uint32) * 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        mapping = mapper.build_coarse_mapping()

        coronal_val = list(slicer.iter_slices(
            "val", mapping, split_strategy="interleaved",
        ))
        multi_val = list(slicer.iter_slices(
            "val", mapping, split_strategy="interleaved", multi_axis=True,
        ))
        assert len(multi_val) == len(coronal_val)
        for (img1, mask1), (img2, mask2) in zip(coronal_val, multi_val):
            assert np.array_equal(img1, img2)
            assert np.array_equal(mask1, mask2)

    def test_iter_slices_multi_axis_yields_valid_shapes(self, mapper):
        """All multi-axis slices should be valid 2D images."""
        image = np.random.default_rng(42).uniform(
            0, 100, size=(20, 16, 24)
        ).astype(np.float32)
        annotation = np.ones((20, 16, 24), dtype=np.uint32) * 567
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        mapping = mapper.build_coarse_mapping()

        for img, mask in slicer.iter_slices(
            "train", mapping, split_strategy="interleaved", multi_axis=True,
        ):
            assert img.ndim == 2
            assert mask.ndim == 2
            assert img.shape == mask.shape
            assert img.dtype == np.uint8


class TestIterSlices:
    """Test iteration over slices with remapping."""

    def test_iter_yields_remapped_masks(self, mapper, synthetic_volume):
        """Iterated masks should use class IDs from the mapping."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        mapping = mapper.build_coarse_mapping()
        slices = list(
            slicer.iter_slices("train", mapping)
        )
        assert len(slices) > 0
        for img, mask in slices:
            assert img.dtype == np.uint8
            # Mask values should be coarse class IDs (0-5)
            assert mask.max() <= 5
            assert mask.min() >= 0

    def test_iter_image_and_mask_same_shape(self, mapper, synthetic_volume):
        """Image and mask from iterator should have matching shapes."""
        image, annotation = synthetic_volume
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)
        mapping = mapper.build_coarse_mapping()
        for img, mask in slicer.iter_slices("train", mapping):
            assert img.shape == mask.shape
