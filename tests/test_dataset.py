"""Tests for BrainSegmentationDataset — written first per TDD."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from histological_image_analysis.dataset import BrainSegmentationDataset


@pytest.fixture
def mock_slicer():
    """Create a mock CCFv3Slicer that yields synthetic slices."""
    slicer = MagicMock()

    # 5 training slices, each 32×40 (DV×ML) to test padding
    slices = []
    for i in range(5):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(32, 40), dtype=np.uint8)
        mask = rng.integers(0, 6, size=(32, 40), dtype=np.int64)
        slices.append((img, mask))

    def iter_slices_fn(split, mapping, split_strategy="spatial"):
        if split == "train":
            yield from slices
        elif split == "val":
            yield from slices[:1]

    slicer.iter_slices = iter_slices_fn
    slicer.get_split_indices.return_value = {
        "train": list(range(5)),
        "val": [5],
        "test": [6],
    }
    return slicer


@pytest.fixture
def mock_slicer_large():
    """Mock slicer with slices larger than crop_size (518)."""
    slicer = MagicMock()

    slices = []
    for i in range(3):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(800, 1140), dtype=np.uint8)
        mask = rng.integers(0, 6, size=(800, 1140), dtype=np.int64)
        slices.append((img, mask))

    def iter_slices_fn(split, mapping, split_strategy="spatial"):
        yield from slices

    slicer.iter_slices = iter_slices_fn
    return slicer


@pytest.fixture
def mapping():
    """Simple coarse mapping for testing."""
    return {0: 0, 567: 1, 343: 2, 512: 3, 1009: 4, 73: 5}


class TestDatasetInit:
    """Test dataset creation and length."""

    def test_len_matches_slice_count(self, mock_slicer, mapping):
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        assert len(ds) == 5

    def test_len_val_split(self, mock_slicer, mapping):
        ds = BrainSegmentationDataset(
            mock_slicer, "val", mapping, crop_size=64, augment=False
        )
        assert len(ds) == 1


class TestGetItem:
    """Test __getitem__ output format."""

    def test_output_keys(self, mock_slicer, mapping):
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        assert "pixel_values" in item
        assert "labels" in item

    def test_pixel_values_shape(self, mock_slicer, mapping):
        """pixel_values should be (3, crop_size, crop_size)."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        assert item["pixel_values"].shape == (3, 64, 64)

    def test_labels_shape(self, mock_slicer, mapping):
        """labels should be (crop_size, crop_size)."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        assert item["labels"].shape == (64, 64)

    def test_pixel_values_dtype(self, mock_slicer, mapping):
        """pixel_values should be float32."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        assert item["pixel_values"].dtype == torch.float32

    def test_labels_dtype(self, mock_slicer, mapping):
        """labels should be long (int64)."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        assert item["labels"].dtype == torch.long

    def test_pixel_values_three_channels(self, mock_slicer, mapping):
        """Grayscale input should be replicated to 3 channels."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        # All three channels should have the same base values (before normalization)
        # After ImageNet normalization they'll differ slightly due to different mean/std per channel
        assert item["pixel_values"].shape[0] == 3

    def test_labels_valid_range(self, mock_slicer, mapping):
        """Label values should be in [0, num_classes-1]."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        assert item["labels"].min() >= 0
        assert item["labels"].max() <= 5  # 6 coarse classes (0-5)


class TestPaddingAndCropping:
    """Test that slices smaller than crop_size are padded."""

    def test_small_slice_padded_to_crop_size(self, mock_slicer, mapping):
        """32×40 slices should be padded to crop_size before cropping."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        # Even though input is 32×40, output should be 64×64
        assert item["pixel_values"].shape == (3, 64, 64)
        assert item["labels"].shape == (64, 64)

    def test_large_slice_cropped(self, mock_slicer_large, mapping):
        """800×1140 slices should be center-cropped when augment=False."""
        ds = BrainSegmentationDataset(
            mock_slicer_large, "train", mapping, crop_size=518, augment=False
        )
        item = ds[0]
        assert item["pixel_values"].shape == (3, 518, 518)
        assert item["labels"].shape == (518, 518)


class TestAugmentation:
    """Test that augmentation changes output stochastically."""

    def test_augmentation_changes_output(self, mock_slicer_large, mapping):
        """With augmentation, multiple calls should sometimes produce different results."""
        ds = BrainSegmentationDataset(
            mock_slicer_large, "train", mapping, crop_size=518, augment=True
        )
        # Get same item multiple times — random crop should vary
        results = [ds[0]["pixel_values"] for _ in range(5)]
        # At least some should differ (random crop on 800×1140 > 518×518)
        all_same = all(torch.equal(results[0], r) for r in results[1:])
        assert not all_same, "Augmented outputs should vary"

    def test_rotation_preserves_mask_integrity(self, mock_slicer_large, mapping):
        """Rotation should preserve integer mask values (nearest-neighbor, no artifacts)."""
        ds = BrainSegmentationDataset(
            mock_slicer_large, "train", mapping, crop_size=518, augment=True
        )
        for _ in range(10):
            item = ds[0]
            labels = item["labels"]
            assert labels.dtype == torch.long
            assert labels.min() >= 0

    def test_no_augmentation_deterministic(self, mock_slicer, mapping):
        """Without augmentation, same index should give same result."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        r1 = ds[0]["pixel_values"]
        r2 = ds[0]["pixel_values"]
        assert torch.equal(r1, r2)


class TestNormalization:
    """Test ImageNet normalization is applied correctly."""

    def test_normalized_values_reasonable_range(self, mock_slicer, mapping):
        """After ImageNet normalization, values should be roughly in [-3, 3]."""
        ds = BrainSegmentationDataset(
            mock_slicer, "train", mapping, crop_size=64, augment=False
        )
        item = ds[0]
        pv = item["pixel_values"]
        # ImageNet normalized values typically fall in [-2.5, 2.5]
        assert pv.min() >= -5.0
        assert pv.max() <= 5.0
