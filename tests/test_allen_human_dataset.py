"""Tests for AllenHumanDataset — written first per TDD."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from histological_image_analysis.dataset import (
    AllenHumanDataset,
    split_by_donor,
)


@pytest.fixture
def sample_human_image_path(fixtures_dir):
    return fixtures_dir / "sample_human_image.jpg"


@pytest.fixture
def sample_human_svg_path(fixtures_dir):
    return fixtures_dir / "sample_human.svg"


@pytest.fixture
def human_mapping():
    """Mapping from structure IDs in sample_human.svg to class IDs.

    sample_human.svg has structure_ids: 4039, 4045, 9366, 9364.
    Map them to contiguous class IDs 1-4.
    """
    return {
        0: 0,
        4039: 1,
        4045: 2,
        9366: 3,
        9364: 4,
    }


@pytest.fixture
def mock_rasterizer():
    """Mock SVGRasterizer that returns a mask with known structure IDs."""
    rasterizer = MagicMock()

    def rasterize_fn(svg_path, target_width, target_height, sparse=False):
        bg = 255 if sparse else 0
        mask = np.full((target_height, target_width), bg, dtype=np.int64)
        # Fill a region with structure_id 4039
        h4 = target_height // 4
        w4 = target_width // 4
        mask[h4:h4*2, w4:w4*2] = 4039
        mask[h4*2:h4*3, w4*2:w4*3] = 4045
        return mask

    rasterizer.rasterize = rasterize_fn
    return rasterizer


@pytest.fixture
def image_svg_pairs(sample_human_image_path, sample_human_svg_path):
    """List of (image_path, svg_path) pairs for dataset."""
    return [(sample_human_image_path, sample_human_svg_path)] * 5


class TestAllenHumanDatasetInit:
    """Test dataset creation."""

    def test_len_matches_pairs(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        assert len(ds) == 5

    def test_empty_pairs(self, mock_rasterizer, human_mapping):
        ds = AllenHumanDataset(
            image_svg_pairs=[],
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        assert len(ds) == 0


class TestAllenHumanGetItem:
    """Test __getitem__ output format."""

    def test_output_keys(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        assert "pixel_values" in item
        assert "labels" in item

    def test_pixel_values_shape(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """pixel_values should be (3, crop_size, crop_size)."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        assert item["pixel_values"].shape == (3, 64, 64)

    def test_labels_shape(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """labels should be (crop_size, crop_size)."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        assert item["labels"].shape == (64, 64)

    def test_pixel_values_dtype(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """pixel_values should be float32."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        assert item["pixel_values"].dtype == torch.float32

    def test_labels_dtype(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """labels should be long (int64)."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        assert item["labels"].dtype == torch.long


class TestAllenHumanSparseLabels:
    """Test that sparse annotation labels are handled correctly."""

    def test_labels_contain_ignore_index(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """Labels should contain 255 (ignore_index) for unlabeled pixels."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        unique = set(item["labels"].numpy().ravel())
        assert 255 in unique, f"Expected 255 in labels, got {unique}"

    def test_labels_contain_class_ids(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """Labels should contain mapped class IDs (not raw structure IDs)."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        unique = set(item["labels"].numpy().ravel())
        # Should have class IDs from mapping (1, 2) and ignore (255)
        # Should NOT have raw structure IDs (4039, 4045)
        assert 4039 not in unique
        assert 4045 not in unique
        class_ids_found = unique - {255}
        assert len(class_ids_found) >= 1


class TestAllenHumanRGBNormalization:
    """Test RGB normalization (not grayscale replication)."""

    def test_three_channels_distinct(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """RGB channels should have distinct values after normalization.

        Unlike grayscale (same base value per channel), RGB images have
        different values per channel BEFORE normalization, so after
        applying different ImageNet mean/std per channel, they diverge.
        """
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        pv = item["pixel_values"]
        # RGB channels should not all be identical
        # (grayscale replicated would make them identical before normalization,
        # but here they come from a true RGB image)
        ch0 = pv[0].numpy()
        ch1 = pv[1].numpy()
        ch2 = pv[2].numpy()
        # At least one pair of channels should differ
        assert not (
            np.allclose(ch0, ch1) and np.allclose(ch1, ch2)
        ), "RGB channels should not all be identical"

    def test_normalized_values_reasonable_range(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """After ImageNet normalization, values should be roughly in [-3, 3]."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        item = ds[0]
        pv = item["pixel_values"]
        assert pv.min() >= -5.0
        assert pv.max() <= 5.0


class TestAllenHumanAugmentation:
    """Test augmentation on RGB images."""

    def test_augmentation_changes_output(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """With augmentation, multiple calls should produce different results."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=True,
        )
        results = [ds[0]["pixel_values"] for _ in range(10)]
        all_same = all(torch.equal(results[0], r) for r in results[1:])
        assert not all_same, "Augmented outputs should vary"

    def test_no_augmentation_deterministic(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """Without augmentation, same index should give same result."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
        )
        r1 = ds[0]["pixel_values"]
        r2 = ds[0]["pixel_values"]
        assert torch.equal(r1, r2)

    def test_augmentation_preserves_ignore_index(
        self, image_svg_pairs, mock_rasterizer, human_mapping
    ):
        """Augmentation fill value for mask should be 255 (ignore), not 0."""
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=True,
        )
        for _ in range(10):
            item = ds[0]
            labels = item["labels"]
            # 255 should still be present after augmentation
            assert 255 in labels.numpy()


class TestAllenHumanCache:
    """Test pre-cache build and cached loading."""

    def test_build_cache_creates_npz_files(
        self, image_svg_pairs, mock_rasterizer, human_mapping, tmp_path,
    ):
        """build_cache should create one .npz file per pair."""
        cache_dir = tmp_path / "cache"
        count = AllenHumanDataset.build_cache(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            cache_dir=cache_dir,
            max_dim=64,
        )
        # All 5 pairs share the same image stem, so only 1 unique file
        assert count == 5
        npz_files = list(cache_dir.glob("*.npz"))
        assert len(npz_files) == 1

    def test_build_cache_npz_contents(
        self, image_svg_pairs, mock_rasterizer, human_mapping, tmp_path,
    ):
        """Cached .npz should contain image (H,W,3) and mask (H,W)."""
        cache_dir = tmp_path / "cache"
        AllenHumanDataset.build_cache(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            cache_dir=cache_dir,
            max_dim=64,
        )
        npz_file = list(cache_dir.glob("*.npz"))[0]
        data = np.load(npz_file)
        assert "image" in data
        assert "mask" in data
        assert data["image"].ndim == 3  # (H, W, 3)
        assert data["image"].shape[2] == 3
        assert data["mask"].ndim == 2  # (H, W)

    def test_build_cache_resizes_large_images(
        self, image_svg_pairs, mock_rasterizer, human_mapping, tmp_path,
    ):
        """Images larger than max_dim should be resized."""
        cache_dir = tmp_path / "cache"
        AllenHumanDataset.build_cache(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            cache_dir=cache_dir,
            max_dim=64,
        )
        npz_file = list(cache_dir.glob("*.npz"))[0]
        data = np.load(npz_file)
        h, w = data["image"].shape[:2]
        assert max(h, w) <= 64

    def test_build_cache_skips_existing(
        self, image_svg_pairs, mock_rasterizer, human_mapping, tmp_path,
    ):
        """build_cache should skip files that already exist."""
        cache_dir = tmp_path / "cache"
        AllenHumanDataset.build_cache(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            cache_dir=cache_dir,
            max_dim=64,
        )
        # Second call should skip (no rasterizer calls)
        mock_rasterizer_2 = MagicMock()
        count = AllenHumanDataset.build_cache(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer_2,
            mapping=human_mapping,
            cache_dir=cache_dir,
            max_dim=64,
        )
        assert count == 5
        # Rasterizer should not have been called (all files exist)
        mock_rasterizer_2.rasterize.assert_not_called()

    def test_cached_dataset_returns_correct_shapes(
        self, image_svg_pairs, mock_rasterizer, human_mapping, tmp_path,
    ):
        """Dataset with cache_dir should return same shapes as uncached."""
        cache_dir = tmp_path / "cache"
        AllenHumanDataset.build_cache(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            cache_dir=cache_dir,
            max_dim=128,
        )
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
            cache_dir=cache_dir,
        )
        item = ds[0]
        assert item["pixel_values"].shape == (3, 64, 64)
        assert item["labels"].shape == (64, 64)

    def test_cached_dataset_contains_mapped_labels(
        self, image_svg_pairs, mock_rasterizer, human_mapping, tmp_path,
    ):
        """Cached dataset labels should have class IDs, not structure IDs."""
        cache_dir = tmp_path / "cache"
        AllenHumanDataset.build_cache(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            cache_dir=cache_dir,
            max_dim=128,
        )
        ds = AllenHumanDataset(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            crop_size=64,
            augment=False,
            cache_dir=cache_dir,
        )
        item = ds[0]
        unique = set(item["labels"].numpy().ravel())
        assert 4039 not in unique
        assert 4045 not in unique

    def test_cached_mask_preserves_255(
        self, image_svg_pairs, mock_rasterizer, human_mapping, tmp_path,
    ):
        """Cached masks should preserve 255 (ignore_index)."""
        cache_dir = tmp_path / "cache"
        AllenHumanDataset.build_cache(
            image_svg_pairs=image_svg_pairs,
            rasterizer=mock_rasterizer,
            mapping=human_mapping,
            cache_dir=cache_dir,
            max_dim=128,
        )
        npz_file = list(cache_dir.glob("*.npz"))[0]
        data = np.load(npz_file)
        assert 255 in data["mask"]


class TestSplitByDonor:
    """Test donor-based splitting function."""

    @pytest.fixture
    def sample_pairs(self, tmp_path):
        """Create mock (image, svg) pairs with donor IDs in paths."""
        pairs = []
        donors = {
            "H0351.2002": 5,
            "H0351.2001": 4,
            "H0351.1012": 3,
            "H0351.1009": 3,
            "H0351.1016": 2,
            "H0351.1015": 2,
        }
        for donor, count in donors.items():
            donor_dir = tmp_path / donor
            donor_dir.mkdir()
            for i in range(count):
                img_path = donor_dir / f"image_{i}.jpg"
                svg_path = donor_dir / f"annotation_{i}.svg"
                img_path.touch()
                svg_path.touch()
                pairs.append((img_path, svg_path, donor))
        return pairs

    def test_splits_by_donor(self, sample_pairs):
        train_donors = ["H0351.2002", "H0351.2001", "H0351.1012", "H0351.1009"]
        val_donors = ["H0351.1016"]
        test_donors = ["H0351.1015"]

        splits = split_by_donor(
            sample_pairs, train_donors, val_donors, test_donors
        )
        assert len(splits["train"]) == 5 + 4 + 3 + 3  # 15
        assert len(splits["val"]) == 2
        assert len(splits["test"]) == 2

    def test_no_overlap(self, sample_pairs):
        train_donors = ["H0351.2002", "H0351.2001"]
        val_donors = ["H0351.1016"]
        test_donors = ["H0351.1015"]

        splits = split_by_donor(
            sample_pairs, train_donors, val_donors, test_donors
        )
        train_imgs = {p[0] for p in splits["train"]}
        val_imgs = {p[0] for p in splits["val"]}
        test_imgs = {p[0] for p in splits["test"]}
        assert train_imgs & val_imgs == set()
        assert train_imgs & test_imgs == set()
        assert val_imgs & test_imgs == set()

    def test_returns_image_svg_tuples(self, sample_pairs):
        train_donors = ["H0351.2002"]
        val_donors = ["H0351.1016"]
        test_donors = ["H0351.1015"]

        splits = split_by_donor(
            sample_pairs, train_donors, val_donors, test_donors
        )
        for img_path, svg_path in splits["train"]:
            assert isinstance(img_path, Path)
            assert isinstance(svg_path, Path)
