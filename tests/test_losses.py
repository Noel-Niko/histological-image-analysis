"""Tests for loss functions — written first per TDD."""

import numpy as np
import pytest
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from histological_image_analysis.losses import (
    CombinedDiceCELoss,
    DiceLoss,
    compute_class_weights_from_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_CLASSES = 4
IMAGE_SIZE = 16


class FakeDataset(Dataset):
    """Dataset with controllable label distributions for testing weights."""

    def __init__(self, labels_list: list[torch.Tensor]):
        self._labels = labels_list

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {"pixel_values": torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
                "labels": self._labels[idx]}


@pytest.fixture
def uniform_dataset() -> FakeDataset:
    """Each of 4 classes gets exactly 64 pixels across 1 sample."""
    labels = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
    # 4x16 horizontal bands: class 0, 1, 2, 3
    for cls in range(NUM_CLASSES):
        labels[cls * 4 : (cls + 1) * 4, :] = cls
    return FakeDataset([labels])


@pytest.fixture
def imbalanced_dataset() -> FakeDataset:
    """Class 0 = 240 px, class 1 = 12 px, class 2 = 4 px, class 3 = 0 px."""
    labels = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
    # Almost all class 0
    labels[:, :] = 0
    # 12 pixels of class 1
    labels[0, :12] = 1
    # 4 pixels of class 2
    labels[1, :4] = 2
    # Class 3 absent
    return FakeDataset([labels])


@pytest.fixture
def ignore_dataset() -> FakeDataset:
    """Half pixels are ignore_index=255."""
    labels = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
    labels[:8, :] = 0
    labels[8:, :] = 255  # ignored
    return FakeDataset([labels])


def _make_logits(predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Create one-hot-like logits from integer predictions.

    Puts 10.0 at the predicted class and 0.0 elsewhere, so softmax
    is very close to 1.0 at the predicted class.
    """
    B, H, W = predictions.shape
    logits = torch.zeros(B, num_classes, H, W)
    for b in range(B):
        for c in range(num_classes):
            logits[b, c][predictions[b] == c] = 10.0
    return logits


# ===========================================================================
# TestComputeClassWeights
# ===========================================================================


class TestComputeClassWeights:
    """Test inverse-frequency weight computation."""

    def test_uniform_distribution_gives_equal_weights(self, uniform_dataset):
        weights = compute_class_weights_from_dataset(
            uniform_dataset, num_labels=NUM_CLASSES,
        )
        # All classes equally frequent → weights should be approximately equal
        assert weights.shape == (NUM_CLASSES,)
        assert torch.allclose(weights, weights[0].expand(NUM_CLASSES), atol=0.01)

    def test_imbalanced_gives_higher_weight_to_rare(self, imbalanced_dataset):
        weights = compute_class_weights_from_dataset(
            imbalanced_dataset, num_labels=NUM_CLASSES,
        )
        # Class 2 (4 px) should have higher weight than class 0 (240 px)
        assert weights[2] > weights[0]
        # Class 1 (12 px) should have higher weight than class 0 (240 px)
        assert weights[1] > weights[0]

    def test_zero_count_class_gets_zero_weight(self, imbalanced_dataset):
        weights = compute_class_weights_from_dataset(
            imbalanced_dataset, num_labels=NUM_CLASSES,
        )
        # Class 3 has 0 pixels → weight should be 0.0
        assert weights[3] == 0.0

    def test_weights_sum_to_num_labels(self, uniform_dataset):
        weights = compute_class_weights_from_dataset(
            uniform_dataset, num_labels=NUM_CLASSES,
        )
        assert weights.sum().item() == pytest.approx(NUM_CLASSES, abs=0.01)

    def test_ignore_index_not_counted(self, ignore_dataset):
        weights = compute_class_weights_from_dataset(
            ignore_dataset, num_labels=2, ignore_index=255,
        )
        # Only class 0 has pixels (128 px), class 1 has 0
        assert weights[0] > 0.0
        assert weights[1] == 0.0

    def test_clip_percentile_caps_extreme_weights(self):
        """Very rare class weight is capped at clip percentile."""
        # Class 0 = 254 px, class 1 = 1 px, class 2 = 1 px
        labels = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
        labels[0, 0] = 1
        labels[0, 1] = 2
        ds = FakeDataset([labels])
        weights_clipped = compute_class_weights_from_dataset(
            ds, num_labels=3, clip_percentile=50.0,
        )
        weights_unclipped = compute_class_weights_from_dataset(
            ds, num_labels=3, clip_percentile=100.0,
        )
        # Clipped weights for rare classes should be <= unclipped
        assert weights_clipped[1] <= weights_unclipped[1]

    def test_output_shape_matches_num_labels(self, uniform_dataset):
        for n in [2, 6, 10]:
            weights = compute_class_weights_from_dataset(
                uniform_dataset, num_labels=n,
            )
            assert weights.shape == (n,)

    def test_output_dtype_is_float32(self, uniform_dataset):
        weights = compute_class_weights_from_dataset(
            uniform_dataset, num_labels=NUM_CLASSES,
        )
        assert weights.dtype == torch.float32


# ===========================================================================
# TestDiceLoss
# ===========================================================================


class TestDiceLoss:
    """Test Dice loss computation."""

    def test_perfect_predictions_give_near_zero_loss(self):
        """When predictions match labels exactly, Dice loss is close to 0."""
        labels = torch.tensor([[[0, 0, 1, 1], [2, 2, 3, 3]]])
        logits = _make_logits(labels, num_classes=4)
        loss_fn = DiceLoss(ignore_index=255)
        loss = loss_fn(logits, labels)
        assert loss.item() < 0.05

    def test_completely_wrong_predictions_give_high_loss(self):
        """All-wrong predictions give Dice loss close to 1."""
        labels = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
        preds = torch.tensor([[[1, 1, 1, 1], [0, 0, 0, 0]]])
        logits = _make_logits(preds, num_classes=2)
        loss_fn = DiceLoss(ignore_index=255)
        loss = loss_fn(logits, labels)
        assert loss.item() > 0.8

    def test_known_dice_value(self):
        """Verify exact Dice score for a known configuration.

        Class 0: pred={0,1}, gt={0,1,2,3} → intersection=2, union_sum=4+2=6
                 Dice = 2*2 / 6 = 0.667
        Class 1: pred={2,3}, gt=empty → excluded
        So only class 0 contributes. We predict all as class 0 but GT is
        half class 0, half class 1.
        """
        labels = torch.tensor([[[0, 0], [1, 1]]])  # (1, 2, 2)
        preds = torch.tensor([[[0, 0], [0, 0]]])
        logits = _make_logits(preds, num_classes=2)
        loss_fn = DiceLoss(ignore_index=255, smooth=0.0)
        loss = loss_fn(logits, labels)
        # Class 0: gt has 2 px, pred has 4 px → intersection prob ~1.0 for 2 px
        # Class 1: gt has 2 px, pred has 0 prob → Dice ~0
        # Mean Dice ~ (0.667 + 0) / 2, so loss ~ 1 - 0.333 = 0.667
        # With strong logits, probs are nearly 1-hot.
        assert 0.6 < loss.item() < 0.75

    def test_ignore_index_excluded(self):
        """Pixels with ignore_index=255 do not affect loss."""
        labels_clean = torch.tensor([[[0, 0], [1, 1]]])
        labels_dirty = torch.tensor([[[0, 0], [1, 255]]])
        preds = torch.tensor([[[0, 0], [0, 0]]])
        logits = _make_logits(preds, num_classes=2)

        loss_fn = DiceLoss(ignore_index=255)
        loss_clean = loss_fn(logits, labels_clean)
        loss_dirty = loss_fn(logits, labels_dirty)
        # Losses differ because dirty version excludes one pixel
        assert not torch.isnan(loss_dirty)
        assert torch.isfinite(loss_dirty)

    def test_absent_class_excluded_from_mean(self):
        """Classes not present in labels do not affect average Dice."""
        labels = torch.tensor([[[0, 0], [0, 0]]])  # Only class 0
        preds = torch.tensor([[[0, 0], [0, 0]]])
        logits = _make_logits(preds, num_classes=4)
        loss_fn = DiceLoss(ignore_index=255)
        loss = loss_fn(logits, labels)
        # Only class 0 present, predicted perfectly → loss ~0.0
        assert loss.item() < 0.05

    def test_output_is_scalar(self):
        labels = torch.randint(0, 3, (2, 8, 8))
        logits = torch.randn(2, 3, 8, 8)
        loss_fn = DiceLoss(ignore_index=255)
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_output_is_finite(self):
        labels = torch.randint(0, 3, (2, 8, 8))
        logits = torch.randn(2, 3, 8, 8)
        loss_fn = DiceLoss(ignore_index=255)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_gradients_flow(self):
        labels = torch.randint(0, 3, (2, 8, 8))
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        loss_fn = DiceLoss(ignore_index=255)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_chunk_size_does_not_affect_result(self):
        """Same result with chunk_size=2 and chunk_size=1000."""
        labels = torch.randint(0, 5, (2, 16, 16))
        logits = torch.randn(2, 5, 16, 16)
        loss_small = DiceLoss(ignore_index=255, chunk_size=2)(logits, labels)
        loss_large = DiceLoss(ignore_index=255, chunk_size=1000)(logits, labels)
        assert torch.allclose(loss_small, loss_large, atol=1e-6)

    def test_batch_size_one(self):
        labels = torch.randint(0, 3, (1, 8, 8))
        logits = torch.randn(1, 3, 8, 8)
        loss_fn = DiceLoss(ignore_index=255)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_many_classes(self):
        """Works with num_labels=100."""
        labels = torch.randint(0, 100, (2, 16, 16))
        logits = torch.randn(2, 100, 16, 16)
        loss_fn = DiceLoss(ignore_index=255, chunk_size=10)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)


# ===========================================================================
# TestCombinedDiceCELoss
# ===========================================================================


class TestCombinedDiceCELoss:
    """Test combined Dice + CE loss."""

    def test_alpha_1_equals_pure_ce(self):
        """alpha=1.0 should equal CrossEntropyLoss (unweighted)."""
        labels = torch.randint(0, 4, (2, 8, 8))
        logits = torch.randn(2, 4, 8, 8)

        combined = CombinedDiceCELoss(alpha=1.0, ignore_index=255)
        ce_only = CrossEntropyLoss(ignore_index=255)

        loss_combined = combined(logits, labels)
        loss_ce = ce_only(logits, labels)
        assert torch.allclose(loss_combined, loss_ce, atol=1e-5)

    def test_alpha_0_equals_pure_dice(self):
        """alpha=0.0 should equal DiceLoss."""
        labels = torch.randint(0, 4, (2, 8, 8))
        logits = torch.randn(2, 4, 8, 8)

        combined = CombinedDiceCELoss(alpha=0.0, ignore_index=255)
        dice_only = DiceLoss(ignore_index=255)

        loss_combined = combined(logits, labels)
        loss_dice = dice_only(logits, labels)
        assert torch.allclose(loss_combined, loss_dice, atol=1e-5)

    def test_default_alpha_mixes_both(self):
        """alpha=0.5 gives value between pure CE and pure Dice."""
        labels = torch.randint(0, 4, (2, 8, 8))
        logits = torch.randn(2, 4, 8, 8)

        ce_loss = CrossEntropyLoss(ignore_index=255)(logits, labels)
        dice_loss = DiceLoss(ignore_index=255)(logits, labels)
        combined_loss = CombinedDiceCELoss(alpha=0.5, ignore_index=255)(logits, labels)

        expected = 0.5 * ce_loss + 0.5 * dice_loss
        assert torch.allclose(combined_loss, expected, atol=1e-5)

    def test_class_weights_affect_ce_component(self):
        """Weighted CE produces different loss than unweighted CE."""
        labels = torch.randint(0, 4, (2, 8, 8))
        logits = torch.randn(2, 4, 8, 8)

        weights = torch.tensor([0.1, 1.0, 5.0, 10.0])
        weighted = CombinedDiceCELoss(
            class_weights=weights, alpha=1.0, ignore_index=255,
        )
        unweighted = CombinedDiceCELoss(alpha=1.0, ignore_index=255)

        loss_w = weighted(logits, labels)
        loss_u = unweighted(logits, labels)
        assert not torch.allclose(loss_w, loss_u)

    def test_ignore_index_respected_in_both(self):
        """ignore_index=255 excluded from both CE and Dice."""
        labels = torch.tensor([[[0, 0], [255, 255]]])
        logits = torch.randn(1, 2, 2, 2)

        loss_fn = CombinedDiceCELoss(alpha=0.5, ignore_index=255)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_output_is_scalar_and_finite(self):
        labels = torch.randint(0, 4, (2, 8, 8))
        logits = torch.randn(2, 4, 8, 8)
        loss_fn = CombinedDiceCELoss(ignore_index=255)
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_gradients_flow_through_both_components(self):
        labels = torch.randint(0, 4, (2, 8, 8))
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        loss_fn = CombinedDiceCELoss(alpha=0.5, ignore_index=255)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_no_class_weights_uses_unweighted_ce(self):
        """When class_weights=None, CE is unweighted."""
        labels = torch.randint(0, 4, (2, 8, 8))
        logits = torch.randn(2, 4, 8, 8)

        loss_none = CombinedDiceCELoss(
            class_weights=None, alpha=1.0, ignore_index=255,
        )
        loss_ref = CrossEntropyLoss(ignore_index=255)

        assert torch.allclose(
            loss_none(logits, labels), loss_ref(logits, labels), atol=1e-5,
        )
