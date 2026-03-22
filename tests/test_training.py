"""Tests for training utilities — written first per TDD."""

import numpy as np
import pytest
import torch
from transformers import (
    Dinov2Config,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    UperNetForSemanticSegmentation,
)

from histological_image_analysis.losses import CombinedDiceCELoss
from histological_image_analysis.training import (
    WeightedLossUperNet,
    compute_metrics,
    create_model,
    create_trainer,
    get_training_args,
    make_compute_metrics,
    preprocess_logits_for_metrics,
)

# Tiny DINOv2 config for fast unit tests (no HuggingFace download)
TINY_NUM_LABELS = 6
TINY_IMAGE_SIZE = 64


@pytest.fixture
def tiny_backbone_config() -> Dinov2Config:
    """Tiny DINOv2 config — 4 layers, 32 hidden, 64×64 input."""
    return Dinov2Config(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=TINY_IMAGE_SIZE,
        patch_size=8,
        out_features=["stage1", "stage2", "stage3", "stage4"],
        reshape_hidden_states=True,
        apply_layernorm=True,
    )


@pytest.fixture
def tiny_model(tiny_backbone_config) -> UperNetForSemanticSegmentation:
    """Tiny UperNet model with random weights."""
    return create_model(
        num_labels=TINY_NUM_LABELS,
        freeze_backbone=False,
        backbone_config=tiny_backbone_config,
        hidden_size=64,
        auxiliary_channels=32,
    )


@pytest.fixture
def synthetic_batch() -> dict[str, torch.Tensor]:
    """Synthetic batch matching dataset output format."""
    rng = torch.Generator().manual_seed(42)
    return {
        "pixel_values": torch.randn(
            2, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE, generator=rng
        ),
        "labels": torch.randint(
            0, TINY_NUM_LABELS, (2, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE), generator=rng
        ),
    }


class TestCreateModel:
    """Test model creation with various configurations."""

    def test_returns_upernet_model(self, tiny_backbone_config):
        model = create_model(
            num_labels=6,
            freeze_backbone=False,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        assert isinstance(model, UperNetForSemanticSegmentation)

    def test_correct_num_labels(self, tiny_backbone_config):
        model = create_model(
            num_labels=6,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        assert model.config.num_labels == 6

    def test_different_num_labels(self, tiny_backbone_config):
        for n in [2, 6, 10]:
            model = create_model(
                num_labels=n,
                backbone_config=tiny_backbone_config,
                hidden_size=64,
                auxiliary_channels=32,
            )
            assert model.config.num_labels == n

    def test_has_auxiliary_head(self, tiny_backbone_config):
        model = create_model(
            num_labels=6,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        assert model.auxiliary_head is not None

    def test_loss_ignore_index_set(self, tiny_backbone_config):
        model = create_model(
            num_labels=6,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
            loss_ignore_index=255,
        )
        assert model.config.loss_ignore_index == 255


class TestBackboneFreeze:
    """Test backbone parameter freezing."""

    def test_frozen_backbone_requires_grad_false(self, tiny_backbone_config):
        model = create_model(
            num_labels=6,
            freeze_backbone=True,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad, f"Backbone param {name} should be frozen"

    def test_unfrozen_backbone_requires_grad_true(self, tiny_backbone_config):
        model = create_model(
            num_labels=6,
            freeze_backbone=False,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        for name, param in model.backbone.named_parameters():
            assert param.requires_grad, f"Backbone param {name} should be trainable"

    def test_head_always_trainable(self, tiny_backbone_config):
        model = create_model(
            num_labels=6,
            freeze_backbone=True,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        head_trainable = any(
            p.requires_grad for p in model.decode_head.parameters()
        )
        assert head_trainable

    def test_auxiliary_head_always_trainable(self, tiny_backbone_config):
        model = create_model(
            num_labels=6,
            freeze_backbone=True,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        aux_trainable = any(
            p.requires_grad for p in model.auxiliary_head.parameters()
        )
        assert aux_trainable

    def test_trainable_param_count_decreases_when_frozen(self, tiny_backbone_config):
        model_unfrozen = create_model(
            num_labels=6,
            freeze_backbone=False,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        model_frozen = create_model(
            num_labels=6,
            freeze_backbone=True,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        unfrozen_count = sum(
            p.numel() for p in model_unfrozen.parameters() if p.requires_grad
        )
        frozen_count = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )
        assert frozen_count < unfrozen_count


class TestForwardPass:
    """Test model forward pass with synthetic data."""

    def test_forward_produces_loss(self, tiny_model, synthetic_batch):
        output = tiny_model(**synthetic_batch)
        assert output.loss is not None
        assert output.loss.ndim == 0  # scalar

    def test_forward_loss_is_finite(self, tiny_model, synthetic_batch):
        output = tiny_model(**synthetic_batch)
        assert torch.isfinite(output.loss)

    def test_forward_logits_shape(self, tiny_model, synthetic_batch):
        output = tiny_model(**synthetic_batch)
        b = synthetic_batch["pixel_values"].shape[0]
        h = synthetic_batch["pixel_values"].shape[2]
        w = synthetic_batch["pixel_values"].shape[3]
        assert output.logits.shape == (b, TINY_NUM_LABELS, h, w)

    def test_forward_without_labels_no_loss(self, tiny_model):
        tiny_model.eval()  # BatchNorm requires >1 sample in train mode
        x = torch.randn(1, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)
        output = tiny_model(pixel_values=x)
        assert output.loss is None
        assert output.logits is not None

    def test_forward_with_ignore_index(self, tiny_backbone_config):
        """Loss with ignored pixels should be lower than without."""
        model = create_model(
            num_labels=6,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
            loss_ignore_index=255,
        )
        rng = torch.Generator().manual_seed(99)
        x = torch.randn(2, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE, generator=rng)
        # Labels with half pixels set to ignore_index
        labels = torch.randint(
            0, 6, (2, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE), generator=rng
        )
        labels[:, : TINY_IMAGE_SIZE // 2, :] = 255
        output = model(pixel_values=x, labels=labels)
        assert torch.isfinite(output.loss)


class TestComputeMetrics:
    """Test mIoU computation."""

    def test_perfect_predictions(self):
        labels = np.array([0, 0, 1, 1, 2, 2]).reshape(1, 2, 3)
        preds = labels.copy()
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=3)
        assert result["mean_iou"] == pytest.approx(1.0)

    def test_completely_wrong_predictions(self):
        labels = np.array([0, 0, 0, 1, 1, 1]).reshape(1, 2, 3)
        preds = np.array([1, 1, 1, 0, 0, 0]).reshape(1, 2, 3)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=2)
        assert result["mean_iou"] == pytest.approx(0.0)

    def test_known_iou_values(self):
        """class 0 IoU=1/2, class 1 IoU=2/3, class 2 IoU=1.0."""
        preds = np.array([0, 0, 1, 1, 2, 2]).reshape(1, 2, 3)
        labels = np.array([0, 1, 1, 1, 2, 2]).reshape(1, 2, 3)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=3)
        assert result["iou_class_0"] == pytest.approx(0.5)
        assert result["iou_class_1"] == pytest.approx(2.0 / 3.0)
        assert result["iou_class_2"] == pytest.approx(1.0)
        expected_miou = (0.5 + 2.0 / 3.0 + 1.0) / 3.0
        assert result["mean_iou"] == pytest.approx(expected_miou)

    def test_ignore_index_excluded(self):
        preds = np.array([0, 0, 1, 1, 0, 0]).reshape(1, 2, 3)
        labels = np.array([0, 0, 1, 1, 255, 255]).reshape(1, 2, 3)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=2, ignore_index=255)
        assert result["mean_iou"] == pytest.approx(1.0)

    def test_absent_class_excluded_from_mean(self):
        """Classes not present in ground truth should not affect mIoU."""
        preds = np.array([0, 0, 1, 1]).reshape(1, 2, 2)
        labels = np.array([0, 0, 1, 1]).reshape(1, 2, 2)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=3)
        assert result["mean_iou"] == pytest.approx(1.0)

    def test_overall_accuracy(self):
        preds = np.array([0, 0, 1, 1, 2, 2]).reshape(1, 2, 3)
        labels = np.array([0, 1, 1, 1, 2, 2]).reshape(1, 2, 3)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=3)
        assert result["overall_accuracy"] == pytest.approx(5.0 / 6.0)

    def test_returns_per_class_iou_keys(self):
        preds = np.zeros((1, 4, 4), dtype=np.int64)
        labels = np.zeros((1, 4, 4), dtype=np.int64)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=6)
        assert "mean_iou" in result
        assert "overall_accuracy" in result
        for i in range(6):
            assert f"iou_class_{i}" in result

    def test_class_in_gt_but_never_predicted_returns_zero(self):
        """If class 1 exists in GT but model never predicts it, IoU=0.0 not NaN.

        This is the key diagnostic for the Cerebrum NaN issue: NaN means
        the class is absent from ground truth, not that the model fails.
        """
        # GT has classes 0 and 1, predictions are all class 0
        labels = np.array([0, 0, 1, 1, 1, 1]).reshape(1, 2, 3)
        preds = np.array([0, 0, 0, 0, 0, 0]).reshape(1, 2, 3)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=2)
        # Class 1: intersection=0, union=4, IoU=0.0
        assert result["iou_class_1"] == pytest.approx(0.0)
        assert not np.isnan(result["iou_class_1"])

    def test_absent_class_returns_nan(self):
        """Class absent from GT produces NaN and is excluded from mean."""
        labels = np.array([0, 0, 0, 2, 2, 2]).reshape(1, 2, 3)
        preds = np.array([0, 0, 0, 2, 2, 2]).reshape(1, 2, 3)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=3)
        # Class 1 absent from GT → NaN
        assert np.isnan(result["iou_class_1"])
        # Mean computed from classes 0 and 2 only (both perfect)
        assert result["mean_iou"] == pytest.approx(1.0)

    def test_class_in_gt_never_predicted_lowers_mean(self):
        """Zero IoU for unpredicted-but-present class is included in mean."""
        # Classes 0, 1, 2 in GT; model predicts only 0 and 2
        labels = np.array([0, 0, 1, 1, 2, 2]).reshape(1, 2, 3)
        preds = np.array([0, 0, 0, 0, 2, 2]).reshape(1, 2, 3)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        result = compute_metrics(eval_pred, num_labels=3)
        # Class 0: pred={0,1,2,3}, gt={0,1} → inter=2, union=4, IoU=0.5
        assert result["iou_class_0"] == pytest.approx(0.5)
        # Class 1: pred={}, gt={2,3} → inter=0, union=2, IoU=0.0
        assert result["iou_class_1"] == pytest.approx(0.0)
        # Class 2: pred={4,5}, gt={4,5} → inter=2, union=2, IoU=1.0
        assert result["iou_class_2"] == pytest.approx(1.0)
        # Mean includes all 3: (0.5 + 0.0 + 1.0) / 3 = 0.5
        assert result["mean_iou"] == pytest.approx(0.5)


class TestPreprocessLogitsForMetrics:
    """Test logit preprocessing for eval OOM prevention."""

    def test_reduces_channel_dimension(self):
        logits = torch.randn(4, 6, 64, 64)
        labels = torch.randint(0, 6, (4, 64, 64))
        result = preprocess_logits_for_metrics(logits, labels)
        assert result.shape == (4, 64, 64)

    def test_returns_class_indices(self):
        logits = torch.randn(2, 6, 32, 32)
        labels = torch.randint(0, 6, (2, 32, 32))
        result = preprocess_logits_for_metrics(logits, labels)
        assert result.min() >= 0
        assert result.max() < 6

    def test_argmax_correct(self):
        logits = torch.zeros(1, 3, 2, 2)
        logits[0, 2, 0, 0] = 10.0  # Class 2 strongest at (0,0)
        logits[0, 0, 1, 1] = 10.0  # Class 0 strongest at (1,1)
        labels = torch.zeros(1, 2, 2, dtype=torch.long)
        result = preprocess_logits_for_metrics(logits, labels)
        assert result[0, 0, 0] == 2
        assert result[0, 1, 1] == 0

    def test_output_dtype_is_int(self):
        logits = torch.randn(1, 6, 8, 8)
        labels = torch.zeros(1, 8, 8, dtype=torch.long)
        result = preprocess_logits_for_metrics(logits, labels)
        assert result.dtype == torch.int64


class TestMakeComputeMetrics:
    """Test the compute_metrics factory."""

    def test_returns_callable(self):
        fn = make_compute_metrics(num_labels=6)
        assert callable(fn)

    def test_closure_captures_num_labels(self):
        preds = np.zeros((1, 4, 4), dtype=np.int64)
        labels = np.zeros((1, 4, 4), dtype=np.int64)
        eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        fn = make_compute_metrics(num_labels=6)
        result = fn(eval_pred)
        assert "iou_class_5" in result


class TestGetTrainingArgs:
    """Test training arguments factory."""

    def test_returns_training_arguments(self, tmp_path):
        args = get_training_args(output_dir=str(tmp_path))
        assert isinstance(args, TrainingArguments)

    def test_remove_unused_columns_false(self, tmp_path):
        args = get_training_args(output_dir=str(tmp_path))
        assert args.remove_unused_columns is False

    def test_fp16_enabled_by_default(self, tmp_path):
        args = get_training_args(output_dir=str(tmp_path))
        assert args.fp16 is True

    def test_custom_learning_rate(self, tmp_path):
        args = get_training_args(output_dir=str(tmp_path), learning_rate=5e-5)
        assert args.learning_rate == 5e-5

    def test_label_names_includes_labels(self, tmp_path):
        args = get_training_args(output_dir=str(tmp_path))
        assert "labels" in args.label_names

    def test_metric_for_best_model(self, tmp_path):
        args = get_training_args(output_dir=str(tmp_path))
        assert args.metric_for_best_model == "mean_iou"

    def test_kwargs_passed_through(self, tmp_path):
        args = get_training_args(output_dir=str(tmp_path), seed=42)
        assert args.seed == 42


class TestCreateTrainer:
    """Test trainer creation."""

    def test_returns_trainer(self, tiny_model, tmp_path):
        # Minimal dict-returning dataset
        class MinimalDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "pixel_values": torch.randn(3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE),
                    "labels": torch.randint(
                        0, TINY_NUM_LABELS, (TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)
                    ),
                }

        ds = MinimalDataset()
        args = get_training_args(
            output_dir=str(tmp_path),
            num_train_epochs=1,
            report_to="none",
            fp16=False,
            use_cpu=True,  # Avoid MPS/fp16 singleton issue from prior tests
        )
        trainer = create_trainer(
            model=tiny_model,
            training_args=args,
            train_dataset=ds,
            eval_dataset=ds,
            num_labels=TINY_NUM_LABELS,
        )
        assert isinstance(trainer, Trainer)


# ===========================================================================
# TestWeightedLossUperNet
# ===========================================================================


class TestWeightedLossUperNet:
    """Test UperNet subclass with custom combined loss."""

    @pytest.fixture
    def loss_fn(self):
        weights = torch.ones(TINY_NUM_LABELS)
        weights[0] = 0.1  # low weight for background
        return CombinedDiceCELoss(
            class_weights=weights, alpha=0.5, ignore_index=255,
        )

    @pytest.fixture
    def weighted_model(self, tiny_backbone_config, loss_fn):
        return create_model(
            num_labels=TINY_NUM_LABELS,
            freeze_backbone=False,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
            loss_fn=loss_fn,
        )

    def test_returns_upernet_subclass(self, weighted_model):
        assert isinstance(weighted_model, UperNetForSemanticSegmentation)
        assert isinstance(weighted_model, WeightedLossUperNet)

    def test_forward_produces_loss(self, weighted_model, synthetic_batch):
        output = weighted_model(**synthetic_batch)
        assert output.loss is not None
        assert output.loss.ndim == 0

    def test_forward_loss_is_finite(self, weighted_model, synthetic_batch):
        output = weighted_model(**synthetic_batch)
        assert torch.isfinite(output.loss)

    def test_forward_without_labels_no_loss(self, weighted_model):
        weighted_model.eval()
        x = torch.randn(1, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)
        output = weighted_model(pixel_values=x)
        assert output.loss is None
        assert output.logits is not None

    def test_custom_loss_differs_from_default(
        self, tiny_backbone_config, loss_fn, synthetic_batch,
    ):
        """Loss value differs from default CE-only model."""
        default_model = create_model(
            num_labels=TINY_NUM_LABELS,
            freeze_backbone=False,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        custom_model = create_model(
            num_labels=TINY_NUM_LABELS,
            freeze_backbone=False,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
            loss_fn=loss_fn,
        )
        # Copy weights so only the loss function differs
        custom_model.load_state_dict(default_model.state_dict(), strict=False)

        loss_default = default_model(**synthetic_batch).loss
        loss_custom = custom_model(**synthetic_batch).loss
        # Losses should be different due to Dice component + class weights
        assert not torch.allclose(loss_default, loss_custom)

    def test_logits_shape_unchanged(self, weighted_model, synthetic_batch):
        output = weighted_model(**synthetic_batch)
        b = synthetic_batch["pixel_values"].shape[0]
        h = synthetic_batch["pixel_values"].shape[2]
        w = synthetic_batch["pixel_values"].shape[3]
        assert output.logits.shape == (b, TINY_NUM_LABELS, h, w)

    def test_backward_pass_works(self, weighted_model, synthetic_batch):
        output = weighted_model(**synthetic_batch)
        output.loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in weighted_model.parameters()
            if p.requires_grad
        )
        assert has_grad

    def test_create_model_with_loss_fn_returns_weighted(
        self, tiny_backbone_config, loss_fn,
    ):
        model = create_model(
            num_labels=TINY_NUM_LABELS,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
            loss_fn=loss_fn,
        )
        assert isinstance(model, WeightedLossUperNet)

    def test_create_model_without_loss_fn_returns_base(
        self, tiny_backbone_config,
    ):
        model = create_model(
            num_labels=TINY_NUM_LABELS,
            backbone_config=tiny_backbone_config,
            hidden_size=64,
            auxiliary_channels=32,
        )
        assert type(model) is UperNetForSemanticSegmentation
