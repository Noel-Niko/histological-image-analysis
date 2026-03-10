"""Training utilities for DINOv2-Large + UperNet semantic segmentation.

Provides model creation, metric computation, and training argument
factories for fine-tuning on brain structure segmentation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoBackbone,
    Dinov2Config,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    UperNetConfig,
    UperNetForSemanticSegmentation,
)

logger = logging.getLogger(__name__)

# DINOv2-Large feature stages (evenly spaced across 24 layers)
DINOV2_LARGE_OUT_FEATURES = ["stage6", "stage12", "stage18", "stage24"]


def preprocess_logits_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Argmax logits to prevent OOM during evaluation metric accumulation.

    HuggingFace Trainer accumulates all eval predictions in memory.
    Argmaxing reduces (B, num_labels, H, W) float32 to (B, H, W) int64.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (B, num_labels, H, W).
    labels : torch.Tensor
        Ground truth labels (unused, required by Trainer API).

    Returns
    -------
    torch.Tensor
        Predicted class IDs of shape (B, H, W), dtype int64.
    """
    return logits.argmax(dim=1)


def compute_metrics(
    eval_pred: EvalPrediction,
    num_labels: int,
    ignore_index: int = 255,
) -> dict[str, float]:
    """Compute mean IoU and per-class IoU from evaluation predictions.

    Parameters
    ----------
    eval_pred : EvalPrediction
        predictions: np.ndarray (N, H, W) — argmaxed class IDs.
        label_ids: np.ndarray (N, H, W) — ground truth labels.
    num_labels : int
        Number of classes (including background).
    ignore_index : int
        Label value to exclude from metric computation.

    Returns
    -------
    dict with "mean_iou", "overall_accuracy", and "iou_class_{i}" keys.
    """
    preds = eval_pred.predictions.ravel()
    labels = eval_pred.label_ids.ravel()

    # Mask out ignored pixels
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]

    # Overall accuracy
    total = len(labels)
    correct = (preds == labels).sum()
    accuracy = float(correct) / total if total > 0 else 0.0

    # Per-class IoU
    ious: dict[str, float] = {}
    valid_ious: list[float] = []

    for cls in range(num_labels):
        pred_mask = preds == cls
        label_mask = labels == cls
        intersection = (pred_mask & label_mask).sum()
        union = (pred_mask | label_mask).sum()

        if label_mask.sum() == 0:
            # Class absent from ground truth — exclude from mean
            ious[f"iou_class_{cls}"] = float("nan")
        elif union == 0:
            ious[f"iou_class_{cls}"] = 0.0
        else:
            iou = float(intersection) / float(union)
            ious[f"iou_class_{cls}"] = iou
            valid_ious.append(iou)

    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {"mean_iou": mean_iou, "overall_accuracy": accuracy, **ious}


def make_compute_metrics(
    num_labels: int,
    ignore_index: int = 255,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Create a compute_metrics callable for HuggingFace Trainer.

    Returns a closure capturing num_labels and ignore_index,
    matching Trainer's expected signature: f(EvalPrediction) -> dict.

    Parameters
    ----------
    num_labels : int
        Number of segmentation classes.
    ignore_index : int
        Label value to ignore.
    """

    def _compute(eval_pred: EvalPrediction) -> dict[str, float]:
        return compute_metrics(eval_pred, num_labels, ignore_index)

    return _compute


def create_model(
    num_labels: int,
    freeze_backbone: bool = True,
    pretrained_backbone_path: str | None = None,
    backbone_config: Dinov2Config | None = None,
    hidden_size: int = 512,
    auxiliary_channels: int = 256,
    auxiliary_loss_weight: float = 0.4,
    loss_ignore_index: int = 255,
) -> UperNetForSemanticSegmentation:
    """Create a UperNet model with DINOv2 backbone for semantic segmentation.

    Parameters
    ----------
    num_labels : int
        Number of segmentation classes (including background).
    freeze_backbone : bool
        If True, freeze all backbone parameters (Phase 1 head-only training).
    pretrained_backbone_path : str or None
        Local path to pretrained DINOv2 weights (from snapshot_download).
        If None, backbone uses random initialization (useful for testing).
    backbone_config : Dinov2Config or None
        Custom backbone config. If None, loads from pretrained_backbone_path
        or creates DINOv2-Large defaults.
    hidden_size : int
        UperNet decoder hidden size (default 512).
    auxiliary_channels : int
        Auxiliary head channel count (default 256).
    auxiliary_loss_weight : float
        Weight for auxiliary loss (default 0.4).
    loss_ignore_index : int
        Label value to ignore in loss computation (default 255).

    Returns
    -------
    UperNetForSemanticSegmentation
    """
    # Build backbone config
    if backbone_config is not None:
        bc = backbone_config
    elif pretrained_backbone_path is not None:
        bc = Dinov2Config.from_pretrained(
            pretrained_backbone_path,
            out_features=DINOV2_LARGE_OUT_FEATURES,
            reshape_hidden_states=True,
            apply_layernorm=True,
        )
    else:
        bc = Dinov2Config(
            out_features=DINOV2_LARGE_OUT_FEATURES,
            reshape_hidden_states=True,
            apply_layernorm=True,
        )

    # Build UperNet config
    config = UperNetConfig(
        backbone_config=bc,
        num_labels=num_labels,
        hidden_size=hidden_size,
        auxiliary_in_channels=bc.hidden_size,
        use_auxiliary_head=True,
        auxiliary_channels=auxiliary_channels,
        auxiliary_loss_weight=auxiliary_loss_weight,
        loss_ignore_index=loss_ignore_index,
    )

    # Create model (random weights everywhere)
    model = UperNetForSemanticSegmentation(config)

    # Load pretrained backbone weights if available
    if pretrained_backbone_path is not None:
        pretrained_backbone = AutoBackbone.from_pretrained(
            pretrained_backbone_path,
            out_features=DINOV2_LARGE_OUT_FEATURES,
        )
        model.backbone.load_state_dict(pretrained_backbone.state_dict())
        logger.info("Loaded pretrained backbone from %s", pretrained_backbone_path)

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model created: num_labels=%d, total=%s, trainable=%s (%.1f%%)",
        num_labels,
        f"{total_params:,}",
        f"{trainable_params:,}",
        trainable_params / total_params * 100 if total_params > 0 else 0,
    )

    return model


def get_training_args(
    output_dir: str,
    num_train_epochs: int = 50,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    fp16: bool = True,
    logging_steps: int = 10,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "mean_iou",
    greater_is_better: bool = True,
    report_to: str = "mlflow",
    dataloader_num_workers: int = 4,
    remove_unused_columns: bool = False,
    label_names: list[str] | None = None,
    **kwargs: Any,
) -> TrainingArguments:
    """Create TrainingArguments with defaults for brain segmentation.

    Parameters
    ----------
    output_dir : str
        Directory for checkpoints and logs.
    remove_unused_columns : bool
        MUST be False for dict-style datasets (default False).
    label_names : list[str] or None
        Column names containing labels (default ["labels"]).
    **kwargs
        Additional TrainingArguments parameters.

    Returns
    -------
    TrainingArguments
    """
    if label_names is None:
        label_names = ["labels"]

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to=report_to,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=remove_unused_columns,
        label_names=label_names,
        **kwargs,
    )


def create_trainer(
    model: UperNetForSemanticSegmentation,
    training_args: TrainingArguments,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    num_labels: int,
    ignore_index: int = 255,
) -> Trainer:
    """Create a HuggingFace Trainer for semantic segmentation.

    Parameters
    ----------
    model : UperNetForSemanticSegmentation
        The model to train.
    training_args : TrainingArguments
        Training configuration.
    train_dataset : Dataset
        Training dataset (BrainSegmentationDataset).
    eval_dataset : Dataset
        Evaluation dataset (BrainSegmentationDataset).
    num_labels : int
        Number of segmentation classes.
    ignore_index : int
        Label value to ignore in metrics.

    Returns
    -------
    Trainer
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=make_compute_metrics(num_labels, ignore_index),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
