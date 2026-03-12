"""Loss functions for class-imbalanced semantic segmentation.

Provides Dice loss, class-weighted cross-entropy, and a combined loss
for brain structure segmentation with highly imbalanced classes.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def compute_class_weights_from_dataset(
    dataset: Dataset,
    num_labels: int,
    ignore_index: int = 255,
    clip_percentile: float = 95.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights by scanning all labels.

    Parameters
    ----------
    dataset : Dataset
        Training dataset whose ``__getitem__`` returns
        ``{"labels": Tensor(H, W)}``.
    num_labels : int
        Total number of classes (including background).
    ignore_index : int
        Label value to exclude from counting.
    clip_percentile : float
        Clip weight values at this percentile to prevent extreme weights
        for very rare classes.

    Returns
    -------
    torch.Tensor
        Shape ``(num_labels,)``, dtype float32.  Normalised so the
        non-zero weights sum to ``num_labels``.  Classes with zero
        pixels get weight ``0.0``.
    """
    counts = np.zeros(num_labels, dtype=np.int64)

    for idx in range(len(dataset)):
        labels = dataset[idx]["labels"].numpy().ravel()
        valid = labels[labels != ignore_index]
        # bincount may return fewer bins than num_labels
        bc = np.bincount(valid, minlength=num_labels)
        counts += bc[:num_labels]

    # Inverse frequency
    epsilon = 1e-6
    raw_weights = np.zeros(num_labels, dtype=np.float64)
    nonzero_mask = counts > 0
    raw_weights[nonzero_mask] = 1.0 / (counts[nonzero_mask].astype(np.float64) + epsilon)

    # Clip at percentile (only among nonzero weights)
    if nonzero_mask.any() and clip_percentile < 100.0:
        threshold = np.percentile(raw_weights[nonzero_mask], clip_percentile)
        raw_weights = np.minimum(raw_weights, threshold)

    # Normalise so non-zero weights sum to num_labels
    weight_sum = raw_weights.sum()
    if weight_sum > 0:
        raw_weights = raw_weights / weight_sum * num_labels

    return torch.tensor(raw_weights, dtype=torch.float32)


class DiceLoss(nn.Module):
    """Per-class soft Dice loss for semantic segmentation.

    Uses softmax probabilities for differentiable Dice computation.
    Processes classes in chunks to limit memory when ``num_labels``
    is large (e.g. 1,328 classes).

    Parameters
    ----------
    ignore_index : int
        Label value to mask out before Dice computation.
    smooth : float
        Smoothing factor to prevent division by zero.
    chunk_size : int
        Number of classes to process simultaneously.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        smooth: float = 1.0,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.chunk_size = chunk_size

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean Dice loss over classes present in the batch.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(B, C, H, W)`` — raw model output.
        labels : torch.Tensor
            Shape ``(B, H, W)`` — integer class IDs.

        Returns
        -------
        torch.Tensor
            Scalar loss value ``1 - mean_dice``.
        """
        B, C, H, W = logits.shape

        # Valid pixel mask
        valid = labels != self.ignore_index  # (B, H, W)
        labels_clean = labels.clone()
        labels_clean[~valid] = 0

        # Softmax probabilities (same memory footprint as logits)
        probs = logits.softmax(dim=1)  # (B, C, H, W)

        dice_scores: list[torch.Tensor] = []

        for c_start in range(0, C, self.chunk_size):
            c_end = min(c_start + self.chunk_size, C)
            chunk_ids = torch.arange(c_start, c_end, device=logits.device)

            # One-hot target for this chunk: (B, chunk, H, W)
            target_chunk = (
                labels_clean.unsqueeze(1) == chunk_ids.view(1, -1, 1, 1)
            ).float()

            # Zero out invalid pixels
            valid_mask = valid.unsqueeze(1).float()  # (B, 1, H, W)
            target_chunk = target_chunk * valid_mask

            # Probability slice for this chunk
            prob_chunk = probs[:, c_start:c_end, :, :] * valid_mask

            # Per-class Dice: reduce over batch and spatial dims
            # Cast to fp32 for numerical stability
            intersection = (prob_chunk * target_chunk).to(torch.float32).sum(dim=(0, 2, 3))
            cardinality = (prob_chunk + target_chunk).to(torch.float32).sum(dim=(0, 2, 3))

            # Which classes are present in this batch?
            present = target_chunk.sum(dim=(0, 2, 3)) > 0

            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

            for j in range(len(chunk_ids)):
                if present[j]:
                    dice_scores.append(dice[j])

        if len(dice_scores) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


class CombinedDiceCELoss(nn.Module):
    """Combined weighted cross-entropy + Dice loss.

    ``L = alpha * weighted_CE + (1 - alpha) * Dice``

    Parameters
    ----------
    class_weights : torch.Tensor or None
        Per-class weights for ``CrossEntropyLoss``.  Shape
        ``(num_labels,)``.  If ``None``, unweighted CE is used.
    alpha : float
        Balance between CE (``alpha``) and Dice (``1 - alpha``).
    ignore_index : int
        Label value to ignore in both components.
    dice_smooth : float
        Smoothing factor for Dice loss.
    dice_chunk_size : int
        Number of classes to process at once in Dice computation.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        alpha: float = 0.5,
        ignore_index: int = 255,
        dice_smooth: float = 1.0,
        dice_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

        # CE component
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Dice component
        self.dice = DiceLoss(
            ignore_index=ignore_index,
            smooth=dice_smooth,
            chunk_size=dice_chunk_size,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(B, C, H, W)``.
        labels : torch.Tensor
            Shape ``(B, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        loss = torch.tensor(0.0, device=logits.device)

        if self.alpha > 0.0:
            ce = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights,
                ignore_index=self.ignore_index,
            )
            loss = loss + self.alpha * ce

        if self.alpha < 1.0:
            dice = self.dice(logits, labels)
            loss = loss + (1.0 - self.alpha) * dice

        return loss
