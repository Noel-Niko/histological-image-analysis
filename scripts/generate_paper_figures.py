#!/usr/bin/env python3
"""Generate paper figures for the brain segmentation ablation study.

Generates 6 figures as high-quality PNGs into a figures/ directory.
Some figures require the trained model and CCFv3 data; use --figures-only
to generate only data-independent figures (1, 3) without model loading.

Requires:
    - CCFv3 data at data/allen_brain_data/ccfv3/ (for Figs 2, 4, 5, 6)
    - Ontology at data/allen_brain_data/ontology/structure_graph_1.json
    - Model at ./models/dinov2-upernet-final (for Figs 2, 4, 6; skip with --figures-only)

Usage:
    # All figures (requires model + data)
    python scripts/generate_paper_figures.py --model ./models/dinov2-upernet-final

    # Data-independent figures only (no model needed)
    python scripts/generate_paper_figures.py --figures-only

    # Custom output directory
    python scripts/generate_paper_figures.py --model ./models/dinov2-upernet-final --output-dir ./my_figures

Memory:
    Sliding window inference (Fig 6) allocates a (1328, H, W) float32 buffer.
    For 800x1140 slices this is ~5.7 GB. Requires 16 GB RAM minimum.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "allen_brain_data"
NISSL_PATH = DATA_DIR / "ccfv3" / "ara_nissl_10.nrrd"
ANNOTATION_PATH = DATA_DIR / "ccfv3" / "annotation_10.nrrd"
ONTOLOGY_PATH = DATA_DIR / "ontology" / "structure_graph_1.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 518


@dataclass
class FigureConfig:
    model_path: Optional[str]
    output_dir: Path
    dpi: int
    slice_index: int
    figures_only: bool


def parse_args() -> FigureConfig:
    parser = argparse.ArgumentParser(
        description="Generate paper figures for the brain segmentation study.",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="./models/dinov2-upernet-final",
        help="Path to model directory (default: ./models/dinov2-upernet-final)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="./figures",
        help="Output directory for figures (default: ./figures)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for saved figures (default: 300)",
    )
    parser.add_argument(
        "--slice-index", type=int, default=660,
        help="AP index for representative coronal slice (default: 660, mid-brain)",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Generate only data-independent figures (1, 3) without loading model",
    )
    args = parser.parse_args()
    return FigureConfig(
        model_path=args.model,
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
        slice_index=args.slice_index,
        figures_only=args.figures_only,
    )


# ---------------------------------------------------------------------------
# Figure 1: Architecture diagram
# ---------------------------------------------------------------------------

def generate_fig1_architecture(output_dir: Path, dpi: int) -> None:
    """DINOv2-Large + UperNet architecture block diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.3", linewidth=1.5)

    # Input
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 1.8), 1.6, 1.4, **box_style, facecolor="#E8F5E9", edgecolor="#2E7D32",
    ))
    ax.text(1.0, 2.5, "Input\n518x518x3", ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrow
    ax.annotate("", xy=(2.2, 2.5), xytext=(1.8, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    # DINOv2 backbone
    backbone_x = 2.4
    # Frozen blocks (0-19)
    ax.add_patch(mpatches.FancyBboxPatch(
        (backbone_x, 1.2), 2.4, 2.8, **box_style, facecolor="#E3F2FD", edgecolor="#1565C0",
    ))
    ax.text(backbone_x + 1.2, 4.25, "DINOv2-Large Backbone", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1565C0")
    ax.text(backbone_x + 1.2, 3.6, "304M params", ha="center", va="center",
            fontsize=8, color="#555")

    # Frozen region
    ax.add_patch(mpatches.FancyBboxPatch(
        (backbone_x + 0.15, 1.4), 2.1, 1.1, boxstyle="round,pad=0.15",
        facecolor="#BBDEFB", edgecolor="#90CAF9", linewidth=1,
    ))
    ax.text(backbone_x + 1.2, 2.1, "Blocks 0-19\n(frozen)", ha="center", va="center",
            fontsize=8, color="#1565C0")

    # Unfrozen region
    ax.add_patch(mpatches.FancyBboxPatch(
        (backbone_x + 0.15, 2.7), 2.1, 1.0, boxstyle="round,pad=0.15",
        facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=1.5,
    ))
    ax.text(backbone_x + 1.2, 3.2, "Blocks 20-23\n(trainable, LR=1e-5)", ha="center",
            va="center", fontsize=8, fontweight="bold", color="#E65100")

    # Arrow from backbone to feature maps
    ax.annotate("", xy=(5.2, 2.5), xytext=(4.8, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    # Feature maps
    fm_x = 5.4
    ax.add_patch(mpatches.FancyBboxPatch(
        (fm_x, 1.5), 1.8, 2.0, **box_style, facecolor="#F3E5F5", edgecolor="#7B1FA2",
    ))
    ax.text(fm_x + 0.9, 3.1, "Multi-scale\nFeatures", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#7B1FA2")
    for i, stage in enumerate(["S6", "S12", "S18", "S24"]):
        ax.text(fm_x + 0.9, 2.5 - i * 0.25, stage, ha="center", va="center",
                fontsize=7, color="#7B1FA2")

    # Arrow to UperNet
    ax.annotate("", xy=(7.6, 2.5), xytext=(7.2, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    # UperNet
    upernet_x = 7.8
    ax.add_patch(mpatches.FancyBboxPatch(
        (upernet_x, 1.2), 2.6, 2.8, **box_style, facecolor="#FFF8E1", edgecolor="#F57F17",
    ))
    ax.text(upernet_x + 1.3, 4.25, "UperNet Decode Head", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#F57F17")
    ax.text(upernet_x + 1.3, 3.6, "38M params (LR=1e-4)", ha="center", va="center",
            fontsize=8, color="#555")

    # PSP and FPN sub-boxes
    ax.add_patch(mpatches.FancyBboxPatch(
        (upernet_x + 0.15, 2.7), 2.3, 0.7, boxstyle="round,pad=0.1",
        facecolor="#FFECB3", edgecolor="#FFA000", linewidth=1,
    ))
    ax.text(upernet_x + 1.3, 3.05, "PSP Pooling Module", ha="center", va="center",
            fontsize=8, color="#E65100")

    ax.add_patch(mpatches.FancyBboxPatch(
        (upernet_x + 0.15, 1.5), 2.3, 0.9, boxstyle="round,pad=0.1",
        facecolor="#FFECB3", edgecolor="#FFA000", linewidth=1,
    ))
    ax.text(upernet_x + 1.3, 1.95, "FPN Lateral\nConnections", ha="center", va="center",
            fontsize=8, color="#E65100")

    # Arrow to output
    ax.annotate("", xy=(10.8, 2.5), xytext=(10.4, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    # Output
    ax.add_patch(mpatches.FancyBboxPatch(
        (11.0, 1.8), 2.4, 1.4, **box_style, facecolor="#FFEBEE", edgecolor="#C62828",
    ))
    ax.text(12.2, 2.85, "Output", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#C62828")
    ax.text(12.2, 2.35, "1,328 classes\n518x518", ha="center", va="center",
            fontsize=8, color="#C62828")

    # Summary text
    ax.text(7.0, 0.6, "Total: 342M params | Trainable: 88M (25.8%) | fp16 training on NVIDIA L40S 48GB",
            ha="center", va="center", fontsize=9, color="#555", style="italic")

    plt.tight_layout()
    path = output_dir / "fig1_architecture.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Ablation bar chart (hardcoded data, no model needed)
# ---------------------------------------------------------------------------

def generate_fig3_ablation_barchart(output_dir: Path, dpi: int) -> None:
    """Horizontal bar chart of ablation mIoU deltas."""
    interventions = [
        ("Interleaved split\n(Run 1→2, 6-class)", 37.3, "positive"),
        ("Backbone unfreezing\n(Run 4→5)", 8.5, "positive"),
        ("Extended training 200ep\n(Run 5→9)", 6.0, "positive"),
        ("Sliding window eval\n(Run 9 CC→SW)", 4.4, "method"),
        ("Class pruning\n(Run 5→8a)", 0.2, "neutral"),
        ("Weighted Dice+CE\n(Run 5→6)", -1.6, "negative"),
        ("Extended augmentation\n(Run 5→7)", -6.5, "negative"),
        ("Test-time augmentation\n(Run 5 eval)", -24.4, "negative"),
    ]

    labels = [i[0] for i in interventions]
    values = [i[1] for i in interventions]
    categories = [i[2] for i in interventions]

    color_map = {
        "positive": "#4CAF50",
        "method": "#2196F3",
        "neutral": "#9E9E9E",
        "negative": "#F44336",
    }
    colors = [color_map[c] for c in categories]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Delta mIoU (%)", fontsize=11)
    ax.set_title("Ablation Study: Impact of Each Intervention on mIoU", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.invert_yaxis()

    # Value labels on bars
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        offset = 1.0 if val >= 0 else -1.0
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", ha=ha, va="center", fontsize=9, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", label="Training improvement"),
        mpatches.Patch(facecolor="#2196F3", label="Evaluation methodology"),
        mpatches.Patch(facecolor="#9E9E9E", label="Neutral"),
        mpatches.Patch(facecolor="#F44336", label="Negative result"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_xlim(-30, 45)
    plt.tight_layout()
    path = output_dir / "fig3_ablation_barchart.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: Cross-axis raw slices (data only, no model needed)
# ---------------------------------------------------------------------------

def generate_fig5_cross_axis(slicer, output_dir: Path, dpi: int) -> None:
    """3-panel showing coronal, axial, and sagittal raw slices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    slice_configs = [
        (0, 660, "Coronal (AP=660)\nTrained orientation"),
        (1, 400, "Axial (DV=400)\n3.2% mIoU — unseen"),
        (2, 570, "Sagittal (ML=570)\n0.5% mIoU — unseen"),
    ]

    for ax, (axis, idx, title) in zip(axes, slice_configs):
        img, _ = slicer.get_slice(idx, axis=axis)
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Cross-Axis Generalization: Model is Strictly Orientation-Specific",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = output_dir / "fig5_cross_axis.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Shared inference utilities
# ---------------------------------------------------------------------------

def normalize_tile(tile: np.ndarray):
    """uint8 grayscale (H, W) -> float32 tensor (1, 3, H, W)."""
    import torch

    img = tile.astype(np.float32) / 255.0
    img_3ch = np.stack([img, img, img], axis=0)
    for c in range(3):
        img_3ch[c] = (img_3ch[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return torch.from_numpy(img_3ch).unsqueeze(0)


def center_crop_predict(model, image: np.ndarray, num_labels: int, crop_size: int, device):
    """Run center-crop inference on a single grayscale image."""
    import torch

    h, w = image.shape
    y0 = max(0, (h - crop_size) // 2)
    x0 = max(0, (w - crop_size) // 2)
    crop = image[y0:y0 + crop_size, x0:x0 + crop_size]

    # Pad if image is smaller than crop_size
    if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
        padded = np.zeros((crop_size, crop_size), dtype=crop.dtype)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded

    pixel_values = normalize_tile(crop).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred, (y0, x0)


def sliding_window_predict(model, image: np.ndarray, num_labels: int,
                           crop_size: int, stride: int, device):
    """Predict full-resolution segmentation using overlapping tiles."""
    import torch

    h, w = image.shape
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    ph, pw = image.shape

    logit_sum = np.zeros((num_labels, ph, pw), dtype=np.float32)
    count_map = np.zeros((ph, pw), dtype=np.float32)

    y_starts = list(range(0, ph - crop_size + 1, stride))
    x_starts = list(range(0, pw - crop_size + 1, stride))
    if y_starts[-1] + crop_size < ph:
        y_starts.append(ph - crop_size)
    if x_starts[-1] + crop_size < pw:
        x_starts.append(pw - crop_size)

    model.eval()
    use_cuda = device.type == "cuda"
    for y in y_starts:
        for x in x_starts:
            tile = image[y:y + crop_size, x:x + crop_size]
            pixel_values = normalize_tile(tile).to(device)
            with torch.no_grad():
                if use_cuda:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(pixel_values=pixel_values).logits
                else:
                    logits = model(pixel_values=pixel_values).logits
            tile_logits = logits.squeeze(0).float().cpu().numpy()
            logit_sum[:, y:y + crop_size, x:x + crop_size] += tile_logits
            count_map[y:y + crop_size, x:x + crop_size] += 1.0

    count_map = np.maximum(count_map, 1.0)
    avg_logits = logit_sum / count_map[np.newaxis, :, :]
    pred = avg_logits.argmax(axis=0).astype(np.int64)
    return pred[:h, :w]


# ---------------------------------------------------------------------------
# Figure 2: Segmentation example (input / ground truth / prediction)
# ---------------------------------------------------------------------------

def generate_fig2_segmentation_examples(
    model, slicer, mapper, mapping, num_labels: int,
    device, output_dir: Path, dpi: int,
) -> None:
    """4x3 grid: 4 slices x (input / ground truth / prediction).

    Selects slices from different rostral-caudal positions to show
    diverse brain structures: anterior (olfactory), mid-anterior
    (striatum/cortex), mid-posterior (hippocampus/thalamus), and
    posterior (cerebellum/brainstem).
    """
    # Representative slices spanning the rostral-caudal axis
    # All are validation slices (index % 10 == 0)
    slice_configs = [
        (220, "Anterior (AP=220)\nOlfactory bulb"),
        (500, "Mid-anterior (AP=500)\nStriatum, cortex"),
        (660, "Mid-posterior (AP=660)\nHippocampus, thalamus"),
        (900, "Posterior (AP=900)\nCerebellum, brainstem"),
    ]

    n_rows = len(slice_configs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows))

    for row, (slice_idx, region_label) in enumerate(slice_configs):
        img, annot = slicer.get_slice(slice_idx, axis=0)
        gt_mask = mapper.remap_mask(annot, mapping)
        pred, (y0, x0) = center_crop_predict(model, img, num_labels, CROP_SIZE, device)

        gt_crop = gt_mask[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        img_crop = img[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]

        n_gt_classes = len(np.unique(gt_crop[gt_crop > 0]))
        n_pred_classes = len(np.unique(pred[pred > 0]))

        # Input
        axes[row, 0].imshow(img_crop, cmap="gray")
        axes[row, 0].set_ylabel(region_label, fontsize=10, fontweight="bold", rotation=0,
                                labelpad=120, va="center")
        if row == 0:
            axes[row, 0].set_title("Input (Nissl stain)", fontsize=12, fontweight="bold")
        axes[row, 0].axis("off")

        # Ground truth
        axes[row, 1].imshow(gt_crop, cmap="nipy_spectral", vmin=0, vmax=num_labels,
                            interpolation="nearest")
        if row == 0:
            axes[row, 1].set_title("Ground Truth (CCFv3)", fontsize=12, fontweight="bold")
        axes[row, 1].text(0.02, 0.02, f"{n_gt_classes} structures", transform=axes[row, 1].transAxes,
                          fontsize=8, color="white", fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
        axes[row, 1].axis("off")

        # Prediction
        axes[row, 2].imshow(pred, cmap="nipy_spectral", vmin=0, vmax=num_labels,
                            interpolation="nearest")
        if row == 0:
            axes[row, 2].set_title("Model Prediction", fontsize=12, fontweight="bold")
        axes[row, 2].text(0.02, 0.02, f"{n_pred_classes} structures", transform=axes[row, 2].transAxes,
                          fontsize=8, color="white", fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
        axes[row, 2].axis("off")

    plt.suptitle(
        "Run 9 Segmentation Examples: Input / Ground Truth / Prediction\n"
        "DINOv2-Large + UperNet, 1,328 classes, 200 epochs | 74.8% mIoU (center-crop)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = output_dir / "fig2_segmentation_examples.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Per-class IoU distribution
# ---------------------------------------------------------------------------

def generate_fig4_iou_distribution(
    model, slicer, mapper, mapping, num_labels: int,
    device, output_dir: Path, dpi: int,
) -> None:
    """Histogram of per-class IoU + scatter of log10(pixel_count) vs IoU."""
    # Evaluate on all validation slices (center-crop)
    val_indices = [i for i in range(slicer.image_volume.shape[0]) if i % 10 == 0]

    print("    Running center-crop evaluation on validation set...")
    class_tp = np.zeros(num_labels, dtype=np.int64)
    class_fp = np.zeros(num_labels, dtype=np.int64)
    class_fn = np.zeros(num_labels, dtype=np.int64)
    class_pixels = np.zeros(num_labels, dtype=np.int64)

    for idx in val_indices:
        img, annot = slicer.get_slice(idx, axis=0)
        gt_mask = mapper.remap_mask(annot, mapping)

        pred, (y0, x0) = center_crop_predict(model, img, num_labels, CROP_SIZE, device)
        gt_crop = gt_mask[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]

        for cls in range(num_labels):
            p = pred == cls
            g = gt_crop == cls
            class_tp[cls] += int((p & g).sum())
            class_fp[cls] += int((p & ~g).sum())
            class_fn[cls] += int((~p & g).sum())
            class_pixels[cls] += int(g.sum())

    # Compute IoU per class
    class_ious = {}
    class_pixel_counts = {}
    for cls in range(num_labels):
        union = class_tp[cls] + class_fp[cls] + class_fn[cls]
        if union > 0 and class_pixels[cls] > 0:
            class_ious[cls] = float(class_tp[cls]) / float(union)
            class_pixel_counts[cls] = int(class_pixels[cls])

    iou_values = list(class_ious.values())
    pixel_values = [class_pixel_counts[c] for c in class_ious]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: IoU histogram
    ax1.hist(iou_values, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
    ax1.axvline(x=np.mean(iou_values), color="#F44336", linestyle="--", linewidth=1.5,
                label=f"Mean IoU = {np.mean(iou_values):.1%}")
    ax1.set_xlabel("IoU", fontsize=11)
    ax1.set_ylabel("Number of Classes", fontsize=11)
    ax1.set_title(f"Per-Class IoU Distribution ({len(iou_values)} valid classes)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)

    # Bracket annotations
    high = sum(1 for v in iou_values if v >= 0.5)
    mid = sum(1 for v in iou_values if 0.1 <= v < 0.5)
    low = sum(1 for v in iou_values if v < 0.1)
    ax1.text(0.75, 0.85, f"IoU >= 0.5: {high} ({high/len(iou_values):.0%})",
             transform=ax1.transAxes, fontsize=9, color="#4CAF50", fontweight="bold")
    ax1.text(0.75, 0.78, f"IoU 0.1-0.5: {mid} ({mid/len(iou_values):.0%})",
             transform=ax1.transAxes, fontsize=9, color="#FF9800")
    ax1.text(0.75, 0.71, f"IoU < 0.1: {low} ({low/len(iou_values):.0%})",
             transform=ax1.transAxes, fontsize=9, color="#F44336")

    # Right: scatter of log10(pixel_count) vs IoU
    log_pixels = [np.log10(max(p, 1)) for p in pixel_values]
    ax2.scatter(log_pixels, iou_values, alpha=0.4, s=15, color="#7B1FA2", edgecolor="none")

    # Correlation
    r = np.corrcoef(log_pixels, iou_values)[0, 1]
    z = np.polyfit(log_pixels, iou_values, 1)
    poly = np.poly1d(z)
    x_line = np.linspace(min(log_pixels), max(log_pixels), 100)
    ax2.plot(x_line, poly(x_line), color="#F44336", linewidth=1.5, linestyle="--",
             label=f"r = {r:.3f}")

    ax2.set_xlabel("log₁₀(Validation Pixel Count)", fontsize=11)
    ax2.set_ylabel("IoU", fontsize=11)
    ax2.set_title("Per-Class IoU vs Pixel Count", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower right")
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = output_dir / "fig4_iou_distribution.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6: Sliding window vs center-crop comparison
# ---------------------------------------------------------------------------

def generate_fig6_sliding_vs_centercrop(
    model, slicer, mapper, mapping, num_labels: int,
    device, slice_index: int, output_dir: Path, dpi: int,
) -> None:
    """Side-by-side comparison of center-crop vs sliding window prediction."""
    img, annot = slicer.get_slice(slice_index, axis=0)
    gt_mask = mapper.remap_mask(annot, mapping)

    # Center-crop prediction
    cc_pred, (y0, x0) = center_crop_predict(model, img, num_labels, CROP_SIZE, device)

    # Sliding window prediction (full resolution)
    stride = CROP_SIZE // 2
    print("    Running sliding window inference (may take a minute)...")
    sw_pred = sliding_window_predict(model, img, num_labels, CROP_SIZE, stride, device)

    h, w = img.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: ground truth with center-crop box overlay
    axes[0].imshow(gt_mask, cmap="nipy_spectral", vmin=0, vmax=num_labels, interpolation="nearest")
    rect = plt.Rectangle((x0, y0), CROP_SIZE, CROP_SIZE, linewidth=2,
                          edgecolor="red", facecolor="none", linestyle="--")
    axes[0].add_patch(rect)
    axes[0].set_title(f"Ground Truth ({w}x{h})\nRed box = center crop area", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Middle: center-crop prediction (placed in full-slice context)
    cc_full = np.zeros_like(gt_mask)
    cc_full[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE] = cc_pred
    # Make non-cropped area a distinct value (background)
    cc_display = np.full_like(gt_mask, fill_value=-1, dtype=np.int64)
    cc_display[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE] = cc_pred

    axes[1].imshow(img, cmap="gray", alpha=0.3)
    masked = np.ma.masked_where(cc_display < 0, cc_display)
    axes[1].imshow(masked, cmap="nipy_spectral", vmin=0, vmax=num_labels, interpolation="nearest")
    cc_classes = len(np.unique(cc_pred[cc_pred > 0]))
    axes[1].set_title(f"Center-Crop Prediction\n518x518 ({cc_classes} structures, ~29% coverage)",
                      fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Right: sliding window prediction (full resolution)
    axes[2].imshow(sw_pred, cmap="nipy_spectral", vmin=0, vmax=num_labels, interpolation="nearest")
    sw_classes = len(np.unique(sw_pred[sw_pred > 0]))
    axes[2].set_title(f"Sliding Window Prediction\n{w}x{h} ({sw_classes} structures, 100% coverage)",
                      fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(
        f"Center-Crop vs Sliding Window Evaluation: AP={slice_index}\n"
        f"Sliding window reveals structures at slice edges invisible to center-crop",
        fontsize=13, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    path = output_dir / "fig6_sliding_vs_centercrop.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {config.output_dir}")
    print(f"DPI: {config.dpi}")
    print(f"Figures-only mode: {config.figures_only}")
    print()

    # --- Data-independent figures ---
    print("[1/6] Generating Figure 1: Architecture diagram...")
    generate_fig1_architecture(config.output_dir, config.dpi)

    print("[3/6] Generating Figure 3: Ablation bar chart...")
    generate_fig3_ablation_barchart(config.output_dir, config.dpi)

    if config.figures_only:
        print("\n--figures-only: Skipping model-dependent figures (2, 4, 5, 6).")
        print("To generate all figures, run without --figures-only and provide --model path.")
        return

    # --- Load data and model ---
    print("\nLoading CCFv3 data and ontology...")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from histological_image_analysis.ontology import OntologyMapper
    from histological_image_analysis.ccfv3_slicer import CCFv3Slicer

    if not NISSL_PATH.exists():
        print(f"ERROR: Nissl volume not found at {NISSL_PATH}")
        print("Download with: python scripts/download_allen_data.py")
        sys.exit(1)

    mapper = OntologyMapper(str(ONTOLOGY_PATH))
    mapping = mapper.build_full_mapping()
    num_labels = mapper.get_num_labels(mapping)

    slicer = CCFv3Slicer(
        image_path=str(NISSL_PATH),
        annotation_path=str(ANNOTATION_PATH),
        ontology_mapper=mapper,
    )
    slicer.load_volumes()
    print(f"  Volume loaded: {slicer.image_volume.shape}, {num_labels} classes")

    # Figure 5 only needs data, not model
    print("\n[5/6] Generating Figure 5: Cross-axis raw slices...")
    generate_fig5_cross_axis(slicer, config.output_dir, config.dpi)

    # Load model
    import torch
    from transformers import UperNetForSemanticSegmentation

    model_path = config.model_path
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Download with:")
        print(f"  databricks fs cp -r dbfs:/FileStore/allen_brain_data/models/final-200ep {model_path}")
        sys.exit(1)

    print(f"\nLoading model from {model_path}...")
    model = UperNetForSemanticSegmentation.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device} ({sum(p.numel() for p in model.parameters()):,} params)")

    # --- Model-dependent figures ---
    print("\n[2/6] Generating Figure 2: Segmentation examples (4 slices, input/GT/prediction)...")
    generate_fig2_segmentation_examples(
        model, slicer, mapper, mapping, num_labels, device,
        config.output_dir, config.dpi,
    )

    print("\n[4/6] Generating Figure 4: IoU distribution...")
    generate_fig4_iou_distribution(
        model, slicer, mapper, mapping, num_labels, device,
        config.output_dir, config.dpi,
    )

    print(f"\n[6/6] Generating Figure 6: Sliding window vs center-crop (slice 660)...")
    generate_fig6_sliding_vs_centercrop(
        model, slicer, mapper, mapping, num_labels, device,
        660, config.output_dir, config.dpi,
    )

    print(f"\nAll figures saved to {config.output_dir}/")


if __name__ == "__main__":
    main()
