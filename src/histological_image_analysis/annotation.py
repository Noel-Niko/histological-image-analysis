"""Annotation overlay generation for brain region segmentation.

Creates annotated images with color-coded brain regions overlaid on the
original histological image, plus a legend identifying detected structures.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Overlay transparency for non-background regions
OVERLAY_ALPHA = 0.4

# Maximum number of regions to show in the legend
MAX_LEGEND_REGIONS = 15

# Legend panel width in pixels
LEGEND_WIDTH = 280

# Legend styling
LEGEND_BG_COLOR = (255, 255, 255)
LEGEND_TEXT_COLOR = (30, 30, 30)
LEGEND_SWATCH_SIZE = 16
LEGEND_LINE_HEIGHT = 22
LEGEND_PADDING = 12


def build_annotated_filename(input_path: str, species: str = "mouse") -> str:
    """Build the annotated output filename for a given input image.

    Places the output in the same directory as the input, with format:
    ``{stem}-annotated-{species}-{YYYYMMDDTHHMMSS}.png``

    Parameters
    ----------
    input_path : str
        Path to the original input image.
    species : str
        Model species identifier (e.g. "mouse", "human-allen", "human-bigbrain").

    Returns
    -------
    str
        Absolute path for the annotated output file.
    """
    p = Path(input_path)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_name = f"{p.stem}-annotated-{species}-{timestamp}.png"
    return str(p.parent / output_name)


def resolve_model_path(
    species: str, model_path: Optional[str] = None
) -> str:
    """Resolve model directory path from species flag or explicit path.

    Parameters
    ----------
    species : str
        One of "mouse", "human", or "human-bigbrain".
    model_path : str, optional
        Explicit model path. If provided, returned unchanged.

    Returns
    -------
    str
        Path to the model directory.
    """
    if model_path is not None:
        return model_path
    return str(Path("models") / species)


def _get_region_colors(num_classes: int) -> np.ndarray:
    """Generate distinct colors for each class using nipy_spectral colormap.

    Parameters
    ----------
    num_classes : int
        Total number of classes (including background at index 0).

    Returns
    -------
    np.ndarray
        Shape (num_classes, 3) array of RGB colors as uint8.
    """
    colormap = matplotlib.colormaps.get_cmap("nipy_spectral").resampled(max(num_classes, 2))
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        rgba = colormap(i / max(num_classes - 1, 1))
        colors[i] = (np.array(rgba[:3]) * 255).astype(np.uint8)
    return colors


def _compute_region_stats(
    prediction: np.ndarray,
    id2label: Dict,
) -> List[Tuple[int, str, int]]:
    """Compute per-region pixel areas, sorted descending.

    Parameters
    ----------
    prediction : np.ndarray
        (H, W) array of class IDs.
    id2label : dict
        Mapping from class ID (int or str) to region name.

    Returns
    -------
    list of (class_id, region_name, pixel_count)
        Sorted by pixel_count descending, background (class 0) excluded.
    """
    unique, counts = np.unique(prediction, return_counts=True)
    stats = []
    for cls_id, count in zip(unique, counts):
        if cls_id == 0:
            continue
        # id2label keys may be int or str depending on how HF serializes config
        name = id2label.get(cls_id, id2label.get(str(cls_id), f"Region {cls_id}"))
        stats.append((int(cls_id), name, int(count)))
    stats.sort(key=lambda x: x[2], reverse=True)
    return stats


def _draw_legend(
    height: int,
    region_stats: List[Tuple[int, str, int]],
    colors: np.ndarray,
) -> Image.Image:
    """Draw a legend panel listing top regions with color swatches.

    Parameters
    ----------
    height : int
        Height of the legend panel (matches the image height).
    region_stats : list of (class_id, name, pixel_count)
        Regions to display, pre-sorted by area.
    colors : np.ndarray
        Color array indexed by class_id.

    Returns
    -------
    PIL.Image.Image
        Legend panel as an RGB image.
    """
    legend = Image.new("RGB", (LEGEND_WIDTH, height), LEGEND_BG_COLOR)
    draw = ImageDraw.Draw(legend)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_title = font

    y = LEGEND_PADDING
    draw.text(
        (LEGEND_PADDING, y), "Detected Regions",
        fill=LEGEND_TEXT_COLOR, font=font_title,
    )
    y += LEGEND_LINE_HEIGHT + 4

    # Divider line
    draw.line(
        [(LEGEND_PADDING, y), (LEGEND_WIDTH - LEGEND_PADDING, y)],
        fill=(180, 180, 180), width=1,
    )
    y += 8

    entries = region_stats[:MAX_LEGEND_REGIONS]
    total_pixels = sum(s[2] for s in region_stats)

    for cls_id, name, count in entries:
        if y + LEGEND_LINE_HEIGHT > height - LEGEND_PADDING:
            break

        color = tuple(int(c) for c in colors[cls_id % len(colors)])
        swatch_x = LEGEND_PADDING
        swatch_y = y + 2

        draw.rectangle(
            [swatch_x, swatch_y,
             swatch_x + LEGEND_SWATCH_SIZE, swatch_y + LEGEND_SWATCH_SIZE],
            fill=color, outline=(100, 100, 100),
        )

        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        # Truncate long names
        display_name = name if len(name) <= 22 else name[:20] + ".."
        label = f"{display_name} ({pct:.1f}%)"
        draw.text(
            (swatch_x + LEGEND_SWATCH_SIZE + 6, y),
            label, fill=LEGEND_TEXT_COLOR, font=font,
        )
        y += LEGEND_LINE_HEIGHT

    if len(region_stats) > MAX_LEGEND_REGIONS:
        remaining = len(region_stats) - MAX_LEGEND_REGIONS
        draw.text(
            (LEGEND_PADDING, y + 4),
            f"+ {remaining} more regions",
            fill=(120, 120, 120), font=font,
        )

    return legend


def _draw_contours(overlay: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Draw thin boundary lines between adjacent regions.

    Parameters
    ----------
    overlay : np.ndarray
        (H, W, 3) RGB overlay image to draw on (modified in place).
    prediction : np.ndarray
        (H, W) class ID array.

    Returns
    -------
    np.ndarray
        The overlay with black contour lines drawn at region boundaries.
    """
    h, w = prediction.shape
    boundary = np.zeros((h, w), dtype=bool)

    # Check horizontal and vertical neighbors
    if h > 1:
        boundary[:-1, :] |= prediction[:-1, :] != prediction[1:, :]
        boundary[1:, :] |= prediction[:-1, :] != prediction[1:, :]
    if w > 1:
        boundary[:, :-1] |= prediction[:, :-1] != prediction[:, 1:]
        boundary[:, 1:] |= prediction[:, :-1] != prediction[:, 1:]

    # Only draw contours on non-background pixels to preserve background
    fg_boundary = boundary & (prediction > 0)
    overlay[fg_boundary] = [0, 0, 0]
    return overlay


def create_annotated_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    id2label: Dict,
    alpha: float = OVERLAY_ALPHA,
) -> Image.Image:
    """Create an annotated overlay image with color-coded brain regions and legend.

    Parameters
    ----------
    image : np.ndarray
        Original image as (H, W, 3) uint8 RGB array.
    prediction : np.ndarray
        Segmentation prediction as (H, W) array of class IDs.
        Must be same spatial dimensions as image.
    id2label : dict
        Mapping from class ID to human-readable region name.
    alpha : float
        Overlay transparency (0 = fully transparent, 1 = fully opaque).

    Returns
    -------
    PIL.Image.Image
        Annotated image with overlay and legend panel.
    """
    h, w = image.shape[:2]
    num_classes = int(prediction.max()) + 1
    colors = _get_region_colors(max(num_classes, 2))

    # Build color overlay
    color_overlay = colors[prediction]  # (H, W, 3)

    # Alpha-blend: only non-background pixels
    result = image.copy().astype(np.float32)
    fg_mask = prediction > 0
    result[fg_mask] = (
        result[fg_mask] * (1 - alpha) + color_overlay[fg_mask].astype(np.float32) * alpha
    )
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Draw contour boundaries
    result = _draw_contours(result, prediction)

    # Compute region stats for legend
    region_stats = _compute_region_stats(prediction, id2label)

    # Draw legend panel
    legend = _draw_legend(h, region_stats, colors)

    # Combine image + legend horizontally
    result_pil = Image.fromarray(result)
    combined = Image.new("RGB", (w + LEGEND_WIDTH, h), LEGEND_BG_COLOR)
    combined.paste(result_pil, (0, 0))
    combined.paste(legend, (w, 0))

    return combined
