"""SVG annotation rasterizer for Allen Brain Atlas 2D sections.

Parses SVG path elements with structure_id attributes, converts bezier
curves to polygon points, and rasterizes to pixel-level segmentation masks.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import parse_path

from histological_image_analysis.ontology import OntologyMapper

logger = logging.getLogger(__name__)

# Number of sample points per bezier segment when converting to polygon
BEZIER_SAMPLES = 20


class SVGRasterizer:
    """Convert Allen Brain SVG annotations to pixel-level segmentation masks.

    Parameters
    ----------
    ontology_mapper : OntologyMapper
        Used for structure ID validation (optional warnings).
    """

    def __init__(self, ontology_mapper: OntologyMapper) -> None:
        self._ontology_mapper = ontology_mapper

    def rasterize(
        self,
        svg_path: str | Path,
        target_width: int,
        target_height: int,
    ) -> np.ndarray:
        """Rasterize an SVG file to a segmentation mask.

        1. Parse SVG paths with structure_id attributes
        2. Render at SVG native dimensions (background = 0)
        3. Resize to target dimensions using nearest-neighbor

        Parameters
        ----------
        svg_path : str or Path
            Path to SVG file.
        target_width : int
            Output mask width (columns).
        target_height : int
            Output mask height (rows).

        Returns
        -------
        np.ndarray
            Integer mask of shape (target_height, target_width) with
            structure IDs as pixel values.
        """
        svg_path = Path(svg_path)

        # Parse SVG dimensions
        tree = ET.parse(svg_path)
        root = tree.getroot()
        svg_width = int(root.get("width", str(target_width)))
        svg_height = int(root.get("height", str(target_height)))

        # Parse paths
        parsed_paths = self._parse_paths(str(svg_path))

        if not parsed_paths:
            return np.zeros((target_height, target_width), dtype=np.int64)

        # Create canvas at SVG native dimensions, filled with 0 (background)
        # Use 32-bit mode for structure IDs (PIL "I" mode)
        canvas = Image.new("I", (svg_width, svg_height), 0)
        draw = ImageDraw.Draw(canvas)

        # Draw paths in document order (later paths overwrite)
        for structure_id, polygon_points in parsed_paths:
            if len(polygon_points) < 3:
                continue
            # PIL expects list of (x, y) tuples
            xy = [(float(x), float(y)) for x, y in polygon_points]
            draw.polygon(xy, fill=structure_id)

        # Convert to numpy
        mask = np.array(canvas, dtype=np.int64)

        # Resize to target dimensions using nearest-neighbor
        if mask.shape != (target_height, target_width):
            canvas_resized = canvas.resize(
                (target_width, target_height), Image.NEAREST
            )
            mask = np.array(canvas_resized, dtype=np.int64)

        return mask

    def _parse_paths(
        self, svg_path: str
    ) -> list[tuple[int, list[tuple[float, float]]]]:
        """Parse SVG file and extract (structure_id, polygon_points) pairs.

        Uses namespace fallback chain to handle different SVG formats.
        Skips paths without structure_id or with invalid geometry.

        Parameters
        ----------
        svg_path : str
            Path to SVG file.

        Returns
        -------
        list of (structure_id, polygon_points) tuples.
        """
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Namespace fallback chain
        paths = root.findall(
            ".//svg:path", {"svg": "http://www.w3.org/2000/svg"}
        )
        if not paths:
            paths = root.findall(
                ".//{http://www.w3.org/2000/svg}path"
            )
        if not paths:
            paths = root.findall(".//path")

        result: list[tuple[int, list[tuple[float, float]]]] = []

        for path_elem in paths:
            # Get structure_id
            sid_str = path_elem.get("structure_id") or path_elem.get(
                "data-structure-id"
            )
            if not sid_str:
                logger.debug(
                    "Skipping path without structure_id: %s",
                    path_elem.get("id", "?"),
                )
                continue

            try:
                structure_id = int(sid_str)
            except ValueError:
                logger.warning(
                    "Invalid structure_id '%s' in path %s",
                    sid_str,
                    path_elem.get("id", "?"),
                )
                continue

            # Get path data
            d_attr = path_elem.get("d")
            if not d_attr:
                logger.warning(
                    "Path %s has no 'd' attribute", path_elem.get("id", "?")
                )
                continue

            # Convert bezier path to polygon points
            try:
                points = self._bezier_to_points(d_attr)
            except Exception:
                logger.warning(
                    "Failed to parse path d='%.50s...' for structure_id=%d",
                    d_attr,
                    structure_id,
                    exc_info=True,
                )
                continue

            if len(points) >= 3:
                result.append((structure_id, points))

        logger.info(
            "Parsed %d valid paths from %s (%d total path elements)",
            len(result),
            svg_path,
            len(paths),
        )
        return result

    @staticmethod
    def _bezier_to_points(
        d_string: str, num_samples: int = BEZIER_SAMPLES
    ) -> list[tuple[float, float]]:
        """Convert an SVG path d-string to a list of polygon points.

        Uses svgpathtools to parse the path and sample points along
        each segment.

        Parameters
        ----------
        d_string : str
            SVG path data attribute value.
        num_samples : int
            Number of sample points per path segment.

        Returns
        -------
        list of (x, y) tuples representing the polygon vertices.
        """
        path = parse_path(d_string)
        points: list[tuple[float, float]] = []

        for segment in path:
            t_values = np.linspace(0, 1, num_samples, endpoint=False)
            for t in t_values:
                pt = segment.point(t)
                points.append((pt.real, pt.imag))

        # Add the final endpoint
        if path:
            end = path[-1].point(1.0)
            points.append((end.real, end.imag))

        return points
