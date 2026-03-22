"""Tests for SVGRasterizer — written first per TDD."""

import numpy as np
import pytest

from histological_image_analysis.ontology import OntologyMapper
from histological_image_analysis.svg_rasterizer import SVGRasterizer


@pytest.fixture
def mapper(minimal_ontology_path):
    return OntologyMapper(minimal_ontology_path)


@pytest.fixture
def rasterizer(mapper):
    return SVGRasterizer(mapper)


class TestRasterize:
    """Test SVG → pixel mask rasterization."""

    def test_output_shape_matches_target(self, rasterizer, sample_svg_path):
        """Output mask should match the requested target dimensions."""
        mask = rasterizer.rasterize(sample_svg_path, target_width=100, target_height=50)
        # PIL convention: width=columns, height=rows
        assert mask.shape == (50, 100)

    def test_background_is_zero(self, rasterizer, sample_svg_path):
        """Pixels not covered by any path should be 0 (background)."""
        mask = rasterizer.rasterize(sample_svg_path, target_width=200, target_height=100)
        # The SVG fixture has paths that don't cover the entire canvas
        # At least some pixels should be background
        assert 0 in mask

    def test_contains_expected_structure_ids(self, rasterizer, sample_svg_path):
        """Mask should contain the structure IDs from the SVG paths."""
        # Rasterize at native SVG dimensions for clearest results
        mask = rasterizer.rasterize(sample_svg_path, target_width=200, target_height=100)
        unique_ids = set(np.unique(mask))
        # sample.svg has structure_ids: 567 (Cerebrum), 343 (Brain stem), 1009 (fiber tracts)
        # At least some of these should appear
        brain_ids = {567, 343, 1009}
        found = unique_ids & brain_ids
        assert len(found) >= 1, f"Expected some of {brain_ids}, got {unique_ids}"

    def test_overlap_later_path_wins(self, rasterizer, sample_svg_path):
        """When paths overlap, the later path in document order should win."""
        # In sample.svg:
        # path1: rect (10,10)-(90,90) with structure_id=567
        # path2: rect (110,10)-(190,90) with structure_id=343
        # path3: rect (50,40)-(150,60) with structure_id=1009 (overlaps both)
        # At SVG native size (200×100), path3 should overwrite parts of path1 and path2
        mask = rasterizer.rasterize(sample_svg_path, target_width=200, target_height=100)
        # Center of path3 overlap area (around x=100, y=50 in SVG coords)
        # should be structure_id 1009
        center_val = mask[50, 100]
        assert center_val == 1009

    def test_dtype_is_integer(self, rasterizer, sample_svg_path):
        """Mask should have an integer dtype."""
        mask = rasterizer.rasterize(sample_svg_path, target_width=200, target_height=100)
        assert np.issubdtype(mask.dtype, np.integer)


class TestParsePaths:
    """Test SVG path parsing."""

    def test_parses_all_paths(self, rasterizer, sample_svg_path):
        """Should find all paths with structure_id in the SVG."""
        paths = rasterizer._parse_paths(str(sample_svg_path))
        assert len(paths) == 3  # sample.svg has 3 paths

    def test_parsed_structure_ids(self, rasterizer, sample_svg_path):
        """Parsed paths should have the correct structure IDs."""
        paths = rasterizer._parse_paths(str(sample_svg_path))
        structure_ids = [sid for sid, _ in paths]
        assert 567 in structure_ids
        assert 343 in structure_ids
        assert 1009 in structure_ids

    def test_parsed_polygons_have_points(self, rasterizer, sample_svg_path):
        """Each parsed path should have polygon points."""
        paths = rasterizer._parse_paths(str(sample_svg_path))
        for _, points in paths:
            assert len(points) >= 3  # At least a triangle


class TestErrorHandling:
    """Test graceful handling of malformed SVG data."""

    def test_missing_structure_id_skipped(self, rasterizer, tmp_path):
        """Paths without structure_id should be skipped."""
        svg_content = """<?xml version="1.0"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <path d="M10,10 L90,10 L90,90 L10,90 Z" style="fill:#FF0000"/>
  <path structure_id="567" d="M20,20 L80,20 L80,80 L20,80 Z" style="fill:#00FF00"/>
</svg>"""
        svg_file = tmp_path / "no_sid.svg"
        svg_file.write_text(svg_content)
        paths = rasterizer._parse_paths(str(svg_file))
        assert len(paths) == 1
        assert paths[0][0] == 567

    def test_empty_svg_returns_zero_mask(self, rasterizer, tmp_path):
        """SVG with no paths should return all-zero mask."""
        svg_content = """<?xml version="1.0"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
</svg>"""
        svg_file = tmp_path / "empty.svg"
        svg_file.write_text(svg_content)
        mask = rasterizer.rasterize(svg_file, target_width=50, target_height=50)
        assert mask.shape == (50, 50)
        assert np.all(mask == 0)


# ──────────────────────────────────────────────────────────────────────
# Human SVG (<polygon>) Tests
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_human_svg_path(fixtures_dir):
    return fixtures_dir / "sample_human.svg"


class TestParsePolygons:
    """Test parsing of human SVG <polygon> elements."""

    def test_parses_polygons_with_structure_id(self, rasterizer, sample_human_svg_path):
        """Should find all <polygon> elements with non-empty structure_id."""
        parsed = rasterizer._parse_polygons(str(sample_human_svg_path))
        # sample_human.svg has 4 polygons with valid structure_ids:
        # p1(4039), p2(4045), p4(9366), p5(9364)
        # p3 has structure_id="" → skipped
        assert len(parsed) == 4

    def test_parsed_structure_ids(self, rasterizer, sample_human_svg_path):
        """Parsed polygons should have correct structure IDs."""
        parsed = rasterizer._parse_polygons(str(sample_human_svg_path))
        structure_ids = [sid for sid, _ in parsed]
        assert 4039 in structure_ids
        assert 4045 in structure_ids
        assert 9366 in structure_ids
        assert 9364 in structure_ids

    def test_empty_structure_id_skipped(self, rasterizer, sample_human_svg_path):
        """Polygons with structure_id="" should be skipped."""
        parsed = rasterizer._parse_polygons(str(sample_human_svg_path))
        structure_ids = [sid for sid, _ in parsed]
        # No structure_id from the empty-sid polygon should appear
        assert len(structure_ids) == 4

    def test_polygon_points_parsed_correctly(self, rasterizer, sample_human_svg_path):
        """Points should be parsed as (x, y) tuples."""
        parsed = rasterizer._parse_polygons(str(sample_human_svg_path))
        for sid, points in parsed:
            assert len(points) >= 3, f"Polygon {sid} has <3 points"
            for x, y in points:
                assert isinstance(x, float) or isinstance(x, int)
                assert isinstance(y, float) or isinstance(y, int)

    def test_first_polygon_points_values(self, rasterizer, sample_human_svg_path):
        """First polygon (4039) should have points: (10,10),(90,10),(90,50),(10,50)."""
        parsed = rasterizer._parse_polygons(str(sample_human_svg_path))
        # Find the polygon with structure_id 4039
        p4039 = [pts for sid, pts in parsed if sid == 4039]
        assert len(p4039) == 1
        points = p4039[0]
        assert len(points) == 4
        assert points[0] == (10.0, 10.0)
        assert points[1] == (90.0, 10.0)
        assert points[2] == (90.0, 50.0)
        assert points[3] == (10.0, 50.0)


class TestRasterizeHumanSVG:
    """Test rasterization of human SVGs with <polygon> elements."""

    def test_output_shape(self, rasterizer, sample_human_svg_path):
        """Output mask should match target dimensions."""
        mask = rasterizer.rasterize(
            sample_human_svg_path, target_width=200, target_height=100
        )
        assert mask.shape == (100, 200)

    def test_contains_structure_ids(self, rasterizer, sample_human_svg_path):
        """Mask should contain the structure IDs from polygons."""
        mask = rasterizer.rasterize(
            sample_human_svg_path, target_width=200, target_height=100
        )
        unique_ids = set(np.unique(mask))
        expected = {4039, 4045, 9366, 9364}
        found = unique_ids & expected
        assert len(found) >= 2, f"Expected some of {expected}, got {unique_ids}"

    def test_overlap_later_polygon_wins(self, rasterizer, sample_human_svg_path):
        """Later polygon in document order should overwrite earlier."""
        mask = rasterizer.rasterize(
            sample_human_svg_path, target_width=200, target_height=100
        )
        # p5 (structure_id=9364) at (50,30)-(150,70) overlaps p1 (4039) and p2 (4045)
        # At center ~(100, 50), p5 should win
        center_val = mask[50, 100]
        assert center_val == 9364

    def test_background_is_zero(self, rasterizer, sample_human_svg_path):
        """Pixels outside all polygons should be 0."""
        mask = rasterizer.rasterize(
            sample_human_svg_path, target_width=200, target_height=100
        )
        # Top-left corner (0,0) is outside all polygons
        assert mask[0, 0] == 0

    def test_dtype_is_integer(self, rasterizer, sample_human_svg_path):
        """Mask dtype should be integer."""
        mask = rasterizer.rasterize(
            sample_human_svg_path, target_width=200, target_height=100
        )
        assert np.issubdtype(mask.dtype, np.integer)


class TestRasterizeWithSparseAnnotations:
    """Test sparse annotation behavior (mask init to 255 for human SVGs)."""

    def test_sparse_mode_unlabeled_is_255(self, rasterizer, sample_human_svg_path):
        """When sparse=True, unlabeled pixels should be 255 (ignore_index)."""
        mask = rasterizer.rasterize(
            sample_human_svg_path,
            target_width=200,
            target_height=100,
            sparse=True,
        )
        # Top-left corner is outside all polygons → should be 255
        assert mask[0, 0] == 255

    def test_sparse_mode_annotated_pixels_have_structure_ids(
        self, rasterizer, sample_human_svg_path
    ):
        """Annotated regions should still have correct structure IDs."""
        mask = rasterizer.rasterize(
            sample_human_svg_path,
            target_width=200,
            target_height=100,
            sparse=True,
        )
        unique_ids = set(np.unique(mask))
        # Should have 255 (unlabeled) + structure IDs
        assert 255 in unique_ids
        assert len(unique_ids & {4039, 4045, 9366, 9364}) >= 2

    def test_default_mode_unlabeled_is_zero(self, rasterizer, sample_human_svg_path):
        """Default (sparse=False) should keep background as 0."""
        mask = rasterizer.rasterize(
            sample_human_svg_path, target_width=200, target_height=100
        )
        assert mask[0, 0] == 0


class TestMixedSVGBackwardCompat:
    """Test that mouse SVGs (with <path> elements) still work after changes."""

    def test_mouse_svg_still_works(self, rasterizer, sample_svg_path):
        """The original mouse SVG fixture should still rasterize correctly."""
        mask = rasterizer.rasterize(sample_svg_path, target_width=200, target_height=100)
        assert mask.shape == (100, 200)
        unique_ids = set(np.unique(mask))
        # Should still contain the mouse structure IDs
        assert len(unique_ids & {567, 343, 1009}) >= 1

    def test_mouse_svg_paths_parsed(self, rasterizer, sample_svg_path):
        """Mouse SVG should still have parseable <path> elements."""
        paths = rasterizer._parse_paths(str(sample_svg_path))
        assert len(paths) == 3
