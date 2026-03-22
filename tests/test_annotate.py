"""Tests for the annotation overlay script."""

import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from histological_image_analysis.inference import CROP_SIZE


# Import annotation functions — these will be implemented in scripts/annotate.py
# We import from the package module that annotate.py will use
from histological_image_analysis.annotation import (
    create_annotated_overlay,
    build_annotated_filename,
    resolve_model_path,
)


ANNOTATED_FILENAME_PATTERN = re.compile(
    r"^.+-annotated-\d{8}T\d{6}\.png$"
)


class TestBuildAnnotatedFilename:
    """Test the annotated filename convention."""

    def test_format_matches_pattern(self):
        result = build_annotated_filename("/path/to/slide_001.jpg")
        assert ANNOTATED_FILENAME_PATTERN.match(Path(result).name)

    def test_preserves_original_stem(self):
        result = build_annotated_filename("/data/brain_slice_42.tif")
        name = Path(result).name
        assert name.startswith("brain_slice_42-annotated-")

    def test_output_in_same_directory_as_input(self):
        result = build_annotated_filename("/some/deep/path/slide.jpg")
        assert Path(result).parent == Path("/some/deep/path")

    def test_always_png_extension(self):
        for ext in [".jpg", ".tif", ".bmp", ".jpeg"]:
            result = build_annotated_filename(f"/data/img{ext}")
            assert result.endswith(".png")

    def test_timestamp_is_recent(self):
        result = build_annotated_filename("/data/slide.jpg")
        name = Path(result).name
        # Extract timestamp from filename
        ts_str = name.split("-annotated-")[1].replace(".png", "")
        ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%S")
        # Should be within last 5 seconds
        assert (datetime.now() - ts).total_seconds() < 5


class TestCreateAnnotatedOverlay:
    """Test the overlay image generation."""

    @pytest.fixture
    def sample_image(self):
        """A 200x300 RGB image."""
        return np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_prediction(self):
        """A 200x300 prediction mask with 4 classes."""
        pred = np.zeros((200, 300), dtype=np.int64)
        pred[0:100, 0:150] = 0   # Background
        pred[0:100, 150:300] = 1  # Region A
        pred[100:200, 0:150] = 2  # Region B
        pred[100:200, 150:300] = 3  # Region C
        return pred

    @pytest.fixture
    def sample_id2label(self):
        return {
            0: "Background",
            1: "Cerebral cortex",
            2: "Hippocampus",
            3: "Thalamus",
        }

    def test_returns_pil_image(
        self, sample_image, sample_prediction, sample_id2label
    ):
        result = create_annotated_overlay(
            sample_image, sample_prediction, sample_id2label
        )
        assert isinstance(result, Image.Image)

    def test_output_is_rgb(
        self, sample_image, sample_prediction, sample_id2label
    ):
        result = create_annotated_overlay(
            sample_image, sample_prediction, sample_id2label
        )
        assert result.mode == "RGB"

    def test_output_height_matches_input(
        self, sample_image, sample_prediction, sample_id2label
    ):
        result = create_annotated_overlay(
            sample_image, sample_prediction, sample_id2label
        )
        # Height should match input; width may be wider due to legend
        assert result.size[1] == sample_image.shape[0]

    def test_output_wider_than_input_due_to_legend(
        self, sample_image, sample_prediction, sample_id2label
    ):
        result = create_annotated_overlay(
            sample_image, sample_prediction, sample_id2label
        )
        # Legend panel adds width
        assert result.size[0] > sample_image.shape[1]

    def test_background_not_overlaid(
        self, sample_image, sample_prediction, sample_id2label
    ):
        # Where prediction == 0 (background), overlay should not alter original
        result = create_annotated_overlay(
            sample_image, sample_prediction, sample_id2label
        )
        result_arr = np.array(result)

        # The left portion of the result (before legend) at background pixels
        # should be close to original (within alpha blending tolerance)
        bg_mask = sample_prediction == 0
        original_bg = sample_image[bg_mask]
        result_bg = result_arr[:200, :300][bg_mask]

        # Background pixels should be identical to original (no overlay)
        np.testing.assert_array_equal(original_bg, result_bg)

    def test_non_background_pixels_are_modified(
        self, sample_image, sample_prediction, sample_id2label
    ):
        result = create_annotated_overlay(
            sample_image, sample_prediction, sample_id2label
        )
        result_arr = np.array(result)

        # Non-background pixels should differ from original
        fg_mask = sample_prediction > 0
        original_fg = sample_image[fg_mask]
        result_fg = result_arr[:200, :300][fg_mask]

        # At least some pixels should be different (alpha blended)
        assert not np.array_equal(original_fg, result_fg)

    def test_handles_single_class_prediction(self, sample_image, sample_id2label):
        # All background — should still produce valid output
        pred = np.zeros((200, 300), dtype=np.int64)
        result = create_annotated_overlay(sample_image, pred, sample_id2label)
        assert isinstance(result, Image.Image)

    def test_handles_missing_id2label_gracefully(self, sample_image):
        pred = np.ones((200, 300), dtype=np.int64) * 999
        # id2label doesn't contain class 999
        id2label = {0: "Background"}
        result = create_annotated_overlay(sample_image, pred, id2label)
        assert isinstance(result, Image.Image)


class TestResolveModelPath:
    """Test model path resolution from species flag."""

    def test_mouse_resolves_to_models_mouse(self):
        path = resolve_model_path("mouse")
        assert "models" in path
        assert "mouse" in path

    def test_human_resolves_to_models_human(self):
        path = resolve_model_path("human")
        assert "models" in path
        assert "human" in path

    def test_human_bigbrain_resolves_to_models_human_bigbrain(self):
        path = resolve_model_path("human-bigbrain")
        assert "models" in path
        assert "human-bigbrain" in path

    def test_explicit_path_returned_unchanged(self):
        path = resolve_model_path("mouse", model_path="/custom/model")
        assert path == "/custom/model"
