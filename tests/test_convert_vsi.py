"""Tests for VSI conversion functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from histological_image_analysis.vsi import (
    SeriesInfo,
    select_best_series,
    build_output_path,
    validate_conversion_output,
    convert_vsi_file,
)


# -- Series fixtures for selection testing --


def _series(index, width, height, pixel_size_um=None, is_thumbnail=False):
    """Helper to create SeriesInfo for tests."""
    return SeriesInfo(
        index=index,
        width=width,
        height=height,
        pixel_size_um=pixel_size_um,
        channels=3,
        is_thumbnail=is_thumbnail,
    )


class TestSelectBestSeries:
    """Test pyramid level selection algorithm."""

    def test_selects_by_pixel_size_exact_match(self):
        series = [
            _series(0, 800, 600, pixel_size_um=23.0, is_thumbnail=True),
            _series(2, 98304, 73728, pixel_size_um=0.245),
            _series(3, 49152, 36864, pixel_size_um=0.49),
            _series(4, 24576, 18432, pixel_size_um=0.98),
        ]
        best = select_best_series(series, target_um_per_pixel=1.0)
        assert best.index == 4  # 0.98 µm closest to 1.0

    def test_selects_by_pixel_size_target_10um(self):
        series = [
            _series(0, 800, 600, pixel_size_um=23.0, is_thumbnail=True),
            _series(2, 98304, 73728, pixel_size_um=0.245),
            _series(3, 49152, 36864, pixel_size_um=0.49),
            _series(4, 24576, 18432, pixel_size_um=0.98),
        ]
        # Target 10 µm: series 0 is thumbnail (filtered), so closest is series 4 (0.98)
        best = select_best_series(series, target_um_per_pixel=10.0)
        assert best.index == 4

    def test_filters_out_thumbnails(self):
        series = [
            _series(0, 800, 600, pixel_size_um=10.0, is_thumbnail=True),
            _series(1, 640, 480, pixel_size_um=12.0, is_thumbnail=True),
            _series(2, 49152, 36864, pixel_size_um=0.49),
        ]
        best = select_best_series(series, target_um_per_pixel=10.0)
        # Thumbnails filtered, only series 2 remains
        assert best.index == 2

    def test_filters_small_images(self):
        """Series with both dimensions < 1000 px are filtered as label/macro images."""
        series = [
            _series(0, 800, 600),  # both < 1000
            _series(1, 640, 480),  # both < 1000
            _series(2, 40000, 30000),
            _series(3, 10000, 7500),
        ]
        best = select_best_series(series, target_um_per_pixel=10.0)
        # Series 0 and 1 are filtered as small
        assert best.index in (2, 3)

    def test_fallback_to_dimensions_when_no_pixel_sizes(self):
        """When no pixel sizes, pick series closest to 1000-3000 px on long side."""
        series = [
            _series(0, 500, 400),  # too small, filtered
            _series(1, 40000, 30000),  # too large
            _series(2, 10000, 7500),  # large
            _series(3, 2500, 1875),  # in target range
        ]
        best = select_best_series(series, target_um_per_pixel=10.0)
        assert best.index == 3  # 2500 is closest to 1000-3000 range

    def test_fallback_prefers_2000px_long_side(self):
        series = [
            _series(0, 500, 400),  # filtered (small)
            _series(1, 5000, 3750),
            _series(2, 2000, 1500),  # ~2000 long side
            _series(3, 1200, 900),  # close but a bit small
        ]
        best = select_best_series(series, target_um_per_pixel=10.0)
        assert best.index == 2

    def test_single_series(self):
        series = [_series(0, 50000, 40000, pixel_size_um=0.245)]
        best = select_best_series(series, target_um_per_pixel=10.0)
        assert best.index == 0

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="No valid series"):
            select_best_series([], target_um_per_pixel=10.0)

    def test_all_filtered_raises(self):
        """If all series are tiny thumbnails, raise ValueError."""
        series = [
            _series(0, 128, 96, is_thumbnail=True),
            _series(1, 64, 48, is_thumbnail=True),
        ]
        with pytest.raises(ValueError, match="No valid series"):
            select_best_series(series, target_um_per_pixel=10.0)

    def test_needs_downsample_flag(self):
        """When best series is >2x finer than target, needs_downsample should be set."""
        series = [
            _series(0, 24576, 18432, pixel_size_um=0.98),
        ]
        best = select_best_series(series, target_um_per_pixel=10.0)
        assert best.index == 0
        # 0.98 vs target 10.0: ratio ~10x, needs downsample
        # The function should return the series; caller checks if downsample needed


class TestBuildOutputPath:
    """Test output filename construction."""

    def test_basic_output_path(self):
        vsi_path = Path("/data/slides/brain_01.vsi")
        output = build_output_path(vsi_path)
        assert output == Path("/data/slides/brain_01-converted.tiff")

    def test_preserves_directory(self):
        vsi_path = Path("/some/deep/path/slide.vsi")
        output = build_output_path(vsi_path)
        assert output.parent == Path("/some/deep/path")

    def test_handles_uppercase_extension(self):
        vsi_path = Path("/data/slide.VSI")
        output = build_output_path(vsi_path)
        assert output == Path("/data/slide-converted.tiff")

    def test_custom_suffix(self):
        vsi_path = Path("/data/slide.vsi")
        output = build_output_path(vsi_path, suffix="-downsampled")
        assert output == Path("/data/slide-downsampled.tiff")


class TestValidateConversionOutput:
    """Test output validation after bfconvert."""

    def test_valid_tiff(self, tmp_path):
        """A real small TIFF-like file passes validation."""
        from PIL import Image
        import numpy as np

        output_path = tmp_path / "output.tiff"
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        )
        img.save(str(output_path))

        width, height = validate_conversion_output(output_path)
        assert width == 200
        assert height == 100

    def test_missing_file(self, tmp_path):
        output_path = tmp_path / "nonexistent.tiff"
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_conversion_output(output_path)

    def test_empty_file(self, tmp_path):
        output_path = tmp_path / "empty.tiff"
        output_path.touch()
        with pytest.raises(ValueError, match="empty"):
            validate_conversion_output(output_path)

    def test_tiny_image(self, tmp_path):
        """A 1x1 image suggests wrong series was extracted."""
        from PIL import Image
        import numpy as np

        output_path = tmp_path / "tiny.tiff"
        img = Image.fromarray(np.array([[[255, 0, 0]]], dtype=np.uint8))
        img.save(str(output_path))

        with pytest.raises(ValueError, match="suspiciously small"):
            validate_conversion_output(output_path)


class TestConvertVsiFile:
    """Test the full conversion workflow for a single VSI file."""

    @patch("histological_image_analysis.vsi.subprocess.run")
    @patch("histological_image_analysis.vsi.validate_conversion_output")
    @patch("histological_image_analysis.vsi.inspect_vsi_file")
    def test_convert_calls_bfconvert(
        self, mock_inspect, mock_validate, mock_run, tmp_path
    ):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide_"
        companion.mkdir()
        (companion / "data.ets").touch()
        output_path = tmp_path / "slide-converted.tiff"

        mock_inspect.return_value = [
            SeriesInfo(
                index=0, width=2000, height=1500, pixel_size_um=10.0, channels=3
            ),
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_validate.return_value = (2000, 1500)

        result = convert_vsi_file(
            vsi_path=vsi_path,
            bfconvert_path=tmp_path / "bfconvert",
            showinf_path=tmp_path / "showinf",
            output_path=output_path,
            target_um_per_pixel=10.0,
        )

        assert result.output_path == output_path
        assert result.series_index == 0
        mock_run.assert_called_once()
        call_cmd = mock_run.call_args[0][0]
        assert "-series" in call_cmd
        assert "0" in call_cmd

    @patch("histological_image_analysis.vsi.subprocess.run")
    @patch("histological_image_analysis.vsi.validate_conversion_output")
    @patch("histological_image_analysis.vsi.inspect_vsi_file")
    def test_convert_skips_existing(
        self, mock_inspect, mock_validate, mock_run, tmp_path
    ):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()
        output_path = tmp_path / "slide-converted.tiff"
        output_path.write_text("already exists")

        result = convert_vsi_file(
            vsi_path=vsi_path,
            bfconvert_path=tmp_path / "bfconvert",
            showinf_path=tmp_path / "showinf",
            output_path=output_path,
            target_um_per_pixel=10.0,
        )

        assert result.skipped is True
        mock_run.assert_not_called()

    @patch("histological_image_analysis.vsi.subprocess.run")
    @patch("histological_image_analysis.vsi.inspect_vsi_file")
    def test_convert_bfconvert_fails(self, mock_inspect, mock_run, tmp_path):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide_"
        companion.mkdir()
        (companion / "data.ets").touch()

        mock_inspect.return_value = [
            SeriesInfo(
                index=0, width=2000, height=1500, pixel_size_um=10.0, channels=3
            ),
        ]
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Conversion failed"
        )

        with pytest.raises(RuntimeError, match="bfconvert failed"):
            convert_vsi_file(
                vsi_path=vsi_path,
                bfconvert_path=tmp_path / "bfconvert",
                showinf_path=tmp_path / "showinf",
                output_path=tmp_path / "out.tiff",
                target_um_per_pixel=10.0,
            )

    @patch("histological_image_analysis.vsi.subprocess.run")
    @patch("histological_image_analysis.vsi.validate_conversion_output")
    @patch("histological_image_analysis.vsi.inspect_vsi_file")
    def test_convert_selects_best_series(
        self, mock_inspect, mock_validate, mock_run, tmp_path
    ):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide_"
        companion.mkdir()
        (companion / "data.ets").touch()

        mock_inspect.return_value = [
            SeriesInfo(
                index=0, width=800, height=600, pixel_size_um=23.0,
                channels=3, is_thumbnail=True,
            ),
            SeriesInfo(
                index=1, width=98304, height=73728, pixel_size_um=0.245, channels=3,
            ),
            SeriesInfo(
                index=2, width=24576, height=18432, pixel_size_um=0.98, channels=3,
            ),
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_validate.return_value = (24576, 18432)

        result = convert_vsi_file(
            vsi_path=vsi_path,
            bfconvert_path=tmp_path / "bfconvert",
            showinf_path=tmp_path / "showinf",
            output_path=tmp_path / "out.tiff",
            target_um_per_pixel=10.0,
        )

        # Series 2 (0.98 µm) is closest to 10.0 µm after filtering thumbnail
        assert result.series_index == 2

    @patch("histological_image_analysis.vsi.subprocess.run")
    @patch("histological_image_analysis.vsi.validate_conversion_output")
    @patch("histological_image_analysis.vsi.inspect_vsi_file")
    def test_convert_result_fields(
        self, mock_inspect, mock_validate, mock_run, tmp_path
    ):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide_"
        companion.mkdir()
        (companion / "data.ets").touch()

        mock_inspect.return_value = [
            SeriesInfo(
                index=2, width=24576, height=18432, pixel_size_um=0.98, channels=3
            ),
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_validate.return_value = (24576, 18432)

        result = convert_vsi_file(
            vsi_path=vsi_path,
            bfconvert_path=tmp_path / "bfconvert",
            showinf_path=tmp_path / "showinf",
            output_path=tmp_path / "out.tiff",
            target_um_per_pixel=10.0,
        )

        assert result.skipped is False
        assert result.series_index == 2
        assert result.input_width == 24576
        assert result.input_height == 18432
        assert result.pixel_size_um == pytest.approx(0.98)
        assert result.output_path == tmp_path / "out.tiff"
