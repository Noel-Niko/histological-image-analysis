"""Tests for VSI inspection functionality."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from histological_image_analysis.vsi import (
    SeriesInfo,
    find_vsi_files,
    parse_showinf_output,
    validate_companion_dir,
    validate_java,
    validate_bftools,
    inspect_vsi_file,
)


# -- Realistic showinf output fixtures --

SHOWINF_OUTPUT_WITH_PIXEL_SIZES = """\
Checking file format [OME-TIFF]
Checking file format [Olympus VSI]
Initializing reader
Initialization took 2.35s

Reading core metadata
Series count = 5
Series #0:
\tImage count = 1
\tRGB = true
\tWidth = 800
\tHeight = 600
\tSizeC = 3
\tPixel type = uint8
\tThumbnail series = true
Series #1:
\tImage count = 1
\tRGB = true
\tWidth = 640
\tHeight = 480
\tSizeC = 3
\tPixel type = uint8
Series #2:
\tImage count = 1
\tRGB = true
\tWidth = 98304
\tHeight = 73728
\tSizeC = 3
\tPixel type = uint8

Reading global metadata
PhysicalSizeX #0: 23.456
PhysicalSizeY #0: 23.456
PhysicalSizeX #2: 0.2450
PhysicalSizeY #2: 0.2450
PhysicalSizeX #3: 0.4900
PhysicalSizeY #3: 0.4900
PhysicalSizeX #4: 0.9800
PhysicalSizeY #4: 0.9800

Series #3:
\tImage count = 1
\tRGB = true
\tWidth = 49152
\tHeight = 36864
\tSizeC = 3
\tPixel type = uint8
Series #4:
\tImage count = 1
\tRGB = true
\tWidth = 24576
\tHeight = 18432
\tSizeC = 3
\tPixel type = uint8
"""

SHOWINF_OUTPUT_NO_PIXEL_SIZES = """\
Checking file format [Olympus VSI]
Initializing reader

Reading core metadata
Series count = 3
Series #0:
\tImage count = 1
\tRGB = true
\tWidth = 500
\tHeight = 400
\tSizeC = 3
\tPixel type = uint8
Series #1:
\tImage count = 1
\tRGB = true
\tWidth = 40000
\tHeight = 30000
\tSizeC = 3
\tPixel type = uint8
Series #2:
\tImage count = 1
\tRGB = true
\tWidth = 10000
\tHeight = 7500
\tSizeC = 3
\tPixel type = uint8
"""

SHOWINF_OUTPUT_SINGLE_SERIES = """\
Reading core metadata
Series count = 1
Series #0:
\tImage count = 1
\tRGB = true
\tWidth = 2048
\tHeight = 1536
\tSizeC = 3
\tPixel type = uint8
"""


class TestValidateJava:
    """Test Java availability check."""

    @patch("histological_image_analysis.vsi.subprocess.run")
    def test_java_available(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr='openjdk version "11.0.20"')
        assert validate_java() is True

    @patch("histological_image_analysis.vsi.subprocess.run")
    def test_java_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert validate_java() is False

    @patch("histological_image_analysis.vsi.subprocess.run")
    def test_java_returns_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "java")
        assert validate_java() is False


class TestValidateBftools:
    """Test Bio-Formats tools availability check."""

    def test_bftools_found(self, tmp_path):
        bftools_dir = tmp_path / "bftools"
        bftools_dir.mkdir()
        showinf = bftools_dir / "showinf"
        showinf.touch()
        showinf.chmod(0o755)
        bfconvert = bftools_dir / "bfconvert"
        bfconvert.touch()
        bfconvert.chmod(0o755)

        showinf_path, bfconvert_path = validate_bftools(bftools_dir)
        assert showinf_path == showinf
        assert bfconvert_path == bfconvert

    def test_bftools_missing_dir(self, tmp_path):
        nonexistent = tmp_path / "no_such_dir"
        with pytest.raises(FileNotFoundError, match="Bio-Formats tools not found"):
            validate_bftools(nonexistent)

    def test_bftools_missing_showinf(self, tmp_path):
        bftools_dir = tmp_path / "bftools"
        bftools_dir.mkdir()
        bfconvert = bftools_dir / "bfconvert"
        bfconvert.touch()
        with pytest.raises(FileNotFoundError, match="showinf"):
            validate_bftools(bftools_dir)

    def test_bftools_missing_bfconvert(self, tmp_path):
        bftools_dir = tmp_path / "bftools"
        bftools_dir.mkdir()
        showinf = bftools_dir / "showinf"
        showinf.touch()
        with pytest.raises(FileNotFoundError, match="bfconvert"):
            validate_bftools(bftools_dir)


class TestFindVsiFiles:
    """Test VSI file discovery in a directory."""

    def test_finds_vsi_files(self, tmp_path):
        (tmp_path / "slide1.vsi").touch()
        (tmp_path / "slide2.vsi").touch()
        (tmp_path / "notes.txt").touch()

        files = find_vsi_files(str(tmp_path))
        assert len(files) == 2
        assert all(f.suffix == ".vsi" for f in files)

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "slide1.VSI").touch()
        (tmp_path / "slide2.Vsi").touch()

        files = find_vsi_files(str(tmp_path))
        assert len(files) == 2

    def test_returns_sorted(self, tmp_path):
        (tmp_path / "c.vsi").touch()
        (tmp_path / "a.vsi").touch()
        (tmp_path / "b.vsi").touch()

        files = find_vsi_files(str(tmp_path))
        names = [f.name for f in files]
        assert names == ["a.vsi", "b.vsi", "c.vsi"]

    def test_empty_directory(self, tmp_path):
        files = find_vsi_files(str(tmp_path))
        assert files == []

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            find_vsi_files("/nonexistent/path")


class TestValidateCompanionDir:
    """Test .ets companion directory validation."""

    def test_companion_dir_exists(self, tmp_path):
        vsi_path = tmp_path / "slide1.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide1_"
        companion.mkdir()
        (companion / "frame_00000.ets").touch()

        result = validate_companion_dir(vsi_path)
        assert result == companion

    def test_companion_dir_missing(self, tmp_path):
        vsi_path = tmp_path / "slide1.vsi"
        vsi_path.touch()

        with pytest.raises(FileNotFoundError, match="companion directory"):
            validate_companion_dir(vsi_path)

    def test_companion_dir_empty(self, tmp_path):
        vsi_path = tmp_path / "slide1.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide1_"
        companion.mkdir()

        with pytest.raises(FileNotFoundError, match="no .ets files"):
            validate_companion_dir(vsi_path)


class TestParseShowinfOutput:
    """Test parsing of showinf -nopix output."""

    def test_parses_multiple_series(self):
        series = parse_showinf_output(SHOWINF_OUTPUT_WITH_PIXEL_SIZES)
        assert len(series) == 5

    def test_series_dimensions(self):
        series = parse_showinf_output(SHOWINF_OUTPUT_WITH_PIXEL_SIZES)
        # Series #0: 800x600
        assert series[0].width == 800
        assert series[0].height == 600
        # Series #2: 98304x73728 (full res)
        assert series[2].width == 98304
        assert series[2].height == 73728
        # Series #4: 24576x18432 (lowest pyramid)
        assert series[4].width == 24576
        assert series[4].height == 18432

    def test_series_indices(self):
        series = parse_showinf_output(SHOWINF_OUTPUT_WITH_PIXEL_SIZES)
        indices = [s.index for s in series]
        assert indices == [0, 1, 2, 3, 4]

    def test_pixel_sizes_parsed(self):
        series = parse_showinf_output(SHOWINF_OUTPUT_WITH_PIXEL_SIZES)
        # Series #2 has 0.2450 µm/pixel
        assert series[2].pixel_size_um == pytest.approx(0.245)
        # Series #3 has 0.49 µm/pixel
        assert series[3].pixel_size_um == pytest.approx(0.49)
        # Series #4 has 0.98 µm/pixel
        assert series[4].pixel_size_um == pytest.approx(0.98)

    def test_missing_pixel_sizes(self):
        series = parse_showinf_output(SHOWINF_OUTPUT_NO_PIXEL_SIZES)
        assert len(series) == 3
        assert all(s.pixel_size_um is None for s in series)

    def test_channels_parsed(self):
        series = parse_showinf_output(SHOWINF_OUTPUT_WITH_PIXEL_SIZES)
        assert all(s.channels == 3 for s in series)

    def test_single_series(self):
        series = parse_showinf_output(SHOWINF_OUTPUT_SINGLE_SERIES)
        assert len(series) == 1
        assert series[0].width == 2048
        assert series[0].height == 1536

    def test_empty_output(self):
        series = parse_showinf_output("")
        assert series == []

    def test_thumbnail_flag_parsed(self):
        series = parse_showinf_output(SHOWINF_OUTPUT_WITH_PIXEL_SIZES)
        assert series[0].is_thumbnail is True
        assert series[1].is_thumbnail is False


class TestInspectVsiFile:
    """Test the full inspect workflow for a single VSI file."""

    @patch("histological_image_analysis.vsi.subprocess.run")
    def test_inspect_calls_showinf(self, mock_run, tmp_path):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide_"
        companion.mkdir()
        (companion / "data.ets").touch()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=SHOWINF_OUTPUT_SINGLE_SERIES,
            stderr="",
        )

        showinf_path = tmp_path / "showinf"
        series = inspect_vsi_file(vsi_path, showinf_path)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert str(showinf_path) in call_args[0][0]
        assert str(vsi_path) in call_args[0][0]

    @patch("histological_image_analysis.vsi.subprocess.run")
    def test_inspect_returns_series(self, mock_run, tmp_path):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide_"
        companion.mkdir()
        (companion / "data.ets").touch()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=SHOWINF_OUTPUT_WITH_PIXEL_SIZES,
            stderr="",
        )

        series = inspect_vsi_file(vsi_path, tmp_path / "showinf")
        assert len(series) == 5

    @patch("histological_image_analysis.vsi.subprocess.run")
    def test_inspect_showinf_fails(self, mock_run, tmp_path):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()
        companion = tmp_path / "_slide_"
        companion.mkdir()
        (companion / "data.ets").touch()

        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error reading file",
        )

        with pytest.raises(RuntimeError, match="showinf failed"):
            inspect_vsi_file(vsi_path, tmp_path / "showinf")

    def test_inspect_missing_companion(self, tmp_path):
        vsi_path = tmp_path / "slide.vsi"
        vsi_path.touch()

        with pytest.raises(FileNotFoundError, match="companion directory"):
            inspect_vsi_file(vsi_path, tmp_path / "showinf")
