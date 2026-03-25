"""VSI (Olympus Virtual Slide Image) inspection and conversion utilities.

Provides functions to inspect VSI file metadata using Bio-Formats ``showinf``,
select the appropriate pyramid level for a target resolution, and convert
VSI files to standard TIFF using ``bfconvert``.

VSI is a two-part format:
- The ``.vsi`` file contains metadata only.
- Actual pixel data lives in a companion directory ``_[filename]_/``
  containing ``.ets`` files. Both parts are required.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_BFTOOLS_DIR = Path("tools/bftools")
DEFAULT_TARGET_UM_PER_PIXEL = 10.0
DIMENSION_TARGET_LONG_SIDE = 2000
MIN_DIMENSION_THRESHOLD = 1000


@dataclass
class SeriesInfo:
    """Metadata for a single series within a VSI file."""

    index: int
    width: int
    height: int
    pixel_size_um: Optional[float] = None
    channels: int = 3
    is_thumbnail: bool = False

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)


@dataclass
class ConversionResult:
    """Result of converting a single VSI file."""

    output_path: Path
    skipped: bool = False
    series_index: int = 0
    input_width: int = 0
    input_height: int = 0
    output_width: int = 0
    output_height: int = 0
    pixel_size_um: Optional[float] = None
    error: Optional[str] = None


def validate_java() -> bool:
    """Check that Java is available on the system."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def validate_bftools(tools_dir: Path = DEFAULT_BFTOOLS_DIR) -> tuple[Path, Path]:
    """Validate that Bio-Formats CLI tools are installed.

    Parameters
    ----------
    tools_dir : Path
        Directory containing bftools (showinf, bfconvert).

    Returns
    -------
    tuple of (showinf_path, bfconvert_path)

    Raises
    ------
    FileNotFoundError
        If tools_dir, showinf, or bfconvert are missing.
    """
    if not tools_dir.exists():
        raise FileNotFoundError(
            f"Bio-Formats tools not found at {tools_dir}\n"
            "Install with: make download-bioformats-cli"
        )

    showinf_path = tools_dir / "showinf"
    if not showinf_path.exists():
        raise FileNotFoundError(
            f"showinf not found in {tools_dir}\n"
            "Install with: make download-bioformats-cli"
        )

    bfconvert_path = tools_dir / "bfconvert"
    if not bfconvert_path.exists():
        raise FileNotFoundError(
            f"bfconvert not found in {tools_dir}\n"
            "Install with: make download-bioformats-cli"
        )

    return showinf_path, bfconvert_path


def find_vsi_files(image_dir: str) -> list[Path]:
    """Find all .vsi files in a directory, sorted by name.

    Parameters
    ----------
    image_dir : str
        Path to directory to scan.

    Returns
    -------
    list of Path
        Sorted list of .vsi file paths. May be empty if none found.

    Raises
    ------
    FileNotFoundError
        If directory does not exist.
    """
    dir_path = Path(image_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    vsi_files = [
        p for p in dir_path.iterdir()
        if p.suffix.lower() == ".vsi"
    ]
    return sorted(vsi_files, key=lambda p: p.name)


def validate_companion_dir(vsi_path: Path) -> Path:
    """Validate that the .ets companion directory exists for a VSI file.

    VSI is a two-part format. The ``.vsi`` file contains metadata only;
    pixel data lives in ``_[stem]_/`` containing ``.ets`` files.

    Parameters
    ----------
    vsi_path : Path
        Path to the .vsi file.

    Returns
    -------
    Path
        Path to the companion directory.

    Raises
    ------
    FileNotFoundError
        If companion directory is missing or contains no .ets files.
    """
    stem = vsi_path.stem
    companion = vsi_path.parent / f"_{stem}_"

    if not companion.exists() or not companion.is_dir():
        raise FileNotFoundError(
            f"VSI companion directory not found: {companion}\n"
            f"\n"
            f"VSI is a two-part format:\n"
            f"  - {vsi_path.name}  (metadata only)\n"
            f"  - _{stem}_/       (pixel data in .ets files)\n"
            f"\n"
            f"Both must be copied together. Ensure the companion directory\n"
            f"'_{stem}_' is in the same folder as the .vsi file."
        )

    ets_files = list(companion.glob("*.ets"))
    if not ets_files:
        raise FileNotFoundError(
            f"VSI companion directory exists but contains no .ets files: {companion}\n"
            f"The directory may be incomplete. Re-copy from the scanner."
        )

    return companion


def parse_showinf_output(output: str) -> list[SeriesInfo]:
    """Parse the text output of ``showinf -nopix`` into series metadata.

    Parameters
    ----------
    output : str
        Full stdout from ``showinf -nopix <file.vsi>``.

    Returns
    -------
    list of SeriesInfo
        One entry per series found in the output.
    """
    if not output.strip():
        return []

    # Parse pixel sizes from global metadata section
    # Format: PhysicalSizeX #N: 0.245
    pixel_sizes: dict[int, float] = {}
    for match in re.finditer(
        r"PhysicalSizeX\s*#(\d+)\s*[:=]\s*([0-9.]+)", output
    ):
        series_idx = int(match.group(1))
        pixel_sizes[series_idx] = float(match.group(2))

    # Also try format: PhysicalSizeX: 0.245 (without series index)
    # and inline format within series blocks
    # These are handled per-series below.

    # Split into series blocks
    series_list: list[SeriesInfo] = []
    series_pattern = re.compile(r"Series\s*#(\d+)\s*:")
    series_starts = list(series_pattern.finditer(output))

    for i, match in enumerate(series_starts):
        series_idx = int(match.group(1))

        # Extract the block of text for this series
        start = match.end()
        end = series_starts[i + 1].start() if i + 1 < len(series_starts) else len(output)
        block = output[start:end]

        # Parse dimensions
        width_match = re.search(r"Width\s*=\s*(\d+)", block)
        height_match = re.search(r"Height\s*=\s*(\d+)", block)
        channels_match = re.search(r"SizeC\s*=\s*(\d+)", block)
        thumbnail_match = re.search(r"Thumbnail series\s*=\s*true", block, re.IGNORECASE)

        if not width_match or not height_match:
            continue

        width = int(width_match.group(1))
        height = int(height_match.group(1))
        channels = int(channels_match.group(1)) if channels_match else 3
        is_thumbnail = thumbnail_match is not None

        # Check for inline pixel size
        pixel_size = pixel_sizes.get(series_idx)
        if pixel_size is None:
            inline_match = re.search(
                r"Physical\s*size[XxWidthwidth]*\s*[:=#]\s*([0-9.]+)", block
            )
            if inline_match:
                pixel_size = float(inline_match.group(1))

        series_list.append(
            SeriesInfo(
                index=series_idx,
                width=width,
                height=height,
                pixel_size_um=pixel_size,
                channels=channels,
                is_thumbnail=is_thumbnail,
            )
        )

    return series_list


def select_best_series(
    series: list[SeriesInfo],
    target_um_per_pixel: float = DEFAULT_TARGET_UM_PER_PIXEL,
) -> SeriesInfo:
    """Select the pyramid level closest to the target resolution.

    Selection logic:
    1. Filter out thumbnail series and series with both dimensions < 1000 px.
    2. If pixel size metadata is available, pick the series closest to target.
    3. If no pixel sizes, pick the series with long side closest to 2000 px.

    Parameters
    ----------
    series : list of SeriesInfo
        Available series from a VSI file.
    target_um_per_pixel : float
        Target resolution in micrometers per pixel (default 10.0).

    Returns
    -------
    SeriesInfo
        The best series for conversion.

    Raises
    ------
    ValueError
        If no valid series remain after filtering.
    """
    # Filter out thumbnails and very small images
    candidates = [
        s for s in series
        if not s.is_thumbnail
        and not (s.width < MIN_DIMENSION_THRESHOLD and s.height < MIN_DIMENSION_THRESHOLD)
    ]

    if not candidates:
        raise ValueError(
            "No valid series found after filtering thumbnails and small images.\n"
            f"All {len(series)} series were filtered out. "
            "The VSI file may not contain usable image data."
        )

    # Try pixel-size-based selection first
    with_pixel_size = [s for s in candidates if s.pixel_size_um is not None]

    if with_pixel_size:
        return min(
            with_pixel_size,
            key=lambda s: abs(s.pixel_size_um - target_um_per_pixel),
        )

    # Fallback: select by dimension proximity to target range
    return min(
        candidates,
        key=lambda s: abs(s.long_side - DIMENSION_TARGET_LONG_SIDE),
    )


def build_output_path(vsi_path: Path, suffix: str = "-converted") -> Path:
    """Build the output TIFF path for a converted VSI file.

    Parameters
    ----------
    vsi_path : Path
        Path to the original .vsi file.
    suffix : str
        Suffix to append before .tiff extension.

    Returns
    -------
    Path
        Output path, e.g. ``/data/slides/brain_01-converted.tiff``.
    """
    return vsi_path.with_name(f"{vsi_path.stem}{suffix}.tiff")


def validate_conversion_output(output_path: Path) -> tuple[int, int]:
    """Validate that a converted TIFF file is reasonable.

    Parameters
    ----------
    output_path : Path
        Path to the output TIFF.

    Returns
    -------
    tuple of (width, height)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or suspiciously small.
    """
    if not output_path.exists():
        raise FileNotFoundError(
            f"Conversion output not found: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Conversion output is empty (0 bytes): {output_path}\n"
            "bfconvert may have failed silently. Check Java and bftools installation."
        )

    from PIL import Image

    with Image.open(output_path) as img:
        width, height = img.size

    if width <= 1 and height <= 1:
        raise ValueError(
            f"Conversion output is suspiciously small ({width}x{height}): {output_path}\n"
            "This usually means the wrong series was extracted. "
            "Run 'make inspect-vsi' to see available series and try a different one."
        )

    return width, height


def inspect_vsi_file(vsi_path: Path, showinf_path: Path) -> list[SeriesInfo]:
    """Inspect a single VSI file and return series metadata.

    Validates the companion directory, runs ``showinf -nopix``, and parses
    the output.

    Parameters
    ----------
    vsi_path : Path
        Path to the .vsi file.
    showinf_path : Path
        Path to the showinf executable.

    Returns
    -------
    list of SeriesInfo

    Raises
    ------
    FileNotFoundError
        If companion directory is missing.
    RuntimeError
        If showinf fails.
    """
    validate_companion_dir(vsi_path)

    result = subprocess.run(
        [str(showinf_path), "-nopix", "-no-upgrade", str(vsi_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"showinf failed for {vsi_path.name} (exit code {result.returncode})\n"
            f"stderr: {result.stderr[:500]}"
        )

    return parse_showinf_output(result.stdout)


def convert_vsi_file(
    vsi_path: Path,
    bfconvert_path: Path,
    showinf_path: Path,
    output_path: Path,
    target_um_per_pixel: float = DEFAULT_TARGET_UM_PER_PIXEL,
) -> ConversionResult:
    """Convert a single VSI file to TIFF at the target resolution.

    Parameters
    ----------
    vsi_path : Path
        Path to the input .vsi file.
    bfconvert_path : Path
        Path to the bfconvert executable.
    showinf_path : Path
        Path to the showinf executable.
    output_path : Path
        Path for the output TIFF file.
    target_um_per_pixel : float
        Target resolution in µm/pixel.

    Returns
    -------
    ConversionResult

    Raises
    ------
    RuntimeError
        If bfconvert fails.
    FileNotFoundError
        If companion directory is missing.
    ValueError
        If no valid series found.
    """
    if output_path.exists():
        return ConversionResult(output_path=output_path, skipped=True)

    series = inspect_vsi_file(vsi_path, showinf_path)
    best = select_best_series(series, target_um_per_pixel)

    result = subprocess.run(
        [
            str(bfconvert_path),
            "-compression", "LZW",
            "-series", str(best.index),
            "-no-upgrade",
            str(vsi_path),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"bfconvert failed for {vsi_path.name} (exit code {result.returncode})\n"
            f"stderr: {result.stderr[:500]}"
        )

    output_width, output_height = validate_conversion_output(output_path)

    return ConversionResult(
        output_path=output_path,
        skipped=False,
        series_index=best.index,
        input_width=best.width,
        input_height=best.height,
        output_width=output_width,
        output_height=output_height,
        pixel_size_um=best.pixel_size_um,
    )
