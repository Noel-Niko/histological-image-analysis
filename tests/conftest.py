"""Shared pytest fixtures for histological image analysis tests."""

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def minimal_ontology_path() -> Path:
    """Path to the minimal ontology JSON fixture."""
    return FIXTURES_DIR / "minimal_ontology.json"


@pytest.fixture
def sample_svg_path() -> Path:
    """Path to the sample SVG fixture."""
    return FIXTURES_DIR / "sample.svg"


@pytest.fixture
def synthetic_volume() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic image + annotation volume pair for CCFv3Slicer tests.

    Returns (image_volume, annotation_volume) with shape (10, 8, 12).
    Image is float32 (like ara_nissl_10.nrrd), annotation is uint32.
    """
    rng = np.random.default_rng(42)

    # Image: float32 with a realistic-ish range (simulating nissl)
    image = rng.uniform(0.0, 5000.0, size=(10, 8, 12)).astype(np.float32)

    # Annotation: uint32 with structure IDs
    # Slices 0-1: mostly background (testing skip logic)
    # Slices 2-9: mix of structures
    annotation = np.zeros((10, 8, 12), dtype=np.uint32)
    annotation[2:9, 1:7, 2:10] = 567   # Cerebrum
    annotation[3:8, 2:5, 3:6] = 688    # Cerebral cortex (under Cerebrum)
    annotation[4:7, 5:7, 7:10] = 343   # Brain stem
    annotation[5:6, 3:4, 8:9] = 1009   # Fiber tracts

    return image, annotation


@pytest.fixture
def synthetic_volume_uint16() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic uint16 image + annotation volume pair.

    Simulates average_template_25.nrrd (uint16 autofluorescence).
    """
    rng = np.random.default_rng(42)

    image = rng.integers(0, 65535, size=(10, 8, 12), dtype=np.uint16)
    annotation = np.zeros((10, 8, 12), dtype=np.uint32)
    annotation[2:9, 1:7, 2:10] = 567
    annotation[4:7, 5:7, 7:10] = 343

    return image, annotation
