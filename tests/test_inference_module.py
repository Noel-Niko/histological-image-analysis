"""Tests for the shared inference module."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from histological_image_analysis.inference import (
    load_model,
    run_inference,
    run_sliding_window_inference,
    get_image_files,
    create_device,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CROP_SIZE,
)


class TestConstants:
    """Verify inference constants match DINOv2 expectations."""

    def test_imagenet_mean_length(self):
        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_length(self):
        assert len(IMAGENET_STD) == 3

    def test_crop_size_is_dinov2_native(self):
        assert CROP_SIZE == 518


class TestCreateDevice:
    """Test device selection logic."""

    def test_force_cpu(self):
        device = create_device(force_cpu=True)
        assert device.type == "cpu"

    @patch("histological_image_analysis.inference.torch")
    def test_no_gpu_falls_back_to_cpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.side_effect = lambda x: MagicMock(type=x)
        device = create_device(force_cpu=False)
        mock_torch.device.assert_called_with("cpu")


class TestLoadModel:
    """Test model loading with error handling."""

    def test_missing_model_path_raises_system_exit(self, tmp_path):
        nonexistent = tmp_path / "no_such_model"
        with pytest.raises(SystemExit):
            import torch

            load_model(str(nonexistent), torch.device("cpu"))


class TestGetImageFiles:
    """Test image file discovery."""

    def test_finds_supported_extensions(self, tmp_path):
        extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
        for ext in extensions:
            (tmp_path / f"image{ext}").write_bytes(b"fake")

        files = get_image_files(str(tmp_path))
        assert len(files) == len(extensions)

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "data.csv").write_text("1,2,3")
        (tmp_path / "brain.jpg").write_bytes(b"fake")

        files = get_image_files(str(tmp_path))
        assert len(files) == 1
        assert "brain.jpg" in files[0]

    def test_nonexistent_dir_raises_system_exit(self):
        with pytest.raises(SystemExit):
            get_image_files("/nonexistent/directory")

    def test_empty_dir_raises_system_exit(self, tmp_path):
        with pytest.raises(SystemExit):
            get_image_files(str(tmp_path))

    def test_returns_sorted_paths(self, tmp_path):
        for name in ["c.png", "a.png", "b.png"]:
            (tmp_path / name).write_bytes(b"fake")

        files = get_image_files(str(tmp_path))
        basenames = [Path(f).name for f in files]
        assert basenames == ["a.png", "b.png", "c.png"]


class TestRunInference:
    """Test inference execution with mocked model."""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.config.num_labels = 10
        # Mock forward pass: return logits shaped (1, 10, 37, 37)
        import torch

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 37, 37)
        model.return_value = mock_output
        return model

    @pytest.fixture
    def mock_processor(self):
        import torch

        processor = MagicMock()
        processor.return_value = {"pixel_values": torch.randn(1, 3, 518, 518)}
        return processor

    def test_returns_prediction_tuple(self, mock_model, mock_processor, tmp_path):
        # Create a small test image
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        import torch

        prediction, resized = run_inference(
            str(img_path), mock_model, mock_processor, torch.device("cpu")
        )

        assert prediction is not None
        assert resized is not None
        assert prediction.ndim == 2
        assert resized.ndim == 2

    def test_resized_matches_original_dimensions(
        self, mock_model, mock_processor, tmp_path
    ):
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        import torch

        _, resized = run_inference(
            str(img_path), mock_model, mock_processor, torch.device("cpu")
        )

        # Resized should match original image dimensions (height, width)
        assert resized.shape == (200, 300)

    def test_invalid_image_returns_none(self, mock_model, mock_processor, tmp_path):
        bad_path = tmp_path / "corrupt.png"
        bad_path.write_bytes(b"not a real image")

        import torch

        prediction, resized = run_inference(
            str(bad_path), mock_model, mock_processor, torch.device("cpu")
        )

        assert prediction is None
        assert resized is None
