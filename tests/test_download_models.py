"""Tests for the model download script."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from histological_image_analysis.download import (
    verify_model_download,
    resolve_repo_ids,
    DEFAULT_HF_USERNAME,
)


class TestVerifyModelDownload:
    """Test download verification logic."""

    def test_valid_download_passes(self, tmp_path):
        model_dir = tmp_path / "mouse"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"num_labels": 1328}')
        (model_dir / "preprocessor_config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1000)

        assert verify_model_download(str(model_dir)) is True

    def test_missing_config_fails(self, tmp_path):
        model_dir = tmp_path / "mouse"
        model_dir.mkdir()
        (model_dir / "preprocessor_config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1000)

        assert verify_model_download(str(model_dir)) is False

    def test_missing_preprocessor_config_fails(self, tmp_path):
        model_dir = tmp_path / "mouse"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"num_labels": 1328}')
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1000)

        assert verify_model_download(str(model_dir)) is False

    def test_missing_weights_fails(self, tmp_path):
        model_dir = tmp_path / "mouse"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"num_labels": 1328}')
        (model_dir / "preprocessor_config.json").write_text("{}")

        assert verify_model_download(str(model_dir)) is False

    def test_accepts_pytorch_model_bin(self, tmp_path):
        """Both safetensors and pytorch_model.bin formats are valid."""
        model_dir = tmp_path / "mouse"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"num_labels": 1328}')
        (model_dir / "preprocessor_config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_bytes(b"\x00" * 1000)

        assert verify_model_download(str(model_dir)) is True

    def test_nonexistent_dir_fails(self, tmp_path):
        assert verify_model_download(str(tmp_path / "nope")) is False


class TestResolveRepoIds:
    """Test HuggingFace repo ID resolution."""

    def test_mouse_only(self):
        repos = resolve_repo_ids("mouse")
        assert len(repos) == 1
        assert "mouse" in repos[0]

    def test_human_only(self):
        repos = resolve_repo_ids("human")
        assert len(repos) == 1
        assert "human" in repos[0]

    def test_all_returns_all_three(self):
        repos = resolve_repo_ids("all")
        assert len(repos) == 3
        names = " ".join(repos)
        assert "mouse" in names
        assert "human" in names
        assert "human-bigbrain" in names

    def test_human_bigbrain(self):
        repos = resolve_repo_ids("human-bigbrain")
        assert len(repos) == 1
        assert "human-bigbrain" in repos[0]

    def test_default_username_is_set(self):
        assert DEFAULT_HF_USERNAME == "Noel-Niko"

    def test_custom_repo_ids_override(self):
        repos = resolve_repo_ids("mouse", repo_id="custom/my-model-mouse")
        assert repos == ["custom/my-model-mouse"]
