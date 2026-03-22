"""Model download and verification utilities.

Downloads trained brain segmentation models from HuggingFace Hub
and verifies the download contains all required files.
"""

import os
from pathlib import Path
from typing import List, Optional

DEFAULT_HF_USERNAME = "Noel-Niko"
DEFAULT_MODEL_BASE = "dinov2-upernet-20260322-histology-annotation"

REQUIRED_CONFIGS = ["config.json", "preprocessor_config.json"]
WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]


def resolve_repo_ids(
    species: str,
    repo_id: Optional[str] = None,
    hf_username: str = DEFAULT_HF_USERNAME,
    model_base: str = DEFAULT_MODEL_BASE,
) -> List[str]:
    """Resolve HuggingFace repo ID(s) for the given species.

    Parameters
    ----------
    species : str
        One of "mouse", "human", or "all".
    repo_id : str, optional
        Explicit repo ID override. If provided, returned as single-element list.
    hf_username : str
        HuggingFace username/org for default repo names.
    model_base : str
        Base model name template.

    Returns
    -------
    list of str
        HuggingFace repo IDs to download.
    """
    if repo_id is not None:
        return [repo_id]

    if species == "all":
        return [
            f"{hf_username}/{model_base}-mouse",
            f"{hf_username}/{model_base}-human",
            f"{hf_username}/{model_base}-human-bigbrain",
        ]
    return [f"{hf_username}/{model_base}-{species}"]


def verify_model_download(model_dir: str) -> bool:
    """Verify a downloaded model directory contains all required files.

    Checks for:
    - config.json (model architecture)
    - preprocessor_config.json (image processor settings)
    - model weights (model.safetensors or pytorch_model.bin)

    Parameters
    ----------
    model_dir : str
        Path to the model directory to verify.

    Returns
    -------
    bool
        True if all required files are present, False otherwise.
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        return False

    for config_file in REQUIRED_CONFIGS:
        if not (model_path / config_file).exists():
            return False

    has_weights = any(
        (model_path / wf).exists() for wf in WEIGHT_FILES
    )
    return has_weights
