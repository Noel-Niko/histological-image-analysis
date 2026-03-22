#!/usr/bin/env python3
"""
Upload trained models to HuggingFace Hub (one-time utility).

Uploads locally-stored brain segmentation models to HuggingFace Hub for
public distribution. Run this once after downloading models from Databricks.

Requires HUGGING_FACE_TOKEN environment variable with write access.

Usage:
    # Upload both mouse and human models
    python scripts/upload_to_hf.py

    # Upload mouse model only
    python scripts/upload_to_hf.py --species mouse

    # Upload from custom local paths
    python scripts/upload_to_hf.py --mouse-model ./models/dinov2-upernet-final --species mouse

    # Custom repo naming
    python scripts/upload_to_hf.py --hf-username my-username
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from histological_image_analysis.download import (
    DEFAULT_HF_USERNAME,
    DEFAULT_MODEL_BASE,
    verify_model_download,
)


REQUIRED_FILES = [
    "config.json",
    "preprocessor_config.json",
]

WEIGHT_FILES = [
    "model.safetensors",
    "pytorch_model.bin",
]

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
  - brain-segmentation
  - histology
  - dinov2
  - upernet
  - neuroscience
  - allen-brain-institute
---

# Brain Region Segmentation — {species_title}

DINOv2-Large + UperNet model fine-tuned for semantic segmentation of
{species} brain structures in Nissl-stained histological sections.

## Model Details

| Attribute | Value |
|-----------|-------|
| Architecture | DINOv2-Large (304M) + UperNet (38M) |
| Classes | {num_classes} |
| Input Size | 518x518 |
| Training Data | Allen Brain Institute {data_source} |

## Usage

```bash
git clone https://github.com/your-repo/histological-image-analysis
cd histological-image-analysis
make install
make download-models
make annotate IMAGES=/path/to/your/slides/
```

## Citation

If you use this model, please cite the Allen Brain Institute data sources.
"""


def upload_model(
    local_dir: str,
    repo_id: str,
    species: str,
    token: str,
) -> bool:
    """Upload a model directory to HuggingFace Hub.

    Parameters
    ----------
    local_dir : str
        Path to local model directory.
    repo_id : str
        HuggingFace repo ID (e.g., "Noel-Niko/dinov2-upernet-20260322-histology-annotation-mouse").
    species : str
        "mouse" or "human" (for model card metadata).
    token : str
        HuggingFace API token with write access.

    Returns
    -------
    bool
        True if upload succeeded.
    """
    if not verify_model_download(local_dir):
        print(f"ERROR: Model at {local_dir} is incomplete. Cannot upload.")
        return False

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id=repo_id, token=token, exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"ERROR: Failed to create repo {repo_id}: {e}")
        return False

    # Upload all model files
    local_path = Path(local_dir)
    files_to_upload = []
    for f in local_path.iterdir():
        if f.is_file() and f.name not in {"optimizer.pt", "scheduler.pt"}:
            if f.name.startswith("rng_state"):
                continue
            files_to_upload.append(f)

    print(f"Uploading {len(files_to_upload)} files from {local_dir}...")
    for filepath in files_to_upload:
        print(f"  Uploading: {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filepath.name,
            repo_id=repo_id,
            token=token,
        )

    # Upload model card
    species_config = {
        "mouse": {
            "species_title": "Mouse Brain",
            "species": "mouse",
            "num_classes": "1,328",
            "data_source": "CCFv3 10um Nissl staining",
        },
        "human": {
            "species_title": "Human Brain",
            "species": "human",
            "num_classes": "44 (depth-3)",
            "data_source": "Human Brain Atlas (6 donors, Nissl staining)",
        },
    }
    config = species_config.get(species, species_config["mouse"])
    model_card = MODEL_CARD_TEMPLATE.format(**config)
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )

    print(f"Upload complete: https://huggingface.co/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained brain segmentation models to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is a one-time utility for the model developer. End users should use
`make download-models` to pull models from HuggingFace Hub.

Requires HUGGING_FACE_TOKEN environment variable with write access.
        """,
    )

    parser.add_argument(
        "--species",
        type=str,
        choices=["mouse", "human", "all"],
        default="all",
        help="Which model(s) to upload (default: all)",
    )
    parser.add_argument(
        "--mouse-model",
        type=str,
        default="./models/dinov2-upernet-final",
        help="Path to mouse model directory",
    )
    parser.add_argument(
        "--human-model",
        type=str,
        default="./models/human-depth3",
        help="Path to human model directory",
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        default=DEFAULT_HF_USERNAME,
        help=f"HuggingFace username (default: {DEFAULT_HF_USERNAME})",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=DEFAULT_MODEL_BASE,
        help=f"Base model name (default: {DEFAULT_MODEL_BASE})",
    )

    args = parser.parse_args()

    token = os.environ.get("HUGGING_FACE_TOKEN")
    if not token:
        print("ERROR: HUGGING_FACE_TOKEN environment variable not set.")
        print("\nSet it in your .env file or export it:")
        print("  export HUGGING_FACE_TOKEN=hf_your_token_here")
        sys.exit(1)

    uploads = []
    if args.species in ("mouse", "all"):
        repo_id = f"{args.hf_username}/{args.model_base}-mouse"
        uploads.append(("mouse", args.mouse_model, repo_id))
    if args.species in ("human", "all"):
        repo_id = f"{args.hf_username}/{args.model_base}-human"
        uploads.append(("human", args.human_model, repo_id))

    print("=" * 60)
    print("HuggingFace Hub Model Upload")
    print("=" * 60)

    success_count = 0
    for species, local_dir, repo_id in uploads:
        print(f"\n--- {species.title()} Model ---")
        print(f"  Local: {local_dir}")
        print(f"  Repo:  {repo_id}")
        if upload_model(local_dir, repo_id, species, token):
            success_count += 1

    print("\n" + "=" * 60)
    if success_count == len(uploads):
        print(f"All {success_count} model(s) uploaded successfully!")
    else:
        print(f"WARNING: {len(uploads) - success_count} upload(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
