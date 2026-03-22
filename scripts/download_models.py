#!/usr/bin/env python3
"""
Download trained brain segmentation models from HuggingFace Hub.

Downloads DINOv2-UperNet models trained on Allen Brain Institute data
for mouse and/or human brain region segmentation.

Usage:
    # Download all models (~2.5 GB total)
    python scripts/download_models.py

    # Download mouse model only (~1.2 GB)
    python scripts/download_models.py --species mouse

    # Download human model only (~1.2 GB)
    python scripts/download_models.py --species human

    # Custom HuggingFace repo
    python scripts/download_models.py --species mouse --repo-id your-user/your-repo
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from histological_image_analysis.download import (
    DEFAULT_HF_USERNAME,
    DEFAULT_MODEL_BASE,
    resolve_repo_ids,
    verify_model_download,
)


SPECIES_MAP = {
    "mouse": "mouse",
    "human": "human",
}


def download_model(repo_id: str, local_dir: str) -> bool:
    """Download a model from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID (e.g., "Noel-Niko/dinov2-upernet-20260322-histology-annotation-mouse").
    local_dir : str
        Local directory to save the model to.

    Returns
    -------
    bool
        True if download and verification succeeded.
    """
    print(f"Downloading {repo_id} -> {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.md", ".gitattributes"],
        )
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Check that the repo exists: https://huggingface.co/{repo_id}")
        print("  2. If the repo is private, set HUGGING_FACE_TOKEN in your .env")
        print("  3. Check your internet connection")
        return False

    if verify_model_download(local_dir):
        print(f"Download verified: {local_dir}")
        return True

    print(f"WARNING: Download may be incomplete at {local_dir}")
    print("  Missing one or more required files (config.json, preprocessor_config.json, weights)")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download trained brain segmentation models from HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Models are downloaded from HuggingFace Hub to the local models/ directory.

Default repos:
  Mouse:          {DEFAULT_HF_USERNAME}/{DEFAULT_MODEL_BASE}-mouse
  Human:          {DEFAULT_HF_USERNAME}/{DEFAULT_MODEL_BASE}-human
  Human BigBrain: {DEFAULT_HF_USERNAME}/{DEFAULT_MODEL_BASE}-human-bigbrain

After downloading, annotate images with:
  python scripts/annotate.py /path/to/slides/
        """,
    )

    parser.add_argument(
        "--species",
        type=str,
        choices=["mouse", "human", "human-bigbrain", "all"],
        default="all",
        help="Which model(s) to download (default: all)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Custom HuggingFace repo ID (overrides default naming)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Base directory for model storage (default: ./models)",
    )

    args = parser.parse_args()

    repo_ids = resolve_repo_ids(args.species, repo_id=args.repo_id)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Brain Segmentation Model Downloader")
    print("=" * 60)
    print(f"Species: {args.species}")
    print(f"Output:  {output_base.resolve()}")
    print()

    success_count = 0
    for repo_id in repo_ids:
        # Determine species from repo name for local directory
        if args.repo_id:
            # Custom repo: use species arg directly for directory name
            species_name = args.species if args.species != "all" else "custom"
            local_dir = str(output_base / species_name)
        else:
            species_name = repo_id.rsplit("-", 1)[-1]  # "mouse" or "human"
            local_dir = str(output_base / species_name)

        if download_model(repo_id, local_dir):
            success_count += 1
            print()

    print("=" * 60)
    if success_count == len(repo_ids):
        print(f"All {success_count} model(s) downloaded successfully!")
        print("\nNext step: annotate your brain images:")
        print("  make annotate-mouse IMAGES=/path/to/slides/")
    else:
        print(f"WARNING: {len(repo_ids) - success_count} download(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
