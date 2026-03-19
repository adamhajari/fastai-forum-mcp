#!/usr/bin/env python3
"""
Upload crawled forum data to Hugging Face.

Compresses posts/ into posts.tar.gz, uploads it along with metadata.json
to the adamhajari/fastai-forum dataset, then removes the local archive.

Usage:
    python upload_to_hub.py
"""

import sys
import tarfile
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("huggingface_hub required: pip install huggingface_hub")
    sys.exit(1)

HF_REPO = "adamhajari/fastai-forum"
DATA_DIR = Path(__file__).parent / "data"
POSTS_DIR = DATA_DIR / "posts"
METADATA_FILE = DATA_DIR / "metadata.json"
ARCHIVE = DATA_DIR / "posts.tar.gz"


def main() -> None:
    if not POSTS_DIR.exists():
        print(f"ERROR: posts directory not found at {POSTS_DIR}")
        print("Run the crawler first: python forum_crawler.py")
        sys.exit(1)

    if not METADATA_FILE.exists():
        print(f"ERROR: metadata.json not found at {METADATA_FILE}")
        sys.exit(1)

    print(f"Compressing {POSTS_DIR} → {ARCHIVE} …")
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(POSTS_DIR, arcname="posts")
    size_mb = ARCHIVE.stat().st_size / 1e6
    print(f"Archive size: {size_mb:.0f} MB")

    api = HfApi()
    print(f"Uploading posts.tar.gz to {HF_REPO} …")
    api.upload_file(
        path_or_fileobj=str(ARCHIVE),
        path_in_repo="posts.tar.gz",
        repo_id=HF_REPO,
        repo_type="dataset",
    )

    print("Uploading metadata.json …")
    api.upload_file(
        path_or_fileobj=str(METADATA_FILE),
        path_in_repo="metadata.json",
        repo_id=HF_REPO,
        repo_type="dataset",
    )

    ARCHIVE.unlink()
    print("Done. Local archive cleaned up.")


if __name__ == "__main__":
    main()
