#!/usr/bin/env python3
"""
Build a semantic vector index from all crawled forum posts.

Uses sentence-transformers to embed each post into a dense vector, then
stores the vectors in a FAISS index for fast approximate nearest-neighbor
search. Post metadata is stored alongside for retrieval.

The embedding model runs on Apple Silicon MPS (GPU) if available, falling
back to CPU. FAISS search runs on CPU (FAISS GPU requires CUDA, not MPS),
but searching 200k vectors on CPU takes only milliseconds.

Usage:
    pip install sentence-transformers faiss-cpu
    python build_embeddings.py

Output:
    data/embeddings/index.faiss   - FAISS vector index
    data/embeddings/posts.pkl     - post metadata + text (parallel to index)
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # prevents loky/OpenMP deadlock on macOS with mpnet

try:
    import faiss
except ImportError:
    print("faiss-cpu required: pip install faiss-cpu")
    sys.exit(1)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers required: pip install sentence-transformers")
    sys.exit(1)

import torch

POSTS_DIR = Path(__file__).parent / "data" / "posts"
EMBEDDINGS_DIR = Path(__file__).parent / "data" / "embeddings"
FAISS_INDEX_FILE = EMBEDDINGS_DIR / "index.faiss"
POSTS_META_FILE = EMBEDDINGS_DIR / "posts.pkl"

# all-MiniLM-L6-v2: 384-dim embeddings, fast, good quality for semantic search
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256   # increase if you have more GPU memory
MAX_TEXT_CHARS = 512  # truncate posts to this length before embedding


HTML_ENTITIES = [
    ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
    ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"),
]


def strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    for entity, char in HTML_ENTITIES:
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()


def select_device() -> str:
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return "mps"
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return "cuda"
    print("Using CPU")
    return "cpu"


def load_posts() -> list[dict]:
    files = sorted(f for f in POSTS_DIR.glob("*.json") if not f.name.startswith("._"))
    if not files:
        print(f"No post files found in {POSTS_DIR}")
        sys.exit(1)

    print(f"Loading posts from {len(files)} topic files …")
    posts = []

    for i, path in enumerate(files):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(files)} topics, {len(posts)} posts so far")
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  skipping {path.name}: {e}")
            continue

        topic = data.get("topic", {})
        topic_id = str(topic.get("id", path.stem))
        topic_title = topic.get("title", "")
        topic_slug = topic.get("slug", "")
        topic_tags = topic.get("tags", [])

        for post_id, post in data.get("posts", {}).items():
            text = strip_html(post.get("cooked", ""))
            if not text.strip():
                continue
            posts.append({
                "topic_id": topic_id,
                "topic_title": topic_title,
                "topic_slug": topic_slug,
                "topic_tags": topic_tags,
                "post_id": post_id,
                "post_number": post.get("post_number", 1),
                "username": post.get("username", ""),
                "created_at": post.get("created_at", ""),
                "like_count": post.get("like_count", 0),
                "reads": post.get("reads", 0),
                "text": text,
            })

    print(f"Loaded {len(posts):,} posts")
    return posts


def build_index(posts: list[dict], device: str, out_dir: Path = None, variant: str = "baseline", model_name: str = None, batch_size: int = None, max_seq_length: int = None) -> None:
    if out_dir is None:
        out_dir = EMBEDDINGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_file = out_dir / "index.faiss"
    posts_file = out_dir / "posts.pkl"

    # Models known to have platform-specific issues on Apple Silicon / MPS:
    #   all-mpnet-base-v2          — hangs at batch 0 unless OMP_NUM_THREADS=1 (already set above)
    #   nomic-ai/nomic-embed-text-v1 — requires `pip install einops`, trust_remote_code=True,
    #                                  and max_seq_length<=512 to avoid 48GB MPS OOM
    # If either model fails, multi-qa-MiniLM-L6-cos-v1 is a strong fallback:
    #   eval scores: R@1=0.290, MRR=0.410 vs nomic R@1=0.290, MRR=0.434
    #   Run: python build_embeddings.py --model multi-qa-MiniLM-L6-cos-v1

    name = model_name or MODEL_NAME
    print(f"Loading embedding model '{name}' …")
    try:
        model = SentenceTransformer(name, device=device, trust_remote_code=True)
    except Exception as e:
        print(f"\nERROR: Failed to load model '{name}': {e}")
        if name in ("all-mpnet-base-v2", "nomic-ai/nomic-embed-text-v1"):
            print("This model is known to have platform-specific issues.")
            print("Fallback: python build_embeddings.py --model multi-qa-MiniLM-L6-cos-v1")
        raise
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
        print(f"Max sequence length: {max_seq_length} tokens")
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    if variant == "title-prefix":
        # Prepend topic title; sentence-transformers handles token-aware truncation
        texts = [f"{p['topic_title']}\n\n{p['text']}" for p in posts]
        print("Variant: title-prefix (topic title prepended, token-aware truncation)")
    else:
        # baseline: character-truncated text, no title
        texts = [p["text"][:MAX_TEXT_CHARS] for p in posts]
        print("Variant: baseline (char-truncated text, no title prefix)")

    effective_batch_size = batch_size or BATCH_SIZE
    print(f"Embedding {len(texts):,} posts in batches of {effective_batch_size} …")
    t0 = time.time()
    try:
        embeddings = model.encode(
            texts,
            batch_size=effective_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # normalise for cosine similarity via dot product
        )
    except Exception as e:
        print(f"\nERROR: Embedding failed for model '{name}': {e}")
        if name == "all-mpnet-base-v2":
            print("Tip: ensure OMP_NUM_THREADS=1 is set (prevents loky deadlock on macOS).")
        elif name == "nomic-ai/nomic-embed-text-v1":
            print("Tip: this model needs max_seq_length<=512 to avoid MPS memory errors.")
            print("     Pass --max-seq-length 512 if calling from the CLI.")
        print("Fallback: python build_embeddings.py --model multi-qa-MiniLM-L6-cos-v1")
        raise
    elapsed = time.time() - t0
    print(f"Embedding took {elapsed:.0f}s ({len(texts)/elapsed:.0f} posts/sec)")

    # Build FAISS index (inner product on normalised vectors = cosine similarity)
    print("Building FAISS index …")
    index = faiss.IndexFlatIP(dim)   # exact search, inner product
    index.add(embeddings.astype(np.float32))
    print(f"Index contains {index.ntotal:,} vectors")

    print(f"Saving index to {faiss_file} …")
    faiss.write_index(index, str(faiss_file))

    print(f"Saving post metadata to {posts_file} …")
    with open(posts_file, "wb") as f:
        pickle.dump(posts, f)

    size_mb = (faiss_file.stat().st_size + posts_file.stat().st_size) / 1e6
    print(f"Done. Total size: {size_mb:.0f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS embedding index")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for index.faiss and posts.pkl (default: data/embeddings/)",
    )
    parser.add_argument(
        "--variant", choices=["baseline", "title-prefix"], default="title-prefix",
        help="Text preparation variant (default: title-prefix)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Embedding model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use for embedding (default: auto-detect mps/cuda/cpu)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=None,
        help="Override model max sequence length (e.g. 512 for nomic on MPS to avoid OOM)",
    )
    args = parser.parse_args()

    device = args.device or select_device()
    posts = load_posts()
    build_index(posts, device, out_dir=args.out_dir, variant=args.variant, model_name=args.model, max_seq_length=args.max_seq_length)
