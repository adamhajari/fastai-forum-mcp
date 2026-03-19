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

import json
import pickle
import re
import sys
import time
from pathlib import Path

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
    files = sorted(POSTS_DIR.glob("*.json"))
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


def build_index(posts: list[dict], device: str) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    # Truncate text before embedding (model has a 256-token limit anyway)
    texts = [p["text"][:MAX_TEXT_CHARS] for p in posts]

    print(f"Embedding {len(texts):,} posts in batches of {BATCH_SIZE} …")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # normalise for cosine similarity via dot product
    )
    elapsed = time.time() - t0
    print(f"Embedding took {elapsed:.0f}s ({len(texts)/elapsed:.0f} posts/sec)")

    # Build FAISS index (inner product on normalised vectors = cosine similarity)
    print("Building FAISS index …")
    index = faiss.IndexFlatIP(dim)   # exact search, inner product
    index.add(embeddings.astype(np.float32))
    print(f"Index contains {index.ntotal:,} vectors")

    print(f"Saving index to {FAISS_INDEX_FILE} …")
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    print(f"Saving post metadata to {POSTS_META_FILE} …")
    with open(POSTS_META_FILE, "wb") as f:
        pickle.dump(posts, f)

    size_mb = (FAISS_INDEX_FILE.stat().st_size + POSTS_META_FILE.stat().st_size) / 1e6
    print(f"Done. Total size: {size_mb:.0f} MB")


if __name__ == "__main__":
    device = select_device()
    posts = load_posts()
    build_index(posts, device)
