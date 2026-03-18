#!/usr/bin/env python3
"""
MCP server exposing a search_forum tool over the indexed fast.ai forum posts.

Supports three search modes:
  - bm25     : keyword search (fast, great for error messages / function names)
  - semantic : vector similarity search (finds conceptually related posts)
  - hybrid   : combines both via Reciprocal Rank Fusion (default, best results)

Both indexes are loaded once on startup and stay resident in memory.

Usage (Claude Code starts this automatically via .mcp.json):
    python mcp_server.py
"""

import math
import pickle
import re
import sys
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("mcp required: pip install 'mcp[cli]'")
    sys.exit(1)

DATA_DIR = Path(__file__).parent / "data"
BM25_INDEX_FILE = DATA_DIR / "search_index.pkl"
FAISS_INDEX_FILE = DATA_DIR / "embeddings" / "index.faiss"
FAISS_POSTS_FILE = DATA_DIR / "embeddings" / "posts.pkl"
BASE_URL = "https://forums.fast.ai"

# ---------------------------------------------------------------------------
# Load BM25 index
# ---------------------------------------------------------------------------

if not BM25_INDEX_FILE.exists():
    print(f"BM25 index not found at {BM25_INDEX_FILE}. Run build_index.py first.")
    sys.exit(1)

print(f"Loading BM25 index …", flush=True)
with open(BM25_INDEX_FILE, "rb") as f:
    _bm25_data = pickle.load(f)
bm25 = _bm25_data["bm25"]
bm25_posts = _bm25_data["posts"]
print(f"BM25: {len(bm25_posts):,} posts loaded.", flush=True)

# ---------------------------------------------------------------------------
# Load semantic (FAISS) index — optional, graceful fallback if not built yet
# ---------------------------------------------------------------------------

faiss_index = None
faiss_posts = None

if FAISS_INDEX_FILE.exists() and FAISS_POSTS_FILE.exists():
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import torch

        print("Loading FAISS index …", flush=True)
        faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(FAISS_POSTS_FILE, "rb") as f:
            faiss_posts = pickle.load(f)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading embedding model (device={device}) …", flush=True)
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        print(f"Semantic search: {faiss_index.ntotal:,} vectors loaded.", flush=True)
    except Exception as e:
        print(f"Semantic index unavailable: {e}", flush=True)
        faiss_index = None
else:
    print("Semantic index not found — run build_embeddings.py to enable semantic/hybrid search.", flush=True)

# ---------------------------------------------------------------------------
# Scoring helpers (shared)
# ---------------------------------------------------------------------------

def _recency_score(created_at: str) -> float:
    """Linear score 0–1. 2025 → 1.0, 2018 or earlier → 0.0."""
    try:
        return max(0.0, min(1.0, (int(created_at[:4]) - 2018) / 7))
    except Exception:
        return 0.5


def _like_score(like_count: int) -> float:
    """Logarithmic like score, ~10 likes → 1.0."""
    return math.log(1 + like_count) / math.log(11)


def _meta_boost(post: dict) -> float:
    """Combined recency + likes bonus (0–1 range)."""
    return _recency_score(post["created_at"]) * 0.25 + _like_score(post["like_count"]) * 0.15


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\.]+", text.lower())


def _format_results(ranked_posts: list[dict], query: str) -> str:
    if not ranked_posts:
        return "No relevant posts found."

    parts = [f"Found {len(ranked_posts)} results for: {query!r}\n"]
    for rank, post in enumerate(ranked_posts, 1):
        url = (
            f"{BASE_URL}/t/{post['topic_slug']}"
            f"/{post['topic_id']}/{post['post_number']}"
        )
        tags = ", ".join(post["topic_tags"]) if post["topic_tags"] else "—"
        date = post["created_at"][:10] if post["created_at"] else "unknown"
        text = post["text"]
        if len(text) > 2000:
            text = text[:2000] + " …[truncated]"

        parts.append(
            f"[{rank}] {post['topic_title']}\n"
            f"URL: {url}\n"
            f"Date: {date}  |  Likes: {post['like_count']}  |"
            f"  By: {post['username']}  |  Tags: {tags}\n\n"
            f"{text}\n"
            f"{'─' * 60}"
        )
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Search implementations
# ---------------------------------------------------------------------------

def _search_bm25(query: str, n: int) -> list[dict]:
    tokens = _tokenize(query)
    raw_scores = bm25.get_scores(tokens)
    max_score = raw_scores.max() or 1.0

    scored = [
        ((raw_scores[i] / max_score) * 0.6 + _meta_boost(bm25_posts[i]), i)
        for i in range(len(bm25_posts))
        if raw_scores[i] > 0
    ]
    scored.sort(reverse=True)
    return [bm25_posts[i] for _, i in scored[:n]]


def _search_semantic(query: str, n: int) -> list[dict]:
    q_vec = _embed_model.encode(
        [query[:512]], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    distances, indices = faiss_index.search(q_vec, n * 3)  # fetch more for re-ranking

    scored = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        post = faiss_posts[idx]
        # dist is cosine similarity (0–1) since vectors are normalised
        score = dist * 0.6 + _meta_boost(post)
        scored.append((score, post))

    scored.sort(reverse=True)
    return [p for _, p in scored[:n]]


def _search_hybrid(query: str, n: int) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF): combine BM25 and semantic rankings.
    Each post's score = 1/(k + bm25_rank) + 1/(k + semantic_rank).
    k=60 is the standard constant that dampens the effect of high ranks.
    """
    k = 60
    fetch = n * 5  # fetch more candidates from each to allow fusion

    bm25_results = _search_bm25(query, fetch)
    sem_results = _search_semantic(query, fetch)

    # Build post_id → RRF score map
    scores: dict[str, float] = {}
    post_by_id: dict[str, dict] = {}

    for rank, post in enumerate(bm25_results, 1):
        pid = post["post_id"]
        scores[pid] = scores.get(pid, 0) + 1 / (k + rank)
        post_by_id[pid] = post

    for rank, post in enumerate(sem_results, 1):
        pid = post["post_id"]
        scores[pid] = scores.get(pid, 0) + 1 / (k + rank)
        post_by_id[pid] = post

    # Add metadata boost on top of RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1] + _meta_boost(post_by_id[x[0]]) * 0.1, reverse=True)
    return [post_by_id[pid] for pid, _ in ranked[:n]]

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("fastai-forum-search")


@mcp.tool()
def search_forum(query: str, n_results: int = 20, mode: str = "hybrid") -> str:
    """Search the fast.ai forum for posts relevant to a question or topic.

    Use this tool whenever the user asks about the fast.ai course, its
    notebooks, or any errors encountered while working through it. Run
    multiple searches with different phrasings if the first result doesn't
    surface a clear answer. Prefer recent posts (2023+) and highly-liked posts.

    Args:
        query: Natural language question or keywords to search for.
        n_results: Number of results to return (default 20, max 50).
        mode: Search mode — "hybrid" (default, best results), "semantic"
              (meaning-based, good for conceptual questions), or "bm25"
              (keyword-based, good for exact error messages / function names).
              Falls back to "bm25" if the semantic index hasn't been built yet.
    """
    n_results = min(n_results, 50)

    if mode in ("semantic", "hybrid") and faiss_index is None:
        mode = "bm25"

    if mode == "semantic":
        results = _search_semantic(query, n_results)
    elif mode == "hybrid":
        results = _search_hybrid(query, n_results)
    else:
        results = _search_bm25(query, n_results)

    return _format_results(results, query)


if __name__ == "__main__":
    mcp.run()
