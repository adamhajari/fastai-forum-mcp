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

import json
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
FAISS_CONFIG_FILE = DATA_DIR / "embeddings" / "config.json"
BASE_URL = "https://forums.fast.ai"

# ---------------------------------------------------------------------------
# Index state — populated by load_indexes() at startup, not at import time, so
# that this module can be imported (e.g. by tests) without the corpus present.
# ---------------------------------------------------------------------------

bm25 = None
bm25_posts = None
faiss_index = None
faiss_posts = None
_embed_model = None


def _load_bm25_index() -> None:
    global bm25, bm25_posts

    if not BM25_INDEX_FILE.exists():
        print(f"BM25 index not found at {BM25_INDEX_FILE}. Run build_index.py first.")
        sys.exit(1)

    print("Loading BM25 index …", flush=True)
    with open(BM25_INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    bm25_posts = data["posts"]
    print(f"BM25: {len(bm25_posts):,} posts loaded.", flush=True)


def _load_semantic_index() -> None:
    """Load the FAISS index — optional, graceful fallback if not built yet."""
    global faiss_index, faiss_posts, _embed_model

    if not (FAISS_INDEX_FILE.exists() and FAISS_POSTS_FILE.exists()):
        print("Semantic index not found — run build_embeddings.py to enable semantic/hybrid search.", flush=True)
        return

    try:
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        print("Loading FAISS index …", flush=True)
        faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(FAISS_POSTS_FILE, "rb") as f:
            faiss_posts = pickle.load(f)

        # The query-time model must be the one the index was built with; the
        # builder records it in config.json next to the index.
        if not FAISS_CONFIG_FILE.exists():
            print(
                f"ERROR: {FAISS_CONFIG_FILE} not found. This index was built by an older "
                "version of build_embeddings.py, so the embedding model it used is unknown.\n"
                "Rebuild it: uv run python build_embeddings.py [--model MODEL]",
                flush=True,
            )
            sys.exit(1)

        cfg = json.loads(FAISS_CONFIG_FILE.read_text())
        model_name = cfg["model_name"]
        if cfg["dim"] != faiss_index.d:
            print(
                f"ERROR: index config mismatch — {FAISS_CONFIG_FILE} says dim={cfg['dim']} "
                f"(model '{model_name}') but {FAISS_INDEX_FILE} has dim={faiss_index.d}.\n"
                "The config and the index are out of sync. Rebuild: "
                "uv run python build_embeddings.py [--model MODEL]",
                flush=True,
            )
            sys.exit(1)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading embedding model '{model_name}' (device={device}) …", flush=True)
        _embed_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        if cfg.get("max_seq_length"):
            _embed_model.max_seq_length = cfg["max_seq_length"]

        loaded_dim = _embed_model.get_sentence_embedding_dimension()
        if loaded_dim != faiss_index.d:
            print(
                f"ERROR: model '{model_name}' produces {loaded_dim}-dim embeddings but the "
                f"FAISS index is {faiss_index.d}-dim. Every semantic query would fail.\n"
                "Rebuild the index: uv run python build_embeddings.py --model "
                f"{model_name}",
                flush=True,
            )
            sys.exit(1)

        print(f"Semantic search: {faiss_index.ntotal:,} vectors loaded.", flush=True)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Semantic index unavailable: {e}", flush=True)
        faiss_index = None


def load_indexes() -> None:
    """Load both indexes into module-level state. Called once at startup."""
    _load_bm25_index()
    _load_semantic_index()


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


def main() -> None:
    load_indexes()
    mcp.run()


if __name__ == "__main__":
    main()
