#!/usr/bin/env python3
"""
Evaluate embedding quality using synthetic queries.

For each (query, post_id) pair in the eval set, embeds the query and searches
the FAISS index. Reports MRR and Recall@K to measure how often the source post
appears near the top of results.

This evaluates raw embedding quality only — no BM25, no metadata boost — so
results reflect what the embedding model contributes on its own.

Usage:
    uv run python eval_embeddings.py
    uv run python eval_embeddings.py --queries data/eval_queries.jsonl
    uv run python eval_embeddings.py --index data/embeddings/index.faiss \\
                                     --posts data/embeddings/posts.pkl \\
                                     --model all-MiniLM-L6-v2
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# Prevent loky (used by sentence-transformers' tokenizer) from spawning parallel
# workers that conflict with FAISS's OpenMP threads on macOS.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # prevents loky/OpenMP deadlock on macOS with mpnet

# Heavy imports (faiss, torch, sentence-transformers) are deferred to the
# functions that need them so that `import eval_embeddings` is fast when only
# compute_pvalues (scipy-only) is needed.

REPO_ROOT = Path(__file__).parent.parent
EVAL_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_QUERIES = EVAL_DATA_DIR / "eval_queries.jsonl"
DEFAULT_INDEX = EVAL_DATA_DIR / "baseline" / "index.faiss"
DEFAULT_POSTS = EVAL_DATA_DIR / "baseline" / "posts.pkl"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

K_VALUES = [1, 5, 10, 20]


def load_eval_queries(path: Path) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_index(index_path: Path, posts_path: Path):
    import faiss
    print(f"Loading FAISS index from {index_path} …")
    index = faiss.read_index(str(index_path))
    print(f"Loading post metadata from {posts_path} …")
    with open(posts_path, "rb") as f:
        posts = pickle.load(f)
    # Build post_id → faiss position map
    post_id_to_pos = {p["post_id"]: i for i, p in enumerate(posts)}
    print(f"Index: {index.ntotal:,} vectors | Posts: {len(posts):,}")
    return index, posts, post_id_to_pos


def select_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate(queries, index, post_id_to_pos, model, max_k=20):
    """
    For each query, embed it and retrieve top max_k results from FAISS.
    Record the rank of the ground-truth post (None if not in top max_k).
    """
    results = []
    not_in_index = 0

    query_texts = [q["query"] for q in queries]
    print(f"Embedding {len(query_texts)} queries …")
    import numpy as np
    embeddings = model.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print(f"Searching FAISS (top {max_k}) …")
    _, indices = index.search(embeddings, max_k)

    for q, idx_row in zip(queries, indices):
        target_pos = post_id_to_pos.get(q["post_id"])
        if target_pos is None:
            not_in_index += 1
            continue

        rank = None
        for r, idx in enumerate(idx_row, 1):
            if idx == target_pos:
                rank = r
                break

        results.append({"query": q["query"], "post_id": q["post_id"], "rank": rank})

    if not_in_index:
        print(f"Warning: {not_in_index} eval posts not found in the index (skipped)")

    return results


def compute_pvalues(results_a: list[dict], results_b: list[dict], k_values: list[int]) -> dict:
    """
    Compute p-values comparing two paired eval result sets.

    Uses McNemar's test for Recall@K (binary hit/miss per query) and Wilcoxon
    signed-rank for MRR (continuous per-query 1/rank scores). Paired by post_id.
    """
    from scipy import stats

    ranks_a = {r["post_id"]: r["rank"] for r in results_a}
    ranks_b = {r["post_id"]: r["rank"] for r in results_b}
    common_ids = sorted(set(ranks_a) & set(ranks_b))

    pvalues = {}

    for k in k_values:
        hits_a = [ranks_a[pid] is not None and ranks_a[pid] <= k for pid in common_ids]
        hits_b = [ranks_b[pid] is not None and ranks_b[pid] <= k for pid in common_ids]
        # Discordant pairs: n01 = b wins, n10 = a wins
        n01 = sum(1 for a, b in zip(hits_a, hits_b) if not a and b)
        n10 = sum(1 for a, b in zip(hits_a, hits_b) if a and not b)
        if n01 + n10 == 0:
            pvalues[f"recall@{k}"] = 1.0
        else:
            # McNemar's chi-squared with Yates' continuity correction
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            pvalues[f"recall@{k}"] = float(stats.chi2.sf(chi2, df=1))

    # Wilcoxon signed-rank on per-query 1/rank contributions
    scores_a = [1 / ranks_a[pid] if ranks_a[pid] is not None else 0.0 for pid in common_ids]
    scores_b = [1 / ranks_b[pid] if ranks_b[pid] is not None else 0.0 for pid in common_ids]
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    if any(d != 0 for d in diffs):
        _, p = stats.wilcoxon(diffs, alternative="two-sided")
        pvalues["mrr"] = float(p)
    else:
        pvalues["mrr"] = 1.0

    return pvalues


def report(results: list[dict]):
    total = len(results)
    found = [r for r in results if r["rank"] is not None]

    print(f"\n{'─' * 50}")
    print(f"Eval set: {total} queries  ({total - len(found)} post not in top results)")
    print()

    # Recall@K
    for k in K_VALUES:
        hits = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k)
        print(f"  Recall@{k:<3} {hits/total:.3f}  ({hits}/{total})")

    # MRR
    mrr = sum(1 / r["rank"] for r in results if r["rank"] is not None) / total
    print(f"\n  MRR       {mrr:.3f}")
    print(f"{'─' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding quality")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--posts", type=Path, default=DEFAULT_POSTS)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    for path, name in [(args.queries, "eval queries"), (args.index, "FAISS index"), (args.posts, "posts metadata")]:
        if not path.exists():
            print(f"{name} not found at {path}")
            sys.exit(1)

    queries = load_eval_queries(args.queries)
    print(f"Loaded {len(queries)} eval queries from {args.queries}")

    index, posts, post_id_to_pos = load_index(args.index, args.posts)

    from sentence_transformers import SentenceTransformer
    device = select_device()
    print(f"Loading model '{args.model}' on {device} …")
    model = SentenceTransformer(args.model, device=device, trust_remote_code=True)

    max_k = max(K_VALUES)
    results = evaluate(queries, index, post_id_to_pos, model, max_k=max_k)
    report(results)


if __name__ == "__main__":
    main()
