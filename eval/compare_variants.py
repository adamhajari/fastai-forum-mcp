#!/usr/bin/env python3
"""
Compare embedding quality across variants using synthetic eval queries.

For each variant, builds a FAISS index (if not already present) and runs the
eval to produce Recall@K and MRR scores. Prints a comparison table at the end.

Variants:
  baseline      - current approach: char-truncated text, no title prefix
  title-prefix  - topic title prepended, token-aware truncation by sentence-transformers

Usage:
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --variants baseline title-prefix
    uv run python eval/run_eval.py --skip-build   # use existing indexes
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # prevents loky/OpenMP deadlock on macOS with mpnet

REPO_ROOT = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
EVAL_DATA_DIR = EVAL_DIR / "data"

# Make repo root importable so we can reuse build_embeddings and eval_embeddings
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

VARIANTS = {
    "baseline": "char-truncated text, no title prefix (current behavior)",
    "title-prefix": "topic title prepended, token-aware truncation",
}

K_VALUES = [1, 5, 10, 20]


def build_variant(variant: str, posts, device) -> Path:
    import build_embeddings as be
    out_dir = EVAL_DATA_DIR / variant
    if (out_dir / "index.faiss").exists():
        print(f"[{variant}] Index already exists at {out_dir}, skipping build.")
        return out_dir
    print(f"\n{'='*60}")
    print(f"Building index for variant: {variant}")
    print(f"  {VARIANTS[variant]}")
    print(f"{'='*60}")
    be.build_index(posts, device, out_dir=out_dir, variant=variant)
    return out_dir


def eval_variant(variant: str, queries) -> dict:
    import eval_embeddings as ee

    out_dir = EVAL_DATA_DIR / variant
    index_path = out_dir / "index.faiss"
    posts_path = out_dir / "posts.pkl"

    if not index_path.exists():
        print(f"[{variant}] Index not found at {index_path}. Run without --skip-build.")
        sys.exit(1)

    print(f"\n[{variant}] Loading index …")
    index, _posts, post_id_to_pos = ee.load_index(index_path, posts_path)

    device = ee.select_device()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ee.DEFAULT_MODEL, device=device)

    max_k = max(K_VALUES)
    results = ee.evaluate(queries, index, post_id_to_pos, model, max_k=max_k)
    return results


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    metrics = {}
    for k in K_VALUES:
        hits = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k)
        metrics[f"recall@{k}"] = hits / total
    metrics["mrr"] = sum(1 / r["rank"] for r in results if r["rank"] is not None) / total
    return metrics


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def print_comparison(all_metrics: dict[str, dict], all_pvalues: dict[str, dict] = None):
    k_cols = [f"R@{k}" for k in K_VALUES]
    col_headers = ["Variant"] + k_cols + ["MRR"]
    col_widths = [max(14, max(len(v) for v in VARIANTS))] + [18] * (len(k_cols) + 1)

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    print(f"\n{'─' * 74}")
    print("Embedding variant comparison  (* p<0.05  ** p<0.01  *** p<0.001)")
    print(f"{'─' * 74}")
    print(fmt_row(col_headers))
    print("  ".join("─" * w for w in col_widths))

    baseline_metrics = all_metrics.get("baseline")
    for variant, metrics in all_metrics.items():
        pvalues = (all_pvalues or {}).get(variant, {})
        row = [variant]
        for k in K_VALUES:
            val = metrics[f"recall@{k}"]
            cell = f"{val:.3f}"
            if baseline_metrics and variant != "baseline":
                delta = val - baseline_metrics[f"recall@{k}"]
                stars = sig_stars(pvalues.get(f"recall@{k}", 1.0))
                cell += f" ({delta:+.3f}){stars}"
            row.append(cell)
        mrr = metrics["mrr"]
        mrr_cell = f"{mrr:.3f}"
        if baseline_metrics and variant != "baseline":
            delta = mrr - baseline_metrics["mrr"]
            stars = sig_stars(pvalues.get("mrr", 1.0))
            mrr_cell += f" ({delta:+.3f}){stars}"
        row.append(mrr_cell)
        print(fmt_row(row))

    print(f"{'─' * 74}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare embedding variants")
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
        help="Which variants to evaluate (default: all)",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip building indexes (use existing ones)",
    )
    parser.add_argument(
        "--queries", type=Path, default=EVAL_DATA_DIR / "eval_queries.jsonl",
    )
    args = parser.parse_args()

    if not args.queries.exists():
        print(f"Eval queries not found at {args.queries}")
        print("Run: uv run python eval/generate_eval_queries.py")
        sys.exit(1)

    import eval_embeddings as ee
    queries = ee.load_eval_queries(args.queries)
    print(f"Loaded {len(queries)} eval queries from {args.queries}")

    if not args.skip_build:
        import build_embeddings as be
        device = be.select_device()
        print(f"Loading posts (shared across all variants) …")
        posts = be.load_posts()
        for variant in args.variants:
            build_variant(variant, posts, device)

    all_results = {}
    all_metrics = {}
    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"Evaluating variant: {variant}")
        print(f"  {VARIANTS[variant]}")
        print(f"{'='*60}")
        results = eval_variant(variant, queries)
        all_results[variant] = results
        all_metrics[variant] = compute_metrics(results)

    # Compute p-values vs baseline for each non-baseline variant
    import eval_embeddings as ee
    baseline_results = all_results.get("baseline")
    all_pvalues = {}
    if baseline_results:
        for variant, results in all_results.items():
            if variant != "baseline":
                all_pvalues[variant] = ee.compute_pvalues(baseline_results, results, K_VALUES)

    print_comparison(all_metrics, all_pvalues)


if __name__ == "__main__":
    main()
