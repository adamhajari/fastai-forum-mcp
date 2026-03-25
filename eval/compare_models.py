#!/usr/bin/env python3
"""
Compare embedding quality across different sentence-transformer models.

For each model, builds a FAISS index (if not already present) and runs the
eval to produce Recall@K and MRR scores. Prints a comparison table at the end.

Text preparation uses the title-prefix variant for all models (prepend topic
title, token-aware truncation) — pass --variant baseline to change this.

Models (from search-index-design.md):
  all-MiniLM-L6-v2               384-dim, 256 tokens  (current)
  multi-qa-MiniLM-L6-cos-v1      384-dim, 512 tokens, QA-tuned
  all-mpnet-base-v2              768-dim, 514 tokens, stronger general model
  nomic-ai/nomic-embed-text-v1   768-dim, 8192 tokens, long context

Usage:
    uv run python eval/compare_models.py
    uv run python eval/compare_models.py --models all-MiniLM-L6-v2 multi-qa-MiniLM-L6-cos-v1
    uv run python eval/compare_models.py --skip-build   # use existing indexes
    uv run python eval/compare_models.py --variant baseline
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
MODELS_DATA_DIR = EVAL_DATA_DIR / "models"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

# Models to compare (name → description)
MODELS = {
    "all-MiniLM-L6-v2": "384-dim, 256 tokens (current)",
    "multi-qa-MiniLM-L6-cos-v1": "384-dim, 512 tokens, QA-tuned",
    "all-mpnet-base-v2": "768-dim, 514 tokens, stronger general model",
    "nomic-ai/nomic-embed-text-v1": "768-dim, 8192 tokens, long context",
}

K_VALUES = [1, 5, 10, 20]

# Long-context models: cap seq length (8192-token attention is O(n²), ~48GB) and batch size
MODEL_BATCH_SIZES = {
    "nomic-ai/nomic-embed-text-v1": 64,
}
MODEL_MAX_SEQ_LENGTHS = {
    "nomic-ai/nomic-embed-text-v1": 512,  # most forum posts fit; full 8192 is impractical
}


def model_dir(model_name: str) -> Path:
    # Sanitize model name for use as a directory name (e.g. org/model → org_model)
    safe = model_name.replace("/", "_")
    return MODELS_DATA_DIR / safe


def build_model(model_name: str, posts, device, variant: str, build_device: str = None) -> Path:
    import build_embeddings as be
    out_dir = model_dir(model_name)
    if (out_dir / "index.faiss").exists():
        print(f"[{model_name}] Index already exists at {out_dir}, skipping build.")
        return out_dir
    print(f"\n{'='*60}")
    print(f"Building index for model: {model_name}")
    print(f"  {MODELS.get(model_name, '')}")
    print(f"  variant: {variant}")
    print(f"{'='*60}")
    effective_device = build_device or device
    batch_size = MODEL_BATCH_SIZES.get(model_name)
    max_seq_length = MODEL_MAX_SEQ_LENGTHS.get(model_name)
    be.build_index(posts, effective_device, out_dir=out_dir, variant=variant, model_name=model_name, batch_size=batch_size, max_seq_length=max_seq_length)
    return out_dir


def eval_model(model_name: str, queries, variant: str) -> list[dict]:
    import eval_embeddings as ee
    from sentence_transformers import SentenceTransformer

    out_dir = model_dir(model_name)
    index_path = out_dir / "index.faiss"
    posts_path = out_dir / "posts.pkl"

    if not index_path.exists():
        print(f"[{model_name}] Index not found at {index_path}. Run without --skip-build.")
        sys.exit(1)

    print(f"\n[{model_name}] Loading index …")
    index, _posts, post_id_to_pos = ee.load_index(index_path, posts_path)

    device = ee.select_device()
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    max_k = max(K_VALUES)
    return ee.evaluate(queries, index, post_id_to_pos, model, max_k=max_k)


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


def print_comparison(all_metrics: dict[str, dict], baseline_model: str, all_pvalues: dict[str, dict] = None):
    max_name_len = max(len(m) for m in all_metrics)
    k_cols = [f"R@{k}" for k in K_VALUES]
    col_headers = ["Model"] + k_cols + ["MRR"]
    col_widths = [max(max_name_len, 5)] + [18] * (len(k_cols) + 1)

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    print(f"\n{'─' * 80}")
    print("Model comparison  (* p<0.05  ** p<0.01  *** p<0.001)")
    print(f"{'─' * 80}")
    print(fmt_row(col_headers))
    print("  ".join("─" * w for w in col_widths))

    baseline_metrics = all_metrics.get(baseline_model)
    for model_name, metrics in all_metrics.items():
        pvalues = (all_pvalues or {}).get(model_name, {})
        row = [model_name]
        for k in K_VALUES:
            val = metrics[f"recall@{k}"]
            cell = f"{val:.3f}"
            if baseline_metrics and model_name != baseline_model:
                delta = val - baseline_metrics[f"recall@{k}"]
                stars = sig_stars(pvalues.get(f"recall@{k}", 1.0))
                cell += f" ({delta:+.3f}){stars}"
            row.append(cell)
        mrr = metrics["mrr"]
        mrr_cell = f"{mrr:.3f}"
        if baseline_metrics and model_name != baseline_model:
            delta = mrr - baseline_metrics["mrr"]
            stars = sig_stars(pvalues.get("mrr", 1.0))
            mrr_cell += f" ({delta:+.3f}){stars}"
        row.append(mrr_cell)
        print(fmt_row(row))

    print(f"{'─' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS),
        help="Which models to evaluate (default: all)",
    )
    parser.add_argument(
        "--variant", choices=["baseline", "title-prefix"], default="title-prefix",
        help="Text preparation variant applied to all models (default: title-prefix)",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip building indexes (use existing ones)",
    )
    parser.add_argument(
        "--build-device", type=str, default=None,
        help="Override device for index building, e.g. --build-device cpu (default: auto-detect)",
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
        print("Loading posts (shared across all models) …")
        posts = be.load_posts()
        for model_name in args.models:
            build_model(model_name, posts, device, args.variant, build_device=args.build_device)

    all_results = {}
    all_metrics = {}
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"  {MODELS.get(model_name, '')}")
        print(f"{'='*60}")
        results = eval_model(model_name, queries, args.variant)
        all_results[model_name] = results
        all_metrics[model_name] = compute_metrics(results)

    # Compute p-values vs the baseline model for each other model
    import eval_embeddings as ee
    baseline_model = args.models[0]
    baseline_results = all_results.get(baseline_model)
    all_pvalues = {}
    if baseline_results:
        for model_name, results in all_results.items():
            if model_name != baseline_model:
                all_pvalues[model_name] = ee.compute_pvalues(baseline_results, results, K_VALUES)

    print_comparison(all_metrics, baseline_model, all_pvalues)


if __name__ == "__main__":
    main()
