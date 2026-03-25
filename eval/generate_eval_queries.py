#!/usr/bin/env python3
"""
Generate synthetic evaluation queries for benchmarking embedding quality.

For each sampled post, asks Claude to write a natural search query that a user
would type to find that post. The (query, post_id) pairs are saved as a JSONL
eval set that eval_embeddings.py uses to measure MRR and Recall@K.

Usage:
    uv run python eval/generate_eval_queries.py [--n 200] [--out data/eval_queries.jsonl]

Requires an Anthropic API key:
    export ANTHROPIC_API_KEY=sk-ant-...

Get a key at https://console.anthropic.com/
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("anthropic required: uv add anthropic")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
EVAL_DATA_DIR = Path(__file__).parent / "data"
POSTS_META_FILE = REPO_ROOT / "data" / "embeddings" / "posts.pkl"
DEFAULT_OUT = EVAL_DATA_DIR / "eval_queries.jsonl"

# Use haiku — fast and cheap for simple query generation
MODEL = "claude-haiku-4-5-20251001"

PROMPT_TEMPLATE = """\
Below is a post from the fast.ai discussion forum. Write a single, natural search \
query that a user would type into a search engine to find this post. The query \
should reflect the post's main topic or question, and sound like something a real \
person would search for — not a summary of the post.

Respond with only the query, no explanation, no quotes.

Topic: {topic_title}

Post:
{text}"""


def load_posts() -> list[dict]:
    if not POSTS_META_FILE.exists():
        print(f"posts.pkl not found at {POSTS_META_FILE}. Run build_embeddings.py first.")
        sys.exit(1)
    print(f"Loading posts from {POSTS_META_FILE} …")
    with open(POSTS_META_FILE, "rb") as f:
        posts = pickle.load(f)
    print(f"Loaded {len(posts):,} posts")
    return posts


def sample_posts(posts: list[dict], n: int, seed: int = 42) -> list[dict]:
    """
    Sample OP posts (post_number == 1) from topics that have at least one reply
    by a different user. This models the real use case: searching for someone
    who had the same problem and received a solution.

    Filters:
    - post_number == 1 (the question being asked)
    - topic has >= 1 reply by a username other than the OP
    - post text is long enough to be a real question (>200 chars)
    """
    # Group all posts by topic_id
    from collections import defaultdict
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for p in posts:
        by_topic[p["topic_id"]].append(p)

    candidates = []
    for topic_id, topic_posts in by_topic.items():
        op = next((p for p in topic_posts if p["post_number"] == 1), None)
        if op is None:
            continue
        if len(op["text"]) < 200:
            continue
        # Check for at least one reply from a different user
        has_other_reply = any(
            p["post_number"] > 1 and p["username"] != op["username"]
            for p in topic_posts
        )
        if has_other_reply:
            candidates.append(op)

    print(f"{len(candidates):,} OP posts from topics with replies by other users")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def generate_query(client: anthropic.Anthropic, post: dict) -> str:
    text_snippet = post["text"][:1500]  # enough context, not the whole post
    prompt = PROMPT_TEMPLATE.format(
        topic_title=post["topic_title"],
        text=text_snippet,
    )
    message = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic eval queries")
    parser.add_argument("--n", type=int, default=200, help="Number of queries to generate")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    client = anthropic.Anthropic()  # fails fast if ANTHROPIC_API_KEY is not set

    posts = load_posts()
    sampled = sample_posts(posts, args.n, seed=args.seed)
    print(f"Sampled {len(sampled)} posts for query generation")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume from where we left off if the file already exists
    existing_ids: set[str] = set()
    if args.out.exists():
        with open(args.out) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["post_id"])
                except Exception:
                    pass
        print(f"Resuming — {len(existing_ids)} queries already generated")

    todo = [p for p in sampled if p["post_id"] not in existing_ids]
    print(f"{len(todo)} queries to generate")

    with open(args.out, "a") as out_f:
        for i, post in enumerate(todo, 1):
            try:
                query = generate_query(client, post)
            except Exception as e:
                print(f"  [{i}/{len(todo)}] ERROR for post {post['post_id']}: {e}")
                continue

            record = {
                "query": query,
                "post_id": post["post_id"],
                "topic_id": post["topic_id"],
                "topic_title": post["topic_title"],
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {query!r}")

    total = len(existing_ids) + len(todo)
    print(f"\nDone. {total} queries saved to {args.out}")


if __name__ == "__main__":
    main()
