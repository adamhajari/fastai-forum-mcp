#!/usr/bin/env python3
"""
Build a BM25 search index from all crawled forum posts.

Run this once after the crawler completes (and re-run after incremental updates).

Usage:
    python build_index.py

Output:
    data/search_index.pkl  (~500MB-1GB depending on corpus size)
"""

import json
import math
import pickle
import re
import sys
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("rank-bm25 required: pip install rank-bm25")
    sys.exit(1)

POSTS_DIR = Path(__file__).parent / "data" / "posts"
INDEX_FILE = Path(__file__).parent / "data" / "search_index.pkl"

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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\.]+", text.lower())


def build_index() -> None:
    files = sorted(f for f in POSTS_DIR.glob("*.json") if not f.name.startswith("._"))
    if not files:
        print(f"No post files found in {POSTS_DIR}")
        sys.exit(1)

    print(f"Processing {len(files)} topic files …")
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

    print(f"Total posts indexed: {len(posts)}")
    print("Building BM25 index (this may take a few minutes) …")

    corpus = [tokenize(p["text"]) for p in posts]
    bm25 = BM25Okapi(corpus)

    print(f"Saving index to {INDEX_FILE} …")
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "posts": posts}, f)

    size_mb = INDEX_FILE.stat().st_size / 1_000_000
    print(f"Done. Index size: {size_mb:.0f} MB")


if __name__ == "__main__":
    build_index()
