#!/usr/bin/env python3
"""
Crawler for forums.fast.ai (Discourse-based forum).

Uses the Discourse public JSON API to incrementally fetch topics and posts.

Storage layout:
  data/metadata.json         - index of all topics with timestamps
  data/posts/{topic_id}.json - all posts for each topic

Usage:
  python forum_crawler.py            # run crawler
  python forum_crawler.py --stats    # show stats without crawling
"""

import json
import time
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests library required: pip install requests")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

BASE_URL = "https://forums.fast.ai"
DATA_DIR = Path(__file__).parent / "data"
METADATA_FILE = DATA_DIR / "metadata.json"
POSTS_DIR = DATA_DIR / "posts"

RATE_LIMIT_DELAY = 1.0   # seconds between requests (~60 req/min)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

session = requests.Session()
session.headers.update({
    "User-Agent": "fastai-course-crawler/1.0 (educational; +https://course.fast.ai)"
})


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str) -> dict | None:
    """Fetch JSON from a URL with rate limiting, retry, and backoff."""
    time.sleep(RATE_LIMIT_DELAY)
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  [rate limited] waiting {wait}s …")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            print(f"  HTTP {resp.status_code} for {url}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {"last_run": None, "topics": {}}


def save_metadata(metadata: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))


def post_path(topic_id: int | str) -> Path:
    """Return the path for a topic's post file, bucketed into subdirectories."""
    bucket = int(topic_id) // 10000
    return POSTS_DIR / str(bucket) / f"{topic_id}.json"


def load_topic_posts(topic_id: int | str) -> dict:
    path = post_path(topic_id)
    if path.exists():
        return json.loads(path.read_text())
    return {"posts": {}}


def save_topic_posts(topic_id: int | str, data: dict) -> None:
    path = post_path(topic_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Discourse API helpers
# ---------------------------------------------------------------------------

def iter_latest_pages():
    """Yield topic dicts from /latest.json, page by page."""
    page = 0
    while True:
        data = fetch_json(f"{BASE_URL}/latest.json?page={page}")
        if not data:
            break
        topic_list = data.get("topic_list", {})
        topics = topic_list.get("topics", [])
        if not topics:
            break
        yield from topics
        if not topic_list.get("more_topics_url"):
            break
        page += 1


def get_topics_to_process(metadata: dict) -> dict[str, dict]:
    """
    Return {topic_id: topic_summary} for topics that need to be fetched or updated.

    First run:  all topics (paginate fully through /latest.json).
    Later runs: only topics whose bumped_at > last_run (stop early once we
                hit topics older than last_run, since /latest is sorted by
                recency).
    """
    last_run = metadata.get("last_run")
    known_topics = metadata.get("topics", {})
    topics_to_update: dict[str, dict] = {}

    if last_run is None:
        print("First run — fetching all topics …")
        for t in iter_latest_pages():
            tid = str(t["id"])
            known = known_topics.get(tid)
            if known and known.get("posts_count") == t.get("posts_count"):
                continue  # already fully fetched in a previous interrupted run
            topics_to_update[tid] = t
            if len(topics_to_update) % 100 == 0:
                print(f"  … {len(topics_to_update)} topics found so far")
    else:
        print(f"Incremental run (last run: {last_run}) …")
        for t in iter_latest_pages():
            bumped_at = t.get("bumped_at") or t.get("last_posted_at") or ""
            if bumped_at <= last_run:
                # /latest is sorted newest-first; everything from here is older
                break
            topics_to_update[str(t["id"])] = t

    return topics_to_update


def fetch_new_posts_for_topic(topic_id: str, existing_posts: dict) -> tuple[dict, dict]:
    """
    Fetch any posts for topic_id that are not already in existing_posts.

    Returns (topic_info, updated_posts_dict).
    """
    data = fetch_json(f"{BASE_URL}/t/{topic_id}.json")
    if not data:
        return {}, existing_posts

    topic_info = {
        "id": data.get("id"),
        "title": data.get("title"),
        "created_at": data.get("created_at"),
        "last_posted_at": data.get("last_posted_at"),
        "bumped_at": data.get("bumped_at"),
        "posts_count": data.get("posts_count"),
        "views": data.get("views"),
        "like_count": data.get("like_count"),
        "category_id": data.get("category_id"),
        "tags": data.get("tags", []),
        "slug": data.get("slug"),
        "archetype": data.get("archetype"),
    }

    # Full list of post IDs in the topic stream
    all_post_ids: list[int] = data.get("post_stream", {}).get("stream", [])
    new_post_ids = [pid for pid in all_post_ids if str(pid) not in existing_posts]

    # Fetch new posts in chunks of 20 (Discourse API limit)
    chunk_size = 20
    for i in range(0, len(new_post_ids), chunk_size):
        chunk = new_post_ids[i : i + chunk_size]
        ids_param = "&".join(f"post_ids[]={pid}" for pid in chunk)
        url = f"{BASE_URL}/t/{topic_id}/posts.json?{ids_param}"
        post_data = fetch_json(url)
        if post_data:
            for post in post_data.get("post_stream", {}).get("posts", []):
                pid = str(post["id"])
                existing_posts[pid] = {
                    "id": post["id"],
                    "post_number": post.get("post_number"),
                    "username": post.get("username"),
                    "created_at": post.get("created_at"),
                    "updated_at": post.get("updated_at"),
                    "reply_to_post_number": post.get("reply_to_post_number"),
                    "cooked": post.get("cooked"),   # rendered HTML
                    "raw": post.get("raw"),          # markdown (may be absent)
                    "like_count": post.get("like_count", 0),
                    "reads": post.get("reads", 0),
                }

    return topic_info, existing_posts


# ---------------------------------------------------------------------------
# Hugging Face bootstrap
# ---------------------------------------------------------------------------

HF_REPO = "adamhajari/fastai-forum"

def download_from_hub() -> None:
    """Download pre-crawled data from Hugging Face on first run."""
    if not HF_AVAILABLE:
        print("ERROR: huggingface_hub is required to download the pre-crawled forum data.")
        print("  Install it with: pip install huggingface_hub")
        print()
        print("  Alternatively, run with --no-download to skip this step and crawl from")
        print("  scratch instead. WARNING: this will take up to 24 hours to complete.")
        sys.exit(1)
    print(f"Downloading pre-crawled data from {HF_REPO} …")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
    )
    print("Download complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_crawler(no_download: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    POSTS_DIR.mkdir(parents=True, exist_ok=True)

    if not no_download and not METADATA_FILE.exists():
        download_from_hub()

    metadata = load_metadata()
    topics_to_process = get_topics_to_process(metadata)
    total = len(topics_to_process)
    print(f"Topics to process: {total}")

    for i, (topic_id, summary) in enumerate(topics_to_process.items(), 1):
        title = summary.get("title", "")[:70]
        print(f"[{i}/{total}] {topic_id}: {title}")

        existing = load_topic_posts(topic_id)
        existing_posts = existing.get("posts", {})

        topic_info, updated_posts = fetch_new_posts_for_topic(topic_id, existing_posts)
        if not topic_info:
            print(f"  skipped (no data)")
            continue

        new_count = len(updated_posts) - len(existing_posts)
        if new_count:
            print(f"  +{new_count} new posts (total {len(updated_posts)})")

        save_topic_posts(topic_id, {"topic": topic_info, "posts": updated_posts})

        metadata["topics"][topic_id] = {
            "title": topic_info["title"],
            "created_at": topic_info["created_at"],
            "last_posted_at": topic_info["last_posted_at"],
            "bumped_at": topic_info["bumped_at"],
            "posts_count": topic_info["posts_count"],
            "category_id": topic_info["category_id"],
            "tags": topic_info["tags"],
            "slug": topic_info["slug"],
        }

        # Checkpoint every 25 topics
        if i % 25 == 0:
            save_metadata(metadata)

    metadata["last_run"] = datetime.now(timezone.utc).isoformat()
    save_metadata(metadata)
    print(f"\nDone. Processed {total} topics. Total indexed: {len(metadata['topics'])}")


def show_stats() -> None:
    metadata = load_metadata()
    topics = metadata.get("topics", {})
    last_run = metadata.get("last_run", "never")
    total_posts = sum(
        len(json.loads(post_path(tid).read_text()).get("posts", {}))
        for tid in topics
        if post_path(tid).exists()
    )
    print(f"Last run   : {last_run}")
    print(f"Topics     : {len(topics)}")
    print(f"Posts      : {total_posts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="forums.fast.ai crawler")
    parser.add_argument("--stats", action="store_true", help="show stats and exit")
    parser.add_argument("--no-download", action="store_true", help="skip Hugging Face bootstrap download")
    args = parser.parse_args()
    if args.stats:
        show_stats()
    else:
        run_crawler(no_download=args.no_download)
