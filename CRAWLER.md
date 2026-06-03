# Forum crawler

The crawler incrementally syncs [forums.fast.ai](https://forums.fast.ai) to local disk using the Discourse public JSON API. It is the data source for the search indexes and MCP server described in [README.md](README.md).

## Usage

```bash
uv run python forum_crawler.py              # download snapshot (first run) + incremental sync
uv run python forum_crawler.py --stats      # show corpus stats without crawling
uv run python forum_crawler.py --no-download  # skip Hugging Face bootstrap; crawl from scratch
```

### First run

If `data/metadata.json` does not exist, the crawler downloads a pre-crawled corpus from [adamhajari/fastai-forum](https://huggingface.co/datasets/adamhajari/fastai-forum) on Hugging Face:

- `metadata.json` — topic index and crawler state
- `posts.tar.gz` — compressed archive of all topic JSON files (~52 MB vs ~237 MB uncompressed)

The archive is extracted automatically and removed locally. A full crawl from scratch with `--no-download` can take up to 24 hours.

### Subsequent runs

Only topics with new activity since the last run are fetched. Typical incremental runs finish in minutes. The crawler checkpoints `metadata.json` every 25 topics and resumes safely if interrupted.

### Rate limiting

Requests are limited to ~1 req/sec (60/min) with automatic backoff on HTTP 429. The user agent identifies the crawler as educational.

## On-disk layout

```
data/
├── metadata.json          # topic index + last_run timestamp
└── posts/
    └── {topic_id}.json    # one file per topic
```

Search indexes (`search_index.pkl`, `embeddings/`) are built separately — see the README.

### `data/metadata.json`

Tracks crawler state and a lightweight summary of every topic:

```json
{
  "last_run": "2024-01-15T10:30:00+00:00",
  "topics": {
    "12345": {
      "title": "How do I train a model?",
      "created_at": "2022-01-15T10:30:00.000Z",
      "last_posted_at": "2022-03-01T14:22:00.000Z",
      "bumped_at": "2022-03-01T14:22:00.000Z",
      "posts_count": 8,
      "category_id": 10,
      "tags": ["fastai", "training"],
      "slug": "how-do-i-train-a-model"
    }
  }
}
```

### `data/posts/{topic_id}.json`

One file per topic containing topic metadata and all posts, keyed by post ID:

```json
{
  "topic": {
    "id": 12345,
    "title": "How do I train a model?",
    "created_at": "2022-01-15T10:30:00.000Z",
    "last_posted_at": "2022-03-01T14:22:00.000Z",
    "posts_count": 8,
    "views": 342,
    "like_count": 12,
    "category_id": 10,
    "tags": ["fastai", "training"],
    "slug": "how-do-i-train-a-model"
  },
  "posts": {
    "98001": {
      "id": 98001,
      "post_number": 1,
      "username": "jsmith",
      "created_at": "2022-01-15T10:30:00.000Z",
      "updated_at": "2022-01-15T10:30:00.000Z",
      "reply_to_post_number": null,
      "cooked": "<p>How do I train a model?</p>",
      "raw": "How do I train a model?",
      "like_count": 2,
      "reads": 45
    }
  }
}
```

- `cooked` — rendered HTML from Discourse
- `raw` — markdown source (may be absent on older posts)
- `reply_to_post_number` — links replies to their parent post within the topic

To resolve a parent post from a reply:

```python
import json

with open("data/posts/12345.json") as f:
    topic = json.load(f)

by_number = {p["post_number"]: p for p in topic["posts"].values()}

for post in topic["posts"].values():
    if post["reply_to_post_number"]:
        parent = by_number.get(post["reply_to_post_number"])
```

## Updating the Hugging Face dataset

After re-crawling, maintainers can publish updated data so others get a current bootstrap snapshot:

```bash
# 1. Crawl new posts
uv run python forum_crawler.py

# 2. Rebuild search indexes (see README)
uv run python build_index.py
uv run python build_embeddings.py

# 3. Upload (requires write access to adamhajari/fastai-forum)
uv run python upload_to_hub.py
```

`upload_to_hub.py` compresses `data/posts/` into `posts.tar.gz`, uploads it with `metadata.json`, then deletes the local archive.

For very large updates, use Hugging Face's `upload_large_folder` API instead of `upload_folder` — the full dataset can exceed single-commit size limits and trigger 504 timeouts.

Posts are stored as a flat archive on Hugging Face (not individual files) to stay under the 10k files-per-directory limit and keep downloads small.
