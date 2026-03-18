# forums.fast.ai Crawler

Incrementally crawls [forums.fast.ai](https://forums.fast.ai) using the Discourse public JSON API and persists all topics and posts to disk.

## Usage

```bash
git clone https://github.com/adamhajari/fastai-forum-mcp
cd fastai-forum-mcp
uv sync

# 1. Crawl the forum
#    First run downloads pre-crawled history from Hugging Face, then fetches
#    any new posts since that snapshot. Subsequent runs are incremental (~minutes).
uv run python forum_crawler.py
uv run python forum_crawler.py --stats  # show stats without crawling

# 2. Build the search index (run once after crawling, ~few minutes)
uv run python build_index.py
```

### Registering the MCP server with Claude Code

The repo includes a `.mcp.json` that works if you run Claude Code from within the `fastai-forum-mcp` directory. If you run Claude Code from a different directory (more common), add the server to your global Claude config instead:

```bash
claude mcp add fastai-forum -- uv --directory /path/to/fastai-forum-mcp run python mcp_server.py
```

Replace `/path/to/fastai-forum-mcp` with the absolute path where you cloned this repo. Then restart Claude Code.

The first run downloads pre-crawled data from [Hugging Face](https://huggingface.co/datasets/adamhajari/fastai-forum), then fetches any posts newer than that snapshot. Subsequent runs only fetch topics that have had new activity since the last run.

## File Structure

```
fastai-forum-mcp/
├── .mcp.json              # tells Claude Code how to start the MCP server
├── forum_crawler.py       # the crawler
├── build_index.py         # builds the BM25 search index from crawled posts
├── mcp_server.py          # MCP server — exposes search_forum tool to Claude
├── pyproject.toml
├── uv.lock
└── data/                  # created after running the crawler (gitignored)
    ├── metadata.json      # index of all known topics
    ├── search_index.pkl   # BM25 index (built by build_index.py)
    └── posts/
        ├── 12345.json     # all posts for topic 12345
        ├── 12346.json
        └── ...
```

### `data/metadata.json`

Tracks crawler state and a lightweight summary of every topic. Checkpointed every 25 topics during a run.

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

One file per topic containing the topic metadata and all its posts.

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

Posts are stored as a flat dict keyed by post ID. Replies are linked to their parent via `reply_to_post_number`. To look up a parent post:

## Search Index

See [search-index-design.md](search-index-design.md) for design notes, current approach, and a documented alternative approach using SQLite FTS5.

---

Posts are stored as a flat dict keyed by post ID. Replies are linked to their parent via `reply_to_post_number`. To look up a parent post:

```python
import json

with open("data/posts/12345.json") as f:
    topic = json.load(f)

by_number = {p["post_number"]: p for p in topic["posts"].values()}

for post in topic["posts"].values():
    if post["reply_to_post_number"]:
        parent = by_number.get(post["reply_to_post_number"])
```
