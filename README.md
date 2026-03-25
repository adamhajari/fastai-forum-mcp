# forums.fast.ai Crawler

Incrementally crawls [forums.fast.ai](https://forums.fast.ai) using the Discourse public JSON API and persists all topics and posts to disk.

## Usage
1. Crawl the forum. First run downloads pre-crawled history from Hugging Face, then fetches any new posts since that snapshot. Subsequent runs are incremental (~minutes).
2. Build the search index (run once after crawling, ~few minutes)
3. Build the semantic vector index (run once after crawling, ~20 minutes)

```bash
git clone https://github.com/adamhajari/fastai-forum-mcp
cd fastai-forum-mcp
uv sync

# fetch history and crawl forum for new posts
uv run python forum_crawler.py
uv run python forum_crawler.py --stats  # show stats without crawling

# build BM25 search index
uv run python build_index.py

# build FAISS semantic vector index (recommended model)
uv run python build_embeddings.py --model nomic-ai/nomic-embed-text-v1 --max-seq-length 512
```

#### Embedding model

`nomic-ai/nomic-embed-text-v1` gave the best eval results (MRR=0.434, R@1=0.290 on 200 held-out queries). It requires the `einops` package (already in `pyproject.toml`) and `--max-seq-length 512` to avoid out-of-memory errors on Apple Silicon MPS.

If you run into issues with nomic, `multi-qa-MiniLM-L6-cos-v1` is a reliable fallback with nearly as strong results (MRR=0.410, R@1=0.290) and no special requirements:

```bash
uv run python build_embeddings.py --model multi-qa-MiniLM-L6-cos-v1
```

### Registering the MCP server with Claude Code

The repo includes a `.mcp.json` that works if you run Claude Code from within the `fastai-forum-mcp` directory. If you run Claude Code from a different directory (more common), add the server to your global Claude config instead:

```bash
claude mcp add fastai-forum -- uv --directory /path/to/fastai-forum-mcp run python mcp_server.py
```

Replace `/path/to/fastai-forum-mcp` with the absolute path where you cloned this repo. Then restart Claude Code.

The first run downloads pre-crawled data from [Hugging Face](https://huggingface.co/datasets/adamhajari/fastai-forum), then fetches any posts newer than that snapshot. Subsequent runs only fetch topics that have had new activity since the last run.

## Updating the Hugging Face dataset

After re-crawling, push the updated data to [huggingface.co/datasets/adamhajari/fastai-forum](https://huggingface.co/datasets/adamhajari/fastai-forum) so others can benefit from the latest posts:

```bash
# 1. Crawl new posts
uv run python forum_crawler.py

# 2. Rebuild indexes
uv run python build_index.py
uv run python build_embeddings.py --model nomic-ai/nomic-embed-text-v1 --max-seq-length 512

# 3. Upload to Hugging Face (requires write access to adamhajari/fastai-forum)
uv run python upload_to_hub.py
```

The compressed archive is ~52MB (vs 237MB uncompressed), making uploads and downloads much faster.

## File Structure

```
fastai-forum-mcp/
├── .mcp.json              # tells Claude Code how to start the MCP server
├── forum_crawler.py       # the crawler
├── build_index.py         # builds the BM25 search index from crawled posts
├── build_embeddings.py    # builds the FAISS semantic vector index from crawled posts
├── mcp_server.py          # MCP server — exposes search_forum tool to Claude
├── upload_to_hub.py       # compresses posts/ and uploads to Hugging Face
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
