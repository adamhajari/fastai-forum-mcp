# Search Index Design

## Current approach: BM25 + pickle (`build_index.py` + `mcp_server.py`)

`build_index.py` reads all `data/posts/*.json` files, strips HTML, tokenizes,
and builds a `BM25Okapi` index (from the `rank-bm25` package). The index and
post metadata are serialised together into `data/search_index.pkl`.

`mcp_server.py` loads the pickle once on startup and keeps it in memory.
Search results are re-ranked by a weighted combination of:
- **60% BM25 score** (keyword relevance)
- **25% recency** (linear scale: 2025 → 1.0, 2018 → 0.0)
- **15% likes** (logarithmic: ~10 likes → max bonus)

### Limitation: no incremental updates

BM25 IDF scores are global statistics computed across the entire corpus.
Adding new documents changes the scores for all existing documents, so the
index must be rebuilt from scratch every time. This means `build_index.py`
must be re-run after every crawler run to include new posts.

---

## Alternative approach: SQLite FTS5 (not yet implemented)

SQLite's built-in FTS5 extension uses BM25 ranking internally and supports
incremental `INSERT`/`UPDATE`/`DELETE` — no full rebuild needed.

To implement this, ask: **"implement the alternative approach to building the
index using SQLite"**.

### What to build

Replace `build_index.py` and the pickle with a single SQLite database at
`data/forum.db`. The JSON post files stay unchanged as the source of truth.

#### Schema

```sql
-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE posts_fts USING fts5(
    post_id,          -- stored, not indexed (use UNINDEXED)
    topic_id,
    text,             -- the searchable content (plain text, HTML stripped)
    tokenize = 'porter unicode61'  -- porter stemming improves recall
);

-- Metadata table for re-ranking (not full-text searched)
CREATE TABLE post_meta (
    post_id      TEXT PRIMARY KEY,
    topic_id     TEXT,
    topic_title  TEXT,
    topic_slug   TEXT,
    topic_tags   TEXT,   -- JSON array stored as string
    post_number  INTEGER,
    username     TEXT,
    created_at   TEXT,
    like_count   INTEGER,
    reads        INTEGER
);
```

#### `build_index.py` changes

- Open (or create) `data/forum.db`
- Iterate all `data/posts/*.json` files
- For each post: `INSERT OR REPLACE INTO posts_fts` and `INSERT OR REPLACE INTO post_meta`
- This is also used to do a full rebuild if the DB is deleted

#### `forum_crawler.py` changes

After `save_topic_posts(...)`, also upsert each new/updated post into the DB:

```python
def upsert_posts_to_db(db_conn, topic_info, posts_dict):
    for post_id, post in posts_dict.items():
        text = strip_html(post.get("cooked", ""))
        db_conn.execute(
            "INSERT OR REPLACE INTO posts_fts(post_id, topic_id, text) VALUES (?,?,?)",
            (post_id, str(topic_info["id"]), text)
        )
        db_conn.execute(
            "INSERT OR REPLACE INTO post_meta VALUES (?,?,?,?,?,?,?,?,?,?)",
            (post_id, str(topic_info["id"]), topic_info["title"],
             topic_info["slug"], json.dumps(topic_info["tags"]),
             post["post_number"], post["username"], post["created_at"],
             post["like_count"], post["reads"])
        )
    db_conn.commit()
```

Open the DB connection once in `run_crawler()` and pass it through.

#### `mcp_server.py` changes

Replace pickle load + BM25 scoring with a DB query:

```python
import sqlite3, math

DB_FILE = Path(__file__).parent / "data" / "forum.db"

def search_forum(query: str, n_results: int = 20) -> str:
    conn = sqlite3.connect(DB_FILE)

    # FTS5 returns BM25 score (negative — lower is better)
    rows = conn.execute("""
        SELECT
            f.post_id,
            bm25(posts_fts) AS bm25_score,
            m.topic_title, m.topic_slug, m.topic_id,
            m.post_number, m.username, m.created_at,
            m.like_count, m.topic_tags,
            snippet(posts_fts, 2, '[', ']', '...', 32) AS snippet
        FROM posts_fts f
        JOIN post_meta m ON f.post_id = m.post_id
        WHERE posts_fts MATCH ?
        ORDER BY bm25_score   -- FTS5 native BM25, ascending (more negative = better)
        LIMIT 200              -- fetch more than needed for re-ranking
    """, (query,)).fetchall()

    # Re-rank by recency + likes (same formula as current approach)
    ...
```

`snippet()` is an FTS5 built-in that returns the matching excerpt with
highlighted terms — useful for showing why a result was returned.

### Advantages over current approach

| | BM25 pickle | SQLite FTS5 |
|---|---|---|
| Incremental updates | No (full rebuild) | Yes (`INSERT OR REPLACE`) |
| Memory on MCP startup | ~500MB–1GB loaded into RAM | Minimal (DB queried on demand) |
| Extra dependencies | `rank-bm25` | None (stdlib `sqlite3`) |
| Stemming support | No | Yes (`porter` tokenizer) |
| Result snippets | No | Yes (`snippet()`) |
| Rebuild if DB deleted | Yes, via `build_index.py` | Yes, via `build_index.py` |

---

## Embedding quality improvements (not yet implemented)

The current `build_embeddings.py` has two issues that likely degrade search quality:

### Problem 1: character-based truncation wastes model capacity

Posts are truncated to 512 characters before embedding:

```python
text = post.get('text', '')[:512]
```

`all-MiniLM-L6-v2` has a 256-token limit. Average English text is ~4 chars/token, so
512 chars ≈ 128 tokens — roughly half the model's capacity is wasted. The truncation
should be token-aware (using the model's own tokenizer) to use the full 256 tokens.

### Problem 2: title is not included in the embedding

The post body alone may not reflect what the thread is about. Prepending the topic title
gives the model essential context, especially for short or ambiguous posts.

### Proposed fix: title-prefixed, token-aware truncation

```python
# Use the model's tokenizer to fill the full context window
text = f"{post['topic_title']}\n\n{post['text']}"
# sentence-transformers truncates to the model's max_seq_length automatically
# when you pass truncate=True (the default), so no manual slicing needed
```

This is a low-effort, high-impact change: no architecture changes, no new dependencies,
no index format changes. Requires a full rebuild of `index.faiss` and `posts.pkl`.

To implement this, ask: **"fix the embedding truncation to be token-aware and include the topic title"**.

### Alternative: switch to a longer-context embedding model

If post bodies are long and contain important information beyond the first 256 tokens,
a model with a larger context window would help more than better truncation:

| Model | Dimensions | Max tokens | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` (current) | 384 | 256 | General similarity, not QA-tuned |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | 512 | Same size, trained on QA pairs |
| `flax-sentence-embeddings/stackoverflow_mpnet-base` | 768 | 512 | StackOverflow domain, larger index |
| `nomic-embed-text` | 768 | 8192 | Long context, can embed entire posts |

Switching models requires rebuilding the FAISS index (dimensions may change) and
updating the model name in both `build_embeddings.py` and `mcp_server.py`.

### Alternative: chunk + store separately

Instead of embedding one vector per post, split each post into overlapping chunks and
store each chunk as its own FAISS entry. Search returns the most relevant chunk;
deduplication groups multiple chunks from the same post. Better recall for long posts,
but increases index size proportionally to average post length.

---

## Alternative approach: incremental FAISS embeddings (not yet implemented)

The current `build_embeddings.py` uses `IndexFlatIP` which requires a full
rebuild every time. FAISS supports incremental adds via `IndexIDMap`.

To implement this, ask: **"implement incremental FAISS embedding updates"**.

### What to build

#### `build_embeddings.py` changes

Switch from `IndexFlatIP` to `IndexIDMap`:

```python
flat = faiss.IndexFlatIP(dim)
index = faiss.IndexIDMap(flat)

# Add vectors with explicit integer IDs
# Use a stable integer ID derived from post_id (e.g. hash or sequential)
ids = np.array([int(post_id) for post_id in post_ids], dtype=np.int64)
index.add_with_ids(embeddings, ids)
```

Save a mapping of `post_id → position` so the updater knows which IDs exist.

#### New `update_embeddings.py`

- Load existing `index.faiss` and `posts.pkl`
- Build a set of already-indexed post IDs from `posts.pkl`
- Scan all `data/posts/*.json` files for posts not yet in the set
- Embed only the new posts (MPS GPU)
- Call `index.add_with_ids(new_embeddings, new_ids)`
- Append new post metadata to `posts.pkl`
- Save updated index and metadata

#### `forum_crawler.py` changes (optional)

After each crawl run, call `update_embeddings.py` as a subprocess, or import
and call its update function directly. This keeps the embedding index in sync
without a full rebuild.

### Tradeoffs

- `IndexIDMap` search is identical speed to `IndexFlatIP` — no performance cost
- Deleted/updated posts are not removed (FAISS flat index has no delete). For
  this use case (append-only forum posts) that's fine.
- The metadata `posts.pkl` grows unboundedly — could switch to a SQLite table
  for metadata if that becomes a concern
