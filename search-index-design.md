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
