# fastai-forum-mcp

An [MCP](https://modelcontextprotocol.io/) server that gives coding agents RAG-style access to the [fast.ai forum](https://forums.fast.ai/). It searches ~200k community posts with hybrid keyword + semantic retrieval, so agents can answer course questions using up-to-date community knowledge instead of only the 2022 course materials.

## Quick start

```bash
git clone https://github.com/adamhajari/fastai-forum-mcp
cd fastai-forum-mcp
uv sync
```

**1. Get forum data** — on first run, the crawler downloads a pre-built snapshot from [Hugging Face](https://huggingface.co/datasets/adamhajari/fastai-forum), then fetches any newer posts incrementally (~minutes on subsequent runs):

```bash
uv run python forum_crawler.py
```

**2. Build search indexes** — required before starting the MCP server. Run both after crawling (or after refreshing data):

```bash
uv run python build_index.py          # BM25 keyword index (~few minutes)
uv run python build_embeddings.py     # FAISS semantic index (~20 minutes)
```

**3. Register the MCP server** with your agent. The repo includes `.mcp.json` for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) when run from this directory. From elsewhere, add it globally:

```bash
claude mcp add fastai-forum -- uv --directory /path/to/fastai-forum-mcp run python mcp_server.py
```

Replace `/path/to/fastai-forum-mcp` with the absolute path to your clone, then restart Claude Code.

## Using the `search_forum` tool

The MCP exposes a single tool, `search_forum(query, n_results=20, mode="hybrid")`, which returns ranked post excerpts with URLs, dates, authors, and tags.

| Mode | Best for |
|------|----------|
| `hybrid` (default) | General questions — combines keyword and semantic ranking via reciprocal rank fusion |
| `semantic` | Conceptual questions where wording may differ from the posts |
| `bm25` | Exact error messages, function names, library versions |

If the semantic index has not been built, `semantic` and `hybrid` fall back to `bm25`.

For agent behavior tips (when to search, how many queries to run), see [instructions.md](instructions.md).

## How it works

```
forums.fast.ai  ──►  forum_crawler.py  ──►  data/posts/*.json
                                              │
                    build_index.py ──────────►│──► data/search_index.pkl (BM25)
                    build_embeddings.py ─────►│──► data/embeddings/ (FAISS)
                                              │
                    mcp_server.py ◄───────────┘
                         │
                    search_forum tool
```

On startup, `mcp_server.py` loads both indexes into memory. Each search strips HTML from stored posts, runs the selected retrieval mode, and re-ranks by relevance plus light recency and like-count boosts. See [search-index-design.md](search-index-design.md) for ranking details and a documented SQLite FTS5 alternative.

## Embedding models

`build_embeddings.py` defaults to `all-MiniLM-L6-v2` (fast, no extra deps). For higher retrieval quality, evals favored `nomic-ai/nomic-embed-text-v1` (requires `einops`, already in dependencies):

```bash
uv run python build_embeddings.py --model nomic-ai/nomic-embed-text-v1 --max-seq-length 512
```

A reliable fallback is `multi-qa-MiniLM-L6-cos-v1`. See `eval/` for comparison scripts.

**Note:** The model used at query time in `mcp_server.py` must match the one used to build the FAISS index. If you build with a different model, update `MODEL_NAME` in `mcp_server.py` accordingly.

## Project layout

```
fastai-forum-mcp/
├── mcp_server.py          # MCP server (search_forum tool)
├── forum_crawler.py       # incremental Discourse crawler
├── build_index.py         # BM25 index builder
├── build_embeddings.py    # FAISS semantic index builder
├── upload_to_hub.py       # publish crawled data to Hugging Face
├── .mcp.json              # Claude Code MCP config
├── instructions.md        # suggested agent instructions
├── CRAWLER.md             # crawler & data format reference
├── search-index-design.md # index design notes
└── data/                  # gitignored; created by setup steps
    ├── metadata.json
    ├── posts/{topic_id}.json
    ├── search_index.pkl
    └── embeddings/
```

## Refreshing data

After new posts are crawled, rebuild both indexes and restart the MCP server so it picks up the new files:

```bash
uv run python forum_crawler.py
uv run python build_index.py
uv run python build_embeddings.py
# restart your agent / MCP client
```

To publish an updated snapshot for others, see [CRAWLER.md](CRAWLER.md#updating-the-hugging-face-dataset).

## Further reading

- [CRAWLER.md](CRAWLER.md) — Discourse API crawler, on-disk data formats, Hugging Face dataset
- [search-index-design.md](search-index-design.md) — BM25 ranking, limitations, SQLite FTS5 alternative
- [instructions.md](instructions.md) — context and prompts for agents using this server
