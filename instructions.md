## Context Aware Instructions

I'm working through the fast.ai 2022 course (https://course.fast.ai/). The notebooks and lectures are from 2022 and some libraries have changed significantly since then. The forum (https://forums.fast.ai/) has suggestions from people who have recently gone through the same materials, but is hard to search manually.

The goal is to give Claude access to the full forum corpus so it can answer questions about the course and notebooks using up-to-date community knowledge.

## Instructions for Claude

When the user asks any question related to the fast.ai course, its notebooks, or errors encountered while working through it, **always search the forum first** using the `search_forum` tool before answering. Run multiple searches with different phrasings if the first doesn't surface a clear answer — the forum contains community solutions to many library compatibility issues that postdate the 2022 course materials. Prefer recent posts (2023+) and highly-liked posts when synthesizing answers.

## Current Status (as of 2026-03-12)

Everything below has been implemented. Do not re-implement from scratch — read the existing files first.

### What's been built

- `forum/forum_crawler.py` — incremental crawler using the Discourse public JSON API. Stores one JSON file per topic in `forum/data/posts/`, with a metadata index at `forum/data/metadata.json`. Checkpoints every 25 topics. Handles rate limiting (1 req/sec), 429 backoff, and resuming interrupted runs.
- `forum/build_index.py` — builds a BM25 search index from all post files. Output: `forum/data/search_index.pkl`. Must be re-run after crawling to include new posts (BM25 cannot be updated incrementally).
- `forum/build_embeddings.py` — generates semantic vector embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim). Uses MPS (Apple Silicon GPU) for fast encoding. Output: `forum/data/embeddings/index.faiss` + `forum/data/embeddings/posts.pkl`. Must be re-run after crawling.
- `forum/mcp_server.py` — MCP server exposing `search_forum(query, n_results, mode)`. Supports three modes: `bm25` (keyword), `semantic` (vector similarity), and `hybrid` (Reciprocal Rank Fusion of both — default). Falls back to BM25 if embeddings index not built yet.
- `.mcp.json` (project root) — tells Claude Code to start the MCP server automatically.
- `forum/search-index-design.md` — documents the current BM25 approach and a fully-specced SQLite FTS5 alternative that supports incremental updates. To implement it, ask: "implement the alternative approach to building the index using SQLite".

### Current state of the data

- The crawler completed its first full run (~21,000 topics)
- Run `python forum/build_index.py` after each crawler run to refresh the search index
- Restart Claude Code after rebuilding the index so the MCP server picks up the new file

### Key decisions made

- One JSON file per topic (not a single monolithic file) — easier to inspect and update incrementally
- BM25 pickle chosen over SQLite FTS5 for simplicity; SQLite alternative is documented in `forum/search-index-design.md` if incremental updates become important
- Rate limit set to 1 req/sec (60 req/min) to stay within Discourse's default limit

## Crawler Instructions (original — already implemented)

Create a crawler to collect all posts from https://forums.fast.ai/

Persists these posts to local disc and store metadata about each post and each topic such as when it was created, when it was last updated, and category tags. This crawler will be run on a regular basis and everytime it runs it should know which topics are new and posts are new, which topics have been updated and which posts have new replies. It should use this information along with the metadata it has stored to only look at new topics and topics that have been updated since the last run and only collect new posts or new replies. 

I'm not opinionated about how this data is stored. The forum corupus is small enough that it should easily be able to fit into memory and it will only be updated by this crawler so we don't need to worry about concurrency. A json file might be fine, but I'm open to other suggestions.

I'm also not opinionated about how the crawler works. I'm not aware of a public API for the forum, but you should check. Try a simple solution first. The site may actively be monitoring for and blocking crawlers so be cautious about rate limiting. If a simple solution doesn't work we can try something more complicated like using a library like Selenium.

