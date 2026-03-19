# Project Notes

## Hugging Face dataset
Pre-crawled forum data is hosted at `adamhajari/fastai-forum` on Hugging Face.
To update it after re-crawling, use `upload_large_folder` (not `upload_folder`) — the dataset
is too large for a single commit and will hit HF's 504 timeout otherwise.

## Post file layout
Posts are stored in `data/posts/{topic_id // 10000}/{topic_id}.json` — bucketed into
subdirectories to stay under Hugging Face's 10,000 files-per-directory limit. Any code
that reads post files must use `**/*.json` glob patterns, not `*.json`.
