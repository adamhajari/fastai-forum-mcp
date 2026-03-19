# Project Notes

## Hugging Face dataset
Pre-crawled forum data is hosted at `adamhajari/fastai-forum` on Hugging Face.
To update it after re-crawling, use `upload_large_folder` (not `upload_folder`) — the dataset
is too large for a single commit and will hit HF's 504 timeout otherwise.

## Post file layout
Posts are stored flat in `data/posts/{topic_id}.json`. On Hugging Face they are stored
as a single compressed archive `posts.tar.gz` (~52MB) to avoid the 10k files/dir limit
and keep downloads small. The download step extracts the archive automatically.
