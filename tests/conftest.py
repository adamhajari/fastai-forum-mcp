"""Shared fixtures and helpers for the test suite.

Every test here runs without the `data/` corpus and without network access.
"""

import pytest


def make_post(post_id="1", **overrides):
    """A minimal post dict shaped like the ones build_index.py produces."""
    post = {
        "topic_id": "12345",
        "topic_title": "How do I fine-tune a model?",
        "topic_slug": "how-do-i-fine-tune-a-model",
        "topic_tags": ["part1", "beginner"],
        "post_id": post_id,
        "post_number": 1,
        "username": "jeremy",
        "created_at": "2024-06-01T12:00:00.000Z",
        "like_count": 3,
        "reads": 42,
        "text": "Use learn.fine_tune(3).",
    }
    post.update(overrides)
    return post


@pytest.fixture
def post_factory():
    return make_post
