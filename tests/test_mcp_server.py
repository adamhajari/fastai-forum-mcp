"""Tests for the pure scoring / formatting / fusion logic in mcp_server.py.

These import mcp_server directly, which is only possible because index loading
happens in load_indexes() rather than at import time.
"""

import math

import pytest
from conftest import make_post

import mcp_server

# ---------------------------------------------------------------------------
# _recency_score
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "created_at, expected",
    [
        ("2025-01-01T00:00:00.000Z", 1.0),   # top of the scale
        ("2018-12-31T23:59:59.000Z", 0.0),   # bottom of the scale
        ("2021-06-01T00:00:00.000Z", 3 / 7),  # linear in between
        ("2030-01-01T00:00:00.000Z", 1.0),   # clamped above
        ("2001-01-01T00:00:00.000Z", 0.0),   # clamped below
    ],
)
def test_recency_score_boundaries(created_at, expected):
    assert mcp_server._recency_score(created_at) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["", "not-a-date", "20XX-01-01", None, 20250101])
def test_recency_score_malformed_falls_back_to_midpoint(bad):
    assert mcp_server._recency_score(bad) == 0.5


# ---------------------------------------------------------------------------
# _like_score
# ---------------------------------------------------------------------------

def test_like_score_zero_likes_is_zero():
    assert mcp_server._like_score(0) == 0.0


def test_like_score_ten_likes_is_one():
    assert mcp_server._like_score(10) == pytest.approx(1.0)


def test_like_score_is_monotonic_and_logarithmic():
    scores = [mcp_server._like_score(n) for n in (0, 1, 5, 10, 100)]
    assert scores == sorted(scores)
    # 100 likes scores well above 1.0 but only ~2x the 10-like score
    assert 1.0 < scores[-1] < 2.5
    assert mcp_server._like_score(100) == pytest.approx(math.log(101) / math.log(11))


# ---------------------------------------------------------------------------
# _meta_boost
# ---------------------------------------------------------------------------

def test_meta_boost_combines_recency_and_likes():
    post = make_post(created_at="2025-03-01T00:00:00.000Z", like_count=10)
    assert mcp_server._meta_boost(post) == pytest.approx(0.25 + 0.15)


def test_meta_boost_floor_is_zero_for_old_unliked_post():
    post = make_post(created_at="2015-01-01T00:00:00.000Z", like_count=0)
    assert mcp_server._meta_boost(post) == pytest.approx(0.0)


def test_meta_boost_recent_beats_old_at_equal_likes():
    recent = make_post(created_at="2025-01-01T00:00:00.000Z", like_count=2)
    old = make_post(created_at="2019-01-01T00:00:00.000Z", like_count=2)
    assert mcp_server._meta_boost(recent) > mcp_server._meta_boost(old)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits_on_punctuation():
    assert mcp_server._tokenize("Fine Tuning a Model!") == ["fine", "tuning", "a", "model"]


def test_tokenize_keeps_underscores_and_dots_for_identifiers():
    assert mcp_server._tokenize("learn.fine_tune(3)") == ["learn.fine_tune", "3"]


def test_tokenize_empty_string():
    assert mcp_server._tokenize("") == []


def test_tokenize_drops_non_ascii_and_symbols():
    assert mcp_server._tokenize("CUDA — out of memory ✅") == ["cuda", "out", "of", "memory"]


# ---------------------------------------------------------------------------
# _format_results
# ---------------------------------------------------------------------------

def test_format_results_empty():
    assert mcp_server._format_results([], "anything") == "No relevant posts found."


def test_format_results_builds_forum_url():
    post = make_post(topic_slug="cuda-oom", topic_id="999", post_number=7)
    out = mcp_server._format_results([post], "cuda")
    assert "https://forums.fast.ai/t/cuda-oom/999/7" in out


def test_format_results_includes_metadata_and_header():
    post = make_post(username="rachel", like_count=12, topic_tags=["part2"])
    out = mcp_server._format_results([post], "fine tune")
    assert "Found 1 results for: 'fine tune'" in out
    assert "Date: 2024-06-01" in out
    assert "Likes: 12" in out
    assert "By: rachel" in out
    assert "Tags: part2" in out


def test_format_results_em_dash_for_missing_tags_and_unknown_date():
    post = make_post(topic_tags=[], created_at="")
    out = mcp_server._format_results([post], "q")
    assert "Tags: —" in out
    assert "Date: unknown" in out


def test_format_results_truncates_at_2000_chars():
    post = make_post(text="x" * 2500)
    out = mcp_server._format_results([post], "q")
    assert "x" * 2000 + " …[truncated]" in out
    assert "x" * 2001 not in out


def test_format_results_does_not_truncate_at_exactly_2000_chars():
    post = make_post(text="y" * 2000)
    out = mcp_server._format_results([post], "q")
    assert "[truncated]" not in out


def test_format_results_numbers_each_result():
    posts = [make_post(str(i), topic_title=f"Topic {i}") for i in range(1, 4)]
    out = mcp_server._format_results(posts, "q")
    assert "[1] Topic 1" in out
    assert "[2] Topic 2" in out
    assert "[3] Topic 3" in out
    assert "Found 3 results" in out


# ---------------------------------------------------------------------------
# _search_hybrid — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def neutral_post(post_id):
    """A post whose metadata boost is identical to every other neutral post,
    so fusion ranking alone determines the order."""
    return make_post(post_id, created_at="2022-01-01T00:00:00.000Z", like_count=1)


@pytest.fixture
def stub_searches(monkeypatch):
    """Install fake BM25/semantic result lists and record the fetch sizes."""
    calls = {}

    def install(bm25_results, sem_results):
        def fake_bm25(query, n):
            calls["bm25_n"] = n
            return bm25_results

        def fake_semantic(query, n):
            calls["semantic_n"] = n
            return sem_results

        monkeypatch.setattr(mcp_server, "_search_bm25", fake_bm25)
        monkeypatch.setattr(mcp_server, "_search_semantic", fake_semantic)
        return calls

    return install


def test_hybrid_fuses_rankings_from_both_retrievers(stub_searches):
    a, b, c = (neutral_post(x) for x in "ABC")
    stub_searches([a, b, c], [b, c, a])

    ranked = mcp_server._search_hybrid("q", 3)

    # RRF (k=60): B = 1/62+1/61, A = 1/61+1/63, C = 1/63+1/62  → B > A > C
    assert [p["post_id"] for p in ranked] == ["B", "A", "C"]


def test_hybrid_prefers_post_ranked_by_both_over_single_list_leader(stub_searches):
    a, b, c = (neutral_post(x) for x in "ABC")
    # A is #2 in both lists; B and C each top exactly one list.
    stub_searches([b, a], [c, a])

    ranked = mcp_server._search_hybrid("q", 3)

    assert ranked[0]["post_id"] == "A"
    assert {p["post_id"] for p in ranked} == {"A", "B", "C"}


def test_hybrid_deduplicates_posts_present_in_both_lists(stub_searches):
    a = neutral_post("A")
    stub_searches([a], [a])

    ranked = mcp_server._search_hybrid("q", 10)

    assert len(ranked) == 1


def test_hybrid_breaks_rrf_ties_with_metadata_boost(stub_searches):
    # Both are rank 1 in exactly one list, so their RRF scores are identical;
    # only the metadata boost separates them.
    popular = make_post("popular", created_at="2025-01-01T00:00:00.000Z", like_count=10)
    obscure = make_post("obscure", created_at="2018-01-01T00:00:00.000Z", like_count=0)
    stub_searches([obscure], [popular])

    ranked = mcp_server._search_hybrid("q", 2)

    assert [p["post_id"] for p in ranked] == ["popular", "obscure"]


def test_hybrid_truncates_to_n_results(stub_searches):
    posts = [neutral_post(str(i)) for i in range(10)]
    stub_searches(posts, list(reversed(posts)))

    assert len(mcp_server._search_hybrid("q", 4)) == 4


def test_hybrid_over_fetches_candidates_from_each_retriever(stub_searches):
    calls = stub_searches([], [])

    mcp_server._search_hybrid("q", 6)

    assert calls["bm25_n"] == 30
    assert calls["semantic_n"] == 30


def test_hybrid_with_no_candidates_returns_empty(stub_searches):
    stub_searches([], [])
    assert mcp_server._search_hybrid("q", 5) == []
