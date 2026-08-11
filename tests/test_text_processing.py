"""Tests for the HTML-stripping and tokenizing helpers used by the indexers.

strip_html is duplicated in build_index.py and build_embeddings.py; the two
copies must behave identically or the BM25 and semantic indexes would be built
over different text, so every case runs against both.
"""

import pytest

import build_embeddings
import build_index
from build_index import tokenize

STRIP_HTML_IMPLS = [build_index.strip_html, build_embeddings.strip_html]


@pytest.fixture(params=STRIP_HTML_IMPLS, ids=["build_index", "build_embeddings"])
def strip_html(request):
    return request.param


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------

def test_strips_tags(strip_html):
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strips_tags_with_attributes(strip_html):
    html = '<a href="https://forums.fast.ai/t/1" class="link">click here</a>'
    assert strip_html(html) == "click here"


def test_strips_multiline_code_blocks(strip_html):
    html = "<pre><code>import fastai\nlearn.fit()\n</code></pre>"
    assert strip_html(html) == "import fastai learn.fit()"


@pytest.mark.parametrize(
    "entity, expected",
    [
        ("a&nbsp;b", "a b"),
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("&lt;div&gt;", "<div>"),
        ("&quot;quoted&quot;", '"quoted"'),
        ("it&#39;s", "it's"),
    ],
)
def test_decodes_html_entities(strip_html, entity, expected):
    assert strip_html(entity) == expected


def test_decodes_double_escaped_ampersand(strip_html):
    # &amp; is replaced before &lt;, so &amp;lt; decodes to a literal "<"
    assert strip_html("&amp;lt;") == "<"


def test_collapses_all_whitespace_runs(strip_html):
    assert strip_html("  a \n\n\t b   c  ") == "a b c"


def test_tag_removal_inserts_a_separator(strip_html):
    # Tags become spaces, so adjacent words never get glued together
    assert strip_html("<p>one</p><p>two</p>") == "one two"


@pytest.mark.parametrize("empty", ["", None, "   ", "<p></p>", "<br/>"])
def test_empty_input_returns_empty_string(strip_html, empty):
    assert strip_html(empty) == ""


def test_plain_text_passes_through_unchanged(strip_html):
    assert strip_html("no markup here") == "no markup here"


def test_strip_html_is_idempotent(strip_html):
    once = strip_html("<p>Tom &amp; Jerry  &nbsp; ride</p>")
    assert strip_html(once) == once


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lowercases():
    assert tokenize("Fastai LEARN") == ["fastai", "learn"]


def test_tokenize_preserves_dotted_and_underscored_identifiers():
    assert tokenize("learn.fine_tune(epochs=3)") == ["learn.fine_tune", "epochs", "3"]


def test_tokenize_drops_punctuation_and_symbols():
    assert tokenize("what?! is  a  *tensor*") == ["what", "is", "a", "tensor"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_tokenize_whitespace_only():
    assert tokenize("   \n\t ") == []


def test_tokenize_matches_the_mcp_server_query_tokenizer():
    # Query tokens must be produced the same way as index tokens or BM25
    # lookups silently miss.
    import mcp_server

    text = "RuntimeError: CUDA out of memory in learn.fit_one_cycle(5)"
    assert tokenize(text) == mcp_server._tokenize(text)
