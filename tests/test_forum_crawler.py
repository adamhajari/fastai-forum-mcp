"""Tests for the crawler's incremental-update logic and HTTP retry behaviour.

No real network calls are made: iter_latest_pages and the requests session are
both stubbed out, and every sleep is recorded instead of performed.
"""

import pytest
import requests

import forum_crawler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def no_sleep(monkeypatch):
    """Record sleep durations instead of actually waiting."""
    slept = []
    monkeypatch.setattr(forum_crawler.time, "sleep", slept.append)
    return slept


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


@pytest.fixture
def fake_session(monkeypatch):
    """Install a session whose get() returns queued responses (or raises)."""
    def install(*responses):
        queue = list(responses)
        calls = []

        def fake_get(url, timeout=None):
            calls.append(url)
            item = queue.pop(0) if queue else FakeResponse(500)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(forum_crawler.session, "get", fake_get)
        return calls

    return install


def topic(tid, bumped_at="2025-01-01T00:00:00.000Z", posts_count=5, **extra):
    t = {"id": tid, "title": f"Topic {tid}", "bumped_at": bumped_at, "posts_count": posts_count}
    t.update(extra)
    return t


@pytest.fixture
def stub_latest(monkeypatch):
    """Replace iter_latest_pages with a fixed sequence of topic summaries."""
    def install(topics):
        yielded = []

        def fake_iter():
            for t in topics:
                yielded.append(t["id"])
                yield t

        monkeypatch.setattr(forum_crawler, "iter_latest_pages", fake_iter)
        return yielded

    return install


# ---------------------------------------------------------------------------
# get_topics_to_process — first run
# ---------------------------------------------------------------------------

def test_first_run_takes_every_topic(stub_latest):
    stub_latest([topic(1), topic(2), topic(3)])

    result = forum_crawler.get_topics_to_process({"last_run": None, "topics": {}})

    assert list(result) == ["1", "2", "3"]


def test_first_run_ignores_bumped_at_entirely(stub_latest):
    # Even ancient topics are fetched when there is no last_run to compare to.
    stub_latest([topic(1, bumped_at="2013-01-01T00:00:00.000Z")])

    assert list(forum_crawler.get_topics_to_process({"last_run": None, "topics": {}})) == ["1"]


def test_first_run_skips_topics_already_fully_fetched(stub_latest):
    """A previous interrupted first run left some topics complete; skip those."""
    metadata = {"last_run": None, "topics": {"1": {"posts_count": 5}, "2": {"posts_count": 5}}}
    stub_latest([topic(1, posts_count=5), topic(2, posts_count=9), topic(3, posts_count=5)])

    result = forum_crawler.get_topics_to_process(metadata)

    # 1 is unchanged and skipped; 2 gained posts; 3 is new.
    assert list(result) == ["2", "3"]


def test_first_run_keys_are_strings_and_values_are_the_summaries(stub_latest):
    stub_latest([topic(42)])

    result = forum_crawler.get_topics_to_process({"last_run": None, "topics": {}})

    assert result["42"]["title"] == "Topic 42"


# ---------------------------------------------------------------------------
# get_topics_to_process — incremental run
# ---------------------------------------------------------------------------

def test_incremental_run_stops_at_first_topic_older_than_last_run(stub_latest):
    yielded = stub_latest([
        topic(1, bumped_at="2025-06-03T00:00:00.000Z"),
        topic(2, bumped_at="2025-06-02T00:00:00.000Z"),
        topic(3, bumped_at="2025-05-01T00:00:00.000Z"),  # older → stop here
        topic(4, bumped_at="2025-04-01T00:00:00.000Z"),
    ])

    result = forum_crawler.get_topics_to_process(
        {"last_run": "2025-06-01T00:00:00.000Z", "topics": {}}
    )

    assert list(result) == ["1", "2"]
    # /latest is sorted newest-first, so pagination must stop rather than
    # walking the whole forum.
    assert 4 not in yielded


def test_incremental_run_treats_equal_timestamp_as_already_seen(stub_latest):
    stub_latest([topic(1, bumped_at="2025-06-01T00:00:00.000Z")])

    result = forum_crawler.get_topics_to_process(
        {"last_run": "2025-06-01T00:00:00.000Z", "topics": {}}
    )

    assert result == {}


def test_incremental_run_falls_back_to_last_posted_at(stub_latest):
    t = topic(1, last_posted_at="2025-06-05T00:00:00.000Z")
    t["bumped_at"] = None
    stub_latest([t])

    result = forum_crawler.get_topics_to_process(
        {"last_run": "2025-06-01T00:00:00.000Z", "topics": {}}
    )

    assert list(result) == ["1"]


def test_incremental_run_stops_on_topic_with_no_timestamps(stub_latest):
    t = topic(1)
    del t["bumped_at"]
    stub_latest([t])

    result = forum_crawler.get_topics_to_process(
        {"last_run": "2025-06-01T00:00:00.000Z", "topics": {}}
    )

    # "" <= last_run, so the topic is treated as old and the scan stops.
    assert result == {}


def test_incremental_run_refetches_known_topics_that_were_bumped(stub_latest):
    """Unlike the first run, an incremental run does not skip on posts_count."""
    metadata = {
        "last_run": "2025-06-01T00:00:00.000Z",
        "topics": {"1": {"posts_count": 5}},
    }
    stub_latest([topic(1, bumped_at="2025-06-02T00:00:00.000Z", posts_count=5)])

    assert list(forum_crawler.get_topics_to_process(metadata)) == ["1"]


# ---------------------------------------------------------------------------
# fetch_json
# ---------------------------------------------------------------------------

def test_fetch_json_returns_payload_on_200(no_sleep, fake_session):
    calls = fake_session(FakeResponse(200, {"topic_list": {"topics": []}}))

    assert forum_crawler.fetch_json("https://example.test/latest.json") == {
        "topic_list": {"topics": []}
    }
    assert len(calls) == 1


def test_fetch_json_rate_limits_before_every_request(no_sleep, fake_session):
    fake_session(FakeResponse(200, {}))

    forum_crawler.fetch_json("https://example.test/x.json")

    assert no_sleep[0] == forum_crawler.RATE_LIMIT_DELAY


def test_fetch_json_returns_none_on_404_without_retrying(no_sleep, fake_session):
    calls = fake_session(FakeResponse(404))

    assert forum_crawler.fetch_json("https://example.test/missing.json") is None
    assert len(calls) == 1


def test_fetch_json_waits_retry_after_on_429_then_succeeds(no_sleep, fake_session):
    calls = fake_session(
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(200, {"ok": True}),
    )

    assert forum_crawler.fetch_json("https://example.test/x.json") == {"ok": True}
    assert len(calls) == 2
    assert 7 in no_sleep


def test_fetch_json_defaults_to_60s_when_retry_after_header_missing(no_sleep, fake_session):
    fake_session(FakeResponse(429), FakeResponse(200, {}))

    forum_crawler.fetch_json("https://example.test/x.json")

    assert 60 in no_sleep


def test_fetch_json_gives_up_after_max_retries_of_429(no_sleep, fake_session):
    calls = fake_session(*[FakeResponse(429, headers={"Retry-After": "1"})] * 5)

    assert forum_crawler.fetch_json("https://example.test/x.json") is None
    assert len(calls) == forum_crawler.MAX_RETRIES


def test_fetch_json_retries_server_errors_with_increasing_backoff(no_sleep, fake_session):
    calls = fake_session(*[FakeResponse(500)] * 5)

    assert forum_crawler.fetch_json("https://example.test/x.json") is None
    assert len(calls) == forum_crawler.MAX_RETRIES
    # Backoff after the first two failures only — no wait after the last try.
    assert [s for s in no_sleep if s in (5, 10)] == [5, 10]


def test_fetch_json_recovers_after_a_transient_server_error(no_sleep, fake_session):
    calls = fake_session(FakeResponse(503), FakeResponse(200, {"ok": 1}))

    assert forum_crawler.fetch_json("https://example.test/x.json") == {"ok": 1}
    assert len(calls) == 2


def test_fetch_json_retries_connection_errors(no_sleep, fake_session):
    calls = fake_session(
        requests.ConnectionError("boom"),
        requests.Timeout("slow"),
        FakeResponse(200, {"ok": 2}),
    )

    assert forum_crawler.fetch_json("https://example.test/x.json") == {"ok": 2}
    assert len(calls) == 3


def test_fetch_json_returns_none_when_every_attempt_raises(no_sleep, fake_session):
    calls = fake_session(*[requests.ConnectionError("boom")] * 5)

    assert forum_crawler.fetch_json("https://example.test/x.json") is None
    assert len(calls) == forum_crawler.MAX_RETRIES
