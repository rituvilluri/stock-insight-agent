"""
Tests for Node 5: News Retriever

All external calls (NewsAPI, Google RSS) are mocked so tests run without
network access or API keys.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.news_retriever import (
    _build_query,
    _fetch_google_rss,
    _fetch_newsapi,
    _parse_newsapi_articles,
    retrieve_news,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_state(**overrides):
    state = {
        "user_message": "What happened to NVDA last month?",
        "user_config": {},
        "ticker": "NVDA",
        "company_name": "NVIDIA",
        "start_date": "2024-06-01",
        "end_date": "2024-06-30",
        "include_current_snapshot": False,
    }
    state.update(overrides)
    return state


_FAKE_NEWSAPI_RESPONSE = {
    "status": "ok",
    "articles": [
        {
            "title": "NVIDIA hits record high",
            "source": {"name": "Reuters"},
            "publishedAt": "2024-06-15T12:00:00Z",
            "url": "https://reuters.com/nvda-record",
            "description": "NVIDIA stock reached a new all-time high.",
            "content": None,
        },
        {
            "title": "AI chip demand surges",
            "source": {"name": "Bloomberg"},
            "publishedAt": "2024-06-20T08:00:00Z",
            "url": "https://bloomberg.com/ai-chips",
            "description": "Demand for AI chips continues to grow.",
            "content": None,
        },
    ],
}

_FAKE_RSS_FEED = {
    "bozo": False,
    "entries": [
        {
            "title": "NVIDIA earnings beat expectations",
            "link": "https://example.com/nvda-earnings",
            "summary": "NVIDIA reported strong Q2 results.",
            "source": {"title": "CNBC"},
            "published_parsed": (2024, 6, 18, 10, 0, 0, 0, 0, 0),
        },
    ],
}


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

def test_build_query_ticker_and_company():
    assert _build_query("NVDA", "NVIDIA") == 'NVDA OR "NVIDIA"'


def test_build_query_same_ticker_and_company():
    # Should not duplicate when they match
    assert _build_query("NVDA", "NVDA") == "NVDA"


def test_build_query_ticker_only():
    assert _build_query("AAPL", "") == "AAPL"


def test_parse_newsapi_articles():
    articles = _parse_newsapi_articles(_FAKE_NEWSAPI_RESPONSE["articles"])
    assert len(articles) == 2
    assert articles[0]["title"] == "NVIDIA hits record high"
    assert articles[0]["source_name"] == "Reuters"
    assert articles[0]["published_date"] == "2024-06-15"
    assert articles[0]["url"] == "https://reuters.com/nvda-record"
    assert "NVIDIA stock" in articles[0]["snippet"]


# ---------------------------------------------------------------------------
# Unit tests: NewsAPI layer
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever.NewsApiClient")
def test_fetch_newsapi_success(mock_client_cls):
    mock_client = MagicMock()
    mock_client.get_everything.return_value = _FAKE_NEWSAPI_RESPONSE
    mock_client_cls.return_value = mock_client

    result = _fetch_newsapi("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")

    assert result is not None
    assert len(result) == 2
    assert result[0]["title"] == "NVIDIA hits record high"


@patch("agent.graph.nodes.news_retriever.NewsApiClient")
def test_fetch_newsapi_empty_returns_none(mock_client_cls):
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {"status": "ok", "articles": []}
    mock_client_cls.return_value = mock_client

    result = _fetch_newsapi("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


@patch("agent.graph.nodes.news_retriever.NewsApiClient")
def test_fetch_newsapi_error_status_returns_none(mock_client_cls):
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {"status": "error", "code": "rateLimited"}
    mock_client_cls.return_value = mock_client

    result = _fetch_newsapi("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


@patch("agent.graph.nodes.news_retriever.NewsApiClient")
def test_fetch_newsapi_exception_returns_none(mock_client_cls):
    mock_client_cls.side_effect = Exception("connection refused")
    result = _fetch_newsapi("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


# ---------------------------------------------------------------------------
# Unit tests: Google RSS layer
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever.feedparser.parse")
def test_fetch_google_rss_success(mock_parse):
    entry = MagicMock()
    entry.published_parsed = (2024, 6, 18, 10, 0, 0, 0, 0, 0)
    entry.get = lambda key, default=None: {
        "title": "NVIDIA earnings beat expectations",
        "link": "https://example.com/nvda-earnings",
        "summary": "NVIDIA reported strong Q2 results.",
        "source": {"title": "CNBC"},
        "published_parsed": entry.published_parsed,
    }.get(key, default)

    mock_parse.return_value = MagicMock()
    mock_parse.return_value.bozo = False
    mock_parse.return_value.entries = [entry]

    result = _fetch_google_rss("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")

    assert result is not None
    assert result[0]["title"] == "NVIDIA earnings beat expectations"
    assert result[0]["url"] == "https://example.com/nvda-earnings"
    assert result[0]["published_date"] == "2024-06-18"


@patch("agent.graph.nodes.news_retriever.feedparser.parse")
def test_fetch_google_rss_filters_out_of_range(mock_parse):
    out_of_range_entry = MagicMock()
    out_of_range_entry.title = "Old news"
    out_of_range_entry.link = "https://example.com/old"
    out_of_range_entry.summary = "This is old."
    out_of_range_entry.source = {"title": "Reuters"}
    out_of_range_entry.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)  # outside range

    mock_parse.return_value = MagicMock()
    mock_parse.return_value.bozo = False
    mock_parse.return_value.entries = [out_of_range_entry]

    result = _fetch_google_rss("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert result is None


@patch("agent.graph.nodes.news_retriever.feedparser.parse")
def test_fetch_google_rss_exception_returns_none(mock_parse):
    mock_parse.side_effect = Exception("network error")
    result = _fetch_google_rss("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert result is None


# ---------------------------------------------------------------------------
# Integration-style tests: retrieve_news node
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever._get_newsapi_key", return_value="fake-key")
@patch("agent.graph.nodes.news_retriever._fetch_newsapi")
def test_retrieve_news_uses_newsapi_when_key_available(mock_newsapi, mock_key):
    mock_newsapi.return_value = [{"title": "NVDA up", "source_name": "Reuters",
                                   "published_date": "2024-06-15", "url": "https://r.com", "snippet": ""}]
    result = retrieve_news(_base_state())

    assert result["news_source_used"] == "newsapi"
    assert len(result["news_articles"]) == 1
    assert result["news_error"] is None


@patch("agent.graph.nodes.news_retriever._get_newsapi_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._fetch_google_rss")
def test_retrieve_news_falls_back_to_rss_when_no_key(mock_rss, mock_key):
    mock_rss.return_value = [{"title": "NVDA earnings", "source_name": "CNBC",
                               "published_date": "2024-06-18", "url": "https://cnbc.com", "snippet": ""}]
    result = retrieve_news(_base_state())

    assert result["news_source_used"] == "google_rss"
    assert len(result["news_articles"]) == 1
    assert result["news_error"] is None


@patch("agent.graph.nodes.news_retriever._get_newsapi_key", return_value="fake-key")
@patch("agent.graph.nodes.news_retriever._fetch_newsapi", return_value=None)
@patch("agent.graph.nodes.news_retriever._fetch_google_rss")
def test_retrieve_news_falls_back_to_rss_when_newsapi_returns_none(mock_rss, mock_newsapi, mock_key):
    mock_rss.return_value = [{"title": "NVDA news", "source_name": "BBC",
                               "published_date": "2024-06-10", "url": "https://bbc.com", "snippet": ""}]
    result = retrieve_news(_base_state())

    assert result["news_source_used"] == "google_rss"
    assert result["news_articles"] is not None


@patch("agent.graph.nodes.news_retriever._get_newsapi_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._fetch_google_rss", return_value=None)
def test_retrieve_news_both_fail_returns_none_articles(mock_rss, mock_key):
    result = retrieve_news(_base_state())

    assert result["news_articles"] is None
    assert result["news_source_used"] == "none"
    assert result["news_error"] is None  # not an error, just no results


@patch("agent.graph.nodes.news_retriever._get_newsapi_key", return_value="fake-key")
@patch("agent.graph.nodes.news_retriever._fetch_newsapi")
def test_retrieve_news_current_snapshot_appends_recent(mock_newsapi, mock_key):
    historical = [{"title": "Old news", "source_name": "Reuters",
                   "published_date": "2024-06-15", "url": "https://r.com/old", "snippet": ""}]
    current = [{"title": "Breaking news", "source_name": "Bloomberg",
                "published_date": "2024-07-20", "url": "https://b.com/new", "snippet": ""}]

    mock_newsapi.side_effect = [historical, current]
    result = retrieve_news(_base_state(include_current_snapshot=True))

    assert len(result["news_articles"]) == 2
    urls = [a["url"] for a in result["news_articles"]]
    assert "https://r.com/old" in urls
    assert "https://b.com/new" in urls


@patch("agent.graph.nodes.news_retriever._get_newsapi_key", return_value="fake-key")
@patch("agent.graph.nodes.news_retriever._fetch_newsapi")
def test_retrieve_news_deduplicates_on_current_snapshot(mock_newsapi, mock_key):
    """Articles appearing in both historical and snapshot fetches should not be duplicated."""
    article = {"title": "Same article", "source_name": "Reuters",
               "published_date": "2024-06-15", "url": "https://r.com/same", "snippet": ""}
    mock_newsapi.side_effect = [[article], [article]]

    result = retrieve_news(_base_state(include_current_snapshot=True))
    assert len(result["news_articles"]) == 1


@patch("agent.graph.nodes.news_retriever._get_newsapi_key", side_effect=Exception("unexpected"))
def test_retrieve_news_unexpected_exception_writes_error(mock_key):
    result = retrieve_news(_base_state())

    assert result["news_articles"] is None
    assert result["news_source_used"] == "none"
    assert result["news_error"] is not None
    assert "unexpected" in result["news_error"]


@patch("agent.graph.nodes.news_retriever.NewsApiClient")
def test_fetch_newsapi_parameter_invalid_returns_none(mock_client_cls):
    """NewsAPI parameterInvalid error (free tier date range limit) returns None."""
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {
        "status": "error",
        "code": "parameterInvalid",
        "message": "You are not allowed to use the from parameter for your plan level.",
    }
    mock_client_cls.return_value = mock_client

    result = _fetch_newsapi("NVDA", "NVIDIA", "2023-01-01", "2023-06-30", "fake-key")
    assert result is None


@patch("agent.graph.nodes.news_retriever.NewsApiClient")
def test_fetch_newsapi_upgrade_message_returns_none(mock_client_cls):
    """NewsAPI 'upgrade' message (paid tier required) returns None."""
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {
        "status": "error",
        "code": "rateLimited",
        "message": "Please upgrade your plan to access this feature.",
    }
    mock_client_cls.return_value = mock_client

    result = _fetch_newsapi("NVDA", "NVIDIA", "2023-01-01", "2023-06-30", "fake-key")
    assert result is None
