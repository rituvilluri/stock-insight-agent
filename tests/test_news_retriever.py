"""
Tests for Node 5: News Retriever

All external calls (Finnhub, You.com, Google RSS) are mocked so tests run without
network access or API keys.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.news_retriever import (
    _build_query,
    _fetch_articles,
    _fetch_finnhub,
    _fetch_youcom,
    _fetch_google_rss,
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


def _make_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    return mock


_FAKE_FINNHUB_RESPONSE = [
    {
        "headline": "NVIDIA hits record high",
        "source": "Reuters",
        "datetime": 1718452800,  # 2024-06-15 12:00:00 UTC
        "url": "https://reuters.com/nvda-record",
        "summary": "NVIDIA stock reached a new all-time high.",
    },
    {
        "headline": "AI chip demand surges",
        "source": "Bloomberg",
        "datetime": 1718870400,  # 2024-06-20 08:00:00 UTC
        "url": "https://bloomberg.com/ai-chips",
        "summary": "Demand for AI chips continues to grow.",
    },
]

_FAKE_YOUCOM_RESPONSE = {
    "results": {
        "news": [
            {
                "title": "NVIDIA earnings beat expectations",
                "description": "NVIDIA reported strong Q2 results.",
                "url": "https://cnbc.com/nvda-earnings",
                "page_age": "2024-06-18T10:00:00",
            },
        ],
        "web": [],
    },
    "metadata": {"query": "NVDA stock news", "search_uuid": "abc123", "latency": 0.5},
}


# ---------------------------------------------------------------------------
# Unit tests: _build_query
# ---------------------------------------------------------------------------

def test_build_query_ticker_and_company():
    assert _build_query("NVDA", "NVIDIA") == 'NVDA OR "NVIDIA"'


def test_build_query_same_ticker_and_company():
    assert _build_query("NVDA", "NVDA") == "NVDA"


def test_build_query_ticker_only():
    assert _build_query("AAPL", "") == "AAPL"


# ---------------------------------------------------------------------------
# Unit tests: _fetch_finnhub
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_finnhub_success(mock_get):
    mock_get.return_value = _make_response(_FAKE_FINNHUB_RESPONSE)

    result = _fetch_finnhub("NVDA", "2024-06-01", "2024-06-30", "fake-key")

    assert result is not None
    assert len(result) == 2
    assert result[0]["title"] == "NVIDIA hits record high"
    assert result[0]["source_name"] == "Reuters"
    assert result[0]["published_date"] == "2024-06-15"
    assert result[0]["url"] == "https://reuters.com/nvda-record"
    assert "NVIDIA stock" in result[0]["snippet"]


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_finnhub_empty_returns_none(mock_get):
    mock_get.return_value = _make_response([])

    result = _fetch_finnhub("NVDA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_finnhub_non_200_returns_none(mock_get):
    mock_get.return_value = _make_response({}, status_code=403)

    result = _fetch_finnhub("NVDA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_finnhub_exception_returns_none(mock_get):
    mock_get.side_effect = Exception("connection refused")

    result = _fetch_finnhub("NVDA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


# ---------------------------------------------------------------------------
# Unit tests: _fetch_youcom
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_youcom_success(mock_get):
    mock_get.return_value = _make_response(_FAKE_YOUCOM_RESPONSE)

    result = _fetch_youcom("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")

    assert result is not None
    assert len(result) == 1
    assert result[0]["title"] == "NVIDIA earnings beat expectations"
    assert result[0]["source_name"] == "cnbc.com"
    assert result[0]["published_date"] == "2024-06-18"
    assert result[0]["url"] == "https://cnbc.com/nvda-earnings"
    assert "NVIDIA reported" in result[0]["snippet"]


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_youcom_empty_returns_none(mock_get):
    mock_get.return_value = _make_response({"results": {"news": [], "web": []}, "metadata": {}})

    result = _fetch_youcom("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_youcom_non_200_returns_none(mock_get):
    mock_get.return_value = _make_response({}, status_code=401)

    result = _fetch_youcom("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_youcom_exception_returns_none(mock_get):
    mock_get.side_effect = Exception("timeout")

    result = _fetch_youcom("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")
    assert result is None


# ---------------------------------------------------------------------------
# Unit tests: _fetch_google_rss
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
    assert result[0]["published_date"] == "2024-06-18"


@patch("agent.graph.nodes.news_retriever.feedparser.parse")
def test_fetch_google_rss_filters_out_of_range(mock_parse):
    entry = MagicMock()
    entry.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)  # outside range
    entry.get = lambda key, default=None: {
        "title": "Old news",
        "link": "https://example.com/old",
        "summary": "This is old.",
        "source": {"title": "Reuters"},
        "published_parsed": entry.published_parsed,
    }.get(key, default)

    mock_parse.return_value = MagicMock()
    mock_parse.return_value.bozo = False
    mock_parse.return_value.entries = [entry]

    result = _fetch_google_rss("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert result is None


@patch("agent.graph.nodes.news_retriever.feedparser.parse")
def test_fetch_google_rss_exception_returns_none(mock_parse):
    mock_parse.side_effect = Exception("network error")
    result = _fetch_google_rss("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert result is None


# ---------------------------------------------------------------------------
# Unit tests: _fetch_articles (fallback chain)
# ---------------------------------------------------------------------------

def test_fetch_articles_uses_finnhub_first():
    articles = [{"title": "from finnhub", "source_name": "Reuters",
                 "published_date": "2024-06-15", "url": "https://r.com", "snippet": ""}]

    with patch("agent.graph.nodes.news_retriever._fetch_finnhub", return_value=articles):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fh-key", "ydc-key")

    assert source == "finnhub"
    assert result == articles


def test_fetch_articles_falls_back_to_youcom_when_finnhub_returns_none():
    articles = [{"title": "from youcom", "source_name": "cnbc.com",
                 "published_date": "2024-06-18", "url": "https://cnbc.com", "snippet": ""}]

    with patch("agent.graph.nodes.news_retriever._fetch_finnhub", return_value=None), \
         patch("agent.graph.nodes.news_retriever._fetch_youcom", return_value=articles):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fh-key", "ydc-key")

    assert source == "youcom"
    assert result == articles


def test_fetch_articles_falls_back_to_rss_when_both_api_keys_missing():
    articles = [{"title": "from rss", "source_name": "Google News",
                 "published_date": "2024-06-18", "url": "https://example.com", "snippet": ""}]

    with patch("agent.graph.nodes.news_retriever._fetch_google_rss", return_value=articles):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", None, None)

    assert source == "google_rss"
    assert result == articles


def test_fetch_articles_returns_none_when_all_fail():
    with patch("agent.graph.nodes.news_retriever._fetch_finnhub", return_value=None), \
         patch("agent.graph.nodes.news_retriever._fetch_youcom", return_value=None), \
         patch("agent.graph.nodes.news_retriever._fetch_google_rss", return_value=None):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fh-key", "ydc-key")

    assert result is None
    assert source == "none"


def test_fetch_articles_skips_finnhub_when_no_key():
    articles = [{"title": "from youcom", "source_name": "cnbc.com",
                 "published_date": "2024-06-18", "url": "https://cnbc.com", "snippet": ""}]

    with patch("agent.graph.nodes.news_retriever._fetch_finnhub") as mock_fh, \
         patch("agent.graph.nodes.news_retriever._fetch_youcom", return_value=articles):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", None, "ydc-key")

    mock_fh.assert_not_called()
    assert source == "youcom"


# ---------------------------------------------------------------------------
# Integration-style tests: retrieve_news node
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value="fh-key")
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value="ydc-key")
@patch("agent.graph.nodes.news_retriever._fetch_articles")
def test_retrieve_news_returns_articles_and_source(mock_fetch, mock_ydc_key, mock_fh_key):
    mock_fetch.return_value = (
        [{"title": "NVDA up", "source_name": "Reuters",
          "published_date": "2024-06-15", "url": "https://r.com", "snippet": "NVDA"}],
        "finnhub",
    )

    result = retrieve_news(_base_state())

    assert result["news_source_used"] == "finnhub"
    assert len(result["news_articles"]) == 1
    assert result["news_error"] is None


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._fetch_articles", return_value=(None, "none"))
def test_retrieve_news_all_sources_fail_returns_none_articles(mock_fetch, mock_ydc_key, mock_fh_key):
    result = retrieve_news(_base_state())

    assert result["news_articles"] is None
    assert result["news_source_used"] == "none"
    assert result["news_error"] is None  # no results is not an error


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value="fh-key")
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value="ydc-key")
@patch("agent.graph.nodes.news_retriever._fetch_articles")
def test_retrieve_news_current_snapshot_appends_recent(mock_fetch, mock_ydc_key, mock_fh_key):
    historical = [{"title": "NVIDIA Q2 earnings beat", "source_name": "Reuters",
                   "published_date": "2024-06-15", "url": "https://r.com/old", "snippet": "NVDA"}]
    current = [{"title": "NVDA price target raised", "source_name": "Bloomberg",
                "published_date": "2024-07-20", "url": "https://b.com/new", "snippet": "NVDA"}]

    mock_fetch.side_effect = [(historical, "finnhub"), (current, "finnhub")]

    result = retrieve_news(_base_state(include_current_snapshot=True))

    assert len(result["news_articles"]) == 2
    urls = [a["url"] for a in result["news_articles"]]
    assert "https://r.com/old" in urls
    assert "https://b.com/new" in urls


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value="fh-key")
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value="ydc-key")
@patch("agent.graph.nodes.news_retriever._fetch_articles")
def test_retrieve_news_deduplicates_on_current_snapshot(mock_fetch, mock_ydc_key, mock_fh_key):
    article = {"title": "NVIDIA same article", "source_name": "Reuters",
               "published_date": "2024-06-15", "url": "https://r.com/same", "snippet": "NVDA"}

    mock_fetch.side_effect = [([article], "finnhub"), ([article], "finnhub")]

    result = retrieve_news(_base_state(include_current_snapshot=True))
    assert len(result["news_articles"]) == 1


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", side_effect=Exception("unexpected"))
def test_retrieve_news_unexpected_exception_writes_error(mock_fh_key):
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
