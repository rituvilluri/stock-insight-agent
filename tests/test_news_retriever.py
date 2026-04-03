"""
Tests for Node 5: News Retriever

All external calls (Finnhub, You.com, Google RSS, Firecrawl) are mocked.
No network access or API keys required.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from agent.graph.nodes.news_retriever import (
    _build_query,
    _enrich_articles,
    _enrich_with_firecrawl,
    _fetch_articles,
    _fetch_finnhub,
    _fetch_google_rss,
    _fetch_youcom,
    _filter_relevant_articles,
    _is_free_domain,
    retrieve_news,
)


# ---------------------------------------------------------------------------
# Helpers
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


def _article(title="NVDA up", url="https://reuters.com/nvda", snippet="NVDA stock rose."):
    return {
        "title": title,
        "source_name": "Reuters",
        "published_date": "2024-06-15",
        "url": url,
        "snippet": snippet,
    }


_FAKE_FINNHUB_RESPONSE = [
    {
        "headline": "NVIDIA hits record high",
        "source": "Reuters",
        "datetime": 1718452800,
        "url": "https://reuters.com/nvda-record",
        "summary": "NVIDIA stock reached a new all-time high.",
    },
    {
        "headline": "AI chip demand surges",
        "source": "Bloomberg",
        "datetime": 1718870400,
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
    },
}


# ---------------------------------------------------------------------------
# _build_query
# ---------------------------------------------------------------------------

def test_build_query_ticker_and_company():
    assert _build_query("NVDA", "NVIDIA") == 'NVDA OR "NVIDIA"'


def test_build_query_same_ticker_and_company():
    assert _build_query("NVDA", "NVDA") == "NVDA"


def test_build_query_ticker_only():
    assert _build_query("AAPL", "") == "AAPL"


# ---------------------------------------------------------------------------
# _fetch_finnhub
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_finnhub_success(mock_get):
    mock_get.return_value = _make_response(_FAKE_FINNHUB_RESPONSE)
    result = _fetch_finnhub("NVDA", "2024-06-01", "2024-06-30", "fake-key")

    assert result is not None
    assert len(result) == 2
    assert result[0]["title"] == "NVIDIA hits record high"
    assert result[0]["source_name"] == "Reuters"
    assert result[0]["url"] == "https://reuters.com/nvda-record"
    assert "NVIDIA stock" in result[0]["snippet"]


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_finnhub_empty_returns_none(mock_get):
    mock_get.return_value = _make_response([])
    assert _fetch_finnhub("NVDA", "2024-06-01", "2024-06-30", "key") is None


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_finnhub_non_200_returns_none(mock_get):
    mock_get.return_value = _make_response({}, status_code=403)
    assert _fetch_finnhub("NVDA", "2024-06-01", "2024-06-30", "key") is None


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_finnhub_exception_returns_none(mock_get):
    mock_get.side_effect = Exception("connection refused")
    assert _fetch_finnhub("NVDA", "2024-06-01", "2024-06-30", "key") is None


# ---------------------------------------------------------------------------
# _fetch_youcom
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_youcom_success(mock_get):
    mock_get.return_value = _make_response(_FAKE_YOUCOM_RESPONSE)
    result = _fetch_youcom("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fake-key")

    assert result is not None
    assert len(result) == 1
    assert result[0]["title"] == "NVIDIA earnings beat expectations"
    assert result[0]["published_date"] == "2024-06-18"
    assert result[0]["url"] == "https://cnbc.com/nvda-earnings"


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_youcom_empty_returns_none(mock_get):
    mock_get.return_value = _make_response({"results": {"news": []}})
    assert _fetch_youcom("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "key") is None


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_youcom_non_200_returns_none(mock_get):
    mock_get.return_value = _make_response({}, status_code=401)
    assert _fetch_youcom("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "key") is None


@patch("agent.graph.nodes.news_retriever.requests.get")
def test_fetch_youcom_exception_returns_none(mock_get):
    mock_get.side_effect = Exception("timeout")
    assert _fetch_youcom("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "key") is None


# ---------------------------------------------------------------------------
# _fetch_google_rss
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

    mock_parse.return_value = MagicMock(bozo=False, entries=[entry])
    result = _fetch_google_rss("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")

    assert result is not None
    assert result[0]["title"] == "NVIDIA earnings beat expectations"
    assert result[0]["published_date"] == "2024-06-18"


@patch("agent.graph.nodes.news_retriever.feedparser.parse")
def test_fetch_google_rss_filters_out_of_range(mock_parse):
    entry = MagicMock()
    entry.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
    entry.get = lambda key, default=None: {
        "title": "Old news", "link": "https://example.com/old",
        "summary": "Old.", "source": {"title": "Reuters"},
        "published_parsed": entry.published_parsed,
    }.get(key, default)

    mock_parse.return_value = MagicMock(bozo=False, entries=[entry])
    assert _fetch_google_rss("NVDA", "NVIDIA", "2024-06-01", "2024-06-30") is None


@patch("agent.graph.nodes.news_retriever.feedparser.parse")
def test_fetch_google_rss_exception_returns_none(mock_parse):
    mock_parse.side_effect = Exception("network error")
    assert _fetch_google_rss("NVDA", "NVIDIA", "2024-06-01", "2024-06-30") is None


# ---------------------------------------------------------------------------
# _fetch_articles — parallel behavior
# ---------------------------------------------------------------------------

def test_fetch_articles_uses_only_finnhub_when_youcom_key_absent():
    articles = [_article(url="https://reuters.com/a")]
    with patch("agent.graph.nodes.news_retriever._fetch_finnhub", return_value=articles) as mock_fh, \
         patch("agent.graph.nodes.news_retriever._fetch_youcom") as mock_ydc:
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fh-key", None)

    mock_fh.assert_called_once()
    mock_ydc.assert_not_called()
    assert source == "finnhub"
    assert result == articles


def test_fetch_articles_uses_only_youcom_when_finnhub_key_absent():
    articles = [_article(url="https://cnbc.com/a")]
    with patch("agent.graph.nodes.news_retriever._fetch_finnhub") as mock_fh, \
         patch("agent.graph.nodes.news_retriever._fetch_youcom", return_value=articles):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", None, "ydc-key")

    mock_fh.assert_not_called()
    assert source == "youcom"


def test_fetch_articles_merges_both_sources():
    fh_articles = [_article(title="Finnhub A", url="https://reuters.com/a")]
    ydc_articles = [_article(title="YouCom B", url="https://cnbc.com/b")]

    with patch("agent.graph.nodes.news_retriever._fetch_finnhub", return_value=fh_articles), \
         patch("agent.graph.nodes.news_retriever._fetch_youcom", return_value=ydc_articles):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fh-key", "ydc-key")

    assert source == "finnhub+youcom"
    assert len(result) == 2
    titles = [a["title"] for a in result]
    assert "Finnhub A" in titles
    assert "YouCom B" in titles


def test_fetch_articles_deduplicates_same_url():
    shared = _article(url="https://reuters.com/same")
    with patch("agent.graph.nodes.news_retriever._fetch_finnhub", return_value=[shared]), \
         patch("agent.graph.nodes.news_retriever._fetch_youcom", return_value=[shared]):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fh-key", "ydc-key")

    assert len(result) == 1


def test_fetch_articles_falls_back_to_rss_when_both_empty():
    rss_articles = [_article(url="https://example.com/rss")]
    with patch("agent.graph.nodes.news_retriever._fetch_finnhub", return_value=None), \
         patch("agent.graph.nodes.news_retriever._fetch_youcom", return_value=None), \
         patch("agent.graph.nodes.news_retriever._fetch_google_rss", return_value=rss_articles):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fh-key", "ydc-key")

    assert source == "google_rss"
    assert result == rss_articles


def test_fetch_articles_falls_back_to_rss_when_no_keys():
    rss_articles = [_article(url="https://example.com/rss")]
    with patch("agent.graph.nodes.news_retriever._fetch_google_rss", return_value=rss_articles):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", None, None)

    assert source == "google_rss"


def test_fetch_articles_returns_none_when_all_fail():
    with patch("agent.graph.nodes.news_retriever._fetch_finnhub", return_value=None), \
         patch("agent.graph.nodes.news_retriever._fetch_youcom", return_value=None), \
         patch("agent.graph.nodes.news_retriever._fetch_google_rss", return_value=None):
        result, source = _fetch_articles("NVDA", "NVIDIA", "2024-06-01", "2024-06-30", "fh-key", "ydc-key")

    assert result is None
    assert source == "none"


# ---------------------------------------------------------------------------
# _is_free_domain
# ---------------------------------------------------------------------------

def test_is_free_domain_recognises_whitelisted_domains():
    assert _is_free_domain("https://reuters.com/article/nvda") is True
    assert _is_free_domain("https://cnbc.com/2024/06/nvda") is True
    assert _is_free_domain("https://finance.yahoo.com/nvda") is True


def test_is_free_domain_rejects_paywalled_domains():
    assert _is_free_domain("https://bloomberg.com/nvda") is False
    assert _is_free_domain("https://wsj.com/articles/nvda") is False


# ---------------------------------------------------------------------------
# _enrich_with_firecrawl
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever.requests.post")
def test_enrich_with_firecrawl_replaces_snippet_for_free_domain(mock_post):
    mock_post.return_value = _make_response({
        "success": True,
        "data": {"markdown": "# NVDA Article\n\nFull article content here. " * 50},
    })
    article = _article(url="https://reuters.com/nvda")
    enriched = _enrich_with_firecrawl(article, "fc-key")

    assert enriched["snippet"] != article["snippet"]
    assert len(enriched["snippet"]) <= 2000
    assert "NVDA Article" in enriched["snippet"]


@patch("agent.graph.nodes.news_retriever.requests.post")
def test_enrich_with_firecrawl_skips_paywalled_domain(mock_post):
    article = _article(url="https://bloomberg.com/nvda")
    result = _enrich_with_firecrawl(article, "fc-key")

    mock_post.assert_not_called()
    assert result is article  # unchanged


@patch("agent.graph.nodes.news_retriever.requests.post")
def test_enrich_with_firecrawl_returns_original_on_http_error(mock_post):
    mock_post.return_value = _make_response({}, status_code=500)
    article = _article(url="https://reuters.com/nvda")
    result = _enrich_with_firecrawl(article, "fc-key")
    assert result["snippet"] == article["snippet"]


@patch("agent.graph.nodes.news_retriever.requests.post")
def test_enrich_with_firecrawl_returns_original_on_exception(mock_post):
    mock_post.side_effect = Exception("timeout")
    article = _article(url="https://reuters.com/nvda")
    result = _enrich_with_firecrawl(article, "fc-key")
    assert result["snippet"] == article["snippet"]


# ---------------------------------------------------------------------------
# _enrich_articles
# ---------------------------------------------------------------------------

def test_enrich_articles_skips_when_no_key():
    articles = [_article(url="https://reuters.com/nvda")]
    result = _enrich_articles(articles, None)
    assert result is articles  # unchanged, same object


@patch("agent.graph.nodes.news_retriever._enrich_with_firecrawl")
def test_enrich_articles_calls_enricher_for_each_article(mock_enrich):
    mock_enrich.side_effect = lambda a, key: a
    articles = [_article(url=f"https://reuters.com/{i}") for i in range(3)]
    _enrich_articles(articles, "fc-key")
    assert mock_enrich.call_count == 3


# ---------------------------------------------------------------------------
# retrieve_news — node integration
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value="fh-key")
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value="ydc-key")
@patch("agent.graph.nodes.news_retriever._get_firecrawl_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._fetch_articles")
def test_retrieve_news_returns_articles_and_source(mock_fetch, *_):
    mock_fetch.return_value = ([_article()], "finnhub")
    result = retrieve_news(_base_state())

    assert result["news_source_used"] == "finnhub"
    assert len(result["news_articles"]) == 1
    assert result["news_error"] is None


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._get_firecrawl_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._fetch_articles", return_value=(None, "none"))
def test_retrieve_news_all_sources_fail_returns_none_articles(mock_fetch, *_):
    result = retrieve_news(_base_state())
    assert result["news_articles"] is None
    assert result["news_source_used"] == "none"
    assert result["news_error"] is None


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value="fh-key")
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value="ydc-key")
@patch("agent.graph.nodes.news_retriever._get_firecrawl_key", return_value="fc-key")
@patch("agent.graph.nodes.news_retriever._fetch_articles")
@patch("agent.graph.nodes.news_retriever._enrich_articles")
def test_retrieve_news_calls_firecrawl_when_key_present(mock_enrich, mock_fetch, *_):
    mock_fetch.return_value = ([_article()], "finnhub")
    mock_enrich.side_effect = lambda articles, key: articles
    retrieve_news(_base_state())
    mock_enrich.assert_called_once()
    assert mock_enrich.call_args[0][1] == "fc-key"


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value="fh-key")
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value="ydc-key")
@patch("agent.graph.nodes.news_retriever._get_firecrawl_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._fetch_articles")
def test_retrieve_news_current_snapshot_appends_recent(mock_fetch, *_):
    historical = [_article(url="https://reuters.com/old", snippet="NVDA")]
    current = [_article(title="New", url="https://cnbc.com/new", snippet="NVDA")]
    mock_fetch.side_effect = [(historical, "finnhub"), (current, "finnhub")]

    result = retrieve_news(_base_state(include_current_snapshot=True))
    assert len(result["news_articles"]) == 2


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", return_value="fh-key")
@patch("agent.graph.nodes.news_retriever._get_youcom_key", return_value="ydc-key")
@patch("agent.graph.nodes.news_retriever._get_firecrawl_key", return_value=None)
@patch("agent.graph.nodes.news_retriever._fetch_articles")
def test_retrieve_news_deduplicates_on_snapshot(mock_fetch, *_):
    article = _article(url="https://reuters.com/same", snippet="NVDA")
    mock_fetch.side_effect = [([article], "finnhub"), ([article], "finnhub")]

    result = retrieve_news(_base_state(include_current_snapshot=True))
    assert len(result["news_articles"]) == 1


@patch("agent.graph.nodes.news_retriever._get_finnhub_key", side_effect=Exception("unexpected"))
def test_retrieve_news_unexpected_exception_writes_error(mock_fh_key):
    result = retrieve_news(_base_state())
    assert result["news_articles"] is None
    assert result["news_error"] is not None
    assert "unexpected" in result["news_error"]
