"""
Tests for Node 6: Reddit Sentiment Analyzer

All HTTP and LLM calls are mocked — no network access or credentials needed.
Covers: Reddit public JSON fetcher, Stocktwits fetcher, LLM classifier,
        pre-label pass-through, and the full node integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.reddit_sentiment import (
    _classify_batch,
    _classify_all,
    _fetch_reddit_posts,
    _fetch_stocktwits_messages,
    analyze_reddit_sentiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides):
    state = {
        "user_message": "What was Reddit saying about NVDA last month?",
        "user_config": {},
        "ticker": "NVDA",
        "company_name": "NVIDIA",
        "start_date": "2024-06-01",
        "end_date": "2024-06-30",
    }
    state.update(overrides)
    return state


def _reddit_response(children: list) -> dict:
    """Build a minimal Reddit public JSON response."""
    return {"data": {"children": [{"data": c} for c in children]}}


def _reddit_post(
    post_id="abc",
    title="NVDA bullish",
    selftext="Great earnings.",
    score=500,
    created_utc=1718000000.0,  # 2024-06-10 — within test date range
    subreddit="stocks",
    permalink="/r/stocks/abc",
):
    return {
        "id": post_id,
        "title": title,
        "selftext": selftext,
        "score": score,
        "created_utc": created_utc,
        "subreddit": subreddit,
        "permalink": permalink,
    }


def _stocktwits_response(messages: list, cursor_max=None) -> dict:
    """Build a minimal Stocktwits API response."""
    return {
        "messages": messages,
        "cursor": {"max": cursor_max, "more": cursor_max is not None},
    }


def _stocktwits_message(
    msg_id=1,
    body="NVDA looks strong here",
    created_at="2024-06-10T12:00:00Z",
    sentiment_basic=None,  # "Bullish", "Bearish", or None
    likes=3,
):
    msg = {
        "id": msg_id,
        "body": body,
        "created_at": created_at,
        "entities": {"sentiment": {"basic": sentiment_basic} if sentiment_basic else {}},
        "likes": {"total": likes},
    }
    return msg


# ---------------------------------------------------------------------------
# _fetch_reddit_posts
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.reddit_sentiment.requests.get")
def test_fetch_reddit_posts_returns_in_range_posts(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: _reddit_response([
            _reddit_post(post_id="in1", created_utc=1718000000.0),   # 2024-06-10 — in range
            _reddit_post(post_id="out1", created_utc=1700000000.0),  # 2023-11-14 — out of range
        ]),
    )
    posts = _fetch_reddit_posts("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert len(posts) == 1
    assert posts[0]["id"] == "in1"
    assert posts[0]["source"] == "reddit"


@patch("agent.graph.nodes.reddit_sentiment.requests.get")
def test_fetch_reddit_posts_returns_expected_fields(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: _reddit_response([
            _reddit_post(title="NVDA earnings", selftext="Very bullish.", score=200, created_utc=1718000000.0),
        ]),
    )
    posts = _fetch_reddit_posts("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    post = posts[0]
    assert post["title"] == "NVDA earnings"
    assert post["score"] == 200
    assert post["snippet"] == "Very bullish."
    assert post["date"] == "2024-06-10"
    assert post["source"] == "reddit"
    assert post["pre_label"] is None


@patch("agent.graph.nodes.reddit_sentiment.requests.get")
def test_fetch_reddit_posts_returns_empty_on_http_error(mock_get):
    mock_get.return_value = MagicMock(status_code=429)
    posts = _fetch_reddit_posts("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert posts == []


@patch("agent.graph.nodes.reddit_sentiment.requests.get")
def test_fetch_reddit_posts_returns_empty_on_exception(mock_get):
    mock_get.side_effect = Exception("network error")
    posts = _fetch_reddit_posts("NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert posts == []


# ---------------------------------------------------------------------------
# _fetch_stocktwits_messages
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.reddit_sentiment.requests.get")
def test_fetch_stocktwits_returns_in_range_messages(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: _stocktwits_response([
            _stocktwits_message(msg_id=1, created_at="2024-06-10T12:00:00Z"),
            _stocktwits_message(msg_id=2, created_at="2023-01-01T12:00:00Z"),  # out of range
        ]),
    )
    messages = _fetch_stocktwits_messages("NVDA", "2024-06-01", "2024-06-30")
    assert len(messages) == 1
    assert messages[0]["id"] == "1"
    assert messages[0]["source"] == "stocktwits"


@patch("agent.graph.nodes.reddit_sentiment.requests.get")
def test_fetch_stocktwits_uses_pre_label(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: _stocktwits_response([
            _stocktwits_message(msg_id=1, created_at="2024-06-10T12:00:00Z", sentiment_basic="Bullish"),
            _stocktwits_message(msg_id=2, created_at="2024-06-11T12:00:00Z", sentiment_basic="Bearish"),
            _stocktwits_message(msg_id=3, created_at="2024-06-12T12:00:00Z", sentiment_basic=None),
        ]),
    )
    messages = _fetch_stocktwits_messages("NVDA", "2024-06-01", "2024-06-30")
    assert messages[0]["pre_label"] == "bullish"
    assert messages[1]["pre_label"] == "bearish"
    assert messages[2]["pre_label"] is None


@patch("agent.graph.nodes.reddit_sentiment.requests.get")
def test_fetch_stocktwits_returns_empty_on_http_error(mock_get):
    mock_get.return_value = MagicMock(status_code=429)
    messages = _fetch_stocktwits_messages("NVDA", "2024-06-01", "2024-06-30")
    assert messages == []


@patch("agent.graph.nodes.reddit_sentiment.requests.get")
def test_fetch_stocktwits_returns_empty_on_exception(mock_get):
    mock_get.side_effect = Exception("connection refused")
    messages = _fetch_stocktwits_messages("NVDA", "2024-06-01", "2024-06-30")
    assert messages == []


# ---------------------------------------------------------------------------
# _classify_batch
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.reddit_sentiment.llm_classifier")
def test_classify_batch_parses_llm_response(mock_llm):
    mock_llm.invoke.return_value = MagicMock(
        content='[{"index": 0, "sentiment": "bullish"}, {"index": 1, "sentiment": "bearish"}]'
    )
    posts = [
        {"title": "NVDA up", "snippet": "Very bullish."},
        {"title": "NVDA down", "snippet": "Bearish outlook."},
    ]
    labels = _classify_batch(posts)
    assert labels == ["bullish", "bearish"]


@patch("agent.graph.nodes.reddit_sentiment.llm_classifier")
def test_classify_batch_falls_back_to_neutral_on_llm_error(mock_llm):
    mock_llm.invoke.side_effect = Exception("LLM unavailable")
    posts = [{"title": "NVDA", "snippet": "something"}]
    labels = _classify_batch(posts)
    assert labels == ["neutral"]


@patch("agent.graph.nodes.reddit_sentiment.llm_classifier")
def test_classify_batch_strips_markdown_fence(mock_llm):
    mock_llm.invoke.return_value = MagicMock(
        content='```json\n[{"index": 0, "sentiment": "neutral"}]\n```'
    )
    labels = _classify_batch([{"title": "NVDA", "snippet": "unclear"}])
    assert labels == ["neutral"]


@patch("agent.graph.nodes.reddit_sentiment.llm_classifier")
def test_classify_batch_strips_uppercase_json_fence(mock_llm):
    mock_llm.invoke.return_value = MagicMock(
        content='```JSON\n[{"index": 0, "sentiment": "bullish"}]\n```'
    )
    labels = _classify_batch([{"title": "NVDA moon", "snippet": "great"}])
    assert labels == ["bullish"]


# ---------------------------------------------------------------------------
# _classify_all — pre-label pass-through and batching
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.reddit_sentiment._classify_batch")
def test_classify_all_skips_llm_for_pre_labeled_posts(mock_batch):
    mock_batch.return_value = ["neutral"]  # called for the one unlabeled post
    posts = [
        {"title": "A", "snippet": "", "pre_label": "bullish"},
        {"title": "B", "snippet": "", "pre_label": "bearish"},
        {"title": "C", "snippet": "", "pre_label": None},  # needs LLM
    ]
    labels = _classify_all(posts)
    assert labels == ["bullish", "bearish", "neutral"]
    mock_batch.assert_called_once()
    assert len(mock_batch.call_args[0][0]) == 1  # only 1 post sent to LLM


@patch("agent.graph.nodes.reddit_sentiment._classify_batch")
def test_classify_all_batches_correctly(mock_batch):
    mock_batch.side_effect = lambda batch: ["bullish"] * len(batch)
    posts = [{"title": f"post {i}", "snippet": "", "pre_label": None} for i in range(12)]
    labels = _classify_all(posts)
    # 12 posts, batch size 5 → 3 calls (5+5+2)
    assert mock_batch.call_count == 3
    assert len(labels) == 12


# ---------------------------------------------------------------------------
# analyze_reddit_sentiment — node integration
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.reddit_sentiment._fetch_reddit_posts", return_value=[])
@patch("agent.graph.nodes.reddit_sentiment._fetch_stocktwits_messages", return_value=[])
def test_node_returns_zero_summary_when_no_posts(mock_st, mock_reddit):
    result = analyze_reddit_sentiment(_base_state())
    assert result["sentiment_error"] is None
    assert result["sentiment_summary"]["total_posts_analyzed"] == 0
    assert result["sentiment_posts"] == []
    assert "reddit" in result["sentiment_summary"]["sources"]
    assert "stocktwits" in result["sentiment_summary"]["sources"]


@patch("agent.graph.nodes.reddit_sentiment._fetch_reddit_posts")
@patch("agent.graph.nodes.reddit_sentiment._fetch_stocktwits_messages")
@patch("agent.graph.nodes.reddit_sentiment._classify_all")
def test_node_aggregates_totals_correctly(mock_classify, mock_st, mock_reddit):
    mock_reddit.return_value = [
        {"title": "A", "subreddit": "stocks", "date": "2024-06-10",
         "score": 100, "snippet": "", "source": "reddit", "permalink": "", "pre_label": None},
        {"title": "B", "subreddit": "wallstreetbets", "date": "2024-06-12",
         "score": 50, "snippet": "", "source": "reddit", "permalink": "", "pre_label": None},
    ]
    mock_st.return_value = [
        {"title": "C", "subreddit": "stocktwits", "date": "2024-06-13",
         "score": 2, "snippet": "", "source": "stocktwits", "permalink": "", "pre_label": "bullish"},
    ]
    mock_classify.return_value = ["bullish", "bearish", "bullish"]

    result = analyze_reddit_sentiment(_base_state())
    summary = result["sentiment_summary"]

    assert summary["total_posts_analyzed"] == 3
    assert summary["bullish_count"] == 2
    assert summary["bearish_count"] == 1
    assert summary["neutral_count"] == 0
    assert summary["bullish_percentage"] == round(2 / 3 * 100, 1)
    assert result["sentiment_error"] is None


@patch("agent.graph.nodes.reddit_sentiment._fetch_reddit_posts")
@patch("agent.graph.nodes.reddit_sentiment._fetch_stocktwits_messages")
@patch("agent.graph.nodes.reddit_sentiment._classify_all")
def test_node_sources_breakdown_is_correct(mock_classify, mock_st, mock_reddit):
    mock_reddit.return_value = [
        {"title": "R1", "subreddit": "stocks", "date": "2024-06-10",
         "score": 100, "snippet": "", "source": "reddit", "permalink": "", "pre_label": None},
    ]
    mock_st.return_value = [
        {"title": "S1", "subreddit": "stocktwits", "date": "2024-06-10",
         "score": 1, "snippet": "", "source": "stocktwits", "permalink": "", "pre_label": None},
        {"title": "S2", "subreddit": "stocktwits", "date": "2024-06-11",
         "score": 2, "snippet": "", "source": "stocktwits", "permalink": "", "pre_label": None},
    ]
    mock_classify.return_value = ["bullish", "bearish", "neutral"]

    result = analyze_reddit_sentiment(_base_state())
    sources = result["sentiment_summary"]["sources"]

    assert sources["reddit"]["posts"] == 1
    assert sources["reddit"]["bullish"] == 1
    assert sources["stocktwits"]["posts"] == 2
    assert sources["stocktwits"]["bearish"] == 1
    assert sources["stocktwits"]["neutral"] == 1


@patch("agent.graph.nodes.reddit_sentiment._fetch_reddit_posts")
@patch("agent.graph.nodes.reddit_sentiment._fetch_stocktwits_messages")
@patch("agent.graph.nodes.reddit_sentiment._classify_all")
def test_node_preserves_post_fields(mock_classify, mock_st, mock_reddit):
    mock_reddit.return_value = [
        {"title": "NVDA bull", "subreddit": "stocks", "date": "2024-06-10",
         "score": 300, "snippet": "Strong buy.", "source": "reddit",
         "permalink": "/r/stocks/abc", "pre_label": None},
    ]
    mock_st.return_value = []
    mock_classify.return_value = ["bullish"]

    result = analyze_reddit_sentiment(_base_state())
    post = result["sentiment_posts"][0]

    assert post["title"] == "NVDA bull"
    assert post["subreddit"] == "stocks"
    assert post["score"] == 300
    assert post["sentiment_label"] == "bullish"
    assert post["source"] == "reddit"
    assert post["permalink"] == "/r/stocks/abc"


@patch("agent.graph.nodes.reddit_sentiment._fetch_reddit_posts")
def test_node_writes_error_on_unexpected_exception(mock_reddit):
    mock_reddit.side_effect = Exception("unexpected crash")
    result = analyze_reddit_sentiment(_base_state())
    assert result["sentiment_summary"] is None
    assert result["sentiment_posts"] is None
    assert "unexpected crash" in result["sentiment_error"]


def test_returns_only_owned_fields():
    """
    analyze_reddit_sentiment must return ONLY its three owned fields.
    Returning {**state} in a parallel Send() branch causes LangGraph to
    throw InvalidUpdateError when merging results from concurrent nodes.
    """
    state = _base_state(extra_sentinel="should_not_leak")

    with patch("agent.graph.nodes.reddit_sentiment._fetch_reddit_posts", return_value=[]):
        with patch("agent.graph.nodes.reddit_sentiment._fetch_stocktwits_messages", return_value=[]):
            result = analyze_reddit_sentiment(state)

    assert set(result.keys()) == {"sentiment_summary", "sentiment_posts", "sentiment_error"}, (
        f"analyze_reddit_sentiment returned unexpected keys: {set(result.keys())}"
    )
    assert "extra_sentinel" not in result
    assert "user_message" not in result
    assert "ticker" not in result
