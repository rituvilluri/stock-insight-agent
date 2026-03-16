"""
Tests for Node 6: Reddit Sentiment Analyzer

All PRAW and LLM calls are mocked — no network access or credentials needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.reddit_sentiment import (
    _classify_batch,
    _classify_all,
    _build_reddit_client,
    _fetch_posts,
    analyze_reddit_sentiment,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def _make_submission(
    sid="abc123",
    title="NVDA to the moon",
    selftext="Strong earnings incoming.",
    score=500,
    created_utc=1718000000.0,  # 2024-06-10
    subreddit="stocks",
):
    sub = MagicMock()
    sub.id = sid
    sub.title = title
    sub.selftext = selftext
    sub.score = score
    sub.created_utc = created_utc
    sub.subreddit.display_name = subreddit
    return sub


# ---------------------------------------------------------------------------
# _build_reddit_client
# ---------------------------------------------------------------------------

@patch.dict("os.environ", {"REDDIT_CLIENT_ID": "cid", "REDDIT_CLIENT_SECRET": "csec", "REDDIT_USER_AGENT": "test/1.0"})
@patch("agent.graph.nodes.reddit_sentiment.praw.Reddit")
def test_build_reddit_client_returns_client(mock_reddit):
    mock_reddit.return_value = MagicMock()
    client = _build_reddit_client()
    assert client is not None
    mock_reddit.assert_called_once_with(
        client_id="cid",
        client_secret="csec",
        user_agent="test/1.0",
        read_only=True,
    )


@patch.dict("os.environ", {}, clear=True)
def test_build_reddit_client_missing_credentials_returns_none():
    # Ensure env vars are absent
    import os
    os.environ.pop("REDDIT_CLIENT_ID", None)
    os.environ.pop("REDDIT_CLIENT_SECRET", None)
    client = _build_reddit_client()
    assert client is None


# ---------------------------------------------------------------------------
# _fetch_posts
# ---------------------------------------------------------------------------

def test_fetch_posts_filters_by_date():
    reddit = MagicMock()
    in_range = _make_submission(sid="in1", created_utc=1718000000.0)   # 2024-06-10 — in range
    out_range = _make_submission(sid="out1", created_utc=1700000000.0)  # 2023-11-14 — out of range

    reddit.subreddit.return_value.search.return_value = [in_range, out_range]

    posts = _fetch_posts(reddit, "NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert len(posts) == 1
    assert posts[0]["date"] == "2024-06-10"


def test_fetch_posts_deduplicates_across_subreddits():
    reddit = MagicMock()
    duplicate = _make_submission(sid="dup1", created_utc=1718000000.0)
    # Return the same submission from every subreddit search
    reddit.subreddit.return_value.search.return_value = [duplicate]

    posts = _fetch_posts(reddit, "NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    # 3 subreddits searched but same ID — should appear only once
    assert len(posts) == 1


def test_fetch_posts_handles_subreddit_error_gracefully():
    reddit = MagicMock()
    reddit.subreddit.return_value.search.side_effect = Exception("API error")

    posts = _fetch_posts(reddit, "NVDA", "NVIDIA", "2024-06-01", "2024-06-30")
    assert posts == []


def test_fetch_posts_returns_expected_fields():
    reddit = MagicMock()
    sub = _make_submission(title="NVDA earnings", selftext="Very bullish.", score=200, created_utc=1718000000.0)
    reddit.subreddit.return_value.search.return_value = [sub]

    posts = _fetch_posts(reddit, "NVDA", "NVIDIA", "2024-06-01", "2024-06-30")

    assert len(posts) == 1
    post = posts[0]
    assert post["title"] == "NVDA earnings"
    assert post["score"] == 200
    assert post["snippet"] == "Very bullish."
    assert "date" in post
    assert "subreddit" in post


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
def test_classify_batch_handles_markdown_fence(mock_llm):
    mock_llm.invoke.return_value = MagicMock(
        content='```json\n[{"index": 0, "sentiment": "neutral"}]\n```'
    )
    posts = [{"title": "NVDA sideways", "snippet": "unclear"}]
    labels = _classify_batch(posts)
    assert labels == ["neutral"]


# ---------------------------------------------------------------------------
# _classify_all — batching
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.reddit_sentiment._classify_batch")
def test_classify_all_batches_correctly(mock_batch):
    # Return labels matching the batch size passed in
    mock_batch.side_effect = lambda batch: ["bullish"] * len(batch)
    posts = [{"title": f"post {i}", "snippet": ""} for i in range(12)]
    labels = _classify_all(posts)

    # 12 posts with batch size 5 → 3 calls (5+5+2)
    assert mock_batch.call_count == 3
    assert len(labels) == 12


# ---------------------------------------------------------------------------
# analyze_reddit_sentiment — node integration
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.reddit_sentiment._build_reddit_client", return_value=None)
def test_node_returns_error_when_no_credentials(mock_client):
    result = analyze_reddit_sentiment(_base_state())
    assert result["sentiment_summary"] is None
    assert result["sentiment_posts"] is None
    assert "credentials" in result["sentiment_error"].lower()


@patch("agent.graph.nodes.reddit_sentiment._build_reddit_client")
@patch("agent.graph.nodes.reddit_sentiment._fetch_posts", return_value=[])
def test_node_returns_zero_summary_when_no_posts(mock_fetch, mock_client):
    mock_client.return_value = MagicMock()
    result = analyze_reddit_sentiment(_base_state())

    assert result["sentiment_error"] is None
    assert result["sentiment_summary"]["total_posts_analyzed"] == 0
    assert result["sentiment_posts"] == []


@patch("agent.graph.nodes.reddit_sentiment._build_reddit_client")
@patch("agent.graph.nodes.reddit_sentiment._fetch_posts")
@patch("agent.graph.nodes.reddit_sentiment._classify_all")
def test_node_aggregates_sentiment_correctly(mock_classify, mock_fetch, mock_client):
    mock_client.return_value = MagicMock()
    mock_fetch.return_value = [
        {"title": "Post A", "subreddit": "stocks", "date": "2024-06-10", "score": 100, "snippet": ""},
        {"title": "Post B", "subreddit": "wallstreetbets", "date": "2024-06-12", "score": 50, "snippet": ""},
        {"title": "Post C", "subreddit": "options", "date": "2024-06-15", "score": 10, "snippet": ""},
        {"title": "Post D", "subreddit": "stocks", "date": "2024-06-18", "score": 5, "snippet": ""},
    ]
    mock_classify.return_value = ["bullish", "bullish", "bearish", "neutral"]

    result = analyze_reddit_sentiment(_base_state())

    summary = result["sentiment_summary"]
    assert summary["total_posts_analyzed"] == 4
    assert summary["bullish_count"] == 2
    assert summary["bearish_count"] == 1
    assert summary["neutral_count"] == 1
    assert summary["bullish_percentage"] == 50.0
    assert summary["bearish_percentage"] == 25.0
    assert summary["neutral_percentage"] == 25.0
    assert summary["subreddits_searched"] == ["wallstreetbets", "stocks", "options"]
    assert result["sentiment_error"] is None


@patch("agent.graph.nodes.reddit_sentiment._build_reddit_client")
@patch("agent.graph.nodes.reddit_sentiment._fetch_posts")
@patch("agent.graph.nodes.reddit_sentiment._classify_all")
def test_node_preserves_individual_post_fields(mock_classify, mock_fetch, mock_client):
    mock_client.return_value = MagicMock()
    mock_fetch.return_value = [
        {"title": "NVDA bull", "subreddit": "stocks", "date": "2024-06-10", "score": 300, "snippet": "Strong buy."},
    ]
    mock_classify.return_value = ["bullish"]

    result = analyze_reddit_sentiment(_base_state())
    post = result["sentiment_posts"][0]

    assert post["title"] == "NVDA bull"
    assert post["subreddit"] == "stocks"
    assert post["date"] == "2024-06-10"
    assert post["score"] == 300
    assert post["sentiment_label"] == "bullish"
    assert post["snippet"] == "Strong buy."


@patch("agent.graph.nodes.reddit_sentiment._build_reddit_client")
def test_node_writes_error_on_unexpected_exception(mock_client):
    mock_client.side_effect = Exception("unexpected crash")
    result = analyze_reddit_sentiment(_base_state())

    assert result["sentiment_summary"] is None
    assert result["sentiment_posts"] is None
    assert "unexpected crash" in result["sentiment_error"]
