"""
Node 6: Reddit Sentiment Analyzer

Reads:  ticker, company_name, start_date, end_date
Writes: sentiment_summary, sentiment_posts, sentiment_error

Uses PRAW to search r/wallstreetbets, r/stocks, and r/options for posts
mentioning the ticker or company name within the date range.  For each
retrieved post, calls the LLM (in batches of 5) to classify sentiment as
bullish, bearish, or neutral.  Aggregates results into summary counts and
preserves individual post data for the Response Synthesizer to cite.

Credentials are read from environment variables:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

If any credential is missing the node writes sentiment_error and returns
gracefully — the rest of the workflow continues without sentiment data.
"""

import json
import logging
import os
from datetime import datetime, timezone

import praw
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_classifier

logger = logging.getLogger(__name__)

_SUBREDDITS = ["wallstreetbets", "stocks", "options"]
_MAX_POSTS = 50
_BATCH_SIZE = 5

_SENTIMENT_PROMPT = """\
You are a financial sentiment classifier.

For each Reddit post below, classify the sentiment toward the stock as:
  "bullish"  — positive, optimistic, expects price to rise
  "bearish"  — negative, pessimistic, expects price to fall
  "neutral"  — informational, mixed, or unclear

Return ONLY a JSON array of objects in the same order as the posts, each with:
  {{"index": <int>, "sentiment": "bullish" | "bearish" | "neutral"}}

No explanation. No markdown. Only the JSON array.

Posts:
{posts_text}
"""


# ---------------------------------------------------------------------------
# PRAW helpers
# ---------------------------------------------------------------------------

def _build_reddit_client() -> praw.Reddit | None:
    """
    Build a read-only PRAW Reddit client from environment variables.
    Returns None if any required credential is missing.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "stock-insight-agent/1.0")

    if not client_id or not client_secret:
        logger.warning("Reddit credentials missing — REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET not set")
        return None

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        read_only=True,
    )


def _fetch_posts(
    reddit: praw.Reddit,
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Search each subreddit for posts mentioning ticker or company_name
    within the date range.  Returns a list of raw post dicts.
    """
    try:
        start_ts = datetime.fromisoformat(start_date).timestamp()
        end_ts = datetime.fromisoformat(end_date).timestamp()
    except ValueError:
        logger.warning("Invalid date format: %s / %s", start_date, end_date)
        start_ts = 0.0
        end_ts = float("inf")

    query_terms = [ticker]
    if company_name and company_name.lower() != ticker.lower():
        query_terms.append(company_name)
    query = " OR ".join(query_terms)

    posts = []
    seen_ids = set()

    for sub_name in _SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.search(query, sort="relevance", limit=_MAX_POSTS):
                if submission.id in seen_ids:
                    continue
                # Filter by date range
                if not (start_ts <= submission.created_utc <= end_ts):
                    continue

                seen_ids.add(submission.id)
                posts.append({
                    "id": submission.id,
                    "title": submission.title,
                    "subreddit": sub_name,
                    "date": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "score": submission.score,
                    "snippet": (submission.selftext or "")[:300],
                })

        except Exception as e:
            logger.warning("PRAW search failed for r/%s: %s", sub_name, e)
            continue

    return posts


# ---------------------------------------------------------------------------
# LLM sentiment classification
# ---------------------------------------------------------------------------

def _classify_batch(batch: list[dict]) -> list[str]:
    """
    Send a batch of posts to the LLM for sentiment classification.
    Returns a list of sentiment labels in the same order as batch.
    Falls back to 'neutral' for any post the LLM fails to classify.
    """
    posts_text = "\n\n".join(
        f"[{i}] {p['title']}\n{p['snippet']}"
        for i, p in enumerate(batch)
    )
    prompt = _SENTIMENT_PROMPT.format(posts_text=posts_text)

    try:
        response = llm_classifier.invoke([
            SystemMessage(content="You are a financial sentiment classifier. Return only JSON."),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        results = json.loads(raw)
        # Build index → sentiment map
        label_map = {r["index"]: r["sentiment"] for r in results}
        return [label_map.get(i, "neutral") for i in range(len(batch))]

    except Exception as e:
        logger.warning("LLM sentiment classification failed for batch: %s", e)
        return ["neutral"] * len(batch)


def _classify_all(posts: list[dict]) -> list[str]:
    """Classify all posts in batches, returning labels in original order."""
    labels = []
    for i in range(0, len(posts), _BATCH_SIZE):
        batch = posts[i: i + _BATCH_SIZE]
        labels.extend(_classify_batch(batch))
    return labels


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def analyze_reddit_sentiment(state: AgentState) -> AgentState:
    """
    Fetch Reddit posts for the ticker/date range and classify sentiment.
    Writes sentiment_summary and sentiment_posts on success.
    Writes sentiment_error on failure; both summary and posts remain None.
    """
    ticker = state.get("ticker", "")
    company_name = state.get("company_name", ticker)
    start_date = state.get("start_date", "")
    end_date = state.get("end_date", "")

    try:
        reddit = _build_reddit_client()
        if reddit is None:
            return {
                **state,
                "sentiment_summary": None,
                "sentiment_posts": None,
                "sentiment_error": "Reddit credentials not configured",
            }

        posts = _fetch_posts(reddit, ticker, company_name, start_date, end_date)

        if not posts:
            logger.info("analyze_reddit_sentiment: no posts found for %s", ticker)
            return {
                **state,
                "sentiment_summary": {
                    "total_posts_analyzed": 0,
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "neutral_count": 0,
                    "bullish_percentage": 0.0,
                    "bearish_percentage": 0.0,
                    "neutral_percentage": 0.0,
                    "subreddits_searched": _SUBREDDITS,
                },
                "sentiment_posts": [],
                "sentiment_error": None,
            }

        labels = _classify_all(posts)

        sentiment_posts = []
        for post, label in zip(posts, labels):
            sentiment_posts.append({
                "title": post["title"],
                "subreddit": post["subreddit"],
                "date": post["date"],
                "score": post["score"],
                "sentiment_label": label,
                "snippet": post["snippet"],
            })

        total = len(sentiment_posts)
        bullish = sum(1 for p in sentiment_posts if p["sentiment_label"] == "bullish")
        bearish = sum(1 for p in sentiment_posts if p["sentiment_label"] == "bearish")
        neutral = total - bullish - bearish

        summary = {
            "total_posts_analyzed": total,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "bullish_percentage": round(bullish / total * 100, 1),
            "bearish_percentage": round(bearish / total * 100, 1),
            "neutral_percentage": round(neutral / total * 100, 1),
            "subreddits_searched": _SUBREDDITS,
        }

        logger.info(
            "analyze_reddit_sentiment: %d posts — %d bullish / %d bearish / %d neutral",
            total, bullish, bearish, neutral,
        )

        return {
            **state,
            "sentiment_summary": summary,
            "sentiment_posts": sentiment_posts,
            "sentiment_error": None,
        }

    except Exception as e:
        logger.error("analyze_reddit_sentiment failed: %s", e)
        return {
            **state,
            "sentiment_summary": None,
            "sentiment_posts": None,
            "sentiment_error": str(e),
        }
