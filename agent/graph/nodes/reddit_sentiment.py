"""
Node 6: Reddit Sentiment Analyzer

Reads:  ticker, company_name, start_date, end_date
Writes: sentiment_summary, sentiment_posts, sentiment_error

Two co-primary sources — both run on every query, results combined:

  1. Reddit public JSON — searches r/wallstreetbets+r/stocks+r/options
     No authentication required.
     URL: reddit.com/r/{subs}/search.json?q={query}&restrict_sr=1
     Date filtering applied post-fetch (Reddit search does not support
     server-side date ranges without OAuth).

  2. Stocktwits public stream — symbol-specific message feed
     No authentication required.
     URL: api.stocktwits.com/api/2/streams/symbol/{ticker}.json
     Pre-labeled sentiment ("Bullish"/"Bearish") used where Stocktwits
     provides it; LLM classifier used for unlabeled messages.
     Best for recent data (< 30 days). Returns 0 results gracefully
     for older date ranges — Reddit carries the historical load.

Posts from both sources are combined and aggregated into sentiment_summary,
which includes a per-source breakdown for transparency in Node 9.
"""

import json
import logging
from datetime import datetime, timezone

import requests
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_classifier

logger = logging.getLogger(__name__)

_SUBREDDITS = "wallstreetbets+stocks+options"
_MAX_REDDIT_POSTS = 100
_MAX_STOCKTWITS_PAGES = 3
_BATCH_SIZE = 5
_REQUEST_TIMEOUT = 10
_REDDIT_HEADERS = {"User-Agent": "stock-insight-agent/1.0"}

_SENTIMENT_PROMPT = """\
You are a financial sentiment classifier.

For each post below, classify the sentiment toward the stock as:
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
# Reddit public JSON fetcher
# ---------------------------------------------------------------------------

def _fetch_reddit_posts(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch posts from the Reddit public JSON search endpoint.
    Searches r/wallstreetbets+r/stocks+r/options in one request.
    Returns a list of post dicts filtered to the given date range.
    Returns [] on any failure — caller handles the empty case.
    """
    try:
        start_ts = datetime.fromisoformat(start_date).timestamp()
        end_ts = datetime.fromisoformat(end_date).timestamp()
    except ValueError:
        start_ts = 0.0
        end_ts = float("inf")

    query_parts = [ticker]
    if company_name and company_name.lower() != ticker.lower():
        query_parts.append(company_name)
    query = " OR ".join(query_parts)

    try:
        resp = requests.get(
            f"https://www.reddit.com/r/{_SUBREDDITS}/search.json",
            headers=_REDDIT_HEADERS,
            params={
                "q": query,
                "sort": "relevance",
                "restrict_sr": "1",
                "t": "all",
                "limit": _MAX_REDDIT_POSTS,
            },
            timeout=_REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            logger.warning("Reddit public JSON returned HTTP %s", resp.status_code)
            return []

        children = resp.json().get("data", {}).get("children", [])

        posts = []
        for child in children:
            post = child.get("data", {})
            created = post.get("created_utc", 0)
            if not (start_ts <= created <= end_ts):
                continue
            posts.append({
                "id": post.get("id", ""),
                "title": post.get("title", ""),
                "subreddit": post.get("subreddit", ""),
                "date": datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d"),
                "score": post.get("score", 0),
                "snippet": (post.get("selftext") or "")[:300],
                "permalink": post.get("permalink", ""),
                "source": "reddit",
                "pre_label": None,
            })

        logger.info("_fetch_reddit_posts → %d posts in date range for %s", len(posts), ticker)
        return posts

    except Exception as e:
        logger.warning("Reddit public JSON fetch failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Stocktwits public stream fetcher
# ---------------------------------------------------------------------------

def _fetch_stocktwits_messages(
    ticker: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Fetch messages from the Stocktwits public symbol stream.
    Paginates up to _MAX_STOCKTWITS_PAGES pages (cursor-based, older-first).
    Filters to the given date range in application code.
    Uses pre-labeled Stocktwits sentiment where available.
    Returns [] on any failure.
    """
    try:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    except ValueError:
        start_dt = None
        end_dt = None

    messages = []
    params: dict = {}

    for _ in range(_MAX_STOCKTWITS_PAGES):
        try:
            resp = requests.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
                params=params,
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning("Stocktwits returned HTTP %s", resp.status_code)
                break

            data = resp.json()
            batch = data.get("messages", [])
            if not batch:
                break

            oldest_in_batch = None
            for msg in batch:
                created_str = msg.get("created_at", "")
                try:
                    created_dt = datetime.fromisoformat(
                        created_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    continue

                oldest_in_batch = created_dt

                if start_dt and end_dt and not (start_dt <= created_dt <= end_dt):
                    continue

                # Use Stocktwits pre-label if available ("Bullish" / "Bearish")
                sentiment_entity = (msg.get("entities") or {}).get("sentiment") or {}
                raw_label = sentiment_entity.get("basic")
                pre_label = raw_label.lower() if raw_label else None

                messages.append({
                    "id": str(msg.get("id", "")),
                    "title": (msg.get("body") or "")[:150],
                    "subreddit": "stocktwits",
                    "date": created_dt.strftime("%Y-%m-%d"),
                    "score": (msg.get("likes") or {}).get("total", 0),
                    "snippet": (msg.get("body") or "")[:300],
                    "permalink": "",
                    "source": "stocktwits",
                    "pre_label": pre_label,
                })

            # Stop paginating if we've gone past the start of the date range
            if oldest_in_batch and start_dt and oldest_in_batch < start_dt:
                break

            cursor = (data.get("cursor") or {}).get("max")
            if not cursor:
                break
            params["max"] = cursor

        except Exception as e:
            logger.warning("Stocktwits fetch failed on page: %s", e)
            break

    logger.info(
        "_fetch_stocktwits_messages → %d messages in date range for %s",
        len(messages), ticker,
    )
    return messages


# ---------------------------------------------------------------------------
# LLM sentiment classification
# ---------------------------------------------------------------------------

def _classify_batch(batch: list[dict]) -> list[str]:
    """
    Send a batch of posts to the LLM for sentiment classification.
    Returns labels in the same order as batch.
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
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        results = json.loads(raw)
        label_map = {r["index"]: r["sentiment"] for r in results}
        return [label_map.get(i, "neutral") for i in range(len(batch))]

    except Exception as e:
        logger.warning("LLM sentiment classification failed for batch: %s", e)
        return ["neutral"] * len(batch)


def _classify_all(posts: list[dict]) -> list[str]:
    """
    Classify all posts. Uses Stocktwits pre-label where available to skip
    unnecessary LLM calls. Remaining posts are classified in batches of
    _BATCH_SIZE. Returns labels in original post order.
    """
    pre_assigned: dict[int, str] = {}
    indices_needing_llm: list[int] = []

    for i, post in enumerate(posts):
        pre_label = post.get("pre_label")
        if pre_label in ("bullish", "bearish", "neutral"):
            pre_assigned[i] = pre_label
        else:
            indices_needing_llm.append(i)

    llm_posts = [posts[i] for i in indices_needing_llm]
    llm_labels: list[str] = []
    for batch_start in range(0, len(llm_posts), _BATCH_SIZE):
        batch = llm_posts[batch_start: batch_start + _BATCH_SIZE]
        llm_labels.extend(_classify_batch(batch))

    labels: list[str] = []
    llm_idx = 0
    for i in range(len(posts)):
        if i in pre_assigned:
            labels.append(pre_assigned[i])
        else:
            labels.append(llm_labels[llm_idx] if llm_idx < len(llm_labels) else "neutral")
            llm_idx += 1

    return labels


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def analyze_reddit_sentiment(state: AgentState) -> AgentState:
    """
    Fetch posts from Reddit and Stocktwits, classify sentiment, and aggregate.
    Writes sentiment_summary (with per-source breakdown) and sentiment_posts.
    Writes sentiment_error on failure; both summary and posts remain None.
    """
    ticker = state.get("ticker", "")
    company_name = state.get("company_name", ticker)
    start_date = state.get("start_date", "")
    end_date = state.get("end_date", "")

    try:
        reddit_posts = _fetch_reddit_posts(ticker, company_name, start_date, end_date)
        stocktwits_messages = _fetch_stocktwits_messages(ticker, start_date, end_date)
        all_posts = reddit_posts + stocktwits_messages

        if not all_posts:
            logger.info("analyze_reddit_sentiment: no posts found for %s", ticker)
            return {
                "sentiment_summary": {
                    "total_posts_analyzed": 0,
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "neutral_count": 0,
                    "bullish_percentage": 0.0,
                    "bearish_percentage": 0.0,
                    "neutral_percentage": 0.0,
                    "subreddits_searched": _SUBREDDITS.split("+"),
                    "sources": {
                        "reddit": {"posts": 0, "bullish": 0, "bearish": 0, "neutral": 0},
                        "stocktwits": {"posts": 0, "bullish": 0, "bearish": 0, "neutral": 0},
                    },
                },
                "sentiment_posts": [],
                "sentiment_error": None,
            }

        labels = _classify_all(all_posts)

        sentiment_posts = []
        for post, label in zip(all_posts, labels):
            sentiment_posts.append({
                "title": post["title"],
                "subreddit": post["subreddit"],
                "date": post["date"],
                "score": post["score"],
                "sentiment_label": label,
                "snippet": post["snippet"],
                "source": post["source"],
                "permalink": post.get("permalink", ""),
            })

        total = len(sentiment_posts)
        bullish = sum(1 for p in sentiment_posts if p["sentiment_label"] == "bullish")
        bearish = sum(1 for p in sentiment_posts if p["sentiment_label"] == "bearish")
        neutral = total - bullish - bearish

        reddit_classified = [p for p in sentiment_posts if p["source"] == "reddit"]
        st_classified = [p for p in sentiment_posts if p["source"] == "stocktwits"]

        summary = {
            "total_posts_analyzed": total,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "bullish_percentage": round(bullish / total * 100, 1),
            "bearish_percentage": round(bearish / total * 100, 1),
            "neutral_percentage": round(neutral / total * 100, 1),
            "subreddits_searched": _SUBREDDITS.split("+"),
            "sources": {
                "reddit": {
                    "posts": len(reddit_classified),
                    "bullish": sum(1 for p in reddit_classified if p["sentiment_label"] == "bullish"),
                    "bearish": sum(1 for p in reddit_classified if p["sentiment_label"] == "bearish"),
                    "neutral": sum(1 for p in reddit_classified if p["sentiment_label"] == "neutral"),
                },
                "stocktwits": {
                    "posts": len(st_classified),
                    "bullish": sum(1 for p in st_classified if p["sentiment_label"] == "bullish"),
                    "bearish": sum(1 for p in st_classified if p["sentiment_label"] == "bearish"),
                    "neutral": sum(1 for p in st_classified if p["sentiment_label"] == "neutral"),
                },
            },
        }

        logger.info(
            "analyze_reddit_sentiment: %d posts (reddit=%d, stocktwits=%d) "
            "— %d bullish / %d bearish / %d neutral",
            total, len(reddit_classified), len(st_classified), bullish, bearish, neutral,
        )

        return {
            "sentiment_summary": summary,
            "sentiment_posts": sentiment_posts,
            "sentiment_error": None,
        }

    except Exception as e:
        logger.error("analyze_reddit_sentiment failed: %s", e)
        return {
            "sentiment_summary": None,
            "sentiment_posts": None,
            "sentiment_error": str(e),
        }
