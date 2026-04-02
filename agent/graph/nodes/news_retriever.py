"""
Node 5: News Retriever

Reads:  ticker, company_name, start_date, end_date, user_config,
        include_current_snapshot
Writes: news_articles, news_source_used, news_error

Three-layer architecture, tried in order:
  1. Finnhub — REST API, ticker-filtered, date-ranged. Key from
               user_config["finnhub_key"] or FINNHUB_API_KEY env var.
               Free tier: 60 calls/minute, ~2 years of history.
               Falls back to Layer 2 on missing key, empty result, or failure.
  2. You.com Search API — Web search index with date-range filtering via
               freshness parameter (YYYY-MM-DDtoYYYY-MM-DD). Key from
               YOUCOM_API_KEY env var. Free tier: $100 credits.
               Returns results.news array. Falls back to Layer 3 on missing
               key or failure.
  3. Google News RSS — No API key needed. Best-effort date filtering via
               post-parse range check. Emergency fallback only.

If include_current_snapshot is True, a second fetch for the last 7 days
is appended to news_articles alongside the historical set (deduped by URL).

Each article dict contains:
  title, source_name, published_date (ISO string YYYY-MM-DD), url, snippet
"""

import logging
import os
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import requests

from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)

_MAX_ARTICLES = 10
_CURRENT_SNAPSHOT_DAYS = 7
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_YOUCOM_SEARCH_URL = "https://ydc-index.io/v1/search"


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _get_finnhub_key(user_config: dict) -> str | None:
    return user_config.get("finnhub_key") or os.getenv("FINNHUB_API_KEY")


def _get_youcom_key(user_config: dict) -> str | None:
    return user_config.get("youcom_api_key") or os.getenv("YOUCOM_API_KEY")


# ---------------------------------------------------------------------------
# Query builder (shared)
# ---------------------------------------------------------------------------

def _build_query(ticker: str, company_name: str) -> str:
    """Build a search query that matches ticker OR company name (for RSS)."""
    parts = []
    if ticker:
        parts.append(ticker)
    if company_name and company_name.lower() != ticker.lower():
        parts.append(f'"{company_name}"')
    return " OR ".join(parts) if parts else ticker


# ---------------------------------------------------------------------------
# Layer 1: Finnhub
# ---------------------------------------------------------------------------

def _fetch_finnhub(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> list | None:
    """
    Query Finnhub /company-news endpoint.
    Returns a normalised article list on success, None on failure or empty.
    """
    try:
        resp = requests.get(
            _FINNHUB_NEWS_URL,
            params={
                "symbol": ticker,
                "from": start_date,
                "to": end_date,
                "token": api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Finnhub returned HTTP %s", resp.status_code)
            return None

        data = resp.json()
        if not isinstance(data, list) or not data:
            logger.info("Finnhub returned 0 articles for %s", ticker)
            return None

        articles = []
        for item in data[:_MAX_ARTICLES]:
            # datetime is a Unix timestamp
            try:
                published = datetime.fromtimestamp(item["datetime"], tz=datetime.now().astimezone().tzinfo).strftime("%Y-%m-%d")
            except (KeyError, TypeError, OSError):
                published = ""

            articles.append({
                "title": item.get("headline") or "",
                "source_name": item.get("source") or "",
                "published_date": published,
                "url": item.get("url") or "",
                "snippet": (item.get("summary") or "")[:300],
            })

        return articles if articles else None

    except Exception as e:
        logger.warning("Finnhub fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Layer 2: You.com Search API
# ---------------------------------------------------------------------------

def _fetch_youcom(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> list | None:
    """
    Query the You.com Search API for news articles in the given date range.

    Uses the freshness parameter (YYYY-MM-DDtoYYYY-MM-DD) for server-side
    date filtering. Returns results.news from the response.
    """
    name_part = f'"{company_name}"' if company_name and company_name.lower() != ticker.lower() else ""
    query = f"{ticker} {name_part} stock news".strip()
    freshness = f"{start_date}to{end_date}"

    try:
        resp = requests.get(
            _YOUCOM_SEARCH_URL,
            headers={"X-API-Key": api_key},
            params={
                "query": query,
                "freshness": freshness,
                "count": _MAX_ARTICLES,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("You.com returned HTTP %s", resp.status_code)
            return None

        news_items = (resp.json().get("results") or {}).get("news") or []
        if not news_items:
            logger.info("You.com returned 0 news results for %s", ticker)
            return None

        articles = []
        for item in news_items:
            # page_age is ISO 8601 datetime — trim to YYYY-MM-DD
            page_age = item.get("page_age") or ""
            published_date = page_age[:10] if len(page_age) >= 10 else ""

            articles.append({
                "title": item.get("title") or "",
                "source_name": item.get("url", "").split("/")[2] if item.get("url") else "",
                "published_date": published_date,
                "url": item.get("url") or "",
                "snippet": (item.get("description") or "")[:300],
            })

        logger.info("You.com → %d news articles for %s", len(articles), ticker)
        return articles if articles else None

    except Exception as e:
        logger.warning("You.com fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Layer 3: Google News RSS (emergency fallback)
# ---------------------------------------------------------------------------

def _fetch_google_rss(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
) -> list | None:
    """
    Query Google News RSS. Date filtering is best-effort (post-parse range check).
    Returns a normalised article list on success, None on failure or empty.
    """
    try:
        query = _build_query(ticker, company_name)
        url = _GOOGLE_NEWS_RSS.format(query=quote_plus(query))
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            logger.warning("Google RSS feed parse error: %s", feed.bozo_exception)
            return None

        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
        except ValueError:
            start_dt = None
            end_dt = None

        articles = []
        total_entries = len(feed.entries[:_MAX_ARTICLES * 2])
        skipped_out_of_range = 0

        for entry in feed.entries[:_MAX_ARTICLES * 2]:
            published_dt = None
            if entry.get("published_parsed"):
                published_dt = datetime(*entry.published_parsed[:6])

            if start_dt and end_dt and published_dt:
                if not (start_dt <= published_dt <= end_dt):
                    continue

            published_str = published_dt.strftime("%Y-%m-%d") if published_dt else ""

            articles.append({
                "title": entry.get("title") or "",
                "source_name": (entry.get("source") or {}).get("title") or "Google News",
                "published_date": published_str,
                "url": entry.get("link") or "",
                "snippet": (entry.get("summary") or "")[:300],
            })

            if len(articles) >= _MAX_ARTICLES:
                break

        if not articles:
            logger.info("Google RSS returned 0 articles in range for %s", query)
            return None

        return articles

    except Exception as e:
        logger.warning("Google RSS fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Relevance filter — keep only articles mentioning the stock
# ---------------------------------------------------------------------------

def _filter_relevant_articles(
    articles: list,
    ticker: str,
    company_name: str,
) -> list:
    """
    Remove articles that don't mention the stock in their title or snippet.

    Finnhub's /company-news endpoint tags articles at the sector level, not
    strictly at the company level — so queries for NVDA regularly return
    articles about Tesla, Palantir, Camping World, etc.  This filter ensures
    only articles with at least one mention of the ticker OR the company name
    (case-insensitive) pass through.

    We check title first (higher signal) then snippet (lower signal but still
    acceptable — many articles mention the company briefly in the lede).
    Returns the filtered list; never returns None.
    """
    if not articles:
        return []

    ticker_lower = ticker.lower()
    # Strip common suffixes for partial match: "NVIDIA Corporation" → also match "NVIDIA"
    company_lower = company_name.lower() if company_name else ticker_lower
    # Use the first word of the company name as a shorter match anchor
    # e.g. "Alphabet (Google)" → "alphabet", "NVIDIA" → "nvidia"
    company_primary = company_lower.split()[0] if company_lower else ticker_lower

    relevant = []
    for article in articles:
        title = (article.get("title") or "").lower()
        snippet = (article.get("snippet") or "").lower()
        haystack = title + " " + snippet
        if ticker_lower in haystack or company_primary in haystack:
            relevant.append(article)

    return relevant


# ---------------------------------------------------------------------------
# Shared fetch orchestrator
# ---------------------------------------------------------------------------

def _fetch_articles(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    finnhub_key: str | None,
    youcom_key: str | None,
) -> tuple[list | None, str]:
    """
    Try each layer in order. Returns (articles, source_label).
    """
    if finnhub_key:
        articles = _fetch_finnhub(ticker, start_date, end_date, finnhub_key)
        if articles is not None:
            logger.info("_fetch_articles [finnhub] → %d articles", len(articles))
            return articles, "finnhub"

    if youcom_key:
        articles = _fetch_youcom(ticker, company_name, start_date, end_date, youcom_key)
        if articles is not None:
            logger.info("_fetch_articles [youcom] → %d articles", len(articles))
            return articles, "youcom"

    articles = _fetch_google_rss(ticker, company_name, start_date, end_date)
    if articles is not None:
        logger.info("_fetch_articles [google_rss] → %d articles", len(articles))
        return articles, "google_rss"

    return None, "none"


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def retrieve_news(state: AgentState) -> AgentState:
    """
    Fetch news articles for the given ticker and date range.
    Tries Finnhub → FMP → Google RSS in order.
    Appends current-snapshot articles if include_current_snapshot is True.
    """
    ticker = state.get("ticker", "")
    company_name = state.get("company_name", ticker)
    start_date = state.get("start_date", "")
    end_date = state.get("end_date", "")
    user_config = state.get("user_config") or {}
    include_current = state.get("include_current_snapshot", False)

    try:
        finnhub_key = _get_finnhub_key(user_config)
        youcom_key = _get_youcom_key(user_config)

        articles, source_used = _fetch_articles(
            ticker, company_name, start_date, end_date, finnhub_key, youcom_key
        )

        if articles is None:
            articles = []
            logger.warning("retrieve_news: all sources returned no articles")

        # Filter to only articles that mention this stock
        articles = _filter_relevant_articles(articles, ticker, company_name)
        logger.info("retrieve_news: %d relevant articles after filter", len(articles))

        # Current snapshot — append last 7 days if requested
        if include_current:
            today = datetime.now().strftime("%Y-%m-%d")
            snapshot_start = (datetime.now() - timedelta(days=_CURRENT_SNAPSHOT_DAYS)).strftime("%Y-%m-%d")

            snapshot_articles, _ = _fetch_articles(
                ticker, company_name, snapshot_start, today, finnhub_key, youcom_key
            )

            if snapshot_articles:
                snapshot_articles = _filter_relevant_articles(
                    snapshot_articles, ticker, company_name
                )
                existing_urls = {a["url"] for a in articles}
                for a in snapshot_articles:
                    if a["url"] not in existing_urls:
                        articles.append(a)
                logger.info(
                    "retrieve_news: appended %d current-snapshot articles after filter",
                    len(snapshot_articles),
                )

        return {
            "news_articles": articles if articles else None,
            "news_source_used": source_used,
            "news_error": None,
        }

    except Exception as e:
        logger.error("retrieve_news failed: %s", e)
        return {
            "news_articles": None,
            "news_source_used": "none",
            "news_error": str(e),
        }
