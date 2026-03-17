"""
Node 5: News Retriever

Reads:  ticker, company_name, start_date, end_date, user_config,
        include_current_snapshot
Writes: news_articles, news_source_used, news_error

Two-layer architecture, tried in order:
  1. NewsAPI — REST API, date-filtered, structured JSON response.
               Key sourced from user_config["newsapi_key"] first,
               then NEWSAPI_KEY env var. Falls back to Layer 2 if the
               key is absent, the tier doesn't cover the date range
               (free tier: last 30 days only), or the call fails.
  2. Google News RSS — No API key needed. Constructs a search URL and
                       parses the RSS feed with feedparser.

If include_current_snapshot is True, a second fetch for the last 7 days
is appended to news_articles alongside the historical set.

Each article dict contains:
  title, source_name, published_date (ISO string), url, snippet
"""

import logging
import os
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import yfinance as yf
from newsapi import NewsApiClient

from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)

_MAX_ARTICLES = 10
_CURRENT_SNAPSHOT_DAYS = 7
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_newsapi_key(user_config: dict) -> str | None:
    return user_config.get("newsapi_key") or os.getenv("NEWSAPI_KEY")


def _get_sector_keywords(ticker: str) -> list[str]:
    """
    Fetch sector/industry from yfinance and return relevant macro search terms.
    Returns an empty list on failure or when no useful terms are known.
    Cached implicitly by yfinance's requests session within a single process.
    """
    try:
        info = yf.Ticker(ticker).info
        industry = (info.get("industry") or "").lower()
        sector = (info.get("sector") or "").lower()

        keywords = []
        if any(w in industry for w in ["shipping", "tanker", "marine"]):
            keywords = ["tanker rates", "shipping sanctions", "Red Sea shipping"]
        elif any(w in industry for w in ["oil", "gas", "energy"]):
            keywords = ["oil prices", "energy sector"]
        elif any(w in industry for w in ["semiconductor", "chip"]):
            keywords = ["semiconductor supply", "chip demand"]
        elif any(w in sector for w in ["technology"]):
            keywords = ["tech sector"]
        elif any(w in sector for w in ["financial"]):
            keywords = ["interest rates", "banking sector"]

        return keywords
    except Exception as e:
        logger.debug("sector keyword lookup failed for %s: %s", ticker, e)
        return []


def _build_query(ticker: str, company_name: str) -> str:
    """Build a search query that matches ticker OR company name."""
    parts = []
    if ticker:
        parts.append(ticker)
    if company_name and company_name.lower() != ticker.lower():
        parts.append(f'"{company_name}"')
    return " OR ".join(parts) if parts else ticker



def _parse_newsapi_articles(articles: list) -> list:
    """Normalise NewsAPI article dicts to the project schema."""
    result = []
    for a in articles:
        published = a.get("publishedAt", "")
        if published:
            # Strip trailing 'Z' and reformat to YYYY-MM-DD
            published = published[:10]
        result.append({
            "title": a.get("title") or "",
            "source_name": (a.get("source") or {}).get("name") or "",
            "published_date": published,
            "url": a.get("url") or "",
            "snippet": (a.get("description") or a.get("content") or "")[:300],
        })
    return result


def _fetch_newsapi(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> list | None:
    """
    Query the NewsAPI 'everything' endpoint.
    Returns a normalised article list on success, None on failure or empty.
    """
    try:
        client = NewsApiClient(api_key=api_key)
        query = _build_query(ticker, company_name)
        response = client.get_everything(
            q=query,
            from_param=start_date,
            to=end_date,
            language="en",
            sort_by="relevancy",
            page_size=_MAX_ARTICLES,
        )
        if response.get("status") != "ok":
            logger.warning("NewsAPI returned status: %s", response.get("status"))
            return None

        articles = response.get("articles") or []
        if not articles:
            logger.info("NewsAPI returned 0 articles for %s", query)
            return None

        return _parse_newsapi_articles(articles)

    except Exception as e:
        logger.warning("NewsAPI call failed: %s", e)
        return None


def _fetch_google_rss(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    extra_terms: list[str] | None = None,
) -> list | None:
    """
    Query Google News RSS.
    Date filtering is best-effort: Google RSS doesn't support exact range
    filtering so we filter parsed entries by published date.
    extra_terms: optional sector/macro keywords appended to broaden context.
    Returns a normalised article list on success, None on failure or empty.
    """
    try:
        query = _build_query(ticker, company_name)
        if extra_terms:
            query = f"{query} OR {extra_terms[0]}"
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
        for entry in feed.entries[:_MAX_ARTICLES * 2]:  # parse extra, filter below
            # feedparser normalises published_parsed to a time.struct_time in UTC
            published_dt = None
            if entry.get("published_parsed"):
                published_dt = datetime(*entry.published_parsed[:6])

            # Date-range filter
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
# Node function
# ---------------------------------------------------------------------------

def retrieve_news(state: AgentState) -> AgentState:
    """
    Fetch news articles for the given ticker and date range.
    Tries NewsAPI first; falls back to Google News RSS.
    Appends current-snapshot articles if include_current_snapshot is True.
    """
    ticker = state.get("ticker", "")
    company_name = state.get("company_name", ticker)
    start_date = state.get("start_date", "")
    end_date = state.get("end_date", "")
    user_config = state.get("user_config") or {}
    include_current = state.get("include_current_snapshot", False)

    try:
        api_key = _get_newsapi_key(user_config)
        articles = None
        source_used = "none"

        # Layer 1 — NewsAPI
        if api_key:
            articles = _fetch_newsapi(ticker, company_name, start_date, end_date, api_key)
            if articles is not None:
                source_used = "newsapi"
                logger.info("retrieve_news [newsapi] → %d articles", len(articles))

        # Layer 2 — Google News RSS fallback (enriched with sector context)
        if articles is None:
            sector_terms = _get_sector_keywords(ticker)
            articles = _fetch_google_rss(ticker, company_name, start_date, end_date, extra_terms=sector_terms)
            if articles is not None:
                source_used = "google_rss"
                logger.info("retrieve_news [google_rss] → %d articles", len(articles))

        if articles is None:
            articles = []
            logger.warning("retrieve_news: both sources returned no articles")

        # Current snapshot — append last 7 days if requested
        if include_current:
            today = datetime.now().strftime("%Y-%m-%d")
            snapshot_start = (datetime.now() - timedelta(days=_CURRENT_SNAPSHOT_DAYS)).strftime("%Y-%m-%d")
            snapshot_articles = None

            if api_key:
                snapshot_articles = _fetch_newsapi(ticker, company_name, snapshot_start, today, api_key)
            if snapshot_articles is None:
                snapshot_articles = _fetch_google_rss(ticker, company_name, snapshot_start, today)

            if snapshot_articles:
                # Avoid duplicates by URL
                existing_urls = {a["url"] for a in articles}
                for a in snapshot_articles:
                    if a["url"] not in existing_urls:
                        articles.append(a)
                logger.info(
                    "retrieve_news: appended %d current-snapshot articles",
                    len(snapshot_articles),
                )

        return {
            **state,
            "news_articles": articles or None,
            "news_source_used": source_used,
            "news_error": None,
        }

    except Exception as e:
        logger.error("retrieve_news failed: %s", e)
        return {
            **state,
            "news_articles": None,
            "news_source_used": "none",
            "news_error": str(e),
        }
