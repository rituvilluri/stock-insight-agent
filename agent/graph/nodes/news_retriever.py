"""
Node 5: News Retriever

Reads:  ticker, company_name, start_date, end_date, user_config,
        include_current_snapshot
Writes: news_articles, news_source_used, news_error

Three-layer architecture:
  1. Finnhub + You.com — run in parallel when both keys are present.
     Results are merged and deduplicated by URL (Finnhub order preserved).
     Finnhub: REST API, ticker-filtered, date-ranged. Key from
              user_config["finnhub_key"] or FINNHUB_API_KEY env var.
     You.com: Web search with date freshness filter. Key from
              YOUCOM_API_KEY env var.
  2. Google News RSS — no API key. Emergency fallback when both
     Layer 1 sources are absent or return nothing.

After retrieval, free-domain articles (reuters.com, cnbc.com, apnews.com,
finance.yahoo.com, benzinga.com, marketwatch.com) are enriched with full
article text via Firecrawl. Key from FIRECRAWL_API_KEY env var.
Enrichment is skipped gracefully if no key is set — snippets fall back
to the 600-char source excerpt.

If include_current_snapshot is True, a second parallel fetch for the last
7 days is appended alongside the historical set (deduped by URL).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import requests

from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)

_MAX_ARTICLES = 10
_CURRENT_SNAPSHOT_DAYS = 7
_SNIPPET_MAX_CHARS = 600
_FIRECRAWL_MAX_CHARS = 2000
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_YOUCOM_SEARCH_URL = "https://ydc-index.io/v1/search"
_FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"

_FREE_DOMAINS = {
    "reuters.com",
    "cnbc.com",
    "apnews.com",
    "finance.yahoo.com",
    "benzinga.com",
    "marketwatch.com",
}


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _get_finnhub_key(user_config: dict) -> str | None:
    return user_config.get("finnhub_key") or os.getenv("FINNHUB_API_KEY")


def _get_youcom_key(user_config: dict) -> str | None:
    return user_config.get("youcom_api_key") or os.getenv("YOUCOM_API_KEY")


def _get_firecrawl_key(user_config: dict) -> str | None:
    return user_config.get("firecrawl_key") or os.getenv("FIRECRAWL_API_KEY")


# ---------------------------------------------------------------------------
# Query builder (shared)
# ---------------------------------------------------------------------------

def _build_query(ticker: str, company_name: str) -> str:
    """Build a search query that matches ticker OR company name."""
    parts = []
    if ticker:
        parts.append(ticker)
    if company_name and company_name.lower() != ticker.lower():
        parts.append(f'"{company_name}"')
    return " OR ".join(parts) if parts else ticker


# ---------------------------------------------------------------------------
# Layer 1a: Finnhub
# ---------------------------------------------------------------------------

def _fetch_finnhub(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> list | None:
    """
    Query Finnhub /company-news. Returns normalised articles or None on failure/empty.
    """
    try:
        resp = requests.get(
            _FINNHUB_NEWS_URL,
            params={"symbol": ticker, "from": start_date, "to": end_date, "token": api_key},
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
            try:
                published = datetime.fromtimestamp(
                    item["datetime"],
                    tz=datetime.now().astimezone().tzinfo,
                ).strftime("%Y-%m-%d")
            except (KeyError, TypeError, OSError):
                published = ""

            articles.append({
                "title": item.get("headline") or "",
                "source_name": item.get("source") or "",
                "published_date": published,
                "url": item.get("url") or "",
                "snippet": (item.get("summary") or "")[:_SNIPPET_MAX_CHARS],
            })

        return articles if articles else None

    except Exception as e:
        logger.warning("Finnhub fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Layer 1b: You.com Search API
# ---------------------------------------------------------------------------

def _fetch_youcom(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> list | None:
    """
    Query the You.com Search API with date freshness filter.
    Returns normalised articles or None on failure/empty.
    """
    name_part = f'"{company_name}"' if company_name and company_name.lower() != ticker.lower() else ""
    query = f"{ticker} {name_part} stock news".strip()
    freshness = f"{start_date}to{end_date}"

    try:
        resp = requests.get(
            _YOUCOM_SEARCH_URL,
            headers={"X-API-Key": api_key},
            params={"query": query, "freshness": freshness, "count": _MAX_ARTICLES},
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
            page_age = item.get("page_age") or ""
            published_date = page_age[:10] if len(page_age) >= 10 else ""
            articles.append({
                "title": item.get("title") or "",
                "source_name": item.get("url", "").split("/")[2] if item.get("url") else "",
                "published_date": published_date,
                "url": item.get("url") or "",
                "snippet": (item.get("description") or "")[:_SNIPPET_MAX_CHARS],
            })

        logger.info("You.com → %d news articles for %s", len(articles), ticker)
        return articles if articles else None

    except Exception as e:
        logger.warning("You.com fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Layer 2: Google News RSS (emergency fallback)
# ---------------------------------------------------------------------------

def _fetch_google_rss(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
) -> list | None:
    """
    Query Google News RSS. Date filtering is best-effort (post-parse check).
    Returns normalised articles or None on failure/empty.
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
        for entry in feed.entries[:_MAX_ARTICLES * 2]:
            published_dt = None
            if entry.get("published_parsed"):
                published_dt = datetime(*entry.published_parsed[:6])

            if start_dt and end_dt and published_dt:
                if not (start_dt <= published_dt <= end_dt):
                    continue

            articles.append({
                "title": entry.get("title") or "",
                "source_name": (entry.get("source") or {}).get("title") or "Google News",
                "published_date": published_dt.strftime("%Y-%m-%d") if published_dt else "",
                "url": entry.get("link") or "",
                "snippet": (entry.get("summary") or "")[:_SNIPPET_MAX_CHARS],
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
# Relevance filter
# ---------------------------------------------------------------------------

def _filter_relevant_articles(
    articles: list,
    ticker: str,
    company_name: str,
) -> list:
    """
    Remove articles that don't mention the stock in their title or snippet.
    Finnhub tags articles at sector level, so off-topic articles regularly appear.
    """
    if not articles:
        return []

    ticker_lower = ticker.lower()
    company_lower = company_name.lower() if company_name else ticker_lower
    company_primary = company_lower.split()[0] if company_lower else ticker_lower

    return [
        a for a in articles
        if ticker_lower in (a.get("title") or "").lower() + " " + (a.get("snippet") or "").lower()
        or company_primary in (a.get("title") or "").lower() + " " + (a.get("snippet") or "").lower()
    ]


# ---------------------------------------------------------------------------
# Firecrawl enrichment (free-domain articles only)
# ---------------------------------------------------------------------------

def _is_free_domain(url: str) -> bool:
    """Return True if the article URL belongs to a free-access domain."""
    return any(domain in url for domain in _FREE_DOMAINS)


def _enrich_with_firecrawl(article: dict, api_key: str) -> dict:
    """
    Fetch full article text via Firecrawl for a single free-domain article.
    Returns the article with snippet replaced by full markdown text.
    Returns the original article unchanged on any failure.
    """
    url = article.get("url", "")
    if not url or not _is_free_domain(url):
        return article

    try:
        resp = requests.post(
            _FIRECRAWL_SCRAPE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown"]},
            timeout=15,
        )
        if resp.status_code != 200:
            return article

        markdown = (resp.json().get("data") or {}).get("markdown") or ""
        if markdown:
            return {**article, "snippet": markdown[:_FIRECRAWL_MAX_CHARS]}
        return article

    except Exception as e:
        logger.warning("Firecrawl enrichment failed for %s: %s", url, e)
        return article


def _enrich_articles(articles: list, api_key: str | None) -> list:
    """
    Enrich free-domain articles with full text via Firecrawl (parallel).
    Skips enrichment entirely if no API key is configured.
    """
    if not api_key or not articles:
        return articles

    with ThreadPoolExecutor(max_workers=5) as executor:
        enriched = list(executor.map(lambda a: _enrich_with_firecrawl(a, api_key), articles))

    enriched_count = sum(
        1 for orig, enr in zip(articles, enriched)
        if orig.get("snippet") != enr.get("snippet")
    )
    if enriched_count:
        logger.info("_enrich_articles: enriched %d/%d articles via Firecrawl", enriched_count, len(articles))

    return enriched


# ---------------------------------------------------------------------------
# Parallel fetch orchestrator
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
    Run Finnhub and You.com in parallel. Merge results, deduplicate by URL
    (Finnhub order preserved). Fall back to Google RSS if both return nothing.
    Returns (articles, source_label).
    """
    finnhub_articles = None
    youcom_articles = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        fh_future = (
            executor.submit(_fetch_finnhub, ticker, start_date, end_date, finnhub_key)
            if finnhub_key else None
        )
        ydc_future = (
            executor.submit(_fetch_youcom, ticker, company_name, start_date, end_date, youcom_key)
            if youcom_key else None
        )

        if fh_future:
            try:
                finnhub_articles = fh_future.result()
            except Exception as e:
                logger.warning("Finnhub parallel fetch error: %s", e)

        if ydc_future:
            try:
                youcom_articles = ydc_future.result()
            except Exception as e:
                logger.warning("You.com parallel fetch error: %s", e)

    # Merge, dedup by URL
    merged = []
    seen_urls: set[str] = set()
    sources_used = []

    for source_name, articles in [("finnhub", finnhub_articles), ("youcom", youcom_articles)]:
        if articles:
            sources_used.append(source_name)
            for a in articles:
                url = a.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    merged.append(a)

    if merged:
        label = "+".join(sources_used)
        logger.info("_fetch_articles [%s] → %d merged articles", label, len(merged))
        return merged, label

    # Fall back to Google RSS
    rss = _fetch_google_rss(ticker, company_name, start_date, end_date)
    if rss:
        logger.info("_fetch_articles [google_rss] → %d articles", len(rss))
        return rss, "google_rss"

    return None, "none"


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def retrieve_news(state: AgentState) -> AgentState:
    """
    Fetch news articles for the given ticker and date range.
    Runs Finnhub + You.com in parallel, falls back to Google RSS.
    Enriches free-domain articles with full text via Firecrawl.
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
        firecrawl_key = _get_firecrawl_key(user_config)

        articles, source_used = _fetch_articles(
            ticker, company_name, start_date, end_date, finnhub_key, youcom_key
        )

        if articles is None:
            articles = []
            logger.warning("retrieve_news: all sources returned no articles")

        articles = _filter_relevant_articles(articles, ticker, company_name)
        articles = _enrich_articles(articles, firecrawl_key)
        logger.info("retrieve_news: %d relevant articles after filter+enrich", len(articles))

        # Current snapshot — append last 7 days if requested
        if include_current:
            today = datetime.now().strftime("%Y-%m-%d")
            snapshot_start = (datetime.now() - timedelta(days=_CURRENT_SNAPSHOT_DAYS)).strftime("%Y-%m-%d")

            snapshot_articles, _ = _fetch_articles(
                ticker, company_name, snapshot_start, today, finnhub_key, youcom_key
            )

            if snapshot_articles:
                snapshot_articles = _filter_relevant_articles(snapshot_articles, ticker, company_name)
                snapshot_articles = _enrich_articles(snapshot_articles, firecrawl_key)
                existing_urls = {a["url"] for a in articles}
                for a in snapshot_articles:
                    if a["url"] not in existing_urls:
                        articles.append(a)
                logger.info(
                    "retrieve_news: appended %d current-snapshot articles",
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
