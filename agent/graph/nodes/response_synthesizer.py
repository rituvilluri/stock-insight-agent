"""
Node 9: Response Synthesizer

Reads:  All state fields set by Nodes 1-8
Writes: response_text, sources_cited, synthesizer_context, synthesizer_error

Two execution paths:

1. Clarification path — intent="unknown" OR date_missing=True
   Skips LLM synthesis; returns a clarification question directly.
   sources_cited = []; synthesizer_context not set.

2. Synthesis path — all other intents with a resolved date range
   Builds a data block from every non-None state field (price, volume,
   analyst consensus, short interest, earnings timing, news, sentiment,
   filings, options) and sends it to Gemini 2.5 Flash with thinking
   enabled. The prompt instructs the model to reason about causality
   (connect price action to catalysts) rather than recite data points.

sources_cited is built deterministically from raw state — no LLM
involvement, making citations testable and reliable.
"""

import logging
from typing import Optional

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_synthesizer


def _fmt_volume(vol) -> str:
    """Format a raw share volume number into a human-readable string."""
    if vol is None:
        return "N/A"
    try:
        vol = float(vol)
    except (TypeError, ValueError):
        return str(vol)
    if vol >= 1_000_000_000:
        return f"{vol / 1_000_000_000:.2f}B"
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.2f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return f"{vol:,.0f}"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder helpers
# ---------------------------------------------------------------------------

def _build_clarification_prompt(state: AgentState) -> str:
    """Return a clarification question for unknown intent or missing date."""
    intent = state.get("intent", "unknown")
    date_missing = state.get("date_missing", False)
    user_message = state.get("user_message", "")

    if date_missing:
        ticker = state.get("ticker", "")
        company = state.get("company_name", ticker)
        if company:
            return (
                f"I can look up information about {company}, but I need a time "
                "period to focus on. Could you specify a date range or event? "
                "For example: \"last 3 weeks\", \"Q2 2024\", \"around earnings last quarter\"."
            )
        return (
            "I need a time period to look up stock information. "
            "Could you specify a date range or event? "
            "For example: \"last 3 weeks\", \"Q2 2024\", \"around earnings last quarter\"."
        )

    # intent == "unknown"
    return (
        "I'm a stock analysis assistant. I can help you with:\n"
        "- Stock price analysis and performance for a specific period\n"
        "- News and sentiment analysis around a stock\n"
        "- Options chain data and put/call ratios\n"
        "- Interactive candlestick charts\n\n"
        f"Your message: \"{user_message}\"\n\n"
        "Could you rephrase your question with a specific stock ticker or "
        "company name and a time period?"
    )


def _build_synthesis_prompt(state: AgentState) -> str:
    """
    Build the full data synthesis prompt.
    Sections are included conditionally — only when data is present.
    The prompt instructs the model to reason (connect data, explain causality)
    rather than recite (list numbers under forced headers).
    """
    ticker = state.get("ticker", "")
    company = state.get("company_name", ticker)
    date_context = state.get("date_context", "the requested period")
    include_snapshot = state.get("include_current_snapshot", False)

    sections = []

    # ------------------------------------------------------------------
    # Price data + volume anomaly + analyst consensus + short interest
    # These are grouped as "market data" so the model sees them together.
    # ------------------------------------------------------------------
    price_data = state.get("price_data")
    price_error = state.get("price_error")

    if price_data:
        dp = price_data
        price_lines = [
            f"Ticker: {dp.get('ticker', ticker)} | Period: {dp.get('start_date')} to {dp.get('end_date')}",
            f"Open: ${dp.get('open_price')}  Close: ${dp.get('close_price')}  "
            f"High: ${dp.get('high_price')}  Low: ${dp.get('low_price')}",
            f"Change: ${dp.get('price_change')} ({dp.get('percent_change')}%)  "
            f"Total volume: {_fmt_volume(dp.get('total_volume'))}",
        ]

        # Daily prices — include all days so model can reason about intra-period moves
        daily = dp.get("daily_prices", [])
        if daily:
            price_lines.append("Daily closes:")
            for d in daily:
                price_lines.append(
                    f"  {d['date']}: O${d['open']} H${d['high']} L${d['low']} C${d['close']} Vol:{_fmt_volume(d['volume'])}"
                )

        # Volume anomaly — inline with price data
        volume_anomaly = state.get("volume_anomaly")
        if volume_anomaly:
            ratio = volume_anomaly.get("anomaly_ratio")
            if ratio is not None:
                ratio_str = f"{ratio:.2f}x"
                anomaly_flag = " ⚠ ELEVATED" if volume_anomaly.get("is_anomalous") else ""
                price_lines.append(
                    f"Volume vs 90-day baseline: {ratio_str}{anomaly_flag} "
                    f"(period avg {_fmt_volume(volume_anomaly.get('average_daily_volume'))} "
                    f"vs historical {_fmt_volume(volume_anomaly.get('historical_average_volume'))})"
                )

        sections.append("## Price Data (source: yfinance)\n" + "\n".join(price_lines))
    elif price_error:
        sections.append(f"## Price Data\nUnavailable — {price_error}")

    # ------------------------------------------------------------------
    # Analyst consensus + short interest + earnings timing
    # Previously fetched but never included in the prompt — critical gap.
    # ------------------------------------------------------------------
    analyst_data = state.get("analyst_data")
    short_interest = state.get("short_interest")
    next_earnings_date = state.get("next_earnings_date")
    days_until_earnings = state.get("days_until_earnings")

    positioning_lines = []

    if analyst_data:
        ad = analyst_data
        rec_parts = []
        for label, key in [("Strong Buy", "strong_buy"), ("Buy", "buy"), ("Hold", "hold"),
                           ("Sell", "sell"), ("Strong Sell", "strong_sell")]:
            val = ad.get(key)
            if val is not None:
                rec_parts.append(f"{label}: {val}")
        positioning_lines.append(
            f"Analyst consensus ({ad.get('num_analysts', '?')} analysts): "
            f"Mean target ${ad.get('mean_target')}  "
            f"High ${ad.get('high_target')}  Low ${ad.get('low_target')}"
        )
        if rec_parts:
            positioning_lines.append("Recommendations: " + "  ".join(rec_parts))

    if short_interest:
        si = short_interest
        pct = si.get("short_percent_of_float")
        ratio = si.get("short_ratio")
        shares = si.get("shares_short")
        prior = si.get("shares_short_prior_month")
        si_line = "Short interest:"
        if pct is not None:
            si_line += f" {pct:.1%} of float"
        if ratio is not None:
            si_line += f"  Days-to-cover: {ratio}"
        if shares is not None and prior is not None:
            change = shares - prior
            direction = "▲" if change > 0 else "▼"
            si_line += f"  Shares short: {_fmt_volume(shares)} ({direction}{_fmt_volume(abs(change))} vs prior month)"
        positioning_lines.append(si_line)

    if next_earnings_date:
        days_str = f"{days_until_earnings} days" if days_until_earnings is not None else "unknown"
        positioning_lines.append(f"Next earnings: {next_earnings_date} ({days_str} away)")

    if positioning_lines:
        sections.append("## Analyst Positioning\n" + "\n".join(positioning_lines))

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------
    news_articles = state.get("news_articles")
    news_error = state.get("news_error")
    news_source = state.get("news_source_used", "")

    if news_articles:
        article_lines = []
        for i, a in enumerate(news_articles[:8], 1):
            pub = a.get("published_date", "")
            snippet = a.get("snippet", "")[:250]
            article_lines.append(
                f"{i}. [{a.get('title')}] — {a.get('source_name')} ({pub})\n   {snippet}"
            )
        sections.append(
            f"## News Articles (source: {news_source})\n" + "\n".join(article_lines)
        )
    elif news_error:
        sections.append(f"## News Articles\nUnavailable — {news_error}")

    # ------------------------------------------------------------------
    # Reddit sentiment
    # ------------------------------------------------------------------
    sentiment_summary = state.get("sentiment_summary")
    sentiment_error = state.get("sentiment_error")

    if sentiment_summary:
        ss = sentiment_summary
        sections.append(
            f"## Reddit Sentiment\n"
            f"Posts analyzed: {ss.get('total_posts_analyzed', 0)}\n"
            f"Bullish: {ss.get('bullish_percentage', 0):.0f}% | "
            f"Bearish: {ss.get('bearish_percentage', 0):.0f}% | "
            f"Neutral: {ss.get('neutral_percentage', 0):.0f}%\n"
            f"Subreddits: {', '.join(ss.get('subreddits_searched', []))}"
        )
    elif sentiment_error:
        sections.append(f"## Reddit Sentiment\nUnavailable — {sentiment_error}")

    # ------------------------------------------------------------------
    # SEC filings (RAG) — only when date-relevant chunks are present
    # ------------------------------------------------------------------
    filing_chunks = state.get("filing_chunks")
    filing_error = state.get("filing_error")
    filing_ingested = state.get("filing_ingested", False)

    if filing_chunks:
        chunk_lines = []
        for i, c in enumerate(filing_chunks[:3], 1):
            text_snippet = c.get("text", "")[:400]
            score = c.get("chunk_relevance_score", "")
            chunk_lines.append(
                f"{i}. {c.get('filing_type')} {c.get('filing_quarter')} "
                f"(relevance: {score}):\n   {text_snippet}"
            )
        ingested_note = " [newly ingested this query]" if filing_ingested else ""
        sections.append(
            f"## SEC Filing Excerpts{ingested_note}\n" + "\n".join(chunk_lines)
        )
    elif filing_error:
        sections.append(f"## SEC Filings\nUnavailable — {filing_error}")

    # ------------------------------------------------------------------
    # Options data
    # ------------------------------------------------------------------
    options_data = state.get("options_data")
    options_error = state.get("options_error")

    if options_data:
        od = options_data
        sections.append(
            f"## Options Data\n"
            f"Put/call ratio: {od.get('put_call_ratio')}\n"
            f"Total call volume: {od.get('total_call_volume'):,}  "
            f"Total put volume: {od.get('total_put_volume'):,}\n"
            f"Average implied volatility: {od.get('average_implied_volatility')}\n"
            f"Max pain strike: {od.get('max_pain')}"
        )
    elif options_error:
        sections.append(f"## Options Data\nUnavailable — {options_error}")

    # ------------------------------------------------------------------
    # Assemble data block
    # ------------------------------------------------------------------
    data_block = "\n\n".join(sections) if sections else "No data was retrieved for this query."

    snapshot_note = ""
    if include_snapshot:
        snapshot_note = (
            "\n\nNote: The user asked for both a historical analysis AND current conditions. "
            "Label these clearly as separate sections. Do not imply historical patterns will repeat."
        )

    # ------------------------------------------------------------------
    # Prompt — instructs reasoning, not recitation
    # ------------------------------------------------------------------
    prompt = f"""\
You are a senior equity research analyst writing a brief for an informed investor.

Your task: explain what drove {company} ({ticker})'s price action during {date_context}.

Do NOT list data — connect it. A good brief explains causality:
  "Volume spiked on [date] because [news event], triggering a [move]..."
is far more useful than "Volume was [X] on [date]."

RULES:
- Only cite facts present in the DATA block. Do not fill gaps from training knowledge.
- Ground every claim in specific numbers from the data (price, %, date, volume).
- Causality requires evidence: only assert that X caused Y if the DATA block contains
  a news article, filing excerpt, or sentiment signal that links X to Y. If news is
  absent, describe what the price did and note that the catalyst is unconfirmed — do
  not infer a cause from the price action alone or from general knowledge about the company.
- If a data dimension is missing, note it once and move on — don't dwell.
- Lead with the most important insight, not a recitation of open/close.
- Write in clear, professional prose. Avoid bullet lists in the narrative sections.
- When analyst targets, short interest, or earnings timing are available, integrate
  them into the forward-looking context — these are signals, not footnotes.{snapshot_note}

Use these markdown sections (omit any section where data is entirely unavailable):
## Price Action
## News & Catalysts
## Market Sentiment
## SEC Filings
## Options Activity

--- DATA ---
{data_block}
--- END DATA ---

Write the analyst brief now:"""

    return prompt


# ---------------------------------------------------------------------------
# sources_cited builder (deterministic — no LLM)
# ---------------------------------------------------------------------------

def _build_sources_cited(state: AgentState) -> list:
    """
    Build the sources_cited list from all non-None data in state.
    Called only on the normal synthesis path.
    """
    sources = []

    news_articles = state.get("news_articles") or []
    for article in news_articles:
        sources.append({
            "type": "news",
            "title": article.get("title", ""),
            "url": article.get("url", ""),
        })

    sentiment_posts = state.get("sentiment_posts") or []
    for post in sentiment_posts[:3]:  # limit to top 3 Reddit posts
        sources.append({
            "type": "reddit",
            "title": post.get("title", ""),
            "url": f"https://reddit.com{post.get('permalink', '')}",
        })

    filing_chunks = state.get("filing_chunks") or []
    seen_filings = set()
    for chunk in filing_chunks:
        ref = f"{chunk.get('filing_type')} {chunk.get('filing_quarter')} ({chunk.get('filing_date')})"
        if ref not in seen_filings:
            seen_filings.add(ref)
            sources.append({
                "type": "filing",
                "title": ref,
                "url": "",
            })

    return sources


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def synthesize_response(state: AgentState) -> AgentState:
    """
    Generate the final natural-language response from all data in state.
    Writes response_text and sources_cited on success.
    Writes synthesizer_error on failure.
    """
    intent = state.get("intent", "unknown")
    date_missing = state.get("date_missing", False)

    # ------------------------------------------------------------------
    # Path 1: Clarification (no LLM synthesis needed)
    # ------------------------------------------------------------------
    if intent == "unknown" or date_missing:
        clarification = _build_clarification_prompt(state)
        logger.info(
            "synthesize_response → clarification path (intent=%s, date_missing=%s)",
            intent, date_missing,
        )
        return {
            **state,
            "response_text": clarification,
            "sources_cited": [],
            "synthesizer_error": None,
        }

    # ------------------------------------------------------------------
    # Path 2: Normal synthesis (LLM call)
    # ------------------------------------------------------------------
    try:
        prompt = _build_synthesis_prompt(state)
        response = llm_synthesizer.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        sources_cited = _build_sources_cited(state)

        ticker = state.get("ticker", "")
        logger.info(
            "synthesize_response → %s | sources=%d",
            ticker, len(sources_cited),
        )

        return {
            **state,
            "response_text": response_text,
            "sources_cited": sources_cited,
            "synthesizer_context": prompt,
            "synthesizer_error": None,
        }

    except Exception as e:
        logger.error("synthesize_response failed: %s", e)
        return {
            **state,
            "response_text": None,
            "sources_cited": [],
            "synthesizer_error": str(e),
        }
