"""
Node 9: Response Synthesizer

Reads:  All state fields set by Nodes 1-8
Writes: response_text, sources_cited, synthesizer_error

Constructs a detailed LLM prompt from all available data in state and
generates a natural-language narrative response with source citations.

Three prompt paths:

1. Clarification path — intent="unknown" OR date_missing=True
   The node skips data synthesis and asks the user for clarification.
   No sources_cited; response_text contains the clarification question.

2. Normal synthesis path — all other intents with date range present
   Provides the LLM with every non-None data field (price, news,
   sentiment, filings, options, volume anomaly).  Error fields are
   also included so the LLM can disclose which sources were unavailable.

3. Current snapshot mode (include_current_snapshot=True, inside path 2)
   The prompt instructs the LLM to separate historical vs. current
   data and present them as distinct sections without implying that
   historical patterns will repeat.

Why build the prompt in Python rather than using a template file?
All the conditional inclusion logic (skip news if news_error is set,
add snapshot section only when include_current_snapshot is true) is
easier to reason about as plain Python conditionals than as a complex
Jinja template.  The trade-off is verbosity here vs. simplicity.

sources_cited is built deterministically from the raw data — no LLM
involvement.  This keeps citations reliable and makes them testable
without an LLM call.
"""

import logging
from typing import Optional

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_synthesizer, llm_synthesizer_deep


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
    Includes only the data sections that are non-None in state.
    Error fields are included when set so the LLM can disclose failures.
    """
    ticker = state.get("ticker", "")
    company = state.get("company_name", ticker)
    date_context = state.get("date_context", "the requested period")
    intent = state.get("intent", "stock_analysis")
    include_snapshot = state.get("include_current_snapshot", False)
    response_depth = state.get("response_depth", "quick")

    sections = []

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------
    price_data = state.get("price_data")
    price_error = state.get("price_error")

    if price_data:
        dp = price_data
        sections.append(
            f"## Price Data (source: {dp.get('source', 'unknown')})\n"
            f"Ticker: {dp.get('ticker', ticker)}\n"
            f"Period: {dp.get('start_date')} to {dp.get('end_date')}\n"
            f"Open: ${dp.get('open_price')}\n"
            f"Close: ${dp.get('close_price')}\n"
            f"High: ${dp.get('high_price')}\n"
            f"Low: ${dp.get('low_price')}\n"
            f"Price change: ${dp.get('price_change')} ({dp.get('percent_change')}%)\n"
            f"Total volume: {_fmt_volume(dp.get('total_volume'))}"
        )
    elif price_error:
        sections.append(
            f"## Price Data\n"
            f"Unavailable — {price_error}"
        )

    # ------------------------------------------------------------------
    # Volume anomaly
    # ------------------------------------------------------------------
    volume_anomaly = state.get("volume_anomaly")
    if volume_anomaly and volume_anomaly.get("is_anomalous"):
        ratio = volume_anomaly.get("anomaly_ratio", "")
        ratio_str = f"{ratio:.1f}x" if isinstance(ratio, (int, float)) else str(ratio)
        sections.append(
            f"## Volume Anomaly\n"
            f"Unusual trading volume detected: {ratio_str} above the 90-day average.\n"
            f"Period average daily volume: {_fmt_volume(volume_anomaly.get('average_daily_volume'))}\n"
            f"Historical average (90-day): {_fmt_volume(volume_anomaly.get('historical_average_volume'))}"
        )

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------
    news_articles = state.get("news_articles")
    news_error = state.get("news_error")
    news_source = state.get("news_source_used", "")

    if news_articles:
        article_lines = []
        for i, a in enumerate(news_articles[:5], 1):  # cap at 5 to stay within token budget
            pub = a.get("published_date", "")
            snippet = a.get("snippet", "")[:200]
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
    # SEC filings (RAG)
    # ------------------------------------------------------------------
    filing_chunks = state.get("filing_chunks")
    filing_error = state.get("filing_error")
    filing_ingested = state.get("filing_ingested", False)

    if filing_chunks:
        chunk_lines = []
        for i, c in enumerate(filing_chunks[:3], 1):  # cap at 3 chunks
            text_snippet = c.get("text", "")[:300]
            chunk_lines.append(
                f"{i}. {c.get('filing_type')} {c.get('filing_quarter')} "
                f"({c.get('filing_date')}):\n   {text_snippet}"
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
            f"Total call volume: {od.get('total_call_volume'):,}\n"
            f"Total put volume: {od.get('total_put_volume'):,}\n"
            f"Average implied volatility: {od.get('average_implied_volatility')}"
        )
    elif options_error:
        sections.append(f"## Options Data\nUnavailable — {options_error}")

    # ------------------------------------------------------------------
    # Assemble final prompt
    # ------------------------------------------------------------------
    data_block = "\n\n".join(sections) if sections else "No data was retrieved for this query."

    snapshot_instruction = ""
    if include_snapshot:
        snapshot_instruction = (
            "\n\nThe user asked for both historical analysis AND current conditions. "
            "Present these as two clearly labelled sections: 'Historical Analysis' "
            "and 'Current Conditions'. Do NOT imply that historical patterns will repeat."
        )

    structure_instruction = (
        "Structure the response: price action first, then contributing factors "
        "(news, sentiment, filings), then trading activity context. "
        "If any data dimension shows 'Unavailable', explicitly state this in the response."
    )

    grounding_instruction = (
        "Only reference dates, prices, and events that appear in the DATA block below. "
        "If a fact is not in the data, say it is unavailable — do not fill gaps from "
        "your training knowledge."
    )

    if response_depth == "deep":
        prompt = (
            f"You are a stock analysis assistant. Generate a comprehensive analyst brief "
            f"for {company} ({ticker}) covering {date_context}.\n\n"
            f"Rules:\n"
            f"- {grounding_instruction}\n"
            f"- Reference specific data points (prices, percentages, dates) from the data below\n"
            f"- If a section's data is unavailable, state this explicitly under that heading\n"
            f"{snapshot_instruction}\n\n"
            f"Structure your response with these exact markdown sections:\n"
            f"## Price Action\n## News & Catalysts\n## Market Sentiment\n"
            f"## SEC Filings\n## Options Activity\n\n"
            f"--- DATA ---\n{data_block}\n--- END DATA ---\n\n"
            f"Generate the analyst brief now:"
        )
    else:
        # quick (default) — any value other than "deep" uses this path
        prompt = (
            f"You are a stock analysis assistant. Generate a concise, factual response "
            f"about {company} ({ticker}) for {date_context}.\n\n"
            f"Rules:\n"
            f"- {grounding_instruction}\n"
            f"- Reference specific data points (prices, percentages, dates) from the data below\n"
            f"- Cite the source for every factual claim\n"
            f"- {structure_instruction}"
            f"{snapshot_instruction}\n\n"
            f"--- DATA ---\n{data_block}\n--- END DATA ---\n\n"
            f"Generate the analysis response now:"
        )

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
        response_depth = state.get("response_depth", "quick")
        llm = llm_synthesizer_deep if response_depth == "deep" else llm_synthesizer
        response = llm.invoke(prompt)
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
