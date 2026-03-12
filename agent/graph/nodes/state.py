"""
AgentState: the single shared data structure for the Stock Insight Agent.

LangGraph passes this dict between every node in the graph. Each node reads
the fields it needs and writes the fields it owns. No node should write to
a field owned by a different node (except to inspect it for routing).

Design principle: one field per piece of data, typed explicitly. This is
deliberately different from the old `messages: List[BaseMessage]` pattern,
which hid data inside an untyped conversation history. The new approach makes
every data dependency visible in the type signature.
"""

from typing import Optional, Required
from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state for the Stock Insight Agent LangGraph workflow.

    `total=False` means every field is optional at the TypedDict level —
    LangGraph will not complain if a field hasn't been set yet. Only the two
    fields marked `Required` must be present before the graph starts; all
    others are written by individual nodes during execution.

    Field ownership is noted in each comment so it's immediately clear which
    node is responsible for writing — and therefore which node to look at if
    the value is wrong.
    """

    # -------------------------------------------------------------------------
    # Input fields — set by Chainlit BEFORE the graph starts running.
    # These are the only two fields that must always be present.
    # -------------------------------------------------------------------------

    user_message: Required[str]
    # The raw text the user typed in the chat UI.
    # Read by: Nodes 1 (intent), 2 (ticker), 3 (date).
    # Never modified after it is set.

    user_config: Required[dict]
    # Optional API keys the user has provided, e.g. {"newsapi_key": "abc"}.
    # Empty dict {} if the user provided nothing (so nodes can always do
    # `state["user_config"].get("newsapi_key")` safely without a KeyError).
    # Read by: any node that calls an external paid API.
    # Never modified after it is set.

    # -------------------------------------------------------------------------
    # Node 1: Intent Classifier
    # Reads: user_message
    # -------------------------------------------------------------------------

    intent: str
    # Classifies what the user wants. One of:
    #   "stock_analysis"  — what happened with a stock during a time period
    #   "options_view"    — options chain / put-call ratio
    #   "chart_request"   — user primarily wants a visual chart
    #   "general_lookup"  — basic price performance, no deep analysis
    #   "unknown"         — doesn't relate to stocks, or too ambiguous
    # Read by: conditional routing edges after Node 3.

    chart_requested: bool
    # True if the user mentioned a chart/graph/visualization anywhere in their
    # message. Tracked separately from `intent` because a user can ask for
    # stock analysis AND a chart at the same time.
    # Read by: conditional edge after Node 9 (Response Synthesizer) to decide
    # whether to run Node 10 (Chart Generator).

    intent_error: Optional[str]
    # Written by Node 1 if classification fails. None on success.
    # CLAUDE.md rule: all nodes must write to their *_error field on failure.
    # Node 9 checks all error fields to report which steps failed.

    # -------------------------------------------------------------------------
    # Node 2: Ticker Resolver
    # Reads: user_message
    # -------------------------------------------------------------------------

    ticker: str
    # Resolved stock ticker symbol, e.g. "NVDA".
    # Read by: Nodes 4, 5, 6, 7, 8, 10.

    company_name: str
    # Human-readable company name, e.g. "NVIDIA".
    # Read by: Node 9 to produce natural-sounding responses instead of
    # repeating the raw ticker symbol.

    ticker_error: Optional[str]
    # Written by Node 2 if the ticker cannot be resolved. None on success.

    # -------------------------------------------------------------------------
    # Node 3: Date Parser
    # Reads: user_message, ticker
    # -------------------------------------------------------------------------

    start_date: str
    # ISO-format date string, e.g. "2024-06-01".
    # Read by: Nodes 4, 5, 6, 7.

    end_date: str
    # ISO-format date string, e.g. "2024-07-01".
    # Read by: Nodes 4, 5, 6, 7.

    date_context: str
    # Human-readable description of the time period, e.g. "around Q2 2024
    # earnings" or "last 3 weeks". Used by Node 9 to write a natural response.
    # Read by: Node 9.

    date_missing: bool
    # True if no date range could be extracted from the user's message.
    # When true, the routing edge after Node 3 skips all data nodes and goes
    # directly to Node 9, which asks the user to specify a time period.
    # Read by: routing edge after Node 3.

    include_current_snapshot: bool
    # True if the query has both a historical component ("around Q2 earnings")
    # AND a request for current market conditions ("what's happening now?").
    # When true, Nodes 5 and 8 fetch additional current-market data on top
    # of the historical range.
    # Read by: Nodes 5 (News Retriever), 8 (Options Analyzer).

    date_error: Optional[str]
    # Written by Node 3 if date parsing fails entirely. None on success.

    # -------------------------------------------------------------------------
    # Node 4: Price Data Fetcher
    # Reads: ticker, start_date, end_date
    # -------------------------------------------------------------------------

    price_data: Optional[dict]
    # Dictionary containing:
    #   ticker, start_date, end_date,
    #   open_price, close_price, high_price, low_price,
    #   total_volume, percent_change, price_change,
    #   daily_prices (list of daily OHLCV dicts),
    #   source ("yfinance" or "alpha_vantage")
    # None if retrieval failed.
    # Read by: Node 9 (narrative), Node 10 (chart — needs daily_prices).

    volume_anomaly: Optional[dict]
    # Dictionary containing:
    #   average_daily_volume (during the queried period),
    #   historical_average_volume (90-day baseline),
    #   anomaly_ratio (period avg / historical avg),
    #   is_anomalous (True if anomaly_ratio > 1.5)
    # Anomaly threshold of 1.5x is a proxy for unusual trading activity.
    # None if retrieval failed.
    # Read by: Node 9.

    price_error: Optional[str]
    # Written by Node 4 if both yfinance and Alpha Vantage fail. None on
    # success. Node 9 checks this to note unavailability in the response.

    # -------------------------------------------------------------------------
    # Node 5: News Retriever
    # Reads: ticker, company_name, start_date, end_date, user_config,
    #         include_current_snapshot
    # -------------------------------------------------------------------------

    news_articles: Optional[list]
    # List of article dicts, each containing:
    #   title, source_name, published_date, url, snippet
    # None if retrieval failed or returned nothing.
    # Read by: Node 9.

    news_source_used: Optional[str]
    # Which news provider returned data: "newsapi", "google_rss", or "none".
    # Separate from news_error so Node 9 can cite the source even when the
    # call succeeded.
    # Read by: Node 9.

    news_error: Optional[str]
    # Written by Node 5 on failure. None on success.

    # -------------------------------------------------------------------------
    # Node 6: Reddit Sentiment Analyzer
    # Reads: ticker, company_name, start_date, end_date
    # -------------------------------------------------------------------------

    sentiment_summary: Optional[dict]
    # Aggregate sentiment counts:
    #   total_posts_analyzed, bullish_count, bearish_count, neutral_count,
    #   bullish_percentage, bearish_percentage, neutral_percentage,
    #   subreddits_searched
    # None if retrieval failed.
    # Read by: Node 9.

    sentiment_posts: Optional[list]
    # List of individual post dicts, each containing:
    #   title, subreddit, date, score (upvotes), sentiment_label, snippet
    # None if retrieval failed.
    # Read by: Node 9.

    sentiment_error: Optional[str]
    # Written by Node 6 on failure. None on success.

    # -------------------------------------------------------------------------
    # Node 7: RAG Retriever (SEC Filings via ChromaDB)
    # Reads: ticker, start_date, end_date
    # -------------------------------------------------------------------------

    filing_chunks: Optional[list]
    # List of relevant SEC filing text chunks, each containing:
    #   text, filing_type (e.g. "10-Q"), filing_quarter, filing_date,
    #   chunk_relevance_score
    # None if retrieval failed or no filings found.
    # Read by: Node 9.

    filing_ingested: Optional[bool]
    # True if a new filing was downloaded and embedded during this query
    # (vs retrieved from the existing vector store). Node 9 notes this in
    # the response ("Note: this filing was just processed for the first time").
    # Read by: Node 9.

    filing_error: Optional[str]
    # Written by Node 7 on failure. None on success.

    # -------------------------------------------------------------------------
    # Node 8: Options Analyzer
    # Reads: ticker, include_current_snapshot
    # -------------------------------------------------------------------------

    options_data: Optional[dict]
    # Dictionary containing:
    #   expiration_dates (list), put_call_ratio,
    #   highest_volume_calls (list of strike prices),
    #   highest_volume_puts (list of strike prices),
    #   total_call_volume, total_put_volume,
    #   average_implied_volatility, notable_positions (list)
    # None if retrieval failed.
    # Read by: Node 9.

    options_error: Optional[str]
    # Written by Node 8 on failure. None on success.

    # -------------------------------------------------------------------------
    # Node 9: Response Synthesizer
    # Reads: all data fields above, all *_error fields
    # -------------------------------------------------------------------------

    response_text: Optional[str]
    # The final narrative response with inline source citations.
    # This is what Chainlit displays as the agent's reply.
    # Read by: Chainlit app.py.

    sources_cited: Optional[list]
    # List of source dicts, each containing:
    #   type ("news", "reddit", "filing"), title, url (or reference string)
    # Kept separate from response_text so Chainlit can render them as
    # clickable links below the narrative, not buried inside it.
    # Read by: Chainlit app.py.

    # -------------------------------------------------------------------------
    # Node 10: Chart Generator
    # Reads: ticker, price_data (specifically daily_prices)
    # -------------------------------------------------------------------------

    chart_data: Optional[str]
    # Plotly chart serialized as a JSON string (fig.to_json()).
    # Chainlit detects this field and renders it as an interactive chart.
    # None if chart_requested was False, or if generation failed.
    # Read by: Chainlit app.py.

    chart_error: Optional[str]
    # Written by Node 10 if chart generation fails. None on success.
    # Added post-Step-1 to comply with CLAUDE.md rule: all nodes must
    # write to their *_error state field on failure.
