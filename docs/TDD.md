# Technical Design Document

**Product:** Stock Insight Agent 

**Author:** Ritu Villuri

**Status:** In Progress 

**Last Updated:** March 19, 2026

**Version:** 1.0

---

## Table of Contents

---

## 1. System Architecture Overview

The Stock Insight Agent is a single-agent system built on LangGraph. It uses a directed graph workflow where each processing step is an explicit node, connected by edges that control execution order and conditional routing. The agent coordinates multiple external data sources, synthesizes their outputs through an LLM, and returns a unified response through a Chainlit chat interface.

The system has four layers:

**Interface Layer:** Chainlit provides the chat-based UI. It receives user messages, passes them into the LangGraph workflow, and renders the agent's response. It also handles rich content like Plotly charts and source links.

**Orchestration Layer:** LangGraph manages the workflow. It defines which nodes execute in what order, how state flows between them, and how the agent routes between different analysis paths based on user intent. This is the core of the architecture and the layer that makes this a structured agent rather than a simple chatbot.

**Data Layer:** External APIs and the local ChromaDB vector store provide the raw data. This includes yfinance for price and options data, NewsAPI and Google News RSS for articles, Reddit API for social posts, SEC EDGAR for filings, and ChromaDB for previously embedded documents. Each data source has its own retrieval logic and fallback strategy.

**Intelligence Layer:** The LLM (Groq-hosted LLaMA 3.1 8B by default, or a user-provided model) handles the tasks that require reasoning: intent classification, ticker resolution, sentiment scoring, and multi-source narrative synthesis. The LLM is never the source of factual data. It only processes and synthesizes data retrieved by the data layer.

The key architectural principle is that the orchestration layer controls the flow, the data layer provides the facts, and the intelligence layer provides the reasoning. These three concerns are kept separate so that any component can be modified or replaced without affecting the others. Swapping the LLM provider, adding a new data source, or changing the graph routing does not require rewriting unrelated parts of the system.

---

## 2. LangGraph Workflow Design

The workflow has 10 nodes, each handling a single focused responsibility. The graph uses conditional edges to route user queries through different paths depending on the classified intent.

### Node List

| Node | Name | LLM Call | External API Call |
| --- | --- | --- | --- |
| 1 | Intent Classifier | Yes | No |
| 2 | Ticker Resolver | Conditional | No |
| 3 | Date Parser | Conditional | Conditional |
| 4 | Price Data Fetcher | No | Yes |
| 5 | News Retriever | No | Yes |
| 6 | Reddit Sentiment Analyzer | Yes | Yes |
| 7 | RAG Retriever | No | Yes (ChromaDB, potentially EDGAR and Gemini) |
| 8 | Options Analyzer | No | Yes |
| 9 | Response Synthesizer | Yes | No |
| 10 | Chart Generator | No | No |

### Workflow Routing

```nix
User Message
     |
[Intent Classifier] (Node 1)
     |
[Ticker Resolver] (Node 2)
     |
[Date Parser] (Node 3)
     |
     |--- intent = "stock_analysis" ----→ [Price Fetcher] (4) ──────────────────┐
     |                                    Send() fan-out:                       │
     |                                    ├─→ [News Retriever] (5) ──────────┐  │
     |                                    └─→ [Reddit Sentiment] (6) ────────┘  │
     |                                    [RAG Retriever] (7) ─────────────────┘
     |                                           │
     |                                    All results merge into state
     |                                           │
     |                                    [Response Synthesizer] (9)
     |                                           |
     |                                    chart_requested?
     |                                      Yes → [Chart Generator] (10) → END
     |                                      No  → END
     |
     |--- intent = "options_view" ------→ [Options Analyzer] (8)
     |                                           |
     |                                    [Response Synthesizer] (9)
     |                                           |
     |                                        → END
     |
     |--- intent = "chart_request" -----→ [Price Fetcher] (4)
     |                                           |
     |                                    [Chart Generator] (10)
     |                                           |
     |                                        → END
     |
     |--- intent = "general_lookup" ----→ [Price Fetcher] (4)
     |                                           |
     |                                    [Response Synthesizer] (9)
     |                                           |
     |                                        → END
     |
     |--- intent = "unknown" -----------→ [Response Synthesizer] (9)
                                          (generates clarification message)
                                                 |
                                              → END
```

### Key Design Properties

For the `stock_analysis` path, Nodes 4 through 7 are independent of each other. None requires the output of another to do its work. They all read ticker and date range from state. This means they can execute in parallel, reducing total response time.

The Intent Classifier, Ticker Resolver, and Date Parser always run in sequence at the start regardless of intent. Every path needs a ticker, and most need a date range. These three nodes form the common prefix of the graph.

The Response Synthesizer (Node 9) is reused across multiple paths. It adapts based on what data is available in state. For `stock_analysis`, it receives rich multi-source data. For `general_lookup`, it receives only price data. For `unknown`, it receives nothing and generates a clarification message.

---

## 3. State Schema

The state is the single data structure that flows through the entire graph. Every node reads from it and writes to it. In implementation, this becomes a Pydantic BaseModel with typed, validated fields.

```nix
StockInsightState:

  # --- Set by Chainlit before the graph starts ---
  user_message: str
      What the user typed. Set once, never modified.
      Read by: Intent Classifier, Ticker Resolver, Date Parser

  user_config: dict
      API keys the user has optionally provided.
      Example: {"openai_key": "sk-...", "newsapi_key": "abc123"}
      Empty dict if no keys provided.
      Read by: Any node that calls an external API

  response_depth: str
      Chat profile selected by the user. Either "quick" or "deep".
      "quick" (default): concise summary, llm_synthesizer (max_tokens=1024)
      "deep": structured analyst brief with markdown sections, llm_synthesizer_deep (max_tokens=2048)
      Not Required — nodes use state.get("response_depth", "quick") as fallback.
      Read by: Node 9

  # --- Set by Node 1: Intent Classifier ---
  intent: str
      One of: "stock_analysis", "options_view", "chart_request",
      "general_lookup", "unknown"
      Read by: Router edges after Node 3

  chart_requested: bool
      Whether the user asked for a chart or visualization.
      Can be true alongside any intent.
      Read by: Conditional edge after Response Synthesizer

  # --- Set by Node 2: Ticker Resolver ---
  ticker: str
      Resolved stock ticker symbol. Example: "NVDA"
      Read by: Nodes 4, 5, 6, 7, 8, 10

  company_name: str
      The human-readable company name. Example: "NVIDIA"
      Read by: Node 9 (for natural language response)

  # --- Set by Node 3: Date Parser ---
  start_date: str
      ISO format date. Example: "2024-06-15"
      Read by: Nodes 4, 5, 6, 7

  end_date: str
      ISO format date. Example: "2024-07-15"
      Read by: Nodes 4, 5, 6, 7

  date_context: str
      Human-readable description of the period.
      Example: "around Q2 2024 earnings" or "last 3 weeks"
      Read by: Node 9 (for natural language response)

  date_missing: bool
      True if no date range could be determined from the message.
      Read by: Router edge after Node 3 (routes to Response Synthesizer
      to ask user for clarification)

  include_current_snapshot: bool
      True if the user's query includes both a historical component
      and a request for current market conditions.
      Read by: Nodes 5, 8

  # --- Set by Node 4: Price Data Fetcher ---
  price_data: dict
      Contains: ticker, start_date, end_date, open_price, close_price,
      high_price, low_price, total_volume, percent_change, price_change,
      daily_prices (list of daily OHLCV data), source
      Read by: Nodes 9, 10

  volume_anomaly: dict
      Contains: average_daily_volume, historical_average_volume,
      anomaly_ratio (period avg / historical avg), is_anomalous (bool,
      true if anomaly_ratio > 1.5, threshold is configurable)
      Read by: Node 9

  price_error: str or None
      Error message if price retrieval failed. None if successful.
      Read by: Node 9

  analyst_data: dict or None
      Analyst consensus from yfinance. Contains: mean_target, high_target,
      low_target, num_analysts, strong_buy, buy, hold, sell, strong_sell.
      None if unavailable or retrieval failed.
      Read by: Node 9

  short_interest: dict or None
      Short selling data from yfinance. Contains: short_percent_of_float,
      short_ratio (days to cover), shares_short, shares_short_prior_month.
      None if unavailable.
      Read by: Node 9

  next_earnings_date: str or None
      ISO date of the next scheduled earnings release. Example: "2026-05-28"
      None if unavailable.
      Read by: Nodes 8, 9

  days_until_earnings: int or None
      Number of calendar days from today until next_earnings_date.
      Negative values indicate the date has passed.
      None if next_earnings_date is None.
      Read by: Nodes 8, 9

  # --- Set by Node 5: News Retriever ---
  news_articles: list[dict]
      Each article contains: title, source_name, published_date,
      url, snippet
      Read by: Node 9

  news_source_used: str
      Which provider returned the data. "newsapi", "google_rss", or "none"
      Read by: Node 9

  news_error: str or None
      Error message if news retrieval failed. None if successful.
      Read by: Node 9

  # --- Set by Node 6: Reddit Sentiment Analyzer ---
  sentiment_summary: dict
      Contains: total_posts_analyzed, bullish_count, bearish_count,
      neutral_count, bullish_percentage, bearish_percentage,
      neutral_percentage, subreddits_searched
      Read by: Node 9

  sentiment_posts: list[dict]
      Each post contains: title, subreddit, date, score (upvotes),
      sentiment_label, snippet
      Read by: Node 9

  sentiment_error: str or None
      Error message if sentiment retrieval failed. None if successful.
      Read by: Node 9

  # --- Set by Node 7: RAG Retriever ---
  filing_chunks: list[dict]
      Each chunk contains: text, filing_type (10-Q, 10-K, 8-K, transcript),
      filing_quarter, filing_date, chunk_relevance_score
      Read by: Node 9

  filing_ingested: bool
      Whether a new filing was ingested during this query
      (vs retrieved from existing vector store).
      Read by: Node 9 (to note "first-time processing" if relevant)

  filing_error: str or None
      Error message if RAG retrieval failed. None if successful.
      Read by: Node 9

  # --- Set by Node 8: Options Analyzer ---
  options_data: dict
      Contains: expiration_dates (list), put_call_ratio,
      highest_volume_calls (list of strikes),
      highest_volume_puts (list of strikes),
      total_call_volume, total_put_volume,
      average_implied_volatility, notable_positions (list)
      Read by: Node 9

  options_error: str or None
      Error message if options retrieval failed. None if successful.
      Read by: Node 9

  # --- Set by Node 9: Response Synthesizer ---
  response_text: str
      The final narrative response with source citations.
      Read by: Chainlit (to display to user)

  sources_cited: list[dict]
      Each source contains: type (news, reddit, filing),
      title, url or reference
      Read by: Chainlit (to display as clickable links below response)

  synthesizer_error: str or None
      Error message if synthesis failed. None if successful.
      Read by: Chainlit (to display fallback message)

  # --- Set by Node 10: Chart Generator ---
  chart_data: str or None
      Plotly JSON payload for inline chart rendering.
      None if no chart was generated.
      Read by: Chainlit (to render Plotly chart)

  chart_error: str or None
      Error message if chart generation failed. None if successful.
      Read by: Chainlit (to display fallback message)
```

### Design Notes

Every retrieval node has a corresponding error field. When a node fails, it writes None to its data field and writes an error message to its error field. The Response Synthesizer checks all error fields and transparently communicates which data sources were unavailable.

The state separates raw data from the synthesized response. Nodes 4 through 8 write raw data. Node 9 reads all of it and produces the narrative. Changing how the response is written only requires modifying Node 9's prompt.

The `user_config` field flows through the entire graph but is never modified. Every node that calls an external API checks for a relevant premium key.

The `sources_cited` field is separate from `response_text` to allow Chainlit to render source links as clickable elements below the main response.

The `date_missing` field enables the graph to route to the Response Synthesizer for a clarification prompt when no date range can be determined from the user's message.

The `include_current_snapshot` field enables the comparative analysis use case where the user wants both historical and current data in a single query.

---

## 4. Node Specifications

### Node 1: Intent Classifier

**Reads:** `user_message` **Writes:** `intent`, `chart_requested` **LLM call:** Yes **External API call:** No

The node sends the user's message to the LLM with a system prompt that instructs it to classify the message into one of the five intent categories. The LLM returns a structured JSON response with the intent and a boolean for whether a chart was mentioned.

Classification criteria:

- `stock_analysis`: the user is asking about what happened with a stock during a time period, especially around events. Keywords: "what happened," "how did it do," "around earnings," "why did it move"
- `options_view`: the user is asking about current options positioning. Keywords: "options chain," "put/call," "options for," "how are options"
- `chart_request`: the user primarily wants a visual chart. Keywords: "show me a chart," "graph," "visualize," "plot"
- `general_lookup`: the user wants basic price performance without deep analysis. Keywords: "how did X perform," "what's the price," "stock data for"
- `unknown`: the message doesn't relate to stock analysis or is too ambiguous to classify

The `chart_requested` flag is set independently of intent. A user can say "What happened with NVIDIA around earnings? Show me a chart too." That's `intent: stock_analysis` with `chart_requested: true`.

### Node 2: Ticker Resolver

**Reads:** `user_message` **Writes:** `ticker`, `company_name` **LLM call:** Conditional (only if lookup table doesn't match) **External API call:** No

The node first checks the user's message against a hardcoded lookup table of common companies and their tickers. This table covers the most frequently discussed stocks: NVIDIA/NVDA, Apple/AAPL, Tesla/TSLA, Microsoft/MSFT, Google/GOOGL, Amazon/AMZN, Meta/META, GameStop/GME, AMD/AMD, and others. If the user typed a ticker symbol directly (all caps, 1-5 characters), it passes through as-is.

If the lookup table doesn't match, the node falls back to an LLM call to extract the company name and resolve the ticker.

### Node 3: Date Parser

**Reads:** `user_message`, `ticker` **Writes:** `start_date`, `end_date`, `date_context`, `date_missing`, `include_current_snapshot` **LLM call:** Conditional (only for complex date expressions) **External API call:** Conditional (yfinance for earnings date lookup)

Handles several categories of date expressions:

Simple relative ranges use deterministic parsing with no LLM: "last week," "last 3 months," "past 30 days." These are regex pattern matching and date arithmetic.

Earnings-relative ranges require an external lookup: "around Q2 2024 earnings" triggers a yfinance earnings calendar lookup to find the actual earnings date, then creates a window of 14 days before through 7 days after.

Complex or ambiguous expressions fall back to the LLM: "during the COVID crash" or "when the tariffs were announced."

If the user's message includes both a historical component and a current component ("How did NVIDIA do around Q2 earnings last year? And what's happening with it right now?"), the node sets `include_current_snapshot` to true in addition to the historical date range.

If no date range or event can be determined from the message, the node sets `date_missing` to true. The graph routes to the Response Synthesizer which asks the user for a time period or event.

### Node 4: Price Data Fetcher

**Reads:** `ticker`, `start_date`, `end_date` **Writes:** `price_data`, `volume_anomaly`, `analyst_data`, `short_interest`, `next_earnings_date`, `days_until_earnings`, `price_error` **LLM call:** No **External API call:** Yes (yfinance, Alpha Vantage as fallback)

Calls yfinance to get daily OHLCV data for the ticker and date range. If yfinance fails, falls back to Alpha Vantage.

Calculates: opening price (first day's open), closing price (last day's close), period high, period low, total volume, percent change, and dollar change.

Calculates volume anomaly by comparing the average daily volume during the queried period against the stock's 90-day historical average volume. If the period's average exceeds the historical average by more than 1.5x, `is_anomalous` is set to true. This threshold is configurable and serves as a proxy signal for unusual trading activity.

The daily_prices list is preserved in state for the Chart Generator.

Also fetches three enrichment signals from yfinance in the same node pass:

**Analyst data:** `ticker.analyst_price_targets` and `ticker.recommendations_summary` return analyst consensus price targets and buy/hold/sell breakdowns. Written to `analyst_data`. Failures are non-fatal — sets to None and continues.

**Short interest:** `ticker.info` fields `shortPercentOfFloat`, `shortRatio`, `sharesShort`, and `sharesShortPriorMonth`. Written to `short_interest`. Non-fatal on failure.

**Earnings date:** `ticker.calendar` returns the next scheduled earnings date. Calculates `days_until_earnings` from today. Written to `next_earnings_date` and `days_until_earnings`. Non-fatal on failure.

### Node 5: News Retriever

**Reads:** `ticker`, `company_name`, `start_date`, `end_date`, `user_config`, `include_current_snapshot` **Writes:** `news_articles`, `news_source_used`, `news_error` **LLM call:** No **External API call:** Yes (NewsAPI or Google News RSS)

Checks `user_config` for a NewsAPI key. If present, uses the user's key. Otherwise uses the default free developer key.

Queries NewsAPI with the company name and ticker, filtered to the date range. If NewsAPI returns no results (date range older than 30 days on free tier, or query failure), falls back to Google News RSS.

If `include_current_snapshot` is true, the node also fetches current news articles (from the last 7 days) in addition to the historical articles. Both sets are stored in `news_articles` with their dates, allowing the Response Synthesizer to distinguish between historical and current coverage.

### Node 6: Reddit Sentiment Analyzer

**Reads:** `ticker`, `company_name`, `start_date`, `end_date` **Writes:** `sentiment_summary`, `sentiment_posts`, `sentiment_error` **LLM call:** Yes (for sentiment classification) **External API call:** Yes (Reddit API via PRAW)

Uses PRAW to search for posts mentioning the ticker or company name in the default subreddit list (r/wallstreetbets, r/stocks, r/options). The subreddit list is configurable. Filters by the date range and retrieves up to 50 posts, sorted by relevance.

For each retrieved post, sends the text to the LLM for sentiment classification: bullish, bearish, or neutral. Posts are batched (5-10 per LLM call) to minimize API calls.

Aggregates results into `sentiment_summary` with counts, percentages, and subreddits searched. Preserves individual post data in `sentiment_posts`.

### Node 7: RAG Retriever

**Reads:** `ticker`, `start_date`, `end_date`, `user_message` **Writes:** `filing_chunks`, `filing_ingested`, `filing_error` **LLM call:** No **External API call:** Yes (ChromaDB local, potentially SEC EDGAR and Google Gemini)

Queries ChromaDB for existing chunks matching the ticker and date range using metadata filtering combined with semantic search.

If ChromaDB returns relevant results, writes them to `filing_chunks` with `filing_ingested` set to false.

If no results exist and the date range overlaps with an earnings period, triggers on-demand ingestion. The corpus covers four SEC EDGAR document types: 10-K (annual report), 10-Q (quarterly report), 8-K (material event filing — earnings releases, guidance updates, executive changes), and earnings call transcripts (filed as 8-K exhibits). Downloads the document from SEC EDGAR, chunks, embeds via Google Gemini API, stores in ChromaDB with metadata and document-level IDs for deduplication, then queries freshly stored chunks.

Full ingestion workflow documented in Section 6: RAG Pipeline Design.

**Parallel fan-out contract:** Node 7 executes in parallel with Nodes 5 and 6 via LangGraph `Send()`. All three nodes must return **only their owned fields** — never `{**state, ...}`. Returning shared fields (e.g., `user_message`, `ticker`) causes `InvalidUpdateError` when LangGraph merges the parallel branches at Node 9. Node 7 returns exactly: `{"filing_chunks": ..., "filing_ingested": ..., "filing_error": ...}`.

### Node 8: Options Analyzer

**Reads:** `ticker`, `intent`, `include_current_snapshot`, `next_earnings_date`, `days_until_earnings` **Writes:** `options_data`, `options_error` **LLM call:** No **External API call:** Yes (yfinance)

Executes when `intent` is `options_view` or when `include_current_snapshot` is true (to provide current market positioning alongside historical analysis).

Calls yfinance for the options chain at the nearest expiration date. Calculates put/call ratio by volume, identifies strikes with highest volume and open interest, and calculates average implied volatility.

Calculates Greeks (Delta, Gamma, Theta, Vega) using Black-Scholes with yfinance-supplied implied volatility, current underlying price, risk-free rate (approximated from 3-month T-bill), and time to expiry.

Calculates Max Pain: the strike price where total dollar loss for option buyers is maximized at expiry, derived from open interest across all strikes.

Incorporates `days_until_earnings` to flag elevated pre-earnings IV context or expected post-earnings IV crush.

For historical queries without the current snapshot flag, this node does not execute. The volume anomaly data from Node 4 serves as the historical proxy.

### Node 9: Response Synthesizer

**Reads:** All state fields from Nodes 1-8, `response_depth` **Writes:** `response_text`, `sources_cited`, `synthesizer_error` **LLM call:** Yes **External API call:** No

Constructs a prompt providing the LLM with all available data from state and routes between two response modes based on `response_depth`.

**Depth routing:**

- `"quick"` (default — any value other than `"deep"`): concise summary using `llm_synthesizer` (`llama-3.1-8b-instant`, `max_tokens=1024`, `streaming=True`). Covers price action, key news, and sentiment in a single narrative paragraph.
- `"deep"`: structured analyst brief using `llm_synthesizer_deep` (`llama-3.3-70b-versatile`, `max_tokens=2048`, `streaming=True`). Output uses five markdown sections: `## Price Action`, `## News & Catalysts`, `## Market Sentiment`, `## SEC Filings`, `## Options Activity`. If a section's data is unavailable, the heading is still included with an explicit note.

Both modes include a grounding instruction: *"Only reference dates, prices, and events that appear in the DATA block below. If a fact is not in the data, say it is unavailable — do not fill gaps from your training knowledge."* This prevents the LLM from substituting hallucinated dates or prices when data is absent.

Both modes also apply these rules:
- Reference specific data points from the provided state (prices, percentages, dates)
- Cite the source for every factual claim (article title and source, Reddit subreddit, filing type and quarter)
- If any data dimension has an error (check the error fields), explicitly state what was unavailable
- Use `date_context` and `company_name` for natural language framing
- If `include_current_snapshot` is true, present historical and current sections distinctly

Also constructs `sources_cited` by extracting all referenced sources from the data for Chainlit to render as clickable links.

For `unknown` intent or `date_missing`, generates a clarification prompt asking the user for more information.

### Node 10: Chart Generator

**Reads:** `ticker`, `price_data`, `volume_anomaly`, `date_context` **Writes:** `chart_data`, `chart_error` **LLM call:** No **External API call:** No

Generates a TradingView-style Plotly chart from the daily price data in state.

**Visual style:** Dark background (`#0f1117`), up candles `#00c896` (green), down candles `#ff4d6d` (red), matching the UI theme.

**Volume subplot:** Always shown in the bottom 25% of the chart (row heights `[0.75, 0.25]`). Bar colors match candle direction at 40% opacity. This is unconditional — `volume_anomaly.is_anomalous` no longer controls volume visibility.

**20-day SMA overlay:** Plotted as an amber line (`#f0b429`) when `daily_prices` has 20 or more data points. Omitted entirely when fewer than 20 data points exist.

**Chart title:** `f"{ticker} — {date_context}"` using `date_context` from state.

**Hover template:** Shows OHLC prices and humanized volume (e.g., `42.3M`) on mouse-over.

The chart is serialized to Plotly's JSON format and written to `chart_data`. Chainlit deserializes the JSON and renders it as an interactive inline chart.

---

## 5. Data Source Integration

This section documents how each external data source is called, what data comes back, and how failures are handled.

### yfinance (Stock Price Data and Options)

Used by: Node 4 (Price Data Fetcher), Node 8 (Options Analyzer), Node 3 (Date Parser for earnings date lookup)

For historical prices, a Ticker object is created with the stock symbol, then its history method is called with start and end dates. Returns a pandas DataFrame with columns for Open, High, Low, Close, and Volume, indexed by date.

For current options chains, the options property returns available expiration dates, and option_chain returns two DataFrames (calls and puts) with strike price, last price, bid, ask, volume, open interest, and implied volatility.

For earnings dates, the earnings_dates property returns a DataFrame of upcoming and recent earnings dates.

Failure modes: Empty DataFrames for invalid tickers or date ranges with no trading days. Connection errors if Yahoo Finance is unreachable. The node checks for empty data before proceeding and catches connection errors explicitly.

### Alpha Vantage (Fallback Price Data)

Used by: Node 4 (Price Data Fetcher, only when yfinance fails)

REST API with URL-based queries. The response is JSON containing a dictionary of dates mapped to price data. Column names are numbered strings ("1. open", "2. high", etc.) that require mapping to clean names.

Free tier allows 5 calls per minute and 500 calls per day.

Failure modes: Rate limit exceeded and invalid ticker errors are returned inside the JSON response. The node checks for the presence of "Time Series (Daily)" before parsing.

### NewsAPI (News Articles)

Used by: Node 5 (News Retriever)

REST API queried via the "everything" endpoint with search terms, date range, language, and sort order. Returns JSON with an array of articles containing title, source name, author, description, url, publishedAt, and content.

Free developer tier: 100 requests per day, articles from the last 30 days only, content truncated to 200 characters.

Failure modes: Status field indicates errors with codes including rateLimited, apiKeyInvalid, and parametersMissing.

### Google News RSS (Fallback News)

Used by: Node 5 (News Retriever, when NewsAPI fails or returns empty)

RSS feed searched by constructing a URL with search terms. Response is XML containing items with title, link, publication date, and source.

No official API, no rate limit documentation, no guaranteed behavior.

Failure modes: Malformed XML, empty results, CAPTCHA redirects on automated access. The node handles XML parsing errors gracefully.

### Reddit API via PRAW (Social Sentiment)

Used by: Node 6 (Reddit Sentiment Analyzer)

PRAW authenticates with a client ID and client secret (free). The search method on subreddit objects finds posts matching a query. Posts contain title, selftext, score, created_utc, num_comments, subreddit name, and permalink.

Reddit's search API does not support precise date filtering natively. Results are filtered by date in application code after retrieval. Historical depth is not guaranteed for very old posts.

Rate limit: 60 requests per minute for authenticated users.

Failure modes: PRAW raises specific exceptions for authentication errors, rate limits, and server errors. Empty search results are not an error.

### SEC EDGAR (Earnings Filings)

Used by: Node 7 (RAG Retriever)

Free REST API, no authentication required. Company lookup by ticker returns the CIK (Central Index Key). Submissions endpoint returns filing lists filtered by type (10-Q, 10-K) and date. Each filing has an accession number and document URL.

Filings are HTML documents requiring HTML tag stripping to extract plain text.

EDGAR requires a User-Agent header with name and email. Courtesy rate limit of 10 requests per second.

Failure modes: 404 for invalid CIK, empty filing lists, timeout on large downloads.

### Google Gemini API (Text Embeddings)

Used by: Node 7 (RAG Retriever)

POST request with text content and model name (text-embedding-004). Returns a vector for each text input.

Free tier: 1,500 requests per day, batch limit of 100 texts per request. A typical filing (100-200 chunks) requires 1-2 API calls.

Failure modes: 429 for rate limit, 400 for oversized input. Per-chunk token limit is approximately 2,048 tokens.

### Groq (LLM Inference)

Used by: Nodes 1, 2, 3 (conditionally), 6, 9

OpenAI-compatible API format, integrated via LangChain's ChatGroq class. Model: llama-3.1-8b-instant.

Free tier rate limits vary based on demand (approximately 30 requests per minute, 6,000 tokens per minute).

Multiple nodes make LLM calls per query, so prompts are kept concise and deterministic logic is preferred over LLM calls where possible.

Failure modes: 429 for rate limiting, 401 for invalid key, 503 for model overload (more common with Groq's free tier during high demand). Retry with backoff for 503 errors.

### ChromaDB (Vector Store)

Used by: Node 7 (RAG Retriever)

Runs in embedded mode within the application process. Stores vectors on disk in a specified directory.

Collections are organized by embedding model. All embeddings use Google Gemini's text-embedding-004, so a single collection handles all documents.

Documents stored with unique IDs based on source (format: `{ticker}-{filing_type}-{period}-chunk-{number}`). Duplicate ID inserts are no-ops, providing automatic deduplication.

Failure modes: Minimal in embedded mode. Disk space exhaustion is the primary risk, unlikely given data volumes.

**Phase 5 migration:** At deployment, ChromaDB transitions from local embedded mode to ChromaDB Cloud free tier (1GB hosted storage). The API is identical — one configuration change. This eliminates local disk dependency on the deployed container and survives container restarts without a mounted persistent volume.

---

## 6. RAG Pipeline Design

The RAG pipeline has two workflows: ingestion (getting documents into the vector store) and retrieval (searching the vector store during a query).

### Ingestion Workflow

Triggered when a user query involves an earnings period for a company whose filing hasn't been processed before.

**Step 1: Filing Discovery.** Query SEC EDGAR for the company's CIK using the ticker, then query the submissions endpoint for the relevant filing type and date range.

**Step 2: Document Download and Cleaning.** Download the filing HTML from EDGAR. Strip HTML tags, remove excessive whitespace, remove boilerplate sections (table of contents, signature pages, exhibit lists). Produce clean plain text. Entirely deterministic, no LLM involved.

**Step 3: Chunking.** Split clean text into chunks of approximately 500-800 tokens. Chunking is paragraph-aware, breaking at natural paragraph or sentence boundaries rather than mid-sentence. Each chunk overlaps with the previous chunk by approximately 100 tokens to preserve information spanning chunk boundaries.

Each chunk is assigned a unique ID: `{ticker}-{filing_type}-{period}-chunk-{number}` (e.g., `NVDA-10Q-2024Q2-chunk-014`). This ID is the deduplication key.

**Step 4: Embedding.** Chunks are sent to the Google Gemini embedding API in batches of up to 100 per request. Each chunk receives a vector representation of its semantic meaning.

**Step 5: Storage.** Each chunk is stored in ChromaDB with three components: the embedding vector, the original text, and metadata (ticker, filing_type, filing_period, filing_date, accession_number, chunk_index). Metadata enables filtered searches.

### Retrieval Workflow

Executes during every query with `stock_analysis` intent.

**Step 1: Query Construction.** The user's original message serves as the search query, potentially refined to extract the core question.

**Step 2: Metadata Filtering.** Narrow the search space by filtering on ticker and filing period before semantic search.

**Step 3: Semantic Search.** The query is embedded using the same Google Gemini model used for storage. ChromaDB computes cosine similarity and returns the top 5 most similar chunks ranked by relevance score.

**Step 4: Result Packaging.** Retrieved chunks are written to state with original text, metadata, and relevance scores.

### Edge Cases

If the query doesn't involve an earnings-related time period, the node skips retrieval and ingestion, writes an empty list to `filing_chunks`.

If the query involves earnings but EDGAR has no matching filing, the node writes the error to `filing_error`.

If ChromaDB has no results but a filing exists on EDGAR, the ingestion workflow is triggered, followed by immediate retrieval from the freshly stored chunks.

---

## 7. Provider Abstraction Layer

Every external service has a default free provider and one or more premium alternatives. The system never requires a user to provide any API key. If a user provides their own key for a service, the agent uses that provider instead.

### Provider Map

| Capability | Default (Free) | Premium (User Provides Key) |
| --- | --- | --- |
| LLM Inference | Groq (LLaMA 3.1 8B) | OpenAI (GPT-4), Anthropic (Claude) |
| Text Embeddings | Google Gemini (text-embedding-004) | Google Gemini (same for all users) |
| News Retrieval | NewsAPI free tier, Google News RSS fallback | NewsAPI with user's key (higher limits, archive access) |
| Social Sentiment | Reddit API (PRAW) | Reddit API (same) |
| Stock Data | yfinance, Alpha Vantage fallback | yfinance (same) |
| Options Data | yfinance | yfinance (same) |
| SEC Filings | EDGAR (free, no key needed) | EDGAR (same) |

### Implementation Pattern

Each node that calls an external service checks the `user_config` dictionary in state. If a relevant key exists, the node initializes the premium provider client. If not, it uses the default.

### Embedding Consistency Constraint

All embeddings must come from the same model. The embedding model is not configurable. All users use Google Gemini for embeddings regardless of other API keys provided. This keeps the vector store consistent and avoids maintaining separate collections per embedding model.

### UI Configuration

The Chainlit settings panel exposes optional text input fields for: OpenAI API Key, Anthropic API Key, and NewsAPI Key. Each field includes a brief explanation of what it unlocks. Keys are stored only in the user's session, not persisted to disk, not logged, and not transmitted anywhere except to the respective API provider.

---

## 8. Error Handling Strategy

### Principle: Fail Gracefully, Never Silently

Every failure must be communicated to the user. The agent never silently omits a data source and presents an incomplete analysis as if it were complete.

### Node-Level Error Handling

Each data retrieval node follows the same pattern:

1. Attempt the primary data source
2. If it fails, attempt the fallback (if one exists)
3. If all sources fail, write None to the data field and a descriptive, user-friendly error message to the corresponding error field
4. Never crash the graph. Never raise an unhandled exception. Always return a valid state update.

Error messages are written for the Response Synthesizer to incorporate into the narrative. Not "HTTPError 429 Too Many Requests" but "Reddit's API rate limit was reached, so sentiment data is unavailable for this query."

### Workflow-Level Error Handling

The graph wraps each node execution in a try/except. On unhandled exceptions, the graph writes a generic error to state and routes to the Response Synthesizer, which generates a response acknowledging the partial failure and listing whatever data was successfully retrieved.

### LLM-Specific Error Handling

**Rate limiting (429) and model overload (503):** Retry once after a 2-second wait. If the retry fails, write the error to state and continue.

**Malformed output:** If the LLM returns unstructured text instead of expected JSON, attempt basic string parsing to extract the relevant information. If that fails, default to safe fallbacks: Intent Classifier defaults to "unknown" (prompting user clarification), Ticker Resolver returns an error asking the user to specify the ticker directly.

**Timeout:** If an LLM call exceeds 30 seconds, cancel and write a timeout error. Prevents a single slow call from blocking the workflow.

### Data Validation

Each retrieval node validates data before writing to state. Price Data Fetcher checks for non-empty DataFrames and positive price values. News Retriever checks that articles have titles and URLs. Reddit Sentiment Analyzer checks that posts have text content. Invalid data is discarded and treated as a partial failure.

---

## 9. UI Integration

### Chat Profiles

Two profiles are presented at session start: **Quick Analysis** and **Deep Dive**. The selected profile maps to `response_depth` in state:

- "Quick Analysis" → `response_depth = "quick"` (default)
- "Deep Dive" → `response_depth = "deep"`

The welcome message adjusts to mention the active mode.

### Message Flow

1. User selects a chat profile and types a message in Chainlit
2. Chainlit's `on_message` handler creates the initial state with `user_message`, `user_config`, `response_depth`, and any persisted context from the previous turn
3. State is streamed through the compiled LangGraph workflow via `graph.astream_events(state, version="v2")`
4. Synthesizer tokens are streamed token-by-token via `on_chat_model_stream` events as they arrive
5. Completed node outputs are collected from `on_chain_end` events into `final_state`
6. Chainlit reads output fields from `final_state` and renders sources and chart after streaming completes

### Rendering Logic

**`response_text` / streaming:** The synthesizer streams tokens directly into a `cl.Message` object. An initial `"Analyzing your query..."` placeholder is shown while the graph runs; `stream_token()` calls overwrite this content as tokens arrive.

**`sources_cited`:** Rendered as a collapsible side panel using `cl.Text(name="📎 View Sources", content=..., display="side")`. Each source shows its icon (📰 news, 💬 reddit, 📄 filing), title, and URL.

**`chart_data`:** If present, the Plotly JSON is deserialized and rendered as an interactive inline chart. The message content is `"📊 Interactive Chart"` so the bubble is not blank.

### Context Persistence

After each turn, `ticker`, `company_name`, `start_date`, `end_date`, and `date_context` are saved to `cl.user_session`. These are injected into `initial_state` on the next turn so follow-up questions resolve correctly without re-parsing.

### Theme

Dark finance aesthetic: `#0f1117` background, `#00c896` green accent, Inter font. Configured via `.chainlit/config.toml` (`default_theme = "dark"`, `cot = "hidden"`) and `public/stylesheet.css`.

### Welcome Message

On chat start, the agent displays a welcome message that mentions the active profile mode and example queries.

---

## 10. System Scope and Boundaries

### Current System Scope

The Stock Insight Agent is a functional MVP and proof of concept. It demonstrates end-to-end AI agent architecture, multi-source data retrieval, RAG integration, and cloud deployment. It is designed for single-user usage, demo scenarios, and portfolio presentation. It is not designed to handle concurrent users, sustained high traffic, or the operational demands of a publicly available service.

### Production Requirements

The following outlines what a production version of this system would require beyond the current MVP.

**Infrastructure:** The MVP runs the entire application in a single Docker container. The web server, agent workflow, and vector store share one process. Production would require separating these components. The Chainlit web server would run independently. ChromaDB would be replaced by a managed vector database (Pinecone, Weaviate Cloud, or Azure AI Search) with backup and replication. A task queue (Celery with Redis or similar) would sit between the web server and workflow engine to handle concurrent requests.

**Authentication and User Management:** The MVP has no user accounts. Production would require user registration, login, session management, and encryption of user-provided API keys at rest.

**Data Persistence:** The MVP stores ChromaDB data on the container's disk, which can be lost during redeployment without a mounted persistent volume. Production would require managed databases with automated backups, replication, and disaster recovery.

**Monitoring and Observability:** The MVP has basic error handling but no centralized logging or performance monitoring. Production would integrate LangSmith or a similar platform for agent workflow tracing, structured logging to a service like Datadog or CloudWatch, uptime monitoring, and error rate alerting.

**Rate Limiting and Cost Management:** The MVP relies on free tier rate limits to naturally throttle usage. Production would require an application-level rate limiting layer and cost monitoring across API providers.

**Content Safety and Legal:** The MVP includes a disclaimer that it does not provide financial advice. Production would require formal terms of service, a privacy policy, and a more robust content safety layer.

**Performance Optimization:** The MVP processes nodes sequentially. Production would execute independent data retrieval nodes in parallel, add caching for frequently queried stocks, and pre-ingest earnings filings for popular companies.

---

## 11. Testing Strategy

Testing is organized into three categories: deterministic tests (unit and integration), LLM evals, and demo reliability checks. Each category serves a different purpose and runs at a different cadence.

### Unit Tests

Unit tests verify that individual nodes produce correct output given known input. Each test creates a mock state, calls the node function, and asserts the output fields are correctly populated. All external dependencies (APIs, LLM, ChromaDB) are mocked to ensure tests are fast, deterministic, and independent of network access.

Representative examples (the actual test suite covers additional edge cases and error conditions for each node):

- **Intent Classifier:** Given "What happened with NVIDIA around Q2 earnings?", assert `intent` is "stock_analysis" and `chart_requested` is false. Given "Show me a chart of Apple", assert `intent` is "chart_request" and `chart_requested` is true. Given "hello", assert `intent` is "unknown".
- **Ticker Resolver:** Given "How did Apple perform?", assert `ticker` is "AAPL". Given "MSFT last week", assert `ticker`is "MSFT". Given "the iPhone company", assert fallback LLM call is triggered.
- **Date Parser:** Given "last 3 weeks", assert `start_date` and `end_date` span 3 weeks. Given "tell me about NVIDIA" with no date, assert `date_missing` is true. Given "around Q2 2024 earnings and what's happening now", assert `include_current_snapshot` is true.
- **Price Data Fetcher:** Given valid ticker and date range with mocked yfinance response, assert `price_data` contains all required fields and `price_error` is None. Given mocked yfinance failure, assert Alpha Vantage fallback is attempted. Given both sources failing, assert `price_error` contains a descriptive message.
- **News Retriever:** Given mocked NewsAPI response, assert `news_articles` contains expected articles with title, URL, and date. Given mocked NewsAPI failure, assert Google News RSS fallback is attempted.
- **Response Synthesizer:** Given state with populated price_data but news_error set, assert `response_text` includes a disclosure about unavailable news data.

### Integration Tests

Integration tests verify that multiple nodes work together correctly through the graph. These test the routing logic, state passing between nodes, and end-to-end data flow.

The CI pipeline runs integration tests with mocked external services to ensure fast, reliable execution. A separate integration test suite can run against live APIs in a development environment to catch real-world issues like API format changes or rate limit behavior.

Representative scenarios:

- **Full stock_analysis path:** Send "What happened with NVIDIA last month?" through the complete workflow with mocked data sources. Verify final state contains populated price_data, news_articles, sentiment_summary, and response_text.
- **Graceful degradation:** Mock NewsAPI and Google News RSS to both fail. Verify the workflow completes, response_text acknowledges missing news data, and all other data dimensions are present.
- **Unknown intent path:** Send "hello" and verify intent is "unknown" and response_text asks for clarification.
- **Missing date path:** Send "tell me about NVIDIA" with no date context. Verify `date_missing` is true and response asks the user for a time period.
- **Chart generation path:** Send "Show me a chart for Tesla" and verify `chart_data` contains valid Plotly JSON.
- **Comparative analysis path:** Send "How did NVIDIA do around Q2 earnings last year? And what's happening now?" Verify `include_current_snapshot` is true and response contains both historical and current sections.

### LLM Evals

Evals run on a dedicated schedule: before major releases, after prompt changes, and periodically during development. They require live LLM calls and are not part of the automated CI pipeline because non-deterministic results would cause intermittent CI failures unrelated to code quality.

### Eval Types

**Exact Match Evals:** For nodes with structured outputs that have clear correct answers.

- Intent classification accuracy across 30+ sample queries. Target: 90%+ correct classification.
- Ticker resolution accuracy across common names, ticker symbols, and edge cases. Target: 95%+ correct.
- Sentiment scoring consistency: run the same 10-15 Reddit posts through the scorer multiple times and verify the majority label is stable across runs.

**LLM-as-a-Judge Evals:** For the Response Synthesizer where output quality is subjective and cannot be evaluated by exact match. A second LLM (or the same LLM with a judge prompt) evaluates the generated response against the raw data provided to the node.

The judge LLM scores each response on:

- **Factual accuracy:** Does the response match the data provided in state? Are prices, percentages, and dates correct?
- **Source attribution:** Does the response cite sources for its claims? Are the citations traceable to specific articles, posts, or filings in the state?
- **Completeness:** Does the response address all available data dimensions? If price data, news, and sentiment were all available, does the response cover all three?
- **Hallucination detection:** Does the response contain any claims not supported by the data in state?
- **Disclosure of gaps:** If a data source was unavailable (error field populated), does the response transparently communicate this?

Each criterion is scored 1-5. Scores are tracked across eval runs to detect regressions when prompts are modified.

**RAG-Specific Evals:** For the RAG Retriever node, using metrics from the Ragas evaluation framework:

- **Context Precision:** Of the chunks retrieved from ChromaDB, what percentage were relevant to the query?
- **Context Recall:** Of all relevant chunks in the vector store for the query, what percentage were retrieved?
- **Faithfulness:** Does the Response Synthesizer's output accurately represent what the retrieved filing chunks say?
- **Answer Relevancy:** Does the final response address the specific question the user asked about the filing?

### Eval Tooling

LangSmith (free tier) is the primary eval platform, providing test dataset management, automated eval runs, result tracking over time, and regression detection. Ragas integrates with LangSmith for RAG-specific metrics. Custom eval scripts supplement these tools for project-specific evaluation criteria.

### Demo Reliability Checks

These are manual verification procedures run before interviews or portfolio presentations.

- **Golden path queries:** A set of 5-10 known-good queries exercising the primary use cases, run against the live deployed system. Examples: "What happened with NVIDIA around Q2 2024 earnings?", "What was going on with GME during the WallStreetBets rally?", "What does the options chain look like for Tesla right now?", "How did Apple perform over the last 3 weeks?"
- **Response quality review:** Manual review of each golden path response for accuracy, source citations, and absence of hallucinated claims.
- **Failure recovery verification:** Intentionally disconnect one data source and verify the agent still produces a useful response with remaining sources and discloses what is missing.