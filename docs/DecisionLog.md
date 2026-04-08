# Decision Log

**Product:** Stock Insight Agent 

**Author:** Ritu Villuri

**Status:** Active

**Last Updated:** March 29, 2026

---

## Table of Contents

## Decision 1: Agent Framework

**Date:** January 2026 **Status:** Accepted

**Decision:** Migrate from LangChain to LangGraph for agent orchestration.

**Context:** The system requires stateful, multi-step workflows with conditional routing based on classified user intent. The initial implementation used LangChain.

**Options Considered:**

- **LangChain:** Chain-based paradigm. Linear execution without native support for branching, conditional routing, or persistent state across processing steps.
- **LangGraph:** Graph-based orchestration built on LangChain. Explicit nodes, conditional edges, typed state management, and cyclical graph support.
- **CrewAI:** Multi-agent role-based collaboration. Designed for scenarios requiring inter-agent dialogue and negotiation.
- **AutoGen (Microsoft):** Multi-agent conversation framework for collaborative problem-solving through dialogue.

**Choice and Rationale:** LangGraph. The workflow requires intent classification followed by conditional routing to different data retrieval paths, which LangChain's linear chain paradigm doesn't support natively. LangGraph was purpose-built for this pattern. LangChain remains in use under the hood as part of the LangGraph ecosystem. CrewAI and AutoGen solve a different problem (multi-agent collaboration) that this system doesn't require.

**Tradeoffs Accepted:** Steeper learning curve and more upfront code than LangChain's chain abstraction or CrewAI's higher-level configuration. The explicitness of the graph definition is a net benefit for debugging and architectural clarity.

---

## Decision 2: Single-Agent vs. Multi-Agent Architecture

**Date:** January 2026 **Status:** Accepted

**Decision:** Single agent with a multi-node graph.

**Context:** The system coordinates multiple data sources. This could be one agent with many nodes sharing state, or multiple specialized agents communicating through a coordinator.

**Options Considered:**

- **Single agent, multi-node graph:** Shared state, explicit node-to-node data flow.
- **Multi-agent system:** Specialist agents (price, news, sentiment) coordinated by a supervisor agent.

**Choice and Rationale:** Single agent. The data retrieval tasks are independent. No node requires another node's output to do its work. They all write to shared state, and the Response Synthesizer reads from it at the end. Multi-agent coordination would add LLM calls for inter-agent communication (1-3 seconds per coordination step) with no quality benefit, since there is no iterative feedback loop between data sources. The single-agent graph is faster, cheaper, and simpler to debug.

**Tradeoffs Accepted:** Cross-referencing between data dimensions (e.g., correlating a news headline with a sentiment spike) happens in the Response Synthesizer prompt rather than through inter-agent dialogue. If synthesis quality needed significant improvement, structured pre-processing nodes would be a more appropriate solution than a multi-agent architecture.

---

## Decision 3: LLM Provider

**Date:** January 2026 **Status:** Accepted

**Decision:** Groq (LLaMA 3.1 8B) as default, with optional user-configured premium alternatives via provider abstraction.

**Context:** An LLM is needed for intent classification, sentiment scoring, and response synthesis. Zero-cost budget. Development machine (8GB RAM) cannot host models locally.

**Options Considered:**

- **Groq (LLaMA 3.1 8B):** Free tier, fast inference, OpenAI-compatible API.
- **OpenAI (GPT-4o / GPT-4o-mini):** Superior quality, paid. Trial credits limited and temporary.
- **Anthropic (Claude):** High quality, paid with limited free tier.
- **Google Gemini:** Free tier available, but using Gemini for both LLM and embeddings creates a single point of failure across the reasoning engine and RAG pipeline.
- **Ollama (local):** Free, requires 16GB+ RAM. Not viable.

**Choice and Rationale:** Groq. Free API access with fast inference. The 8B model is sufficient for the focused tasks in this system. OpenAI-compatible API format simplifies LangChain integration. Provider abstraction layer allows users to optionally upgrade to GPT-4 or Claude with their own keys.

**Tradeoffs Accepted:** 8B model is less capable than GPT-4 or Claude for complex reasoning. Response Synthesizer output quality is lower than what a larger model would produce. Groq's free tier has rate limits and occasional 503 errors during high demand.

---

## Decision 4: Embedding Model

**Date:** January 2026 **Status:** Accepted

**Decision:** Google Gemini API (text-embedding-004) for all text embeddings.

**Context:** The RAG pipeline requires an embedding model for document chunks and search queries. Must be API-accessible and free.

**Options Considered:**

- **Google Gemini (text-embedding-004):** Free tier, 1,500 requests/day, batch support (100 texts/request).
- **OpenAI Embeddings (text-embedding-3-small):** High quality, paid per token.
- **Sentence Transformers (local):** Free, requires local compute beyond hardware constraints.
- **Cohere Embeddings:** Free trial, limited and temporary.

**Choice and Rationale:** Google Gemini. Free tier capacity (1,500 requests/day with batching) is sufficient for ingesting multiple filings per day. Strong embedding quality for financial text. Cloud-based, eliminating local compute requirements.

**Tradeoffs Accepted:** Dependency on Google's free tier terms. The embedding model is intentionally not configurable because all vectors in ChromaDB must use the same model for cosine similarity to be meaningful. Switching providers would require re-embedding all stored documents.

---

## Decision 5: Vector Database and Search Strategy

**Date:** January 2026 **Status:** Accepted

**Decision:** ChromaDB in embedded mode with semantic search and metadata pre-filtering.

**Context:** The RAG pipeline needs a vector database with persistence and search capabilities. The system ingests SEC filings on-demand and needs embeddings to survive application restarts.

**Options Considered:**

- **ChromaDB (embedded):** In-process, automatic disk persistence, zero network latency. Semantic search only.
- **Pinecone:** Managed cloud. Free Starter tier (2GB, 5 indexes) sufficient for data volume. Adds network latency and external dependency per query.
- **Weaviate:** Managed cloud with native hybrid search (semantic + BM25 keyword). Free sandbox tier with limited storage.
- **pgvector:** Requires a running PostgreSQL instance.
- **FAISS:** No built-in persistence. Requires custom save/load logic for the on-demand ingestion use case.

**Choice and Rationale:** ChromaDB. In-process execution provides zero network latency and eliminates external dependencies. Automatic persistence supports the on-demand ingestion workflow. Pinecone and Weaviate free tiers are technically sufficient but add network latency and an external point of failure per query.

The search strategy uses semantic search with metadata pre-filtering (by ticker and filing period) to narrow the search space before the semantic component runs. Hybrid search (semantic + BM25) would improve precision for term-specific financial queries but would require either a separate BM25 index alongside ChromaDB or migration to Weaviate.

**Tradeoffs Accepted:** No concurrent access support. No managed backup or replication. Semantic-only search may miss chunks where exact financial terminology matters. For production, migration to a managed vector database with native hybrid search would address all three limitations. The retrieval interface is abstracted to support this migration.

---

## Decision 6: Stock Data Source

**Date:** January 2026 **Status:** Accepted

**Decision:** yfinance as primary, Alpha Vantage as fallback.

**Context:** The system needs historical OHLCV data, current options chains, and earnings dates. Must be free.

**Options Considered:**

- **yfinance:** Unofficial Yahoo Finance wrapper. Free, comprehensive (prices, options, earnings calendars), no API key.
- **Alpha Vantage:** Official API, free tier (5 calls/min, 500/day). Reliable but limited.
- **Polygon.io:** Professional-grade. Free tier too limited for historical data.
- **IEX Cloud:** Free credits deplete over time.

**Choice and Rationale:** yfinance provides the broadest data coverage in a single library with no API key management. Alpha Vantage as fallback provides an independent data path when Yahoo Finance is unreachable.

**Tradeoffs Accepted:** yfinance is unofficial with no uptime guarantees. Yahoo could block automated access without notice. yfinance is the sole source for options data with no free fallback. Alpha Vantage's rate limits restrict it to occasional fallback use only.

---

## Decision 7: News Data Sources

**Date:** January 2026 **Status:** Accepted

**Decision:** NewsAPI as primary, Google News RSS as fallback.

**Context:** The system needs news articles with structured metadata (title, source, date, URL) for specific stocks and time periods.

**Options Considered:**

- **NewsAPI:** Structured JSON, good search quality. Free tier: 100 requests/day, last 30 days only.
- **Google News RSS:** Free, no date restriction. Unstructured XML, inconsistent results.
- **Bing News Search API:** Limited free tier.
- **Web scraping:** Legally gray (violates most sites' ToS), fragile (breaks on layout changes), requires per-site parsing logic. Maintenance burden exceeds benefit.

**Choice and Rationale:** NewsAPI for structured, consistent data. Google News RSS as fallback to cover dates beyond NewsAPI's 30-day free tier limit, critical for historical queries.

**Tradeoffs Accepted:** NewsAPI free tier limits (30 days, 100 requests/day). RSS fallback produces less structured, less reliable results. Provider abstraction allows users to supply their own NewsAPI key for expanded coverage.

---

## Decision 8: Social Sentiment Source

**Date:** January 2026 **Status:** Accepted

**Decision:** Reddit (via PRAW) as the sole social sentiment source for the MVP.

**Context:** The system needs social media sentiment from retail investor communities around specific market events.

**Options Considered:**

- **Reddit (PRAW):** Free API. Access to r/wallstreetbets, r/stocks, r/options. Rich text for sentiment analysis. Well-documented library.
- **Twitter/X API:** Free tier too limited to be useful. Paid tiers prohibitively expensive.
- **StockTwits:** Purpose-built for stock sentiment with pre-tagged ticker cashtags. Less documented API, smaller developer community, stricter free tier rate limits.
- **Discord:** No practical API for historical message search.

**Choice and Rationale:** Reddit. r/wallstreetbets and related subreddits are the epicenter of retail investor discussion. PRAW is well-maintained with strong documentation. StockTwits is a viable future addition.

**Tradeoffs Accepted:** Single-platform sentiment skews toward Reddit's user demographics. Historical search depth is limited for older posts. Subreddit list is configurable for future expansion; StockTwits is a candidate for a subsequent iteration.

---

## Decision 9: Cloud Provider

**Date:** February 2026 **Status:** Accepted

**Decision:** Azure for Students.

**Context:** The deployed application needs cloud hosting. Zero-cost budget.

**Options Considered:**

- **Azure for Students:** $100 credit tied to student status (typically valid for one year).
- **AWS Free Tier:** 12-month t2.micro (1 vCPU, 1GB RAM). Insufficient RAM for running Chainlit, LangGraph, and ChromaDB in a single process.
- **Google Cloud Platform:** $300 credit, 90-day expiration. Time pressure is undesirable for a project that needs to remain deployed throughout an extended interview period.
- **Railway / Render / Fly.io:** Simpler deployment, less interview value than major cloud provider experience.

**Choice and Rationale:** Azure for Students. Longest credit duration (year-long student status vs. GCP's 90-day window). Sufficient credit for several months of a small VM. AWS was less attractive due to the RAM constraint on the free tier instance.

**Tradeoffs Accepted:** Azure has a smaller deployment community and is generally considered less intuitive than AWS for container workflows. AWS experience is more universally valued. Cloud skills transfer across providers.

---

## Decision 10: Containerization and Deployment Strategy

**Date:** February 2026 **Status:** Accepted

**Decision:** Single Docker container, CI/CD via GitHub Actions. No orchestration tooling.

**Context:** The application needs reproducible cloud deployment proportionate to a single-developer MVP.

**Options Considered:**

- **Single Docker container + GitHub Actions:** Automated test, build, and deploy pipeline on push to main.
- **Docker Compose:** Multi-container separation. Unnecessary for a single-process application.
- **Kubernetes:** Nothing to orchestrate with one container.
- **Terraform:** Single Azure resource doesn't warrant Infrastructure as Code.

**Choice and Rationale:** Docker provides environment reproducibility. GitHub Actions provides automated testing and deployment. Kubernetes and Terraform solve problems that don't exist at this project's scale.

**Tradeoffs Accepted:** Web server, graph workflow, and ChromaDB share one process. In production, these would be separated for independent scaling and fault isolation, at which point Kubernetes and Terraform would become appropriate.

---

## Decision 11: Eval Strategy

**Date:** February 2026 **Status:** Accepted

**Decision:** LangSmith as primary eval platform, supplemented by Ragas for RAG-specific metrics. Evals run separately from the CI pipeline.

**Context:** Multiple LLM-dependent nodes and a RAG pipeline require quality evaluation beyond deterministic tests.

**Options Considered:**

- **LangSmith:** Native LangGraph integration. Tracing, eval datasets, result tracking over time. Free tier sufficient.
- **Promptfoo:** Provider-agnostic prompt evaluation. No deep LangGraph integration.
- **Ragas:** Purpose-built for RAG evaluation (context precision, context recall, faithfulness, answer relevancy).
- **Custom Python scripts:** Full control, no dashboard or historical tracking.
- **Braintrust:** Experiment tracking, not tied to LangChain ecosystem.

**Choice and Rationale:** LangSmith for native LangGraph integration. Automatic tracing of every node execution simplifies identifying quality issues at the node level. Ragas supplements LangSmith specifically for RAG retrieval quality. Promptfoo would be redundant alongside LangSmith within the LangChain ecosystem.

**Tradeoffs Accepted:** Platform dependency on LangSmith. Custom eval scripts provide a fallback if the free tier changes or the platform becomes unavailable.
---

## Decision 12: Two-Model LLM Strategy

**Date:** March 2026 **Status:** Accepted (Updated March 29, 2026)

**Decision:** Use two separate LLM configurations: `llm_classifier` (Groq llama-3.1-8b-instant) for structured classification nodes, `llm_synthesizer` (Google Gemini 2.5 Flash with thinking) for the Response Synthesizer.

**Context:** LangSmith trace analysis revealed that the synthesizer is the quality bottleneck. It receives 500–1,000+ prompt tokens of multi-source financial data and must produce a coherent, sourced narrative that connects price action to catalysts — reasoning about causality, not reciting data. The synthesizer prompt was designed around this causality-first principle; a thinking model aligns directly with it. Classification tasks (intent, ticker, date) are deterministic and structured — they don't benefit from a larger or reasoning model.

An intermediate step used Groq llama-3.3-70b-versatile as the synthesizer upgrade, but Groq 70B's token-per-minute limits became a constraint with large news payloads, and its output still tended toward recitation rather than synthesis.

**Options Considered:**

- **Single model (8B) for all nodes:** Simple configuration. Classification quality is fine, but synthesis quality is the ceiling.
- **Groq 70B (llama-3.3-70b-versatile) for synthesizer:** Better than 8B. Same provider (Groq), no new API key. Still recites rather than reasons; token-per-minute limits constrain large data payloads.
- **Google Gemini 2.5 Flash (thinking) for synthesizer:** Reasoning model with explicit thinking budget. Higher output token ceiling. Different provider — requires GEMINI_API_KEY.

**Choice and Rationale:** Gemini 2.5 Flash with `thinking_budget=1024`. The thinking budget enables internal chain-of-thought so the model connects data points (price action ↔ news ↔ sentiment ↔ earnings) rather than listing them under headers. `max_output_tokens=4096` allows analyst-quality briefs. Streaming is native via LangChain's Gemini integration. Chainlit's astream_events loop handles Gemini's thinking-phase chunk format (structured content list rather than plain string) with a special handler in app.py.

**Tradeoffs Accepted:** Two API providers now required (Groq + Google). `GEMINI_API_KEY` is a required env variable. Gemini free tier has its own rate limits (distinct from Groq). Thinking tokens are not streamed to the user — app.py filters them and only streams the final text output. The `llm` legacy alias in llm_setup.py (previously kept for the now-deleted tool_caller.py prototype) was removed.

## Decision 13: Parallel Data Retrieval via LangGraph Send()

**Date:** March 2026 **Status:** Accepted

**Decision:** Implement fan-out parallelism for the news_retriever and reddit_sentiment nodes using LangGraph's `Send()` API rather than sequential execution.

**Context:** The TDD identified that nodes 4–7 are independent of each other. LangSmith traces confirmed the real-world execution profile: news retrieval and Reddit sentiment each take 1–3 seconds sequentially, adding 2–6 seconds of latency to every stock_analysis response for no reason. Each node only reads ticker and date from state; neither requires the other's output.

**Options Considered:**

- **Sequential execution (status quo):** Simple graph topology. Total latency = sum of all retrieval times.
- **asyncio.gather within a single node:** Parallelizes HTTP calls within one node but doesn't leverage LangGraph's native parallelism primitives.
- **LangGraph Send() fan-out:** Native LangGraph parallel execution. Each Send() dispatches a node as an independent branch; results merge back into state before the synthesizer.

**Choice and Rationale:** LangGraph Send(). Aligns with the documented TDD design intent ("they can execute in parallel") and uses the framework's native mechanism rather than working around it. Total response latency reduces from sequential sum to the slowest individual node.

**Tradeoffs Accepted:** Slightly more complex workflow topology. Fan-out requires a merge step before the synthesizer. Error handling must account for partial failures — if one branch fails, the other should still complete and the synthesizer should handle the missing data gracefully via the existing error fields.

---

## Decision 14: Phase Ordering — Options Analyzer Before RAG

**Date:** March 2026 **Status:** Accepted

**Decision:** Build the Options Analyzer (originally Phase 4) before the RAG pipeline (originally Phase 3).

**Context:** Both phases are independent of each other — neither requires the other to be complete. Options data retrieval is a single yfinance API call; the integration surface is small and well-understood. The RAG pipeline involves SEC EDGAR ingestion, document chunking, embedding with Gemini, and ChromaDB — significantly higher build complexity and more external dependencies. Options analysis also directly extends the existing `stock_analysis` and `options_view` paths that are already wired in the graph.

**Options Considered:**

- **RAG first (original order):** Follows the original PRD phase numbering. RAG adds the deepest analytical depth but is also the most complex build.
- **Options first (revised order):** Delivers demonstrable multi-dimensional analysis (price + news + sentiment + options) sooner. Simpler build with higher immediate user value per effort unit.

**Choice and Rationale:** Options first. The product is more compelling to demonstrate with all four real-time data dimensions (price, news, sentiment, options) before adding the filing retrieval dimension. RAG doesn't make the existing features better; it adds a fifth dimension. Options makes the existing `stock_analysis` path complete. From an interview portfolio perspective, a working options analyzer is also a stronger demonstration of financial domain depth than a RAG pipeline over SEC filings.

**Tradeoffs Accepted:** The RAG pipeline remains deferred. Queries about earnings management commentary will not be answered until Phase 3 is built. The PRD phase numbering is preserved; only the build order changes.

---

## Decision 15: Trader-Grade Data Enrichment Strategy

**Date:** March 2026 **Status:** Accepted

**Decision:** Extend Node 4 (Price Data Fetcher) with three enrichment signals — analyst ratings and price targets, short interest, and earnings date proximity — and expand the Phase 3 RAG corpus to include 8-K filings and earnings call transcripts alongside 10-K and 10-Q.

**Context:** A review of the existing data strategy from the perspective of a working trader identified a gap between what the agent was retrieving and what traders actually use to make decisions. All three enrichment signals are already available via yfinance with no additional API keys or rate limit concerns. The RAG corpus was limited to 10-K and 10-Q filings, which are backward-looking annual and quarterly reports. 8-K filings capture material events in real time (earnings releases, guidance changes, executive departures). Earnings call transcripts, filed as 8-K exhibits, are what analysts and traders read to assess forward outlook — the tone, specific language, and Q&A cannot be found in the structured filing.

**Options Considered:**

- **Status quo:** Analyst ratings, short interest, and earnings proximity are omitted. RAG corpus limited to 10-K and 10-Q.
- **Add Tradier API for exchange-calculated Greeks:** Real-time Greeks are more precise but require account registration and the free tier does not cover live tickers. Rejected.
- **Self-calculate Greeks via Black-Scholes:** Standard industry formula. Given that yfinance provides implied volatility, the only input that affects accuracy is IV precision (15-min delayed). Sufficient for analysis use case. Accepted for Options Analyzer.
- **Add enrichment signals to a new dedicated node:** More explicit separation of concerns, but adds a sequential node to the graph for work that can be done in a single yfinance pass already happening in Node 4.
- **Add enrichment signals to Node 4:** Same yfinance Ticker object, same network call. All failures are non-fatal and don't block price data retrieval. Accepted.

**Choice and Rationale:** All three enrichment signals added to Node 4. The analyst consensus price target tells traders where professionals think the stock is headed. Short float and days-to-cover tell traders whether there is meaningful short positioning that could fuel a squeeze. Earnings proximity tells options traders whether elevated IV reflects a real event catalyst. None of these require a new API, none block the critical path, and all three materially improve the quality of the synthesizer's output for `stock_analysis` and `options_view` intents.

8-K and earnings call transcripts added to the RAG corpus because they are the most actionable documents in the SEC EDGAR archive for a trader's time horizon. A 10-K tells you what happened last year; an 8-K tells you what happened this week. Both are free, same pipeline.

**Tradeoffs Accepted:** Node 4 takes on more responsibility than strictly "price data." The enrichment calls are best-effort and non-fatal — if any yfinance call fails, the node writes None to that field and continues. The synthesizer handles None gracefully. Black-Scholes Greeks are slightly less precise than exchange-calculated Greeks due to 15-minute IV delay, but the difference is immaterial for the retrospective analysis use case this agent is designed for.

---

## Decision 16: Google Gemini 2.5 Flash as Response Synthesizer

**Date:** March 2026 **Status:** Accepted

**Decision:** Replace Groq llama-3.3-70b-versatile with Google Gemini 2.5 Flash (thinking model) as the Response Synthesizer LLM.

**Context:** See Decision 12 for full context. This entry records the provider-switch rationale separately from the two-model split rationale. The switch was driven by two specific observations in LangSmith traces: (1) the 70B Groq model produced structured recitations rather than causal narratives even with explicit prompting; (2) Groq 70B token-per-minute limits were being hit on stock_analysis queries with large news payloads (8–10 articles + Reddit posts + filing chunks).

**Options Considered:**

- **Groq 70B (status quo):** No new API key. Faster model selection. Recitation tendency persists; rate-limit ceiling remains.
- **OpenAI GPT-4o:** Strongest reasoning. Paid. Provider abstraction planned for Phase 5 user-configurable keys; appropriate there, not as the default.
- **Google Gemini 2.5 Flash (thinking):** Free tier sufficient for development and portfolio demo. `thinking_budget` parameter enables internal chain-of-thought on demand. LangChain `langchain_google_genai` integration supports streaming.

**Choice and Rationale:** Gemini 2.5 Flash. The `thinking_budget=1024` tokens of internal reasoning directly addresses the recitation problem — the model reasons about which data points are causally linked before writing the output. Free tier is adequate for development and demo use. The app.py streaming loop was updated to handle Gemini's thinking-phase content format (list of typed parts vs. plain string chunks from Groq).

**Tradeoffs Accepted:** New required dependency: `GEMINI_API_KEY`. Two distinct API providers now in the critical path. Gemini thinking tokens are invisible to the user (filtered in app.py). Model availability depends on Google's free tier terms.

---

## Decision 17: You.com Search API as News Layer 2

**Date:** March 2026 **Status:** Accepted

**Decision:** Replace Financial Modeling Prep (FMP) with the You.com Search API as the Layer 2 news fallback in news_retriever.py.

**Context:** The news retriever uses a three-layer fallback: Finnhub (primary, ticker-specific) → Layer 2 (broader web search for historical coverage) → Google News RSS (emergency fallback). The original plan used FMP as Layer 2 (250 requests/day free, structured news endpoint). During Phase 3 testing, Google Custom Search JSON API — another candidate — was found to be permanently closed to new customers with no workaround. FMP was re-evaluated and found to have coverage gaps for dates older than 90 days. The You.com Search API was introduced as Layer 2.

**Options Considered:**

- **FMP (original plan):** 250 req/day free, structured JSON. Weak historical coverage beyond 90 days. Limited to news explicitly tagged by FMP's feed.
- **Google Custom Search JSON API:** Closed to new customers. Not viable.
- **Bing News Search API:** Limited free tier, restrictive ToS for financial use.
- **You.com Search API:** Web search index. `freshness=YYYY-MM-DDtoYYYY-MM-DD` parameter for server-side date filtering. `results.news[]` returns title, description, url, page_age. Free tier includes $100 credits. Better historical coverage than FMP for older date ranges.

**Choice and Rationale:** You.com Search API. The `freshness` parameter makes server-side date filtering reliable for the Layer 2 use case (historical queries beyond Finnhub's range). Coverage is broader than FMP's feed-based approach because it indexes the web. $100 free credits are sufficient for development and demo use.

**Tradeoffs Accepted:** New env variable: `YOUCOM_API_KEY`. Credits are finite (not unlimited free tier); monitor usage during sustained testing. Source attribution in responses shows "youcom" as the provider when Layer 2 is used. The `news_source_used` state field now takes values "finnhub", "youcom", "google_rss", or "none" — updated in state.py comments and TDD.md.

---

## Decision 18: Reddit Public JSON + Stocktwits Replacing PRAW

**Date:** April 2026 **Status:** Accepted

**Decision:** Replace PRAW (Reddit's official Python client) with Reddit's public JSON endpoints as the primary sentiment source, and add Stocktwits as a co-primary source running on every query.

**Context:** Node 6 originally used PRAW for Reddit sentiment retrieval. During Phase 5 eval runs, PRAW's OAuth credential requirement and rate-limiting behaviour caused intermittent failures that degraded sentiment coverage in the eval dataset. Reddit's public JSON endpoints (`reddit.com/r/{subs}/search.json`) return the same post data without requiring any authentication. Stocktwits (`api.stocktwits.com/api/2/streams/symbol/{ticker}.json`) provides a purpose-built financial sentiment feed with pre-labeled sentiment tags ("Bullish"/"Bearish") on a significant portion of messages, reducing the number of LLM classification calls required.

**Options Considered:**

- **PRAW (status quo):** Official client, well-documented. Requires OAuth credentials. Rate-limit failures observed during eval runs.
- **Reddit public JSON endpoints:** No authentication. Same underlying data. Date filtering applied post-fetch (Reddit's search API does not support server-side date ranges without OAuth). Simpler, more reliable.
- **Pushshift API:** Historical Reddit archive. Shut down in 2023 — not viable.
- **Reddit API v2 (paid tier):** Extended access at cost. Rejected — zero-cost constraint.

**Choice and Rationale:** Reddit public JSON + Stocktwits as co-primary sources. The public endpoints eliminate auth failures entirely. Stocktwits adds a second signal from a finance-specific community, improving coverage for recent queries (< 30 days) where Stocktwits stream depth is strongest. Reddit carries the historical load for older date ranges where Stocktwits returns sparse results. Pre-labeled Stocktwits sentiment reduces LLM classifier calls. Both sources run on every query; results are combined and aggregated into `sentiment_summary` with a per-source breakdown for transparency in Node 9.

**Tradeoffs Accepted:** No `REDDIT_CLIENT_ID` or `REDDIT_CLIENT_SECRET` env variables required. Post-fetch date filtering on Reddit results is less precise than server-side filtering — posts near the boundary of the date range may be included or excluded inconsistently. Stocktwits stream is best-effort for historical queries; returns zero results gracefully for older date ranges without failing the node.

---

## Decision 19: Firecrawl for News Article Enrichment

**Date:** April 2026 **Status:** Accepted

**Decision:** Add Firecrawl as a post-retrieval enrichment step in Node 5 (News Retriever) to fetch full article text for articles from known free-access domains.

**Context:** Finnhub and You.com both return truncated article excerpts — approximately 200–600 characters of snippet text. For the Response Synthesizer to produce a grounded, cited narrative, it needs enough article content to identify the specific claims, numbers, and context reported. Short snippets produce thin, low-confidence synthesis ("NVIDIA reported strong earnings" without the actual figures or analyst reactions). Full article text enables specific, attributable claims.

**Options Considered:**

- **No enrichment (status quo):** 600-char snippets. Synthesis degrades to surface-level summaries. No additional API required.
- **Direct HTTP scraping:** Legally ambiguous (violates most publishers' ToS). Brittle — breaks on layout changes, blocked by anti-scraping measures. No paywalled content. High maintenance burden.
- **Jina Reader API:** Converts URLs to markdown. Free tier available. Less reliable JavaScript rendering compared to Firecrawl.
- **Firecrawl:** Purpose-built web scraping API. Returns clean markdown. Handles JavaScript-rendered pages. Free tier: 500 credits/month. Gracefully skipped if `FIRECRAWL_API_KEY` is not set.

**Choice and Rationale:** Firecrawl with graceful degradation. Enrichment is applied only to articles from a known set of free-access financial domains (reuters.com, cnbc.com, apnews.com, finance.yahoo.com, benzinga.com, marketwatch.com) — paywalled sources are skipped. 500 credits/month is sufficient for the query volume this agent handles. The enrichment step is entirely non-fatal: if Firecrawl is unavailable or the key is absent, Node 5 falls back to the 600-char snippet with no error surfaced to the user.

**Tradeoffs Accepted:** New optional env variable: `FIRECRAWL_API_KEY`. Without it, synthesis quality degrades to snippet-level. Credits are finite — monitor during sustained testing. Enrichment adds latency to Node 5 (one Firecrawl call per eligible article, run after retrieval). Full text is capped at 2,000 characters per article to limit synthesizer prompt size.

---

## Decision 20: Retrieval Planner Node for Adaptive Fan-Out

**Date:** April 2026 **Status:** Accepted

**Decision:** Insert a Retrieval Planner node between Node 4 (Price Data Fetcher) and the parallel retrieval fan-out (Nodes 5, 6, 7). The planner uses an LLM call to emit a `retrieval_plan` dict that controls which retrieval nodes are activated via `Send()`.

**Context:** Prior to Phase 5, Nodes 5, 6, and 7 ran unconditionally on every `stock_analysis` query via `Send()` fan-out. This meant a query like "how did AAPL perform last week?" triggered full news retrieval, Reddit sentiment analysis, and a ChromaDB RAG search — all of which add latency and API cost without contributing meaningful signal to what is essentially a price performance question. Conversely, a query asking about earnings guidance commentary would benefit more from RAG than from Reddit sentiment. A static fan-out cannot distinguish between these cases.

**Options Considered:**

- **Static fan-out (status quo):** All three nodes always run. Simple graph topology. No LLM overhead for routing. Wastes 3–7 seconds of I/O on irrelevant retrievals.
- **Rule-based routing (if-else on intent/keywords):** Deterministic. No additional LLM call. Fragile — intent alone is not sufficient signal; "stock_analysis" covers queries ranging from simple price lookups to deep fundamental analysis.
- **LLM-driven planner node:** One lightweight LLM call (~100 ms, llama-3.1-8b-instant, max_tokens=512) produces a `retrieval_plan` dict with boolean flags (`fetch_news`, `fetch_sentiment`, `fetch_rag`). The router reads these flags and emits only the relevant `Send()` calls. Falls back to activating all three nodes if the LLM call fails or returns unparseable output — identical to pre-planner behaviour.

**Choice and Rationale:** LLM-driven planner. The planner adds one cheap, fast LLM call before the fan-out and can eliminate several seconds of parallel I/O when retrieval is unnecessary. The fallback guarantee (all nodes activate on failure) means the planner cannot make the system worse — worst case is pre-planner behaviour. Rule-based routing was rejected because intent classification alone is too coarse: both "how did NVDA do last week?" and "what did NVDA management say about margins in Q2 earnings?" classify as `stock_analysis`, but only the second warrants RAG retrieval.

**Tradeoffs Accepted:** One additional LLM call per `stock_analysis` query (~100 ms on Groq free tier). New state fields: `retrieval_plan` (dict) and `planner_error` (str or None). Workflow topology change: `route_after_fetch_price` now routes to `plan_retrieval` instead of directly to the fan-out for all non-chart intents. The planner's decision is logged at INFO level for LangSmith trace inspection.

---

## Decision 21: Migrate Gemini to Vertex AI; Upgrade Synthesizer to Gemini 2.5 Pro

**Date:** April 2026 **Status:** Accepted

**Decision:** Route all Gemini usage (synthesizer + embeddings) through Google Cloud Vertex AI instead of the Google AI Studio API. Simultaneously upgrade the synthesizer model from Gemini 2.5 Flash to Gemini 2.5 Pro.

**Context:** The project was using the Google AI Studio API (`GEMINI_API_KEY`) for both the Response Synthesizer (Node 9) and RAG embeddings (Node 7). GCP signup credits ($300) were available on the `stock-insight-agent` project but are scoped to GCP services — they do not apply to Google AI Studio billing. Routing through Vertex AI unlocks these credits and eliminates the cost constraint on model quality. Gemini 2.5 Pro is the most capable model in the Gemini family; upgrading from Flash is appropriate now that cost is no longer a constraint.

**Options Considered:**

- **Stay on AI Studio (status quo):** No setup changes. Credits remain unused. Free tier rate limits apply.
- **Migrate to Vertex AI, keep Gemini 2.5 Flash:** Uses credits, same model quality.
- **Migrate to Vertex AI, upgrade to Gemini 2.5 Pro:** Uses credits, materially better synthesis quality for analyst briefs.
- **Switch synthesizer to Claude (Vertex AI Model Garden):** Also available via Vertex AI. Deferred — see planned experiment in project notes.

**Choice and Rationale:** Vertex AI with Gemini 2.5 Pro. Both `langchain-google-genai` and `google-genai` support Vertex AI natively via a `vertexai=True` flag — no package swap required. Auth switches from API key to Application Default Credentials (ADC), which is the standard GCP auth pattern. The `thinking_budget` is increased from 1024 to 2048 tokens; Pro benefits more from deeper reasoning passes given its larger capacity. The embedding model (`gemini-embedding-001`) is unchanged — only the client initialization differs (drop `models/` prefix for Vertex AI model naming).

**Tradeoffs Accepted:** `GEMINI_API_KEY` removed; replaced by `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION`. Local development now requires `gcloud auth application-default login`. CI/CD and production deployments will require a service account with Vertex AI permissions. `sentence-transformers` removed from requirements.txt — it was never imported in code; its presence was causing a spurious PyTorch version warning in the test runner.
