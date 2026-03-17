# Decision Log

**Product:** Stock Insight Agent 

**Author:** Ritu Villuri

**Status:** 

**Last Updated:** March 17, 2026

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

**Date:** March 2026 **Status:** Accepted

**Decision:** Use two separate LLM configurations: `llm_classifier` (llama-3.1-8b-instant) for structured classification nodes, `llm_synthesizer` (llama-3.3-70b-versatile) for the Response Synthesizer.

**Context:** LangSmith trace analysis revealed that the synthesizer is the quality bottleneck. It receives 500–1,000+ prompt tokens of multi-source financial data and must produce a coherent, sourced narrative. The 8B model produced adequate but noticeably shallow output for complex multi-source queries. Classification tasks (intent, ticker, date) are deterministic and structured — they don't benefit from a larger model.

**Options Considered:**

- **Single model (8B) for all nodes:** Simple configuration. Classification quality is fine, but synthesis quality is the ceiling.
- **Single model (70B) for all nodes:** Better synthesis quality. Rate limits hit faster; classification calls are wasteful at 70B scale.
- **Two-model split:** Small fast model for classification, large capable model for synthesis.

**Choice and Rationale:** Two-model split. Classification nodes return structured JSON with constrained output — the 8B model handles this reliably at lower token cost and higher speed. The synthesizer is the only node that requires genuine reasoning across conflicting signals from multiple data sources. Upgrading only the synthesizer gets the quality improvement where it matters without burning rate limits on classification tasks.

**Tradeoffs Accepted:** Two model configs to maintain. Groq free tier rate limits apply to each model separately, so the 70B model has tighter per-minute token limits than the 8B. For synthesis tasks with large news payloads, the 70B token/minute limit can be a constraint.

---

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
