# Product Requirement Document

**Product:** Stock Insight Agent 

**Author:** Ritu Villuri

**Status:** In Progress 

**Last Updated:** February 26, 2026 

**Version:** 2.0

---

## Table of Contents

---

## 1. Product Overview

**Product Name:** Stock Insight Agent

**One-Liner:** An AI-powered research assistant that assembles the complete picture of why a stock moved at a particular time by combining price action, news, social sentiment, and options activity into a single conversational interface.

### Problem Statement

When a retail investor wants to understand what happened around a stock event (an earnings report, a price spike, a sudden crash), they currently have to piece together the story manually. This means opening Robinhood or Yahoo Finance for price data, Googling for news articles from that time period, checking Reddit or Twitter for what people were saying, and looking at options activity to understand how traders were positioned. Each source tells part of the story, but no single tool assembles the full picture.

This problem is especially painful for events where the price movement only makes sense when you see all the dimensions together. For example, GameStop's rally in January 2021 looks irrational from a pure price chart. But it becomes completely logical when you see the coordinated social sentiment on r/wallstreetbets, the short squeeze mechanics, and the news cycle that followed. The full picture requires multiple data sources, and today that means multiple tabs, multiple searches, and a lot of manual synthesis.

The impact of this fragmentation goes beyond inconvenience. For many retail investors, the difficulty of assembling a complete picture becomes a reason not to invest at all. The information exists across the internet, but the effort required to gather, cross-reference, and synthesize it feels disproportionate. People who would otherwise make informed, calculated decisions opt out because the research process feels inaccessible. The problem isn't a lack of data. It's the absence of a tool that assembles the data into a coherent, trustworthy narrative.

Existing solutions either serve institutional traders (Bloomberg Terminal, Refinitiv) or provide only one dimension of analysis. TradingView handles charts. Stocktwits covers sentiment. Seeking Alpha focuses on news and analysis. No tool brings all of these dimensions together in a conversational interface where a user can simply ask a question and get a complete, sourced answer.

### Solution

Stock Insight Agent is a conversational AI assistant that, given a stock and a time period, automatically retrieves and synthesizes:

- Historical price action with key metrics
- Relevant news articles from the period with source attribution
- Social media sentiment from Reddit with quantified scoring
- Options market positioning (current data) or volume-based proxy analysis (historical)
- Earnings report context from SEC filings via a RAG pipeline

The agent also supports comparative analysis, allowing users to examine historical events alongside current market conditions in a single query. For example, a user can ask how a stock behaved around last quarter's earnings while also seeing current news and options positioning.

The agent presents a unified narrative that explains not just what happened to the price, but why. All claims are grounded in retrieved data with sources cited. The agent does not predict future movements or provide investment advice. It is a retrospective research tool that deals in facts and data. When presenting current market conditions alongside historical data, it does so without implying that past patterns will repeat.

### Target User

Retail investors who trade occasionally and want to make informed decisions. People who are curious about market dynamics but don't have the time, tools, or expertise to manually research across multiple data sources. Someone who wants to ask a natural language question like "What happened with NVIDIA around Q2 2024 earnings?" and get a complete, sourced answer in seconds instead of spending 30 minutes piecing it together across different websites.

---

## 2. Product Discovery and Competitive Landscape

### Discovery Context

This product originated from my own experience as a retail investor. My partner and I were evaluating whether to buy NVIDIA call options before Q2 earnings and wanted to understand how the stock had behaved around previous earnings. Answering this seemingly simple question required opening Robinhood to track price movement in the days and weeks before the event, Googling for news from that period to see if external factors like tariffs or political events had an obvious impact, checking Reddit for retail sentiment, and trying to find options activity data. Each source provided a fragment of the picture, but assembling the full narrative was time-consuming and the result still felt incomplete.

This experience was not unique. In conversations with friends and other retail investors in my network, a consistent pattern emerged. People who were interested in options trading or investing around events often felt the research burden was disproportionate to the trade. They could find a price chart easily enough, but understanding why the stock moved, what people were thinking at the time, and how traders were positioned required effort that most casual investors weren't willing or able to put in. Some described doing their own research but feeling like the results didn't give them enough context to feel confident. Others opted out of investing entirely, not because they weren't interested, but because the barrier to feeling informed was too high. The potential reward was visible, but the path to making an informed decision felt inaccessible.

This discovery pointed to a clear opportunity: a tool that does the multi-source research automatically and presents the full picture in a single conversational response, so that users can focus on understanding and interpretation rather than data gathering.

### Competitive Analysis

| Tool | Price Data | News Analysis | Social Sentiment | Options Data | Unified Narrative | Conversational Interface | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bloomberg Terminal | Yes | Yes | Limited | Yes | No | No | ~$2,000/month |
| TradingView | Yes | Limited | No | Limited | No | No | Free to $60/month |
| Seeking Alpha | Limited | Yes | Limited | No | No | No | Free to $240/year |
| Stocktwits | No | No | Yes | No | No | No | Free |
| Finviz | Yes | Yes | No | Limited | No | No | Free to $40/month |
| Stock Insight Agent | Yes | Yes | Yes | Yes | Yes | Yes | Free |

### Gap Identification

Every existing tool either specializes in one or two dimensions of analysis or requires significant user effort to approximate a complete picture. General-purpose AI assistants like Perplexity or ChatGPT can synthesize web search results into a qualitative summary, but they don't pull structured financial data from APIs for precise metrics, don't run quantified sentiment analysis across social media posts, and can't perform semantic search over embedded SEC filings. Specialized platforms like TradingView or Seeking Alpha provide depth in their specific domain but don't connect the dots across data sources. Bloomberg Terminal comes closest to comprehensive coverage but serves institutional traders at institutional prices.

Stock Insight Agent automates the multi-source research workflow with structured financial data, quantified sentiment analysis, and SEC filing search, returning precise, sourced results from a single natural language question. The value is not that the information is unavailable elsewhere. It's that assembling it today requires either expert-level prompt engineering across general-purpose AI tools or manual aggregation across multiple specialized platforms. Stock Insight Agent eliminates that effort.

---

## 3. User Stories and Use Cases

### Jobs To Be Done

When I'm considering a trade around an upcoming event and I want to understand how the stock behaved in a similar past situation, help me see the full picture so I can form my own view without spending hours researching across multiple sources.

When I see a stock making a big move and I want to understand why it's moving, help me quickly find the news, sentiment, and trading activity driving it so I can understand the context rather than just seeing a number on a screen.

When I'm curious about a historically significant market event like the GME rally and I want to understand what actually happened, help me see all the dimensions of that event in one place so I can learn from it.

### User Stories

As a retail investor, I want to ask a natural language question about a stock's performance during a specific time period so that I can quickly understand what happened without manually researching across multiple sources.

As a retail investor, I want to see what news and social sentiment surrounded a stock's price movement so that I can understand the factors that influenced it, not just the numbers.

As a retail investor, I want to understand how traders were positioned (calls vs puts, bullish vs bearish sentiment) around a past event so that I can see the full picture of market behavior during that period.

As a retail investor, I want every claim the agent makes to be tied to a real source so that I can verify the information myself and trust what the agent tells me.

As a retail investor, I want to be able to view current options market data for a stock so that I can see how traders are positioned right now.

As a retail investor, I want to be able to ask about historically significant market events (like the GME rally or a major earnings surprise) and get a complete narrative that ties together price, news, sentiment, and trading activity from that period.

As a retail investor, I want to examine how a stock behaved around a past event while also seeing current news and market conditions so that I can compare the historical context with the present landscape and form my own informed view.

### Use Cases

**Use Case 1: Event-Based Historical Analysis**

The user asks: "What happened with NVIDIA around Q2 2024 earnings?"

The agent retrieves the stock price action in the weeks surrounding the earnings date, pulls news articles from that period, gathers Reddit sentiment from relevant subreddits, checks stock volume for unusual activity, and searches the embedded earnings filing for management commentary on results. It presents a unified narrative: here's what the price did, here's what was being reported, here's what retail investors were saying, here's what the filing revealed, and here's how trading volume compared to normal levels. All sourced.

**Use Case 2: Social Sentiment-Driven Event Analysis**

The user asks: "What was going on with GME during the WallStreetBets rally?"

The agent identifies the relevant time period (January 2021), retrieves the extreme price action, surfaces the Reddit sentiment data showing the coordinated buying activity on r/wallstreetbets, pulls news articles covering the short squeeze narrative, and shows the volume spike that accompanied the rally. The narrative ties all of these together to paint the full picture of a price movement that only makes sense when the social dimension is included.

**Use Case 3: Historical Comparison Research**

The user asks: "How did NVIDIA calls do around last quarter's earnings?"

The agent retrieves the price movement around that earnings date, shows how much the stock moved in the days before and after, surfaces the options volume data or volume-based proxy analysis for that period, and pulls relevant news and sentiment. The user receives a factual summary of what happened, including how the stock moved relative to how traders were positioned. No recommendation is made. The agent presents historical facts and the user draws their own conclusions.

**Use Case 4: Current Options Market View**

The user asks: "What does the options chain look like for Tesla right now?"

The agent pulls the current options chain data from yfinance, calculates the put/call ratio, identifies strikes with unusually high volume or open interest, and notes the implied volatility levels. It presents this as a snapshot of how the options market is currently positioned for Tesla.

**Use Case 5: General Stock Performance Lookup**

The user asks: "How did Microsoft perform over the last 3 weeks?"

The agent parses the date range, retrieves the historical price data, and presents the key metrics: opening price, closing price, percentage change, high, low, and volume. This is the simplest use case and serves as the foundation that the more complex analyses build on.

**Use Case 6: Historical and Current Comparative Analysis**

The user asks: "How did NVIDIA do around Q2 earnings last year? And what's happening with it right now?"

The agent handles both dimensions in a single response. For the historical component, it retrieves price action around Q2 2024 earnings, news from that period, Reddit sentiment, and relevant earnings filing context. For the current component, it fetches recent news articles about NVIDIA and pulls the current options chain to show how traders are positioned now. The response presents both sections clearly: here's what happened historically, followed by here's what the current landscape looks like. No recommendation is made about whether the historical pattern will repeat. The user receives the facts from both time periods and draws their own conclusions.

### Out of Scope (Explicitly Not Supported)

The agent does not predict future stock movements. It does not provide buy/sell/hold recommendations. It does not offer financial advice of any kind. It does not perform real-time monitoring or alerting. It presents historical facts and current market data, sourced and attributed, and leaves interpretation entirely to the user.

---

## 4. Feature Scope and Prioritization

### Prioritization Framework

Features are organized into phases based on dependency order (what needs to exist before other things can work), user value (what delivers the most meaningful analysis), and complexity (what can be built within the constraints of free-tier APIs and a single developer).

### Phase 1: Core Analysis Foundation

These are the features that must exist for the product to deliver any value. They form the base that all other features build on.

- Natural language query interface via Chainlit chat UI
- Stock ticker resolution from company names (e.g., "NVIDIA" resolves to NVDA)
- Natural language date range parsing (e.g., "last 3 weeks" or "around Q2 2024 earnings")
- Historical stock price retrieval with key metrics: open, close, high, low, volume, percent change
- Dual data source strategy for price data with yfinance as primary and Alpha Vantage as fallback
- Interactive candlestick chart generation via Plotly
- LangGraph multi-node workflow with intent classification and conditional routing
- Structured state management across all nodes
- Basic error handling and graceful failure messaging

*Status: Complete. All Phase 2 nodes built and wired into workflow.*

### Phase 2: Multi-Source Intelligence

These features transform the product from a stock data lookup tool into the multi-dimensional analysis engine described in the product vision. This is the phase that delivers the core value proposition.

- News retrieval with NewsAPI as primary source and Google News RSS as fallback for older dates
- LLM-powered news summarization with source attribution (article title, source, date, URL cited in response)
- Reddit sentiment retrieval from relevant subreddits (r/wallstreetbets, r/stocks, r/options) via PRAW
- LLM-based sentiment scoring of Reddit posts (bullish, bearish, neutral with quantified breakdown)
- Stock volume anomaly detection as a proxy signal for unusual trading activity
- Multi-source narrative synthesis where the response ties together price, news, sentiment, and volume into a coherent explanation
- All factual claims grounded in retrieved data with no LLM fabrication

*Dependency: Requires Phase 1 to be complete. The multi-node graph architecture from Phase 1 is what enables parallel data retrieval and multi-source synthesis.*

### Phase 3: Deep Analysis via RAG

This phase adds depth through earnings report analysis, building a growing knowledge base that makes the agent more capable over time.

- SEC EDGAR integration to retrieve and embed the following document types for any public company:
  - **10-K** (annual report): full-year financial results, MD&A, risk factors
  - **10-Q** (quarterly report): quarterly financials and management commentary
  - **8-K** (material event filing): earnings releases, executive changes, mergers, guidance updates — the real-time signal layer
  - **Earnings call transcripts** (filed as 8-K exhibits): management tone, forward guidance, analyst Q&A — what traders actually read
- Document chunking and embedding pipeline using Google Gemini embedding API
- ChromaDB vector store with metadata tagging (ticker, filing type, quarter, date)
- Deduplication via document-level IDs to prevent redundant storage
- RAG retrieval node in the graph that searches relevant filing chunks based on user query
- On-demand ingestion: first query for a filing triggers download, chunking, and embedding, with subsequent queries served from the vector store
- Integration of filing context into the multi-source narrative so the agent can reference what management specifically said about results
- At Phase 5 deployment, ChromaDB migrates from local embedded mode to ChromaDB Cloud free tier (1GB hosted storage) — same API, one config change, eliminates local disk dependency on the deployed container

*Dependency: Requires Phase 2. The RAG node plugs into the existing multi-source synthesis flow as an additional data dimension. Build order: Phase 4 (Options) is being built before Phase 3 — see Decision 14 in the Decision Log.*

### Phase 4: Options Analysis

This phase adds the options market dimension to complete the full picture described in the product vision.

- Current options chain retrieval from yfinance (strikes, volume, open interest, implied volatility)
- Put/call ratio calculation and interpretation
- Identification of unusual options volume or open interest concentrations
- Implied volatility analysis, particularly around earnings dates
- Greeks calculation (Delta, Gamma, Theta, Vega) using Black-Scholes with yfinance-supplied implied volatility — enables traders to assess directional exposure, time decay cost, and IV sensitivity
- Max Pain calculation from open interest across all strikes — the strike where the most contracts expire worthless, a level market makers are incentivized to pin toward at expiry
- For historical queries, volume-based proxy analysis using stock trading volume as an indicator of elevated options activity
- Analyst ratings and price targets from yfinance (mean target, high/low range, number of analysts, recent recommendation breakdown) — high-signal context for directional bias
- Short interest data from yfinance (short float percentage, days to cover, month-over-month change) — critical for identifying squeeze potential or sustained bearish positioning
- Earnings date awareness: next earnings date and days remaining, surfaced prominently in options responses to contextualize elevated IV before earnings and expected IV crush after
- Integration of all options and enrichment context into the multi-source narrative

*Dependency: Independent of Phase 3. Being built before Phase 3 (see Decision 14). Options data retrieval feeds into the same synthesis node as the RAG pipeline but requires no RAG infrastructure.*

### Phase 5: User Configuration and Infrastructure

This phase focuses on deployment, configurability, and production readiness.

- Docker containerization of the full application
- Azure deployment with environment and secrets management
- GitHub Actions CI/CD pipeline for automated testing and deployment
- Settings UI for users to optionally provide their own API keys (OpenAI, Anthropic, NewsAPI, Twitter)
- Provider abstraction layer: when premium API keys are provided, the agent uses them; otherwise falls back to free defaults (Groq for LLM, Google Gemini for embeddings, RSS for news)
- Test suite covering core tools, graph routing, and data retrieval functions

*Dependency: The application must be functionally complete (Phases 1 through 4) before deployment and configuration are prioritized. Tests should be written incrementally alongside each phase.*

### Explicitly Out of Scope

- Real-time stock monitoring or streaming data
- Push notifications or alerting systems
- User accounts or authentication
- Investment recommendations or predictive analysis of any kind
- Paid subscription tiers or monetization infrastructure
- Mobile application
- Twitter/X integration (free tier too limited to be useful)

---

## 5. Technical Constraints and Dependencies

### API Dependencies and Limitations

The product relies on several external data sources, each with its own access limitations. The architecture is designed so that no single source failure breaks the entire experience. If one source is unavailable, the agent still delivers analysis from the remaining sources and transparently tells the user what it could not retrieve.

| Data Source | Purpose | Limitation | Mitigation |
| --- | --- | --- | --- |
| yfinance | Stock price data, current options chains | Unofficial Yahoo Finance API. No guaranteed uptime. No real-time streaming. | Alpha Vantage as fallback for price data. Acceptable for historical analysis use case. |
| Alpha Vantage | Fallback for stock price data | Free tier limited to 5 API calls per minute, 500 per day | Used only when yfinance fails. Rate limiting logic in data fetcher node. |
| NewsAPI | News article retrieval | Free developer tier only covers the last 30 days of articles | Google News RSS fallback for dates older than 30 days. |
| Google News RSS | Fallback news retrieval | Unstructured results. Inconsistent coverage. No API guarantees. | Best-effort retrieval. Agent discloses when news coverage for a period is limited. |
| Reddit API (PRAW) | Social sentiment data | Rate limited. Historical search has depth limits depending on Reddit's API behavior. | Caching of retrieved posts. Agent discloses when Reddit data for a period is sparse. |
| SEC EDGAR | Earnings filings (10-Q, 10-K) | Free and open, but response times vary. Large documents take time to process. | On-demand ingestion with user-facing loading state. Once ingested, filings are served from local vector store. |
| Groq | LLM inference (LLaMA 3.1 8B) | Free tier with rate limits on requests per minute and tokens per minute. Model is 8B parameters, which is less capable than larger models. | Prompts are designed to be focused and specific to work well within the 8B model's capabilities. Users can optionally provide their own OpenAI or Anthropic keys for a more capable model. |
| Google Gemini API | Text embeddings for RAG pipeline | Free tier allows 1,500 requests per day | Sufficient for on-demand ingestion. Batch chunking to minimize request count per filing. |

### Hardware Constraints

Development is being done on a 2019 MacBook Pro with 8GB RAM and 256GB storage. This means no local model hosting, no large dataset storage, and all heavy computation must happen through cloud APIs or on the deployed Azure instance. The architecture reflects this: all LLM inference and embedding generation happens via cloud APIs, and ChromaDB runs in embedded mode with a minimal disk footprint.

### Cost Constraints

The project operates on a zero-cost budget for development. All APIs and services used are free tier. The Azure deployment uses student credits. This constraint influenced several architectural decisions documented in the Decision Log, including the choice of Groq over OpenAI, Google Gemini over local embedding models, and the fallback strategy for data sources.

### Key Architectural Constraint

The agent does not generate factual claims from the LLM's training data. All factual statements in the agent's responses must originate from data retrieved by the tool nodes (price data, news articles, Reddit posts, earnings filings). The LLM's role is limited to intent classification, sentiment scoring, and narrative synthesis over retrieved data. This is a deliberate design choice to prevent hallucination in a domain where accuracy matters.

---

## 6. Success Metrics

### Product Functional Metrics

These metrics measure whether the agent is delivering on its core value proposition of assembling a complete, accurate, multi-source analysis.

**Data Retrieval Completeness:** For any given query, how many of the available data dimensions did the agent successfully retrieve? If a user asks about NVIDIA around earnings and the agent returns price data, news, sentiment, and earnings filing context, that's 4 out of 4. If it only returns price data because the other sources failed, that's 1 out of 4. The target is that the agent successfully retrieves at least 3 out of 4 data dimensions for any query involving a major stock and a date range within the last 2 years.

**Source Attribution Rate:** What percentage of factual claims in the agent's response are tied to a specific source (article URL, filing reference, Reddit post)? The target is 100%. Any factual claim without attribution is a failure of the synthesis node's prompting.

**Factual Grounding Rate:** What percentage of the agent's factual claims are actually supported by the retrieved data versus hallucinated by the LLM? This can be tested by manually comparing the agent's response against the raw data it received. The target is 100%. Any hallucinated claim is a critical bug, not an acceptable error rate.

**Response Relevance:** When a user asks about a specific event or time period, does the agent's response actually address that event and time period, or does it return generic information? This is evaluated qualitatively during testing. The target is that every response directly addresses the specific stock, date range, and context of the user's question.

**Graceful Degradation:** When a data source fails or returns no results, does the agent transparently communicate what it could not retrieve rather than silently omitting it or fabricating information? The target is that every partial failure is disclosed to the user. For example: "I was able to retrieve price data and news coverage for this period, but Reddit data was not available for dates this far back."

**RAG Retrieval Relevance:** When the agent searches the vector store for earnings filing context, are the returned chunks actually relevant to the user's question? For example, if the user asks about revenue growth, the RAG node should return chunks discussing revenue, not unrelated sections about legal disclaimers. This is evaluated by reviewing the chunks retrieved versus the query. The target is that at least 3 out of 5 returned chunks are directly relevant.

**Response Time:** For a standard query (stock price lookup with news and sentiment), the agent should return a complete response within 30 seconds. For queries that trigger on-demand filing ingestion, the agent should communicate a loading state within 5 seconds and complete the full response within 90 seconds.

### Portfolio and Learning Metrics

These metrics measure whether the project achieves its secondary purpose as a learning exercise and interview portfolio piece.

**Architecture Explainability:** Can the builder whiteboard the full LangGraph workflow from memory, explain what each node does, describe the state schema, and articulate why the graph is structured the way it is? This is tested by practicing the explanation without looking at code or documentation.

**Tradeoff Articulation:** For every major technical decision (framework choice, vector database, embedding model, cloud provider, data source), can the builder explain what alternatives were considered, what the tradeoffs were, and why this choice was made? This is tested by simulating interview questions.

**End-to-End Demo Reliability:** When demonstrating the product live (in an interview or portfolio review), does the agent successfully handle at least 5 consecutive queries without errors, timeouts, or hallucinated responses? Live demos failing is the single biggest risk for portfolio projects. The target is that the core demo flow works reliably every time.

**Code Comprehension:** Can the builder read any file in the codebase and explain what it does, why it's structured that way, and how it connects to the rest of the system? This is critical because the code was not written from scratch by the builder. The builder needs to understand it thoroughly enough that this distinction becomes irrelevant.

**Documentation Completeness:** Does the PM documentation (PRD, Technical Design Document, Decision Log, Roadmap) tell a coherent story from problem identification through technical design through implementation? Could someone read these documents and fully understand what the product does, why it exists, and how it was built?

---

## 7. Assumptions and Risks

### Assumptions

These are things we are assuming to be true that, if proven wrong, would require us to revisit parts of the product design.

**Free API availability remains stable.** The product depends on free tiers from Groq, yfinance, Google Gemini, Reddit, and SEC EDGAR. We assume these services will continue offering free access at current rate limits. If any of these providers eliminate their free tier or significantly reduce rate limits, the affected data dimension would need an alternative source or would degrade in quality.

**yfinance continues to function reliably.** yfinance is an unofficial wrapper around Yahoo Finance data. It is not an officially supported API and Yahoo could block or restrict access at any time. The Alpha Vantage fallback mitigates this partially for price data, but yfinance is also the sole source for current options chain data. If yfinance breaks, options analysis would be unavailable until an alternative source is integrated.

**Reddit remains a meaningful proxy for retail investor sentiment.** The product uses Reddit as its primary social sentiment source. We assume that subreddits like r/wallstreetbets, r/stocks, and r/options continue to be active communities where retail investors discuss stock positions and market events. If retail investor discussion migrates to another platform (Discord, a new social network), the sentiment data would become less representative.

**The 8B parameter LLM is sufficient for the required reasoning tasks.** The free Groq tier provides access to LLaMA 3.1 8B. We assume this model is capable enough to handle intent classification, sentiment scoring, and narrative synthesis. If the model consistently produces poor quality analysis or struggles with multi-source synthesis, the product would need a more capable model, which may require paid API access or a user-provided key.

**SEC EDGAR filings are parseable at query time.** We assume that earnings filings can be downloaded, chunked, embedded, and stored within a reasonable timeframe (under 90 seconds) during a user's query. If filings are significantly larger or more complex than expected, on-demand ingestion may be too slow and we would need to shift to pre-ingestion for common stocks.

**Users ask questions about stocks with sufficient publicly available data.** The product works best for large-cap, widely covered stocks (NVIDIA, Apple, Tesla, etc.) where news coverage, Reddit discussion, and options data are plentiful. For small-cap or obscure stocks, some data dimensions may return little or no results. We assume the primary user base is interested in well-known, actively traded stocks.

### Risks

These are things that could go wrong and how we would respond.

**Risk: LLM hallucination in financial context.**

**Severity: High.** Financial misinformation can lead to real monetary harm for users.

Mitigation: The architecture enforces a strict separation between data retrieval and narrative synthesis. The LLM never generates factual claims from its training data. All factual statements must originate from tool-retrieved data. The synthesis prompt explicitly instructs the LLM to only reference provided data and to state when information is unavailable. Source attribution is required for every claim. Testing includes manual comparison of agent responses against raw retrieved data to catch any hallucination.

**Risk: Data source outage during a live demo or interview.**

**Severity: High.** A failed demo undermines the entire portfolio.

Mitigation: The agent is designed for graceful degradation. If one source fails, the others still return data and the agent communicates what it could not retrieve. For demo purposes, we can prepare a set of known-good queries for well-covered stocks and recent time periods where data availability is reliable. Additionally, having cached responses from previous successful queries (stored in ChromaDB) provides a safety net if live retrieval fails.

**Risk: Rate limiting during sustained use.**

Severity: Medium. If multiple queries are made in quick succession, free tier rate limits on Groq, NewsAPI, or Reddit could throttle responses.

Mitigation: Implement basic rate limiting awareness in the data retrieval nodes. If a rate limit is hit, the agent communicates the delay to the user rather than failing silently. For the RAG pipeline, previously ingested filings are served from the local vector store without any API calls, reducing load on external services.

**Risk: Reddit historical data is sparse for older events.**

Severity: Medium. Reddit's search API does not guarantee deep historical coverage. Posts from 2021 (like the GME rally) may be retrievable, but older events may have limited data.

Mitigation: The agent discloses when Reddit data is limited for a given time period. The analysis still proceeds with the other available dimensions (price, news, filings). The response clearly indicates which sources contributed to the analysis and which were unavailable.

**Risk: Earnings filing ingestion is too slow for good user experience.**

Severity: Medium. A 60-90 second wait for first-time filing ingestion may frustrate users.

Mitigation: The agent communicates a clear loading state explaining that it is processing the filing for the first time and that subsequent queries about the same company will be faster. The ingestion pipeline is optimized for chunk size and batch embedding to minimize processing time.

**Risk: Sentiment scoring accuracy is unreliable.**

Severity: Low to Medium. If the LLM misclassifies sentiment (calling a bearish post bullish or vice versa), the sentiment summary could misrepresent actual market mood.

Mitigation: Sentiment scoring uses a simple three-category classification (bullish, bearish, neutral) rather than attempting fine-grained scoring, which reduces error. Aggregate scores across many posts are more reliable than any individual classification. The agent reports the number of posts analyzed so the user can gauge sample size. If sentiment data is available for only a handful of posts, the agent flags that the sample is small and the scoring may not be representative.