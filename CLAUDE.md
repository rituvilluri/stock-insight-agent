# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Collaboration Style
- User is learning Claude Code best practices. When a more optimal tool, workflow, or approach is available, suggest it proactively with a brief explanation of why.
- After any meaningful implementation, prompt the user to update docs/TDD.md, docs/PRD.md, or docs/DecisionLog.md where relevant. "Meaningful" means: new node, changed routing logic, new state fields, new external dependency, architectural trade-off made, or scope change. Do not prompt for minor bug fixes or styling changes.

## Project
AI-powered stock analysis agent. Portfolio project targeting AI engineering and AI PM roles.
Single-agent LangGraph architecture. NOT multi-agent.

## Documentation (read before implementing anything)
- docs/TDD.md: Source of truth for architecture, node specs, state schema, routing logic
- docs/PRD.md: Source of truth for feature scope and acceptance criteria
- docs/DecisionLog.md: 17 ADR-format architectural decisions

## Current Codebase State
All Phase 1–4 nodes are complete and wired into the workflow. The codebase is in active eval and improvement mode (phase3/testing-langsmith branch).

## Build Sequence (completed through Phase 4)
Read docs/TDD.md for full node specs before implementing any new node.
Done when: all tests in tests/test_<node>.py pass and the node is committed.

### Phase 1: Core Foundation ✅
1. ✅ agent/graph/nodes/state.py (AgentState TypedDict, all fields)
2. ✅ agent/graph/nodes/intent_classifier.py + tests/test_intent_classifier.py
3. ✅ agent/graph/nodes/ticker_resolver.py + tests/test_ticker_resolver.py
4. ✅ agent/graph/nodes/date_parser.py + tests/test_date_parser.py
5. ✅ agent/graph/nodes/data_fetcher.py + tests/test_data_fetcher.py
6. ✅ agent/graph/nodes/chart_generator.py + tests/test_chart_generator.py
7. ✅ agent/graph/nodes/response_synthesizer.py + tests/test_response_synthesizer.py
8. ✅ agent/graph/workflow.py (all nodes wired with conditional edges)
9. ✅ app/chainlit/app.py (streaming via astream_events, Gemini thinking-chunk handling)
10. ✅ LangSmith tracing (LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT set in .env)

### Phase 2: Multi-Source Intelligence ✅
- ✅ Session context memory (app.py + date_parser.py guard)
- ✅ Node 5: news_retriever.py + tests/test_news_retriever.py (Finnhub → You.com → Google RSS)
- ✅ Node 6: reddit_sentiment.py + tests/test_reddit_sentiment.py
- ✅ workflow.py updated (Nodes 5 and 6 wired; parallel fan-out via Send())

### Phase 3: RAG Pipeline ✅
- ✅ Node 7: rag_retriever.py + tests/test_rag_retriever.py
  - SEC EDGAR on-demand ingestion (10-K, 10-Q; 8-K deferred to next iteration)
  - ChromaDB embedded mode (data/vector_store), Google Gemini embeddings
  - Wired into stock_analysis fan-out alongside nodes 5 and 6

### Phase 4: Options + Enrichment ✅
- ✅ Node 8: options_analyzer.py + tests/test_options_analyzer.py
  - Black-Scholes Greeks (stdlib only), Max Pain, put/call ratio, top-volume strikes
- ✅ Node 4 enrichments: analyst_data, short_interest, next_earnings_date, days_until_earnings

### Phase 5: Deployment (upcoming)
- ⬜ Docker containerization
- ⬜ Azure deployment + GitHub Actions CI/CD
- ⬜ ChromaDB Cloud migration (1GB free tier, same API)

## LLM Config (llm/llm_setup.py)
- `llm_classifier`: Groq `llama-3.1-8b-instant`, temperature=0, max_tokens=256
  - Used by: Intent Classifier, Ticker Resolver, Date Parser (structured JSON output)
- `llm_synthesizer`: Google Gemini 2.5 Flash, temperature=0.3, max_output_tokens=4096, thinking_budget=1024
  - Used by: Response Synthesizer (narrative synthesis across multi-source data)
  - Streaming enabled; Chainlit handles thinking-phase chunk format from Gemini

## Rules
- Read docs/TDD.md before implementing any node
- One node per session; write unit tests alongside each node
- Do NOT touch workflow.py while implementing individual nodes
- Commit after each node passes tests (no Co-Authored-By in commit messages or PRs)
- No LangChain Tool wrappers; use direct Python function calls
- No print statements in node files; use logging
- All nodes must write to their *_error state field on failure

Every node file must follow this structure:
```python
import logging
from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)

def <node_name>(state: AgentState) -> AgentState:
    try:
        # ... logic here ...
        return {**state, "<output_field>": result, "<node>_error": None}
    except Exception as e:
        logger.error(f"<node_name> failed: {e}")
        return {**state, "<node>_error": str(e)}
```

## Commands
```bash
# Activate virtual environment first (required before any other command)
source .venv/bin/activate

# Run the app
PYTHONPATH=. chainlit run app/chainlit/app.py

# Run all tests
PYTHONPATH=. pytest tests/

# Run a single node's tests
PYTHONPATH=. pytest tests/test_intent_classifier.py -v

# Run LangSmith experiment
PYTHONPATH=. python tests/evaluators/run_experiment.py
```

## LangSmith Experiment Naming
Before running `run_experiment.py`, always update `EXPERIMENT_NAME` in that file to a descriptive name reflecting what changed. Format: `<what-changed>` (kebab-case, no version numbers).

Examples:
- `post-intent-label-fix`
- `post-rag-threshold-tuning`
- `vertex-claude-synthesizer-trial`
- `post-hallucination-prompt-update`

Never run an experiment with the stale name from the previous run. A good name makes the LangSmith experiment list readable without opening each one.

## Environment Variables (.env)
```
GROQ_API_KEY=                    # Required — llm_classifier (llama-3.1-8b-instant)
GEMINI_API_KEY=                  # Required — llm_synthesizer (Gemini 2.5 Flash) + RAG embeddings
ALPHA_VANTAGE_API_KEY=           # Optional — fallback for stock price data
FINNHUB_API_KEY=                 # Node 5 Layer 1 — primary news (60 calls/min free, ~2yr history)
YOUCOM_API_KEY=                  # Node 5 Layer 2 — You.com Search API (fallback when Finnhub misses)
REDDIT_CLIENT_ID=                # Node 6
REDDIT_CLIENT_SECRET=            # Node 6
REDDIT_USER_AGENT=stock-insight-agent/1.0  # Node 6
CHROMA_PERSIST_DIR=data/vector_store       # Node 7 — local ChromaDB (migrates to Cloud at Phase 5)
LANGSMITH_API_KEY=               # LangSmith tracing + evals
LANGCHAIN_TRACING_V2=true        # LangSmith tracing
LANGCHAIN_PROJECT=stock-insight-agent      # LangSmith project name
```

## Git Commit Format
```
feat: implement <node_name> node
refactor: migrate <tool> into <node_name> node
test: add tests for <node_name>
fix: <what was broken>
chore: <housekeeping>
```

## Tech Stack
Python 3.11+, LangGraph, Chainlit, Groq (llama-3.1-8b-instant classifier),
Google Gemini 2.5 Flash (synthesizer + embeddings), yfinance, Alpha Vantage (fallback),
ChromaDB (Phase 3 RAG), LangSmith, pytest
