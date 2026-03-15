# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
AI-powered stock analysis agent. Portfolio project targeting AI engineering and AI PM roles.
Single-agent LangGraph architecture. NOT multi-agent.

## Documentation (read before implementing anything)
- docs/TDD.md: Source of truth for architecture, node specs, state schema, routing logic
- docs/PRD.md: Source of truth for feature scope and acceptance criteria
- docs/DecisionLog.md: 11 ADR-format architectural decisions

## Current Codebase State
Phase 1 rebuild in progress. Do not treat existing prototype files as sacred.

Phase 1 nodes completed:
- agent/graph/nodes/state.py: ✅ AgentState TypedDict (28 fields, all nodes documented)

Prototype files (will be replaced, do not modify):
- agent/graph/nodes/tool_caller.py: monolithic god-node; also contains inline chart logic
- agent/graph/edges/decision_router.py: routes to END if last message is AIMessage; duplicates AgentState
- agent/graph/workflow.py: single-node graph, tool_caller + conditional edge to END
- app/chainlit/app.py: Chainlit UI; detects PLOTLY_JSON: prefix for chart rendering
- tools/stockprice/stock_analyzer.py: yfinance primary, Alpha Vantage fallback
- tools/date/date_parser_tool.py: regex-based relative date parsing
- tools/news/news_scraper.py: RSS-based scraper; not wired into graph

LLM config (llm/llm_setup.py) — needs two configs during rebuild:
- llm_classifier: temperature=0, max_tokens=256 (intent, ticker, date nodes — structured JSON output)
- llm_synthesizer: temperature=0.3, max_tokens=1024 (response_synthesizer — narrative output)
- AgentState duplication in tool_caller.py and decision_router.py is resolved; both will import from state.py

## Build Sequence (Phase 1 - current focus)
Read docs/TDD.md for full node specs before implementing any step.
Done when: all tests in tests/test_<node>.py pass and the node is committed.

1. ✅ agent/graph/nodes/state.py (AgentState TypedDict, all fields) — committed
2. ✅ agent/graph/nodes/intent_classifier.py + tests/test_intent_classifier.py — committed
3. ✅ agent/graph/nodes/ticker_resolver.py + tests/test_ticker_resolver.py — committed
4. ✅ agent/graph/nodes/date_parser.py + tests/test_date_parser.py — committed
5. ✅ agent/graph/nodes/data_fetcher.py + tests/test_data_fetcher.py — committed
6. ✅ agent/graph/nodes/chart_generator.py + tests/test_chart_generator.py — committed
7. ✅ agent/graph/nodes/response_synthesizer.py + tests/test_response_synthesizer.py — committed
8. ✅ workflow.py rebuild (wire all Phase 1 nodes with conditional edges) — committed
9. ✅ app/chainlit/app.py update (reads response_text/chart_data from state) — committed
10. ✅ LangSmith tracing (LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT set in .env)

Phase 2 (after Phase 1 committed and tested): news_retriever, reddit_sentiment
Phase 3: rag_pipeline with ChromaDB

## Rules
- Read docs/TDD.md before implementing any node
- One node per session; write unit tests alongside each node
- Do NOT touch workflow.py while implementing individual nodes
- Do NOT touch app/chainlit/app.py until workflow.py is rebuilt
- Commit after each node passes tests (no Co-Authored-By in commit messages)
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

# Create tests directory (one-time setup before Step 2)
mkdir -p tests && touch tests/__init__.py
```

## Environment Variables (.env)
```
GROQ_API_KEY=                    # Required
ALPHA_VANTAGE_API_KEY=           # Optional fallback
CHROMA_PERSIST_DIR=data/vector_store   # Phase 3, local ChromaDB
LANGSMITH_API_KEY=               # Phase 1 step 11
LANGCHAIN_TRACING_V2=true        # Phase 1 step 11
LANGCHAIN_PROJECT=stock-insight-agent  # Phase 1 step 11
```

## Git Commit Format
```
feat: implement <node_name> node
refactor: migrate <tool> into <node_name> node
test: add tests for <node_name>
fix: <what was broken>
chore: add LangSmith tracing
```

## Tech Stack
Python 3.11+, LangGraph, Chainlit, Groq (llama-3.1-8b-instant),
yfinance, Alpha Vantage (fallback), ChromaDB (Phase 3), LangSmith, pytest