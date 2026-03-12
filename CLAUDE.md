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
Partially working prototype. Needs a full multi-node rebuild per TDD.

What exists:
- agent/graph/nodes/tool_caller.py: monolithic god-node; also contains inline chart logic
- agent/graph/edges/decision_router.py: routes to END if last message is AIMessage; duplicates AgentState
- agent/graph/workflow.py: single-node graph, tool_caller + conditional edge to END
- app/chainlit/app.py: Chainlit UI; detects PLOTLY_JSON: prefix for chart rendering
- tools/stockprice/stock_analyzer.py: yfinance primary, Alpha Vantage fallback
- tools/date/date_parser_tool.py: regex-based relative date parsing
- tools/news/news_scraper.py: RSS-based scraper; not wired into graph
- llm/llm_setup.py: Groq llama-3.1-8b-instant, temperature=0.7, max_tokens=512

Known issues to fix during rebuild:
- temperature=0.7 is wrong for classifier/extractor nodes; use temperature=0 for structured JSON output
- max_tokens=512 is too low for response_synthesizer; use a separate LLM config with higher token limit
- AgentState is duplicated in tool_caller.py and decision_router.py; consolidate into state.py
- Do not treat existing code as sacred; refactor or replace as needed

## Build Sequence (Phase 1 - current focus)
Read docs/TDD.md for full node specs before implementing any step.

1. agent/graph/nodes/state.py (AgentState TypedDict, all fields)
2. intent_classifier.py + tests/test_intent_classifier.py
3. ticker_resolver.py + tests/test_ticker_resolver.py
4. date_parser.py + tests/test_date_parser.py
5. data_fetcher.py + tests/test_data_fetcher.py
6. chart_generator.py + tests/test_chart_generator.py
7. response_synthesizer.py + tests/test_response_synthesizer.py
8. error_handler.py + tests/test_error_handler.py
9. workflow.py rebuild (wire all Phase 1 nodes with conditional edges)
10. app/chainlit/app.py update (read final_response from state, not messages[-1])
11. LangSmith tracing

Phase 2 (after Phase 1 committed and tested): news_retriever, reddit_sentiment
Phase 3: rag_pipeline with ChromaDB

## Rules
- Read docs/TDD.md before implementing any node
- One node per session; write unit tests alongside each node
- Do NOT touch workflow.py while implementing individual nodes
- Do NOT touch app/chainlit/app.py until workflow.py is rebuilt
- Commit after each node passes tests
- No LangChain Tool wrappers; use direct Python function calls
- No print statements in node files; use logging
- All nodes must write to their *_error state field on failure

## Commands
```bash
source .venv/bin/activate
PYTHONPATH=. chainlit run app/chainlit/app.py
PYTHONPATH=. pytest tests/
PYTHONPATH=. pytest tests/test_intent_classifier.py
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