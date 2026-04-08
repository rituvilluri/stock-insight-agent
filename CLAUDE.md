# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project

AI-powered stock analysis agent. Portfolio project targeting AI engineering and AI PM roles.

**Architecture:** Single-agent LangGraph workflow with adaptive parallel retrieval orchestration. One graph drives intent classification → ticker and date resolution → parallel multi-source data retrieval fan-out (news, sentiment, SEC filings, options) → narrative synthesis. Each node is a plain Python function. No sub-agents, no tool-calling loops.

## Current Codebase State

Phases 1–4 complete. Phase 5 is active: quality improvements across Node 6 (sentiment), Node 5 (news), response synthesizer, date parser, and a new retrieval planning node.

**Phase 5 is the improvement and eval phase** — every change to an existing node requires updating its tests and re-running a LangSmith experiment before the story is closed.

## Documentation

Four sources of truth — do not duplicate content between them:

| File | Owns |
|------|------|
| `docs/TDD.md` | Architecture, node specs, state schema, routing logic |
| `docs/PRD.md` | Feature scope and acceptance criteria |
| `docs/DecisionLog.md` | Architectural decisions in ADR format (17 entries) |
| `CLAUDE.md` | Claude Code behavior, workflows, conventions |

Notion mirrors these as PM-level narrative — see Notion Workflow section.

## Build Sequence

### Phases 1–4 ✅ (complete)

All 10 nodes wired and tested. Key files:
- `agent/graph/nodes/state.py` — AgentState TypedDict (all fields)
- `agent/graph/nodes/intent_classifier.py` — Node 1
- `agent/graph/nodes/ticker_resolver.py` — Node 2
- `agent/graph/nodes/date_parser.py` — Node 3
- `agent/graph/nodes/data_fetcher.py` — Node 4 (price, analyst data, short interest, earnings)
- `agent/graph/nodes/news_retriever.py` — Node 5 (Finnhub → You.com → Google RSS)
- `agent/graph/nodes/reddit_sentiment.py` — Node 6 (Reddit public JSON + Stocktwits)
- `agent/graph/nodes/rag_retriever.py` — Node 7 (SEC EDGAR, ChromaDB, Gemini embeddings)
- `agent/graph/nodes/options_analyzer.py` — Node 8 (Black-Scholes, Max Pain, put/call ratio)
- `agent/graph/nodes/response_synthesizer.py` — Node 9 (Gemini 2.5 Flash, thinking enabled)
- `agent/graph/nodes/chart_generator.py` — Node 10 (Plotly candlestick)
- `agent/graph/workflow.py` — LangGraph graph with conditional edges and Send() fan-out

### Phase 5: Quality + Deployment (active)

- ✅ Node 6 rewrite — replace PRAW with Reddit public JSON + Stocktwits as co-primary source
- ✅ Node 5 upgrade — parallel Finnhub+You.com fetch, Firecrawl enrichment for free-domain articles
- ✅ Response synthesizer prompt redesign — sharp narrative tone, inline [1][2] citations, 350–500 words
- ⬜ Date parser fix — fiscal calendar awareness for earnings queries (anchor to next_earnings_date)
- ⬜ Retrieval planning node — LLM-driven; outputs retrieval_plan dict; requires workflow.py changes (coordinate in same session — see Rules)
- ⬜ Docker containerization
- ⬜ Azure deployment + GitHub Actions CI/CD
- ⬜ ChromaDB Cloud migration (1GB free tier, same API)

## Commands

```bash
# Activate virtual environment (required before any other command)
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

## LLM Config (`llm/llm_setup.py`)

Three distinct roles — never swap models between roles without updating this section:

- **`llm_classifier`** — Groq `llama-3.1-8b-instant`, temperature=0, max_tokens=256
  - Nodes: Intent Classifier (1), Ticker Resolver (2), Date Parser (3), Reddit/Stocktwits sentiment batches (6)
  - Produces structured JSON; fast and cheap

- **`llm_synthesizer`** — Google Gemini 2.5 Pro (Vertex AI), temperature=0.3, max_output_tokens=4096, thinking_budget=2048
  - Node: Response Synthesizer (9)
  - Streaming enabled; Chainlit handles Gemini thinking-chunk format
  - `thinking_budget=2048` gives the model an internal reasoning pass before the final response
  - Auth via Application Default Credentials (ADC) — no API key required

- **`llm_planner`** (Phase 5, new) — Groq `llama-3.1-8b-instant`, temperature=0, max_tokens=512
  - Node: Retrieval Planning Node
  - Lightweight decision: which data nodes to activate based on query signals

## MCP Connections

Available integrations — use these proactively when the task maps to their domain. Do not substitute web search or file reads when an MCP handles it directly.

| MCP | Connects to | When to use |
|-----|-------------|-------------|
| `mcp__atlassian` | JIRA project: **STOCK** | Starting/closing stories, checking sprint, adding eval result comments |
| `mcp__notionApi` | Notion: **Stock Insight Agent** workspace | Updating docs at phase milestones — see Notion Workflow |
| `mcp__langsmith` | LangSmith | Fetching experiment runs, comparing eval results, pushing prompts to Hub |
| `mcp__github` | GitHub repo | Opening PRs, checking CI status, reviewing PR comments |
| `mcp__plugin_playwright` | Browser (Chromium) | Fallback when WebFetch returns 403 or hits a paywall; UI testing |
| `mcp__plugin_context7` | Live library docs | Fetch current LangGraph, Chainlit, LangSmith SDK docs before implementing against an unfamiliar API |

## Environment Variables (`.env`)

```
# Required
GROQ_API_KEY=                    # llm_classifier + Node 6 sentiment batches

# Google Cloud (Vertex AI) — llm_synthesizer (Gemini 2.5 Pro) + RAG embeddings
# Auth: Application Default Credentials — run `gcloud auth application-default login`
GOOGLE_CLOUD_PROJECT=            # GCP project ID (e.g. stock-insight-agent)
GOOGLE_CLOUD_LOCATION=           # Vertex AI region (e.g. us-central1)

# Data — Node 4
ALPHA_VANTAGE_API_KEY=           # Optional — fallback price data when yfinance fails

# News — Node 5
FINNHUB_API_KEY=                 # Layer 1 — primary news (60 calls/min free, ~2yr history)
YOUCOM_API_KEY=                  # Layer 2 — runs in parallel when Finnhub returns <5 results
FIRECRAWL_API_KEY=               # Required for quality — full-text article enrichment; without it
                                 # synthesis degrades to 300-char snippets (500 credits/month free)

# Sentiment — Node 6
# Reddit: no key required — uses public JSON endpoints (reddit.com/r/*/search.json)
# Stocktwits: no key required — uses public stream (api.stocktwits.com/api/2/streams/symbol/{ticker}.json)

# RAG — Node 7
CHROMA_PERSIST_DIR=data/vector_store   # Local ChromaDB (migrates to Cloud at Phase 5 deployment)

# Observability
LANGSMITH_API_KEY=               # LangSmith tracing + evals
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=stock-insight-agent
```

## Rules

- Read `docs/TDD.md` before implementing any node
- Write unit tests alongside every node change — done means tests pass
- **workflow.py rule:** Do not touch during isolated node work. Exception: the retrieval planning node requires workflow.py changes by design — coordinate both in the same session
- No Co-Authored-By in commits or PRs
- No LangChain Tool wrappers — use direct Python function calls
- No `print()` in node files — use `logging`
- All nodes must write to their `*_error` state field on failure
- When improving an existing node, update its tests and re-run the LangSmith experiment before closing the story

**Required node file structure:**

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

## Git Practices

**Branch naming:**
```
feat/<node-name>         # new node or major feature
fix/<what>               # targeted bug fix
refactor/<what>          # refactor without behavior change
phase<N>/<description>   # multi-node phase branch (e.g. phase5/sentiment-rewrite)
```

**Commit timing:** After each passing test suite — not at end of session. Small, focused commits make rollback easy and history readable.

**Commit format:**
```
feat: implement <node_name> node
refactor: improve <what> in <node_name>
test: add/update tests for <node_name>
fix: <what was broken>
chore: <housekeeping>
```

**Push + open PR:** When a complete story is done — tests pass, LangSmith eval run complete, docs updated. Not mid-story.

**PR description:**
```
## Summary
- What changed and why (2–3 bullets)
- JIRA: STOCK-XXX

## Test plan
- [ ] Unit tests pass
- [ ] LangSmith experiment: <experiment-name>
- [ ] Relevant docs updated
```

**Main branch:** Never force-push to `master`. Never commit directly — always via PR.

## Story Workflow (end-to-end)

The complete loop for every unit of work — follow this order:

```
1. Pull story from JIRA STOCK sprint
   → mcp__atlassian: transition story to In Progress

2. Read the relevant docs/TDD.md section before writing code

3. Create branch
   → git checkout -b feat/<node-name>

4. Implement + write/update tests
   → mcp__ide__getDiagnostics to catch type errors early

5. Update EXPERIMENT_NAME in run_experiment.py, then run eval
   → PYTHONPATH=. python tests/evaluators/run_experiment.py
   → mcp__langsmith: verify results vs. prior experiment

6. Commit
   → feat/fix/refactor/test/chore: <what changed>

7. Open PR
   → mcp__github: include STOCK-XXX and experiment name in description

8. PR merged → close JIRA story
   → mcp__atlassian: transition to Done, add comment with experiment name + one-line result

9. Update repo docs if applicable
   → docs/TDD.md for node spec changes
   → docs/DecisionLog.md for any architectural decisions made during the story

10. Update Notion if this closes a phase milestone
    → see Notion Workflow below
```

## JIRA Workflow (project: STOCK)

**Structure:**
- Epic = Phase (e.g., "Phase 5: Quality + Deployment")
- Story = one node improvement or feature
- Sub-task = individual implementation piece (node file, test file, doc update)

**Before starting a story it must have:**
- Acceptance criteria — what does "done" look like?
- Link to the relevant `docs/TDD.md` section
- Planned LangSmith experiment name — required only for stories that change node logic, prompts, or data sources. Set before writing code so "done" is measurable, not subjective.

**Status transitions:**
- `Backlog` → `To Do` during sprint planning
- `To Do` → `In Progress` when implementation begins
- `In Progress` → `Done` after PR merges and eval results are acceptable

**Commenting:** After the LangSmith run, add a comment to the JIRA story: experiment name + one-line result summary. This keeps the sprint review self-documenting without needing separate notes.

## Notion Workflow (Space: Stock Insight Agent)

Notion is the PM-level narrative layer — plain English summaries for review and portfolio presentation. Repo `docs/` files are the technical source of truth. Summarize in Notion; do not duplicate specs.

Always fetch the current page content via `mcp__notionApi` before editing to avoid overwriting existing content.

| Notion Page | Update when |
|-------------|-------------|
| **Technical Design Document** | A phase completes; a node interface changes significantly; a new node is added |
| **Decision Log** | Every entry added to `docs/DecisionLog.md` — copy decision + rationale in plain English |
| **Product Requirement Document** | Scope changes; a feature is deferred or removed; acceptance criteria shift |
| **Roadmap** | Phase boundaries; deployment timeline updates; major scope decisions |

**Minimum update at phase close:** Update Technical Design Document with a phase summary (what shipped, what was deferred) and Roadmap with next phase status.

## LangSmith Evaluation Workflow

Update `EXPERIMENT_NAME` in `tests/evaluators/run_experiment.py` before every run. Use kebab-case describing what changed — no version numbers:
- `post-reddit-public-json-rewrite`
- `post-synthesizer-prompt-redesign`
- `post-fiscal-calendar-fix`
- `post-rag-threshold-tuning`

**What triggers a run:** Any change to a node's core logic, prompt, or data source. Do not close a Phase 5 story without a completed run.

**How to interpret results:** Use `mcp__langsmith` to fetch the experiment and compare against the prior run. Key metrics: hallucination score, answer relevance, source grounding. If a metric regresses, do not merge — diagnose the cause first.

**Stale name rule:** A name identical to the previous run makes the experiment list unreadable. Always change it before running.

## Cost Optimization (Claude Code Pro)

- Use lighter model variants for exploratory questions ("where is X defined?", "what does this function do?")
- Reserve full capability for implementation, refactoring, and eval analysis
- Use `/compact` when context grows large mid-session — resets the window without losing history
- Press `#` mid-session to capture non-obvious learnings into CLAUDE.md before context is lost
- Run `/remember` at session end to persist key decisions to memory for future sessions
- `MEMORY.md` is auto-loaded at session start — prior decisions are already available, no need to re-derive context

## Collaboration Style

- User is learning Claude Code best practices. When a more optimal tool, workflow, or approach is available, suggest it proactively with a brief explanation of why.
- After any meaningful implementation, prompt to update the relevant repo docs and Notion pages. "Meaningful" = new node, changed routing logic, new state fields, new external dependency, architectural trade-off, or scope change. Skip for minor bug fixes and styling.

## Tech Stack

Python 3.11+, LangGraph, Chainlit, Groq llama-3.1-8b-instant (classifiers + sentiment batches),
Google Gemini 2.5 Flash (synthesizer + embeddings), yfinance, Alpha Vantage (price fallback),
Finnhub + You.com + Google RSS (news), Reddit public JSON + Stocktwits (sentiment),
Firecrawl (news enrichment), ChromaDB (RAG), LangSmith, pytest
