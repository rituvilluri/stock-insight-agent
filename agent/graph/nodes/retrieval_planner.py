"""
Retrieval Planner (Phase 5)

Reads:  user_message, intent, date_context, ticker
Writes: retrieval_plan, planner_error

Sits between Node 4 (Price Data Fetcher) and the parallel retrieval fan-out
(Nodes 5, 6, 7). Uses llm_planner (Groq llama-3.1-8b-instant) to decide
which retrieval nodes are worth activating for the current query.

Why a planning node?
  Not every query benefits from all three retrieval paths:
  - RAG (SEC filings) adds ~3-5 s for simple price queries that mention
    nothing about earnings, guidance, or management commentary.
  - Sentiment (Reddit + Stocktwits) adds noise for earnings-focused queries
    where filing excerpts are more signal-dense.
  - News is almost always useful, but can be skipped for pure chart requests
    (those are routed past the planner entirely in workflow.py).

  The planner adds one cheap LLM call (~100 ms) before the fan-out and
  can save several seconds of parallel I/O when retrieval is unnecessary.

Fallback:
  If the LLM call fails or returns invalid/unparseable JSON, all three
  retrieval nodes are activated — identical to the pre-planner baseline.
  planner_error is written but does not halt execution.
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_planner

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """\
You are a retrieval planning assistant for a stock analysis agent.
Given a user query, decide which data sources to fetch.
Respond with ONLY a JSON object — no explanation, no markdown.

Decision rules:
- fetch_news: true unless the query is purely about chart rendering with no
  interest in events or catalysts
- fetch_sentiment: true if the query mentions sentiment, Reddit, retail
  interest, OR if it is a broad stock_analysis that would benefit from
  social context
- fetch_rag: true for all stock_analysis intent queries — SEC filings
  add fundamental context even when not explicitly requested; false only
  for general_lookup intent or queries with no historical date range

JSON format (exact keys, boolean values):
{"fetch_news": true, "fetch_sentiment": true, "fetch_rag": false}
"""


def plan_retrieval(state: AgentState) -> AgentState:
    """
    Ask the LLM which retrieval nodes to activate for this query.
    Falls back to activating all three nodes on any failure.
    """
    user_message = state.get("user_message", "")
    intent = state.get("intent", "stock_analysis")
    date_context = state.get("date_context", "")
    ticker = state.get("ticker", "")

    try:
        user_content = (
            f"Ticker: {ticker}\n"
            f"Intent: {intent}\n"
            f"Date context: {date_context}\n"
            f"User message: {user_message}"
        )

        response = llm_planner.invoke([
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        raw = response.content.strip()

        # Strip markdown fences defensively
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        retrieval_plan = {
            "fetch_news":      bool(parsed.get("fetch_news", True)),
            "fetch_sentiment": bool(parsed.get("fetch_sentiment", True)),
            "fetch_rag":       bool(parsed.get("fetch_rag", True)),
        }

        logger.info(
            "plan_retrieval [%s] -> news=%s sentiment=%s rag=%s",
            ticker,
            retrieval_plan["fetch_news"],
            retrieval_plan["fetch_sentiment"],
            retrieval_plan["fetch_rag"],
        )

        return {
            **state,
            "retrieval_plan": retrieval_plan,
            "planner_error": None,
        }

    except Exception as e:
        logger.warning(
            "plan_retrieval failed (%s) — activating all retrieval nodes as fallback", e
        )
        return {
            **state,
            "retrieval_plan": {"fetch_news": True, "fetch_sentiment": True, "fetch_rag": True},
            "planner_error": str(e),
        }
