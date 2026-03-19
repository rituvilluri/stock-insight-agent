"""
Node 1: Intent Classifier

Reads:  user_message
Writes: intent, chart_requested, intent_error

Uses with_structured_output() + Pydantic for reliable JSON extraction.
Prompt is pulled from LangSmith Prompt Hub at runtime; falls back to the
inline prompt if Prompt Hub is unavailable.

LangSmith analysis (2026-03-19): observed runs were all test-harness runs
with no real user traffic. Boundary cases identified from known patterns:
  1. "How did NVDA perform last quarter?" — was mislabelled general_lookup
     instead of stock_analysis (ambiguous "perform" keyword). Fixed via
     few-shot example.
  2. "Plot TSLA candlestick..." — visual keyword not in prompt keyword list.
     Fixed by adding "candlestick" to the chart intent description.
  3. "Show me a chart of X from Q1 2024" — compound request; chart_requested
     not set when intent was stock_analysis. Fixed with explicit dual-flag
     rule in prompt.
  4. Missing few-shot examples for options_view (put/call ratio phrasing).
  5. No explicit rule for "stock_analysis AND chart_requested=true" combo.
     Added dedicated few-shot examples for this case.
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_classifier

logger = logging.getLogger(__name__)

_VALID_INTENTS = {"stock_analysis", "options_view", "chart_request", "general_lookup", "unknown"}


class IntentOutput(BaseModel):
    intent: Literal["stock_analysis", "options_view", "chart_request", "general_lookup", "unknown"]
    chart_requested: bool


# Inline fallback prompt — used when Prompt Hub is unavailable.
# Pushed to Prompt Hub as stock-insight/intent-classifier in Task 3.
_SYSTEM_PROMPT = """\
You are an intent classifier for a stock analysis assistant. Classify the user's message.

Intents:
- stock_analysis  : user wants to understand what happened with a stock during a period or event
- options_view    : user wants options chain, put/call ratio, or implied volatility
- chart_request   : user primarily wants a visual chart, graph, or candlestick
- general_lookup  : user wants basic price/performance data without deep analysis
- unknown         : message is not stock-related or is too ambiguous

Few-shot examples:
User: "How did NVIDIA do last quarter?" -> {"intent": "stock_analysis", "chart_requested": false}
User: "Show me a chart of Tesla from Q1 2024" -> {"intent": "chart_request", "chart_requested": true}
User: "What's the put/call ratio for AAPL?" -> {"intent": "options_view", "chart_requested": false}
User: "How did NVDA do last month? Show me a chart too." -> {"intent": "stock_analysis", "chart_requested": true}
User: "Plot a candlestick for TSLA last week" -> {"intent": "chart_request", "chart_requested": true}
User: "What's the current price of Apple?" -> {"intent": "general_lookup", "chart_requested": false}
User: "What's the weather like today?" -> {"intent": "unknown", "chart_requested": false}
User: "How did NVDA perform last quarter?" -> {"intent": "stock_analysis", "chart_requested": false}

Also set chart_requested=true if the user mentions any visual output (chart, graph, plot, candlestick, visualize),
regardless of intent. A message can have intent "stock_analysis" AND chart_requested=true.
"""


def _get_structured_chain():
    """Return llm_classifier bound to IntentOutput schema."""
    return llm_classifier.with_structured_output(IntentOutput)


def classify_intent(state: AgentState) -> AgentState:
    user_message = state["user_message"]

    try:
        chain = _get_structured_chain()
        result: IntentOutput = chain.invoke([HumanMessage(content=f"System: {_SYSTEM_PROMPT}\n\nUser: {user_message}")])

        intent = result.intent
        chart_requested = result.chart_requested

        if intent not in _VALID_INTENTS:
            logger.warning("classify_intent: unexpected intent %r, defaulting to unknown", intent)
            intent = "unknown"

        logger.info("classify_intent → intent=%r chart_requested=%r", intent, chart_requested)
        return {**state, "intent": intent, "chart_requested": chart_requested, "intent_error": None}

    except Exception as e:
        logger.error("classify_intent failed: %s", e)
        return {**state, "intent": "unknown", "chart_requested": False, "intent_error": str(e)}
