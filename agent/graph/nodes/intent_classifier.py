"""
Node 1: Intent Classifier

Reads:  user_message
Writes: intent, chart_requested, intent_error

Sends the user's message to the LLM and asks it to classify the intent
into one of five categories, plus flag whether a chart was requested.
No external API calls — pure LLM classification.

Why a separate node for this?
Separating classification from data retrieval means the routing logic
has a clean, typed value to branch on. The old god-node inferred intent
implicitly inside a single function; bugs there were invisible. Here,
if classification is wrong, this is the only place to look.
"""

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_classifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# The system prompt is kept here (not in a separate file) because it is
# tightly coupled to this node's parsing logic. Changing one without the
# other would break the node. Keeping them co-located makes that dependency
# obvious.
_SYSTEM_PROMPT = """\
You are an intent classifier for a stock analysis assistant.

Classify the user's message into exactly one of these intents:
- stock_analysis   : user wants to understand what happened with a stock
                     during a specific time period or around an event
                     (keywords: "what happened", "how did it do",
                     "around earnings", "why did it move", "last quarter")
- options_view     : user wants current options positioning data
                     (keywords: "options chain", "put/call", "options for",
                     "calls", "puts", "implied volatility")
- chart_request    : user primarily wants a visual chart or graph
                     (keywords: "show me a chart", "graph", "visualize",
                     "plot", "candlestick", "draw")
- general_lookup   : user wants basic price/performance data, no deep
                     analysis (keywords: "how did X perform", "what's the
                     price", "stock data", "current price", "52-week high")
- unknown          : message does not relate to stock analysis, or is too
                     ambiguous to classify into the above categories

Also set chart_requested to true if the user mentions wanting any visual
output (chart, graph, plot, visualization), regardless of intent.

A message can have intent "stock_analysis" AND chart_requested true.
Example: "What happened with NVIDIA around earnings? Show me a chart too."

Respond with ONLY a JSON object — no explanation, no markdown, no extra text:
{"intent": "<one of the five values above>", "chart_requested": <true|false>}
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> AgentState:
    """
    Classify the user's message into an intent and detect chart requests.

    Returns a partial state update dict — LangGraph merges this with the
    existing state automatically. We only write the fields this node owns.
    """
    user_message = state["user_message"]

    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = llm_classifier.invoke(messages)
        raw = response.content.strip()

        # The LLM sometimes wraps its JSON in markdown code fences even when
        # told not to. Strip them defensively before parsing.
        # Why: 8B models don't always follow formatting instructions perfectly.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        intent = parsed.get("intent", "unknown")
        chart_requested = bool(parsed.get("chart_requested", False))

        # Validate intent is one of the known values; default to unknown if not.
        # Why: the LLM could hallucinate a value not in our routing logic,
        # which would cause a KeyError later in the conditional edges.
        valid_intents = {
            "stock_analysis",
            "options_view",
            "chart_request",
            "general_lookup",
            "unknown",
        }
        if intent not in valid_intents:
            logger.warning(
                "classify_intent received unexpected intent %r; defaulting to 'unknown'",
                intent,
            )
            intent = "unknown"

        logger.info("classify_intent → intent=%r chart_requested=%r", intent, chart_requested)

        return {
            **state,
            "intent": intent,
            "chart_requested": chart_requested,
            "intent_error": None,
        }

    except Exception as e:
        # On any failure (LLM error, JSON parse error, network error),
        # write a safe default and record the error. The graph will still
        # route — it will take the "unknown" path, which leads to the
        # Response Synthesizer generating a clarification message.
        logger.error("classify_intent failed: %s", e)
        return {
            **state,
            "intent": "unknown",
            "chart_requested": False,
            "intent_error": str(e),
        }
