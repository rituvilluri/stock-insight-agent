"""
LangGraph workflow for the Stock Insight Agent — Phase 2.

Nodes wired (Phase 2):
  1  Intent Classifier    — classify_intent
  2  Ticker Resolver      — resolve_ticker
  3  Date Parser          — parse_dates
  4  Price Data Fetcher   — fetch_price_data
  5  News Retriever       — retrieve_news
  9  Response Synthesizer — synthesize_response
  10 Chart Generator      — generate_chart

Nodes deferred to Phase 3 (stubs, paths route around them):
  7  RAG Retriever
  8  Options Analyzer

Routing overview:
  Node 1 → Node 2 → Node 3 → route_after_date_parser
    date_missing or intent="unknown"  → synthesize
    all other intents                 → fetch_price

  route_after_fetch_price
    intent="chart_request"            → generate_chart → END
    all other intents                 → retrieve_news → synthesize

  route_after_synthesizer
    chart_requested=True              → generate_chart → END
    else                              → END

Why keep chart_request separate from chart_requested?
  intent="chart_request" skips news retrieval and the synthesizer
  entirely — the user explicitly asked for a chart, not a narrative.
  The chart_requested flag (which can be True alongside any intent)
  triggers chart generation AFTER the synthesizer has run.
"""

import logging

from langgraph.graph import StateGraph, END

from agent.graph.nodes.state import AgentState
from agent.graph.nodes.intent_classifier import classify_intent
from agent.graph.nodes.ticker_resolver import resolve_ticker
from agent.graph.nodes.date_parser import parse_dates
from agent.graph.nodes.data_fetcher import fetch_price_data
from agent.graph.nodes.news_retriever import retrieve_news
from agent.graph.nodes.reddit_sentiment import analyze_reddit_sentiment
from agent.graph.nodes.response_synthesizer import synthesize_response
from agent.graph.nodes.chart_generator import generate_chart

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_date_parser(state: AgentState) -> str:
    """
    Decide which node runs after Node 3 (Date Parser).

    Routes to "synthesize" when no useful data can be fetched:
      - date_missing: no time window to query APIs
      - unknown intent: user message is not stock-related

    Routes to "fetch_price" for all data-bearing intents.
    options_view routes to "fetch_price" in Phase 1 because Node 8
    (Options Analyzer) is not yet implemented; fetch_price provides
    the base price context the synthesizer can at least use.

    chart_request also routes to "fetch_price" because the chart needs
    daily price data from Node 4.
    """
    date_missing = state.get("date_missing", False)
    intent = state.get("intent", "unknown")

    if date_missing or intent == "unknown":
        logger.debug("route_after_date_parser → synthesize (date_missing=%s, intent=%s)", date_missing, intent)
        return "synthesize"

    logger.debug("route_after_date_parser → fetch_price (intent=%s)", intent)
    return "fetch_price"


def route_after_fetch_price(state: AgentState) -> str:
    """
    Decide which node runs after Node 4 (Price Data Fetcher).

    chart_request intent: user wants a chart only — skip news retrieval
    and the synthesizer, go straight to chart generation.

    All other intents: proceed to news retrieval (Node 5) before synthesis.
    """
    intent = state.get("intent", "stock_analysis")

    if intent == "chart_request":
        logger.debug("route_after_fetch_price → generate_chart (chart_request intent)")
        return "generate_chart"

    logger.debug("route_after_fetch_price → retrieve_news (intent=%s)", intent)
    return "retrieve_news"


def route_after_synthesizer(state: AgentState) -> str:
    """
    Decide what runs after Node 9 (Response Synthesizer).

    If chart_requested=True, generate an interactive chart to accompany
    the narrative response.  Otherwise the workflow ends.

    chart_request intent never reaches this router (it was routed
    directly to chart generator after fetch_price).
    """
    chart_requested = state.get("chart_requested", False)

    if chart_requested:
        logger.debug("route_after_synthesizer → generate_chart")
        return "generate_chart"

    logger.debug("route_after_synthesizer → END")
    return "end"


# ---------------------------------------------------------------------------
# Workflow factory
# ---------------------------------------------------------------------------

def create_workflow():
    """
    Build and compile the Phase 1 LangGraph workflow.

    Returns the compiled graph object.  Chainlit calls this once at
    startup and stores the result; the same compiled graph handles all
    user sessions.
    """
    graph = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------
    graph.add_node("classify_intent",    classify_intent)
    graph.add_node("resolve_ticker",     resolve_ticker)
    graph.add_node("parse_dates",        parse_dates)
    graph.add_node("fetch_price",        fetch_price_data)
    graph.add_node("retrieve_news",      retrieve_news)
    graph.add_node("reddit_sentiment",   analyze_reddit_sentiment)
    graph.add_node("synthesize",         synthesize_response)
    graph.add_node("generate_chart",     generate_chart)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    graph.set_entry_point("classify_intent")

    # ------------------------------------------------------------------
    # Unconditional edges (common prefix — always runs in sequence)
    # ------------------------------------------------------------------
    graph.add_edge("classify_intent", "resolve_ticker")
    graph.add_edge("resolve_ticker",  "parse_dates")

    # ------------------------------------------------------------------
    # Conditional edges
    # ------------------------------------------------------------------

    # After Node 3: branch on intent + date_missing
    graph.add_conditional_edges(
        "parse_dates",
        route_after_date_parser,
        {
            "fetch_price": "fetch_price",
            "synthesize":  "synthesize",
        },
    )

    # After Node 4: branch on intent (chart_request skips news + synthesizer)
    graph.add_conditional_edges(
        "fetch_price",
        route_after_fetch_price,
        {
            "retrieve_news":  "retrieve_news",
            "generate_chart": "generate_chart",
        },
    )

    # After Node 5: proceed to Reddit sentiment
    graph.add_edge("retrieve_news",    "reddit_sentiment")

    # After Node 6: always proceed to synthesizer
    graph.add_edge("reddit_sentiment", "synthesize")

    # After Node 9: optionally generate chart
    graph.add_conditional_edges(
        "synthesize",
        route_after_synthesizer,
        {
            "generate_chart": "generate_chart",
            "end":            END,
        },
    )

    # After Node 10: always end
    graph.add_edge("generate_chart", END)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    compiled = graph.compile()
    logger.info("Stock Insight Agent workflow compiled (Phase 2)")
    return compiled


# Module-level compiled graph — imported by app/chainlit/app.py
app = create_workflow()
