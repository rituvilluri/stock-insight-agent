"""
LangGraph workflow for the Stock Insight Agent — Phase 2 + Node 8 (Options).

Nodes wired:
  1  Intent Classifier    — classify_intent
  2  Ticker Resolver      — resolve_ticker
  3  Date Parser          — parse_dates
  4  Price Data Fetcher   — fetch_price_data
  5  News Retriever       — retrieve_news      ─┐ parallel via Send()
  6  Reddit Sentiment     — reddit_sentiment   ─┘
  8  Options Analyzer     — analyze_options
  9  Response Synthesizer — synthesize_response
  10 Chart Generator      — generate_chart

Nodes deferred (stub, path routes around it):
  7  RAG Retriever

Routing overview:
  Node 1 → Node 2 → Node 3 → route_after_date_parser
    date_missing or intent="unknown"  → synthesize
    intent="options_view"             → analyze_options → synthesize
    all other intents                 → fetch_price

  route_after_fetch_price
    intent="chart_request"            → generate_chart → END
    all other intents                 → Send(retrieve_news) + Send(reddit_sentiment)
                                        [parallel fan-out; both converge at synthesize]

  route_after_synthesizer
    chart_requested=True              → generate_chart → END
    else                              → END

Why options_view routes directly to analyze_options (skipping fetch_price)?
  Options analysis is forward-looking and options-specific. The price context
  embedded in state.price_data (close_price) is used by the Greeks calculation,
  but a historical OHLCV fetch would add latency without adding relevant
  insight for options_view intent. The current snapshot is enough.

Why keep chart_request separate from chart_requested?
  intent="chart_request" skips news retrieval and the synthesizer
  entirely — the user explicitly asked for a chart, not a narrative.
  The chart_requested flag (which can be True alongside any intent)
  triggers chart generation AFTER the synthesizer has run.
"""

import logging

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from agent.graph.nodes.state import AgentState
from agent.graph.nodes.intent_classifier import classify_intent
from agent.graph.nodes.ticker_resolver import resolve_ticker
from agent.graph.nodes.date_parser import parse_dates
from agent.graph.nodes.data_fetcher import fetch_price_data
from agent.graph.nodes.news_retriever import retrieve_news
from agent.graph.nodes.reddit_sentiment import analyze_reddit_sentiment
from agent.graph.nodes.options_analyzer import analyze_options
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

    Routes to "analyze_options" for options_view intent — Node 8 fetches
    the live options chain and does not need a historical date range.

    Routes to "fetch_price" for all other data-bearing intents, including
    chart_request (chart needs daily OHLCV from Node 4).
    """
    date_missing = state.get("date_missing", False)
    intent = state.get("intent", "unknown")

    if date_missing or intent == "unknown":
        logger.debug("route_after_date_parser → synthesize (date_missing=%s, intent=%s)", date_missing, intent)
        return "synthesize"

    if intent == "options_view":
        logger.debug("route_after_date_parser → analyze_options (intent=options_view)")
        return "analyze_options"

    logger.debug("route_after_date_parser → fetch_price (intent=%s)", intent)
    return "fetch_price"


def route_after_fetch_price(state: AgentState):
    """
    Decide which nodes run after Node 4 (Price Data Fetcher).

    chart_request intent: user wants a chart only — skip news retrieval
    and the synthesizer, go straight to chart generation.

    All other intents: fan-out via Send() to run Node 5 (News Retriever)
    and Node 6 (Reddit Sentiment) in parallel.  Both are independent of
    each other — they only read ticker and date from state.  After both
    complete their results merge into shared state, then synthesize runs.
    """
    intent = state.get("intent", "stock_analysis")

    if intent == "chart_request":
        logger.debug("route_after_fetch_price → generate_chart (chart_request intent)")
        return "generate_chart"

    logger.debug("route_after_fetch_price → Send(retrieve_news) + Send(reddit_sentiment) (intent=%s)", intent)
    return [
        Send("retrieve_news", state),
        Send("reddit_sentiment", state),
    ]


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
    Build and compile the LangGraph workflow with parallel retrieval.

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
    graph.add_node("analyze_options",    analyze_options)
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
            "fetch_price":     "fetch_price",
            "analyze_options": "analyze_options",
            "synthesize":      "synthesize",
        },
    )

    # Node 8: options_view path — analyze options then synthesize
    graph.add_edge("analyze_options", "synthesize")

    # After Node 4: fan-out or direct chart for chart_request
    # Returns [Send("retrieve_news"), Send("reddit_sentiment")] for analysis intents,
    # or "generate_chart" string for chart_request intent.
    graph.add_conditional_edges(
        "fetch_price",
        route_after_fetch_price,
        ["retrieve_news", "reddit_sentiment", "generate_chart"],
    )

    # Nodes 5 and 6 run in parallel (dispatched via Send above).
    # Both converge at synthesize — LangGraph waits for both superstep
    # branches to complete before advancing.
    graph.add_edge("retrieve_news",    "synthesize")
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
    logger.info("Stock Insight Agent workflow compiled (Phase 2 + Node 8 Options)")
    return compiled


# Module-level compiled graph — imported by app/chainlit/app.py
app = create_workflow()
