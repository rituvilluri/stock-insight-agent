"""
LangGraph workflow for the Stock Insight Agent — Phase 2 + Nodes 7 and 8 + Phase 5 Planner.

Nodes wired:
  1  Intent Classifier    — classify_intent
  2  Ticker Resolver      — resolve_ticker
  3  Date Parser          — parse_dates
  4  Price Data Fetcher   — fetch_price_data
  P  Retrieval Planner    — plan_retrieval      (Phase 5 — decides which of 5/6/7 to run)
  5  News Retriever       — retrieve_news      ─┐ parallel via Send() (if plan says fetch_news)
  6  Reddit Sentiment     — reddit_sentiment    ├─ parallel           (if plan says fetch_sentiment)
  7  RAG Retriever        — retrieve_rag        ─┘                    (if plan says fetch_rag)
  8  Options Analyzer     — analyze_options
  9  Response Synthesizer — synthesize_response
  10 Chart Generator      — generate_chart

Routing overview:
  Node 1 → Node 2 → Node 3 → route_after_date_parser
    date_missing or intent="unknown"  → synthesize
    intent="options_view"             → analyze_options → synthesize
    all other intents                 → fetch_price

  route_after_fetch_price
    intent="chart_request"            → generate_chart → END
    all other intents                 → plan_retrieval (Phase 5 planner)

  route_after_plan_retrieval
    reads retrieval_plan flags        → selective Send() fan-out
    fallback (all flags True)         → Send(retrieve_news)
                                        + Send(reddit_sentiment)
                                        + Send(retrieve_rag)
                                        [active branches converge at synthesize]

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
from agent.graph.nodes.rag_retriever import retrieve_rag_context
from agent.graph.nodes.response_synthesizer import synthesize_response
from agent.graph.nodes.chart_generator import generate_chart
from agent.graph.nodes.retrieval_planner import plan_retrieval

logger = logging.getLogger(__name__)

# LangGraph 0.3+ evaluation (2026-03-19):
#
# Command-based routing: NOT adopted. Current standalone routing functions
# (route_after_date_parser, route_after_fetch_price, route_after_synthesizer)
# are easier to test in isolation and keep routing logic decoupled from nodes.
# Command would couple node return values to graph topology.
#
# MemorySaver checkpointer: NOT adopted in this agent. Current session memory
# uses cl.user_session.set("last_context", ...) in app.py and seeds initial_state
# on each message. This works correctly. Migrating to MemorySaver is deferred to
# Agent 4 (Synthesis + UI) which owns app.py. If adopted, Agent 4 should:
#   compiled = graph.compile(checkpointer=MemorySaver())
#   config = {"configurable": {"thread_id": cl.user_session.get("id")}}
#   graph.astream_events(state, config=config, version="v2")
#   Remove last_context seeding in app.py.


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


def route_after_fetch_price(state: AgentState) -> str:
    """
    Decide what runs immediately after Node 4 (Price Data Fetcher).

    chart_request intent: skip news/sentiment/RAG entirely — go straight
    to chart generation.

    All other intents: hand off to the Retrieval Planner, which decides
    which of Nodes 5/6/7 are worth activating for this specific query.
    """
    intent = state.get("intent", "stock_analysis")

    if intent == "chart_request":
        logger.debug("route_after_fetch_price: chart_request -> generate_chart")
        return "generate_chart"

    logger.debug("route_after_fetch_price: intent=%s -> plan_retrieval", intent)
    return "plan_retrieval"


def route_after_plan_retrieval(state: AgentState):
    """
    Build the parallel Send() fan-out using the retrieval_plan written by
    the Retrieval Planner node.

    Reads retrieval_plan flags (fetch_news, fetch_sentiment, fetch_rag)
    and emits a Send() for each active node. Falls back to all three if
    the plan is missing or empty (e.g. planner LLM call failed).

    LangGraph dispatches each Send() as a separate parallel branch. All
    active paths target 'synthesize', so LangGraph waits for every branch
    before advancing past that node.
    """
    plan = state.get("retrieval_plan") or {}

    sends = []
    if plan.get("fetch_news", True):
        sends.append(Send("retrieve_news", state))
    if plan.get("fetch_sentiment", True):
        sends.append(Send("reddit_sentiment", state))
    if plan.get("fetch_rag", True):
        sends.append(Send("retrieve_rag", state))

    # Safety net: never return an empty list — that would leave synthesize
    # waiting for branches that never complete.
    if not sends:
        logger.warning("route_after_plan_retrieval: all flags False — activating all nodes")
        sends = [
            Send("retrieve_news", state),
            Send("reddit_sentiment", state),
            Send("retrieve_rag", state),
        ]

    logger.debug("route_after_plan_retrieval: dispatching %d branch(es)", len(sends))
    return sends


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
    graph.add_node("plan_retrieval",     plan_retrieval)
    graph.add_node("retrieve_news",      retrieve_news)
    graph.add_node("reddit_sentiment",   analyze_reddit_sentiment)
    graph.add_node("analyze_options",    analyze_options)
    graph.add_node("retrieve_rag",       retrieve_rag_context)
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

    # After Node 4: chart_request goes directly to chart; all other intents
    # go to the Retrieval Planner before fan-out.
    graph.add_conditional_edges(
        "fetch_price",
        route_after_fetch_price,
        {
            "plan_retrieval": "plan_retrieval",
            "generate_chart": "generate_chart",
        },
    )

    # After Retrieval Planner: selective Send() fan-out to Nodes 5, 6, 7.
    # Only the branches enabled by retrieval_plan are dispatched.
    graph.add_conditional_edges(
        "plan_retrieval",
        route_after_plan_retrieval,
        ["retrieve_news", "reddit_sentiment", "retrieve_rag"],
    )

    # Active retrieval branches converge at synthesize.
    # LangGraph waits for all dispatched Send() branches before advancing.
    graph.add_edge("retrieve_news",    "synthesize")
    graph.add_edge("reddit_sentiment", "synthesize")
    graph.add_edge("retrieve_rag",     "synthesize")

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
    logger.info("Stock Insight Agent workflow compiled (Phase 5: Retrieval Planner + adaptive fan-out)")
    return compiled


# Module-level compiled graph — imported by app/chainlit/app.py
app = create_workflow()
