"""
Chainlit UI for the Stock Insight Agent.

Reads from AgentState fields written by the workflow nodes:
  - response_text      : the LLM-generated narrative (Node 9)
  - sources_cited      : list of source dicts for clickable links (Node 9)
  - chart_data         : Plotly JSON string (Node 10)
  - synthesizer_error  : set if Node 9 failed
  - chart_error        : set if Node 10 failed

Uses graph.astream_events to stream synthesizer tokens to the UI as they
arrive.  All other state fields are collected from on_chain_end events.

Pipeline progress is surfaced via cl.Step cards — one per node — so users
see incremental progress rather than a blank wait during the 5-10s pipeline.
"""

import logging
from pathlib import Path

import chainlit as cl
import plotly.io as pio

# Chainlit 2.10 requires this directory to exist before rendering any file-backed
# elements (e.g. Plotly charts). Its session.py calls mkdir(exist_ok=True) on a
# subdirectory without parents=True, so it silently fails if the parent is absent.
Path(".files").mkdir(exist_ok=True)

from agent.graph.workflow import app as graph

logger = logging.getLogger(__name__)

AUTHOR = "Stock Insight Agent"

# ---------------------------------------------------------------------------
# Node display names for cl.Step pipeline visibility
# ---------------------------------------------------------------------------
# Maps LangGraph node names to human-readable step labels shown in the UI.
# None means the node is shown via streaming output instead of a step card.
_NODE_DISPLAY_NAMES = {
    "classify_intent": "Classifying intent",
    "resolve_ticker": "Resolving ticker",
    "parse_dates": "Parsing date range",
    "fetch_price": "Fetching price data",
    "retrieve_news": "Retrieving news",
    "reddit_sentiment": "Analyzing Reddit sentiment",
    "retrieve_rag": "Searching SEC filings",
    "analyze_options": "Analyzing options chain",
    "synthesize": None,       # shown via streaming tokens — no step card
    "generate_chart": "Generating chart",
}


# ---------------------------------------------------------------------------
# Chat profiles
# ---------------------------------------------------------------------------

@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Quick Analysis",
            markdown_description="Concise summary with key price, news, and sentiment data.",
            icon="⚡",
        ),
        cl.ChatProfile(
            name="Deep Dive",
            markdown_description="Comprehensive analyst brief with structured sections.",
            icon="🔬",
        ),
    ]


# ---------------------------------------------------------------------------
# Chat start
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def start():
    profile = cl.user_session.get("chat_profile") or "Quick Analysis"
    mode_note = "Deep Dive mode — structured analyst briefs." if profile == "Deep Dive" else "Quick Analysis mode — concise summaries."
    await cl.Message(
        content=(
            f"**Welcome to the Stock Insight Agent** · {mode_note}\n\n"
            "Ask me about any stock's performance over a time period. Examples:\n\n"
            "- *How did NVIDIA do last month?*\n"
            "- *Show me a chart of Tesla from Q1 2024*\n"
            "- *What happened with Apple around Q2 2024 earnings?*\n\n"
            "I can retrieve price data, generate interactive charts, "
            "and provide a narrative analysis with source citations."
        ),
        author=AUTHOR,
    ).send()


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def main(message: cl.Message):
    # Session memory: we seed last_context from cl.user_session into initial_state.
    # LangGraph MemorySaver checkpointer was evaluated (Agent 0) and deferred —
    # current approach is simpler and sufficient for this single-user session model.
    last_context = cl.user_session.get("last_context") or {}
    profile = cl.user_session.get("chat_profile") or "Quick Analysis"
    response_depth = "deep" if profile == "Deep Dive" else "quick"
    initial_state = {
        "user_message": message.content,
        "user_config": {},   # no user-supplied API keys yet (Phase 5)
        "response_depth": response_depth,
        **last_context,
    }

    # Pre-send a streaming message — synthesizer tokens will fill it in.
    streaming_msg = cl.Message(content="", author=AUTHOR)
    await streaming_msg.send()

    final_state: dict = {}
    active_steps: dict[str, cl.Step] = {}

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # Open a step card when a known pipeline node starts.
            if kind == "on_chain_start" and node in _NODE_DISPLAY_NAMES:
                display_name = _NODE_DISPLAY_NAMES.get(node)
                if display_name:
                    step = cl.Step(name=display_name, type="run")
                    await step.__aenter__()
                    # Note: __aenter__/__aexit__ are used instead of `async with`
                    # because the step must span multiple loop iterations — it
                    # opens on on_chain_start and closes on the later on_chain_end.
                    active_steps[node] = step

            # Stream tokens from the synthesizer's LLM call token-by-token.
            elif kind == "on_chat_model_stream" and node == "synthesize":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    await streaming_msg.stream_token(chunk.content)

            # Collect state updates from completed nodes, then close any open step card.
            # Both actions must run on the same on_chain_end event — using separate elif
            # branches would cause the first branch to consume the event and drop state
            # updates for nodes that also have step cards open.
            elif kind == "on_chain_end" and node:
                output = event["data"].get("output", {})
                if isinstance(output, dict):
                    final_state.update(output)
                if node in active_steps:
                    step = active_steps.pop(node)
                    await step.__aexit__(None, None, None)

    except Exception as e:
        # Close any open steps on error.
        for step in active_steps.values():
            await step.__aexit__(None, None, None)
        logger.error("Graph streaming failed: %s", e)
        streaming_msg.content = (
            f"Something went wrong running the analysis:\n\n`{e}`\n\n"
            "Please try rephrasing your question."
        )
        await streaming_msg.update()
        return

    # Finalise the streaming message.
    await streaming_msg.update()

    # For paths that skip the synthesizer (chart_request), streaming_msg
    # will be empty.  Fall back to response_text from final_state if present.
    if not streaming_msg.content:
        response_text = final_state.get("response_text")
        synthesizer_error = final_state.get("synthesizer_error")
        if response_text:
            streaming_msg.content = response_text
            await streaming_msg.update()
        elif synthesizer_error:
            streaming_msg.content = (
                f"I wasn't able to generate a response: {synthesizer_error}"
            )
            await streaming_msg.update()
            return

    # ------------------------------------------------------------------
    # Persist context for follow-up queries
    # ------------------------------------------------------------------
    context_fields = ("ticker", "company_name", "start_date", "end_date", "date_context")
    saved = {k: final_state[k] for k in context_fields if final_state.get(k)}
    if saved:
        cl.user_session.set("last_context", saved)

    # ------------------------------------------------------------------
    # Send sources as a collapsible element (if any)
    # ------------------------------------------------------------------
    sources_cited = final_state.get("sources_cited") or []
    if sources_cited:
        source_lines = []
        for s in sources_cited:
            label = s.get("title", "Source")
            url = s.get("url", "")
            source_type = s.get("type", "")
            icon = {"news": "📰", "reddit": "💬", "filing": "📄"}.get(source_type, "🔗")
            if url:
                source_lines.append(f"{icon} [{label}]({url})")
            else:
                source_lines.append(f"{icon} {label}")

        sources_content = "\n\n".join(source_lines)
        await cl.Message(
            content="",
            author=AUTHOR,
            elements=[
                cl.Text(
                    name="📎 View Sources",
                    content=sources_content,
                    display="side",
                )
            ],
        ).send()

    # ------------------------------------------------------------------
    # Render the Plotly chart (if generated)
    # ------------------------------------------------------------------
    chart_data = final_state.get("chart_data")
    chart_error = final_state.get("chart_error")

    if chart_data:
        try:
            fig = pio.from_json(chart_data)
            await cl.Message(
                content="📊 Interactive Chart",
                author=AUTHOR,
                elements=[cl.Plotly(name="Stock Chart", figure=fig, display="inline")],
            ).send()
        except Exception as e:
            logger.error("Chart rendering failed: %s", e)
            await cl.Message(
                content=f"Chart data was generated but could not be rendered: {e}",
                author=AUTHOR,
            ).send()
    elif chart_error:
        await cl.Message(
            content=f"Chart generation failed: {chart_error}",
            author=AUTHOR,
        ).send()
