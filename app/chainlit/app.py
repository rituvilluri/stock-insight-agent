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
# Chat start
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "**Welcome to the Stock Insight Agent**\n\n"
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
    # Seed initial state with context from the previous turn (if any).
    last_context = cl.user_session.get("last_context") or {}
    initial_state = {
        "user_message": message.content,
        "user_config": {},   # no user-supplied API keys yet (Phase 5)
        **last_context,
    }

    # Start with an empty message — the synthesizer streams tokens into it.
    # Using empty string (not a placeholder) so the `not content` check below
    # correctly detects paths that skip the synthesizer (e.g. chart_request).
    streaming_msg = cl.Message(content="", author=AUTHOR)
    await streaming_msg.send()

    final_state = {}

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # Stream tokens from the synthesizer's LLM call token-by-token.
            # Guard against thinking-phase chunks from Gemini 2.5 Flash:
            # thinking tokens may arrive as structured content lists, not plain strings.
            if kind == "on_chat_model_stream" and node == "synthesize":
                chunk = event["data"]["chunk"]
                content = chunk.content
                if isinstance(content, str) and content:
                    await streaming_msg.stream_token(content)
                elif isinstance(content, list):
                    # Gemini thinking model: content is a list of parts
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                await streaming_msg.stream_token(text)

            # Collect state updates written by completed nodes.
            elif kind == "on_chain_end" and node:
                output = event["data"].get("output", {})
                if isinstance(output, dict):
                    final_state.update(output)

    except Exception as e:
        logger.error("Graph streaming failed: %s", e)
        await cl.Message(
            content=(
                f"Something went wrong running the analysis:\n\n`{e}`\n\n"
                "Please try rephrasing your question."
            ),
            author=AUTHOR,
        ).send()
        return

    # Finalise the streaming message.
    await streaming_msg.update()

    # For paths that skip the synthesizer (chart_request intent), no tokens
    # were streamed so streaming_msg.content is still empty.
    # Fall back to response_text from final_state if present.
    if not streaming_msg.content:
        response_text = final_state.get("response_text")
        synthesizer_error = final_state.get("synthesizer_error")
        if response_text:
            streaming_msg.content = response_text
            await streaming_msg.update()
        elif synthesizer_error:
            await cl.Message(
                content=f"I wasn't able to generate a response: {synthesizer_error}",
                author=AUTHOR,
            ).send()
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
