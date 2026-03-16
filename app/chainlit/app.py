"""
Chainlit UI for the Stock Insight Agent.

Reads from the new AgentState fields written by Phase 1 nodes:
  - response_text      : the LLM-generated narrative (Node 9)
  - sources_cited      : list of source dicts for clickable links (Node 9)
  - chart_data         : Plotly JSON string (Node 10)
  - synthesizer_error  : set if Node 9 failed
  - chart_error        : set if Node 10 failed

No longer reads messages[-1] or the PLOTLY_JSON: prefix hack from
the prototype tool_caller. Those are replaced by explicit state fields.
"""

import logging

import chainlit as cl
import plotly.io as pio

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
    # This allows follow-up queries like "show me a chart of that" to work
    # without the user repeating the ticker and date range.
    last_context = cl.user_session.get("last_context") or {}
    initial_state = {
        "user_message": message.content,
        "user_config": {},   # no user-supplied API keys yet (Phase 5)
        **last_context,
    }

    try:
        # Run the LangGraph workflow synchronously.
        # Chainlit runs on asyncio; graph.invoke is synchronous but fast enough
        # for a portfolio project. Phase 5 can switch to graph.ainvoke.
        result = graph.invoke(initial_state)

    except Exception as e:
        logger.error("Graph invocation failed: %s", e)
        await cl.Message(
            content=(
                f"Something went wrong running the analysis:\n\n`{e}`\n\n"
                "Please try rephrasing your question."
            ),
            author=AUTHOR,
        ).send()
        return

    # ------------------------------------------------------------------
    # 0. Persist context for follow-up queries
    # ------------------------------------------------------------------
    context_fields = ("ticker", "company_name", "start_date", "end_date", "date_context")
    saved = {k: result[k] for k in context_fields if result.get(k)}
    if saved:
        cl.user_session.set("last_context", saved)

    # ------------------------------------------------------------------
    # 1. Send the narrative response
    # ------------------------------------------------------------------
    response_text = result.get("response_text")
    synthesizer_error = result.get("synthesizer_error")

    if response_text:
        await cl.Message(content=response_text, author=AUTHOR).send()
    elif synthesizer_error:
        await cl.Message(
            content=f"I wasn't able to generate a response: {synthesizer_error}",
            author=AUTHOR,
        ).send()
        return
    else:
        # Fallback — should not happen in normal operation
        await cl.Message(
            content="No response was generated. Please try again.",
            author=AUTHOR,
        ).send()
        return

    # ------------------------------------------------------------------
    # 2. Send sources as a formatted list (if any)
    # ------------------------------------------------------------------
    sources_cited = result.get("sources_cited") or []
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

        sources_text = "**Sources**\n\n" + "\n\n".join(source_lines)
        await cl.Message(content=sources_text, author=AUTHOR).send()

    # ------------------------------------------------------------------
    # 3. Render the Plotly chart (if generated)
    # ------------------------------------------------------------------
    chart_data = result.get("chart_data")
    chart_error = result.get("chart_error")

    if chart_data:
        try:
            fig = pio.from_json(chart_data)
            await cl.Message(
                content="",
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
