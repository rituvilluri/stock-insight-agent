"""
Tests for the Phase 1 LangGraph workflow.

Strategy:
  - Unit-test the three routing functions in isolation — they are pure
    functions of state, so no mocking is needed.
  - Smoke-test that create_workflow() compiles without error and
    produces a runnable graph.
  - Run one end-to-end graph invocation with all LLM and external API
    calls mocked to verify the happy-path routing.

We do NOT test narrative quality or LLM output here — that belongs in
test_response_synthesizer.py.  We only care that the right nodes are
reached and that state fields set by each node flow through correctly.
"""

from unittest.mock import MagicMock, patch

import pytest
from langgraph.types import Send

from agent.graph.workflow import (
    route_after_date_parser,
    route_after_fetch_price,
    route_after_plan_retrieval,
    route_after_synthesizer,
    create_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**kwargs) -> dict:
    base = {
        "user_message": "How did NVDA do last week?",
        "user_config": {},
        "ticker": "NVDA",
        "company_name": "NVIDIA",
        "intent": "stock_analysis",
        "chart_requested": False,
        "date_missing": False,
        "date_context": "last week",
        "start_date": "2024-05-01",
        "end_date": "2024-05-07",
        "include_current_snapshot": False,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# route_after_date_parser
# ---------------------------------------------------------------------------

def test_route_date_parser_unknown_intent_to_synthesize():
    state = _state(intent="unknown", date_missing=False)
    assert route_after_date_parser(state) == "synthesize"


def test_route_date_parser_date_missing_to_synthesize():
    state = _state(intent="stock_analysis", date_missing=True)
    assert route_after_date_parser(state) == "synthesize"


def test_route_date_parser_date_missing_overrides_valid_intent():
    """Even if intent is stock_analysis, date_missing sends to synthesize."""
    state = _state(intent="stock_analysis", date_missing=True)
    assert route_after_date_parser(state) == "synthesize"


def test_route_date_parser_stock_analysis_to_fetch_price():
    state = _state(intent="stock_analysis", date_missing=False)
    assert route_after_date_parser(state) == "fetch_price"


def test_route_date_parser_general_lookup_to_fetch_price():
    state = _state(intent="general_lookup", date_missing=False)
    assert route_after_date_parser(state) == "fetch_price"


def test_route_date_parser_chart_request_to_fetch_price():
    state = _state(intent="chart_request", date_missing=False)
    assert route_after_date_parser(state) == "fetch_price"


def test_route_date_parser_options_view_to_analyze_options():
    """options_view routes directly to analyze_options (Node 8 is now wired)."""
    state = _state(intent="options_view", date_missing=False)
    assert route_after_date_parser(state) == "analyze_options"


# ---------------------------------------------------------------------------
# route_after_fetch_price
# ---------------------------------------------------------------------------

def test_route_fetch_price_chart_request_to_generate_chart():
    state = _state(intent="chart_request")
    assert route_after_fetch_price(state) == "generate_chart"


def test_route_fetch_price_stock_analysis_to_planner():
    """Phase 5: analysis intents route to plan_retrieval, not directly to fan-out."""
    state = _state(intent="stock_analysis")
    assert route_after_fetch_price(state) == "plan_retrieval"


def test_route_fetch_price_general_lookup_to_planner():
    state = _state(intent="general_lookup")
    assert route_after_fetch_price(state) == "plan_retrieval"


def test_route_fetch_price_options_view_to_planner():
    """options_view reaching route_after_fetch_price also goes to planner."""
    state = _state(intent="options_view")
    assert route_after_fetch_price(state) == "plan_retrieval"


def test_route_plan_retrieval_all_active():
    """All-active plan fans out to all three retrieval nodes."""
    state = _state(retrieval_plan={"fetch_news": True, "fetch_sentiment": True, "fetch_rag": True})
    result = route_after_plan_retrieval(state)
    assert isinstance(result, list)
    assert {s.node for s in result} == {"retrieve_news", "reddit_sentiment", "retrieve_rag"}


def test_route_plan_retrieval_selective():
    """Selective plan emits only enabled nodes."""
    state = _state(retrieval_plan={"fetch_news": True, "fetch_sentiment": False, "fetch_rag": False})
    result = route_after_plan_retrieval(state)
    assert len(result) == 1
    assert result[0].node == "retrieve_news"


# ---------------------------------------------------------------------------
# route_after_synthesizer
# ---------------------------------------------------------------------------

def test_route_synthesizer_chart_requested_to_generate_chart():
    state = _state(chart_requested=True)
    assert route_after_synthesizer(state) == "generate_chart"


def test_route_synthesizer_no_chart_to_end():
    state = _state(chart_requested=False)
    assert route_after_synthesizer(state) == "end"


def test_route_synthesizer_missing_chart_requested_defaults_to_end():
    """If chart_requested is absent from state, default to END (no chart)."""
    state = {
        "user_message": "How did NVDA do?",
        "user_config": {},
        "ticker": "NVDA",
        "intent": "general_lookup",
    }
    assert route_after_synthesizer(state) == "end"


# ---------------------------------------------------------------------------
# Workflow compilation smoke test
# ---------------------------------------------------------------------------

def test_create_workflow_compiles_without_error():
    """create_workflow() must return a compiled graph object."""
    graph = create_workflow()
    assert graph is not None
    # LangGraph compiled graphs have an invoke method
    assert hasattr(graph, "invoke")


# ---------------------------------------------------------------------------
# End-to-end routing test (all external calls mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("agent.graph.nodes.chart_generator.go")
@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
@patch("agent.graph.nodes.retrieval_planner.llm_planner")
@patch("agent.graph.nodes.date_parser.llm_classifier")
@patch("agent.graph.nodes.ticker_resolver.llm_classifier")
@patch("agent.graph.nodes.intent_classifier.llm_classifier")
async def test_routing_e2e_stock_analysis_no_chart(
    mock_intent_llm,
    mock_ticker_llm,
    mock_date_llm,
    mock_planner_llm,
    mock_synth_llm,
    mock_plotly_go,
):
    """
    Routing integration: verifies full graph wiring with mocked LLMs.
    Confirms state flows through all nodes correctly, but does NOT validate
    LLM output quality — LLMs are mocked, so this is a routing test only.
    """
    # Intent classifier uses with_structured_output().invoke() — must mock at that level
    intent_result = MagicMock()
    intent_result.intent = "stock_analysis"
    intent_result.chart_requested = False
    mock_intent_llm.with_structured_output.return_value.invoke.return_value = intent_result

    # Retrieval planner — return a plan that activates all nodes
    mock_planner_llm.invoke.return_value = MagicMock(
        content='{"fetch_news": true, "fetch_sentiment": true, "fetch_rag": true}'
    )

    # Response synthesizer
    mock_synth_llm.invoke.return_value = MagicMock(
        content="NVDA gained 6% last week driven by AI chip demand."
    )

    with patch("agent.graph.nodes.data_fetcher.yf.Ticker") as mock_yf:
        import pandas as pd
        dates = pd.date_range(start="2024-05-01", periods=5, freq="B")
        hist_df = pd.DataFrame(
            {
                "Open":   [800.0] * 5,
                "High":   [870.0] * 5,
                "Low":    [780.0] * 5,
                "Close":  [850.0] * 5,
                "Volume": [1_000_000] * 5,
            },
            index=dates,
        )
        mock_inst = MagicMock()
        mock_inst.history.return_value = hist_df
        mock_yf.return_value = mock_inst

        graph = create_workflow()
        result = await graph.ainvoke({
            "user_message": "How did NVDA do last week?",
            "user_config": {},
        })

    assert result.get("response_text") is not None
    assert result.get("synthesizer_error") is None
    assert result.get("chart_requested") is False


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_e2e_unknown_intent_returns_clarification(
    mock_intent_llm,
    mock_synth_llm,
):
    """
    unknown intent must return a clarification message via the synthesizer.
    The synthesizer's clarification path doesn't call the LLM, so
    mock_synth_llm.invoke should NOT be called.
    """
    mock_intent_llm.invoke.return_value = MagicMock(
        content='{"intent": "unknown", "chart_requested": false}'
    )

    graph = create_workflow()
    result = graph.invoke({
        "user_message": "What is the capital of France?",
        "user_config": {},
    })

    assert result.get("response_text") is not None
    # Clarification path does not call llm_synthesizer
    mock_synth_llm.invoke.assert_not_called()
    assert result.get("synthesizer_error") is None
