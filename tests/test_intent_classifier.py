"""
Tests for Node 1: Intent Classifier

Strategy: mock the LLM so tests are fast, deterministic, and require no
API key. We test the node's parsing and error-handling logic, not the LLM's
classification quality (that's an eval concern, not a unit test concern).

Why mock instead of calling the real LLM?
- Real calls are slow (1-3s each) and flaky without a valid GROQ_API_KEY
- Unit tests should test OUR code, not a third-party API's output
- We control what the mock returns, so each test case is deterministic
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.intent_classifier import classify_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(message: str) -> dict:
    """Build a minimal AgentState dict with just user_message set."""
    return {
        "user_message": message,
        "user_config": {},
    }


def _mock_llm_response(content: str):
    """Return a MagicMock that looks like a LangChain LLM response."""
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_stock_analysis_intent(mock_llm):
    """A historical event query should classify as stock_analysis."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"intent": "stock_analysis", "chart_requested": false}'
    )

    result = classify_intent(_make_state("What happened with NVIDIA around Q2 earnings?"))

    assert result["intent"] == "stock_analysis"
    assert result["chart_requested"] is False
    assert result["intent_error"] is None


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_general_lookup_intent(mock_llm):
    """A basic price query should classify as general_lookup."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"intent": "general_lookup", "chart_requested": false}'
    )

    result = classify_intent(_make_state("How did Apple perform last week?"))

    assert result["intent"] == "general_lookup"
    assert result["chart_requested"] is False
    assert result["intent_error"] is None


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_chart_request_intent(mock_llm):
    """A chart-focused query should set both chart_request intent and chart_requested=True."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"intent": "chart_request", "chart_requested": true}'
    )

    result = classify_intent(_make_state("Show me a chart of Tesla for the last 3 months"))

    assert result["intent"] == "chart_request"
    assert result["chart_requested"] is True
    assert result["intent_error"] is None


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_chart_flag_independent_of_intent(mock_llm):
    """
    chart_requested can be True even when intent is stock_analysis.
    A user asking for analysis AND a chart is the most common combined case.
    """
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"intent": "stock_analysis", "chart_requested": true}'
    )

    result = classify_intent(
        _make_state("What happened with NVDA around earnings? Show me a chart too.")
    )

    assert result["intent"] == "stock_analysis"
    assert result["chart_requested"] is True
    assert result["intent_error"] is None


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_options_view_intent(mock_llm):
    """An options query should classify as options_view."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"intent": "options_view", "chart_requested": false}'
    )

    result = classify_intent(_make_state("What does the options chain look like for Tesla?"))

    assert result["intent"] == "options_view"
    assert result["chart_requested"] is False
    assert result["intent_error"] is None


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_unknown_intent_on_unrelated_message(mock_llm):
    """A message unrelated to stocks should classify as unknown."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"intent": "unknown", "chart_requested": false}'
    )

    result = classify_intent(_make_state("What is the capital of France?"))

    assert result["intent"] == "unknown"
    assert result["chart_requested"] is False
    assert result["intent_error"] is None


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_llm_json_parse_failure_writes_error(mock_llm):
    """
    If the LLM returns garbage (not valid JSON), the node must not crash.
    It should default to intent='unknown' and write the error to intent_error.
    This tests the CLAUDE.md rule: all nodes must write to *_error on failure.
    """
    mock_llm.invoke.return_value = _mock_llm_response("Sorry, I cannot help with that.")

    result = classify_intent(_make_state("What happened with NVIDIA?"))

    assert result["intent"] == "unknown"
    assert result["chart_requested"] is False
    assert result["intent_error"] is not None  # error message was recorded


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_llm_exception_writes_error(mock_llm):
    """
    If the LLM raises an exception (e.g. network error, rate limit),
    the node must not propagate it — it should catch and write to intent_error.
    """
    mock_llm.invoke.side_effect = Exception("Groq API rate limit exceeded")

    result = classify_intent(_make_state("How did MSFT perform?"))

    assert result["intent"] == "unknown"
    assert result["chart_requested"] is False
    assert "rate limit" in result["intent_error"]


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_invalid_intent_value_defaults_to_unknown(mock_llm):
    """
    If the LLM returns a valid JSON object but with an intent value that isn't
    in our allowed set, the node should default to 'unknown' rather than
    letting an unexpected string reach the routing logic.
    """
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"intent": "buy_recommendation", "chart_requested": false}'
    )

    result = classify_intent(_make_state("Should I buy NVIDIA?"))

    assert result["intent"] == "unknown"
    assert result["intent_error"] is None  # this isn't an error, just a sanitised value


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_markdown_wrapped_json_is_handled(mock_llm):
    """
    8B models sometimes wrap their JSON in markdown code fences despite being
    told not to. The node should strip the fences before parsing.
    """
    mock_llm.invoke.return_value = _mock_llm_response(
        '```json\n{"intent": "general_lookup", "chart_requested": false}\n```'
    )

    result = classify_intent(_make_state("What is the current price of Apple?"))

    assert result["intent"] == "general_lookup"
    assert result["intent_error"] is None


@patch("agent.graph.nodes.intent_classifier.llm_classifier")
def test_state_fields_not_owned_by_node_are_preserved(mock_llm):
    """
    The node must not wipe out state fields it doesn't own.
    We pass in a state that already has ticker set; it must still be there
    in the returned state.
    """
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"intent": "stock_analysis", "chart_requested": false}'
    )

    state_with_extra = {
        "user_message": "What happened with NVDA?",
        "user_config": {},
        "ticker": "NVDA",  # set by a previous node (in tests, set manually)
    }

    result = classify_intent(state_with_extra)

    assert result["ticker"] == "NVDA"  # must still be there
    assert result["intent"] == "stock_analysis"
