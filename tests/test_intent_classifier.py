"""
Tests for Node 1: Intent Classifier

Strategy: mock _get_structured_chain so tests are fast, deterministic,
and require no API key. We test the node's routing and error-handling
logic, not the LLM's classification quality (that's an eval concern).

Uses with_structured_output() + Pydantic so we mock at the chain level,
not at llm.invoke() level.
"""

import pytest
from unittest.mock import MagicMock, patch
from agent.graph.nodes.intent_classifier import classify_intent

BASE_STATE = {"user_message": "", "user_config": {}}


def _mock_structured_llm(intent: str, chart_requested: bool):
    """Returns a mock that behaves like llm.with_structured_output(IntentOutput)."""
    mock_response = MagicMock()
    mock_response.intent = intent
    mock_response.chart_requested = chart_requested
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response
    return mock_chain


@pytest.mark.parametrize("message,expected_intent,expected_chart", [
    ("How did NVIDIA do last quarter?", "stock_analysis", False),
    ("What happened with Apple around Q2 2024 earnings?", "stock_analysis", False),
    ("Show me a chart of Tesla from Q1 2024", "chart_request", True),
    ("What's the put/call ratio for AAPL?", "options_view", False),
    ("How did NVDA do last month? Show me a chart too.", "stock_analysis", True),
    ("What's the weather like today?", "unknown", False),
    ("How did NVDA perform last quarter?", "stock_analysis", False),
    ("Plot TSLA candlestick for last week", "chart_request", True),
])
def test_classify_intent(message, expected_intent, expected_chart):
    state = {**BASE_STATE, "user_message": message}
    mock_chain = _mock_structured_llm(expected_intent, expected_chart)
    with patch("agent.graph.nodes.intent_classifier._get_structured_chain", return_value=mock_chain):
        result = classify_intent(state)
    assert result["intent"] == expected_intent
    assert result["chart_requested"] == expected_chart
    assert result["intent_error"] is None


def test_classify_intent_invalid_value_defaults_to_unknown():
    state = {**BASE_STATE, "user_message": "test"}
    mock_chain = _mock_structured_llm("invalid_intent", False)
    with patch("agent.graph.nodes.intent_classifier._get_structured_chain", return_value=mock_chain):
        result = classify_intent(state)
    assert result["intent"] == "unknown"


def test_classify_intent_llm_failure_returns_error():
    state = {**BASE_STATE, "user_message": "test"}
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("LLM timeout")
    with patch("agent.graph.nodes.intent_classifier._get_structured_chain", return_value=mock_chain):
        result = classify_intent(state)
    assert result["intent"] == "unknown"
    assert result["chart_requested"] is False
    assert "LLM timeout" in result["intent_error"]
