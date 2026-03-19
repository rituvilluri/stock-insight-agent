"""
Tests for Node 2: Ticker Resolver

Strategy: test each of the three resolution layers independently.
LLM calls are mocked — same reasoning as intent classifier tests.
Direct detection and lookup table tests need no mocking since they
are pure Python with no external dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.ticker_resolver import (
    resolve_ticker,
    _detect_direct_ticker,
    _lookup_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(message: str, **extra) -> dict:
    state = {"user_message": message, "user_config": {}}
    state.update(extra)
    return state


def _mock_llm_response(content: str):
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# Layer 1: direct ticker detection (unit tests — no mocking needed)
# ---------------------------------------------------------------------------

def test_direct_ticker_detected_in_message():
    """A 2-5 uppercase letter word is treated as a directly typed ticker."""
    result = _detect_direct_ticker("How did NVDA perform last week?")
    assert result == ("NVDA", "NVDA")


def test_direct_ticker_four_letters():
    result = _detect_direct_ticker("What happened with TSLA around earnings?")
    assert result == ("TSLA", "TSLA")


def test_direct_ticker_not_matched_for_common_words():
    """Common English acronyms like 'AI', 'CEO', 'ETF' must not be picked up."""
    result = _detect_direct_ticker("What is the AI trend affecting CEO pay?")
    assert result is None


def test_direct_ticker_not_matched_single_letter():
    """Single-letter words ('I', 'A') must not be picked up as tickers."""
    result = _detect_direct_ticker("I want to know about A stock")
    assert result is None


# ---------------------------------------------------------------------------
# Layer 2: lookup table (unit tests — no mocking needed)
# ---------------------------------------------------------------------------

def test_lookup_nvidia_lowercase():
    result = _lookup_table("what happened with nvidia around earnings?")
    assert result == ("NVDA", "NVIDIA")


def test_lookup_apple_mixed_case():
    result = _lookup_table("Tell me about Apple last month")
    assert result == ("AAPL", "Apple")


def test_lookup_microsoft():
    result = _lookup_table("How did Microsoft perform over the last quarter?")
    assert result == ("MSFT", "Microsoft")


def test_lookup_google_alias():
    """'google' and 'alphabet' should both resolve to GOOGL."""
    assert _lookup_table("news about google")[0] == "GOOGL"
    assert _lookup_table("alphabet earnings")[0] == "GOOGL"


def test_lookup_meta_alias():
    """'facebook' should resolve to META."""
    result = _lookup_table("what happened with facebook in 2021?")
    assert result == ("META", "Meta")


def test_lookup_returns_none_for_unknown():
    result = _lookup_table("what is the weather like today?")
    assert result is None


# ---------------------------------------------------------------------------
# Full node tests (integration of all three layers)
# ---------------------------------------------------------------------------

def test_node_direct_ticker_end_to_end():
    """Node resolves a directly typed ticker without touching the LLM."""
    result = resolve_ticker(_make_state("How did AAPL do last week?"))
    assert result["ticker"] == "AAPL"
    assert result["ticker_error"] is None


def test_node_lookup_table_end_to_end():
    """Node resolves a company name via lookup without touching the LLM."""
    result = resolve_ticker(_make_state("What happened with Tesla around earnings?"))
    assert result["ticker"] == "TSLA"
    assert result["company_name"] == "Tesla"
    assert result["ticker_error"] is None


@patch("agent.graph.nodes.ticker_resolver.llm_classifier")
def test_node_llm_fallback_called_for_unknown_company(mock_llm):
    """For a company not in the lookup table, the LLM should be called."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"ticker": "SHOP", "company_name": "Shopify"}'
    )

    result = resolve_ticker(_make_state("What happened with Shopify last quarter?"))

    assert result["ticker"] == "SHOP"
    assert result["company_name"] == "Shopify"
    assert result["ticker_error"] is None
    mock_llm.invoke.assert_called_once()


@patch("agent.graph.nodes.ticker_resolver.llm_classifier")
def test_node_llm_returns_null_sets_error(mock_llm):
    """If the LLM cannot identify a stock, ticker_error should be set."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"ticker": null, "company_name": null}'
    )

    result = resolve_ticker(_make_state("What is the weather today?"))

    assert result["ticker"] == ""
    assert result["ticker_error"] is not None


@patch("agent.graph.nodes.ticker_resolver.llm_classifier")
def test_node_llm_json_parse_failure_writes_error(mock_llm):
    """Garbage LLM output should not crash the node — write to ticker_error."""
    mock_llm.invoke.return_value = _mock_llm_response("I cannot determine the ticker.")

    result = resolve_ticker(_make_state("What about that tech company?"))

    assert result["ticker"] == ""
    assert result["ticker_error"] is not None


@patch("agent.graph.nodes.ticker_resolver.llm_classifier")
def test_node_llm_exception_writes_error(mock_llm):
    """An LLM network exception should be caught and written to ticker_error."""
    mock_llm.invoke.side_effect = Exception("Groq rate limit exceeded")

    result = resolve_ticker(_make_state("What about Stripe?"))

    assert result["ticker"] == ""
    assert "rate limit" in result["ticker_error"]


@patch("agent.graph.nodes.ticker_resolver.llm_classifier")
def test_node_llm_markdown_wrapped_json_handled(mock_llm):
    """LLM response wrapped in markdown fences should still be parsed."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '```json\n{"ticker": "SHOP", "company_name": "Shopify"}\n```'
    )

    result = resolve_ticker(_make_state("What happened with Shopify?"))

    assert result["ticker"] == "SHOP"
    assert result["ticker_error"] is None


def test_node_preserves_existing_state_fields():
    """
    Fields written by previous nodes (e.g. intent) must not be lost.
    The node uses {**state, ...} so all existing keys are carried forward.
    """
    state = _make_state(
        "How did NVDA perform?",
        intent="stock_analysis",
        chart_requested=False,
    )

    result = resolve_ticker(state)

    assert result["intent"] == "stock_analysis"
    assert result["chart_requested"] is False
    assert result["ticker"] == "NVDA"


# ---------------------------------------------------------------------------
# Blocklist: common all-caps words must not be extracted as tickers
# ---------------------------------------------------------------------------

def test_ceo_not_extracted_as_ticker():
    """CEO is in the blocklist; NVDA in the same message should still resolve."""
    state = {"user_message": "What is the CEO saying about NVDA?", "user_config": {}}
    result = resolve_ticker(state)
    assert result["ticker"] == "NVDA"


# ---------------------------------------------------------------------------
# Extended lookup table entries
# ---------------------------------------------------------------------------

def test_meta_lookup():
    """'Meta' (company name) should resolve via lookup table to META ticker."""
    state = {"user_message": "How did Meta do last month?", "user_config": {}}
    result = resolve_ticker(state)
    assert result["ticker"] == "META"


def test_broadcom_lookup():
    """'broadcom' should resolve to AVGO."""
    state = {"user_message": "What happened with Broadcom last quarter?", "user_config": {}}
    result = resolve_ticker(state)
    assert result["ticker"] == "AVGO"


def test_avgo_direct_ticker():
    """AVGO typed directly should pass through Layer 1."""
    state = {"user_message": "How did AVGO perform last week?", "user_config": {}}
    result = resolve_ticker(state)
    assert result["ticker"] == "AVGO"
