"""
Tests for Node 3: Date Parser

Strategy:
- Layer 1 (simple regex): pure Python — no mocking needed. We test relative
  properties (is the range approximately right?) rather than exact dates,
  since datetime.now() shifts every time tests run.
- Layer 2 (earnings lookup): mock yfinance so tests are fast and offline.
- Layer 3 (LLM fallback): mock llm_classifier.
- date_missing / include_current_snapshot: checked on full node output.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from agent.graph.nodes.date_parser import (
    parse_dates,
    _parse_simple_range,
    _extract_earnings_quarter_year,
    _get_earnings_date,
    _has_current_snapshot_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(message: str, ticker: str = "NVDA", **extra) -> dict:
    state = {"user_message": message, "user_config": {}, "ticker": ticker}
    state.update(extra)
    return state


def _days_between(start_iso: str, end_iso: str) -> int:
    fmt = "%Y-%m-%d"
    return (datetime.strptime(end_iso, fmt) - datetime.strptime(start_iso, fmt)).days


def _mock_llm_response(content: str):
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# Layer 1: simple relative range (unit tests — no mocking)
# ---------------------------------------------------------------------------

def test_last_week_range():
    result = _parse_simple_range("What happened with NVDA last week?")
    assert result is not None
    start, end, ctx = result
    assert _days_between(start, end) == 7
    assert "week" in ctx


def test_last_n_weeks_range():
    result = _parse_simple_range("How did Tesla do over the last 3 weeks?")
    assert result is not None
    start, end, ctx = result
    assert _days_between(start, end) == 21
    assert "3" in ctx


def test_last_month_range():
    result = _parse_simple_range("Apple performance last month")
    assert result is not None
    start, end, ctx = result
    assert _days_between(start, end) == 30
    assert "month" in ctx


def test_last_n_months_range():
    result = _parse_simple_range("How did Microsoft do over the last 6 months?")
    assert result is not None
    start, end, ctx = result
    assert _days_between(start, end) == 180


def test_last_quarter_range():
    result = _parse_simple_range("NVDA performance last quarter")
    assert result is not None
    start, end, ctx = result
    assert _days_between(start, end) == 90
    assert "quarter" in ctx


def test_last_year_range():
    result = _parse_simple_range("How did META do over the last year?")
    assert result is not None
    start, end, ctx = result
    assert _days_between(start, end) == 365


def test_past_n_days_range():
    result = _parse_simple_range("Show me the past 14 days of AAPL")
    assert result is not None
    start, end, ctx = result
    assert _days_between(start, end) == 14


def test_n_months_ago_style():
    result = _parse_simple_range("What was happening 3 months ago with TSLA?")
    assert result is not None
    start, end, ctx = result
    # end should be approximately today; start ~90 days before
    assert _days_between(start, end) == 90


def test_pattern_mid_sentence():
    """
    Critical regression test: the pattern must match mid-sentence.
    The old tool used re.match() which only matched at position 0.
    With re.search() this should work.
    """
    result = _parse_simple_range(
        "Can you tell me what happened with NVDA performance last week please?"
    )
    assert result is not None
    start, end, ctx = result
    assert _days_between(start, end) == 7


def test_no_date_pattern_returns_none():
    result = _parse_simple_range("What is the capital of France?")
    assert result is None


# ---------------------------------------------------------------------------
# Layer 2: earnings quarter/year extraction (unit tests — no mocking)
# ---------------------------------------------------------------------------

def test_extract_q2_2024_earnings():
    result = _extract_earnings_quarter_year("What happened around Q2 2024 earnings?")
    assert result == (2, 2024)


def test_extract_q1_with_short_year():
    result = _extract_earnings_quarter_year("around Q1 '23 earnings")
    assert result == (1, 2023)


def test_extract_q4_without_earnings_keyword_returns_none():
    """
    Quarter+year alone without 'earnings' should not trigger the
    earnings lookup — it might just be a date reference.
    """
    result = _extract_earnings_quarter_year("How did NVDA do in Q2 2024?")
    assert result is None


def test_extract_no_quarter_returns_none():
    result = _extract_earnings_quarter_year("What happened around the 2024 earnings?")
    assert result is None


# ---------------------------------------------------------------------------
# include_current_snapshot detection (unit tests)
# ---------------------------------------------------------------------------

def test_current_snapshot_detected():
    assert _has_current_snapshot_request(
        "How did NVDA do last month? And what's happening right now?"
    ) is True


def test_current_snapshot_not_detected_for_pure_historical():
    assert _has_current_snapshot_request(
        "What happened with AAPL last quarter?"
    ) is False


# ---------------------------------------------------------------------------
# Full node — Layer 1 end-to-end
# ---------------------------------------------------------------------------

def test_node_simple_range_end_to_end():
    result = parse_dates(_make_state("How did Apple perform last week?", ticker="AAPL"))
    assert result["date_missing"] is False
    assert result["start_date"] != ""
    assert result["end_date"] != ""
    assert _days_between(result["start_date"], result["end_date"]) == 7
    assert result["date_error"] is None


def test_node_sets_include_current_snapshot():
    result = parse_dates(
        _make_state(
            "How did NVDA do last month? And what is happening right now?",
            ticker="NVDA",
        )
    )
    assert result["date_missing"] is False
    assert result["include_current_snapshot"] is True


def test_node_include_current_snapshot_false_for_historical_only():
    result = parse_dates(_make_state("NVDA last quarter", ticker="NVDA"))
    assert result["include_current_snapshot"] is False


# ---------------------------------------------------------------------------
# Full node — Layer 2 (earnings) end-to-end with mocked yfinance
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.date_parser.yf.Ticker")
def test_node_earnings_range_correct_window(mock_ticker_class):
    """
    Earnings date found in yfinance → 14 days before + 7 days after window.
    NVDA Q2 2024 earnings were approximately 2024-05-22.
    Use short-year format (Q2 '24) so the Q{N} YYYY Layer 1 regex does NOT
    intercept this message; Layer 2 earnings lookup must handle it.
    """
    earnings_date = pd.Timestamp("2024-05-22", tz="America/New_York")
    mock_df = pd.DataFrame({"EPS Estimate": [5.59]}, index=[earnings_date])

    mock_instance = MagicMock()
    mock_instance.earnings_dates = mock_df
    mock_ticker_class.return_value = mock_instance

    result = parse_dates(
        _make_state("What happened around Q2 '24 earnings?", ticker="NVDA")
    )

    assert result["date_missing"] is False
    assert result["start_date"] == "2024-05-08"   # 14 days before May 22
    assert result["end_date"] == "2024-05-29"      # 7 days after May 22
    assert "Q2" in result["date_context"]
    assert result["date_error"] is None


@patch("agent.graph.nodes.date_parser.yf.Ticker")
@patch("agent.graph.nodes.date_parser.llm_classifier")
def test_node_earnings_yfinance_failure_falls_through_to_llm(mock_llm, mock_ticker_class):
    """
    If yfinance fails to return an earnings date, the node should fall
    through to Layer 3 (LLM) rather than setting date_missing.
    Use short-year format (Q2 '24) so the Q{N} YYYY Layer 1 regex does NOT
    intercept this message.
    """
    mock_instance = MagicMock()
    mock_instance.earnings_dates = pd.DataFrame()  # empty — no data
    mock_ticker_class.return_value = mock_instance

    mock_llm.invoke.return_value = _mock_llm_response(
        '{"start_date": "2024-05-08", "end_date": "2024-05-29", '
        '"date_context": "around Q2 2024 earnings"}'
    )

    result = parse_dates(
        _make_state("What happened around Q2 '24 earnings?", ticker="NVDA")
    )

    assert result["date_missing"] is False
    assert result["start_date"] == "2024-05-08"
    mock_llm.invoke.assert_called_once()  # LLM was the fallback


# ---------------------------------------------------------------------------
# Full node — Layer 3 (LLM fallback)
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.date_parser.llm_classifier")
def test_node_llm_fallback_for_ambiguous_expression(mock_llm):
    """Complex date expressions the regex can't handle go to the LLM."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"start_date": "2020-02-20", "end_date": "2020-03-23", '
        '"date_context": "COVID market crash"}'
    )

    result = parse_dates(_make_state("What happened during the COVID crash?"))

    assert result["date_missing"] is False
    assert result["start_date"] == "2020-02-20"
    assert result["end_date"] == "2020-03-23"
    assert "COVID" in result["date_context"]
    assert result["date_error"] is None


@patch("agent.graph.nodes.date_parser.llm_classifier")
def test_node_llm_returns_null_sets_date_missing(mock_llm):
    """If LLM can't determine a date range, date_missing must be True."""
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"start_date": null, "end_date": null, "date_context": null}'
    )

    result = parse_dates(_make_state("Tell me something interesting"))

    assert result["date_missing"] is True
    assert result["date_error"] is None  # not an error, just no date found


@patch("agent.graph.nodes.date_parser.llm_classifier")
def test_node_llm_exception_writes_date_error(mock_llm):
    """LLM exceptions must be caught; date_error set; date_missing True."""
    mock_llm.invoke.side_effect = Exception("Groq rate limit exceeded")

    result = parse_dates(_make_state("What happened during the dot-com crash?"))

    assert result["date_missing"] is True
    assert result["date_error"] is not None
    assert "rate limit" in result["date_error"]


# ---------------------------------------------------------------------------
# State preservation
# ---------------------------------------------------------------------------

def test_node_preserves_existing_state_fields():
    """
    Fields written by earlier nodes (intent, ticker, company_name) must
    survive unchanged through the date parser.
    """
    state = _make_state(
        "What happened with NVDA last week?",
        ticker="NVDA",
        intent="stock_analysis",
        company_name="NVIDIA",
        chart_requested=False,
    )

    result = parse_dates(state)

    assert result["intent"] == "stock_analysis"
    assert result["company_name"] == "NVIDIA"
    assert result["chart_requested"] is False
    assert result["ticker"] == "NVDA"


# ---------------------------------------------------------------------------
# Exhaustive Layer 1 edge cases — must never reach LLM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "last 7 days", "last 30 days", "last 1 day", "past 3 weeks", "past 1 week",
    "last 6 months", "past 1 month", "last week", "last month", "last quarter",
    "last year", "this week", "this month", "yesterday", "3 months ago",
    "1 month ago", "2 weeks ago", "1 week ago", "5 days ago",
    "Q1 2024", "Q2 2023", "Q3 2022", "Q4 2025", "Q1 of 2024", "Q3 of 2025",
    "How did NVDA do Q4 2025? Show me a chart too.",
])
def test_layer1_does_not_reach_llm(message):
    """Every Layer 1 pattern must resolve without calling the LLM."""
    state = {"user_message": message, "user_config": {}, "ticker": "NVDA"}
    with patch("agent.graph.nodes.date_parser.llm_classifier") as mock_llm:
        mock_llm.invoke.side_effect = AssertionError("Layer 3 LLM called unexpectedly")
        result = parse_dates(state)
    assert result.get("date_missing") is False, f"Layer 1 missed: {message!r}"
    assert result.get("date_error") is None


# ---------------------------------------------------------------------------
# Layer 2: fiscal calendar handling in _get_earnings_date
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.date_parser.yf.Ticker")
def test_get_earnings_date_fiscal_calendar_q4(mock_ticker_class):
    """
    Q4 fiscal calendar: NVIDIA reports Q4 earnings in February of the
    following year. The old calendar-month filter (Oct-Dec) missed this.
    The extended window (Oct 2024 – Apr 2025) must find it.
    """
    earnings_date = pd.Timestamp("2025-02-26", tz="America/New_York")
    mock_df = pd.DataFrame({"EPS Estimate": [0.89]}, index=[earnings_date])

    mock_instance = MagicMock()
    mock_instance.earnings_dates = mock_df
    mock_ticker_class.return_value = mock_instance

    result = _get_earnings_date("NVDA", quarter=4, year=2024)

    assert result is not None
    assert result.year == 2025
    assert result.month == 2
    assert result.day == 26


@patch("agent.graph.nodes.date_parser.yf.Ticker")
def test_get_earnings_date_calendar_year_company_unaffected(mock_ticker_class):
    """
    Calendar-year companies (report within the same quarter) must still
    resolve correctly after the window widening.
    """
    earnings_date = pd.Timestamp("2024-05-22", tz="America/New_York")
    mock_df = pd.DataFrame({"EPS Estimate": [5.59]}, index=[earnings_date])

    mock_instance = MagicMock()
    mock_instance.earnings_dates = mock_df
    mock_ticker_class.return_value = mock_instance

    result = _get_earnings_date("NVDA", quarter=2, year=2024)

    assert result is not None
    assert result.year == 2024
    assert result.month == 5
    assert result.day == 22


@patch("agent.graph.nodes.date_parser.yf.Ticker")
def test_node_earnings_fiscal_calendar_nvidia_style(mock_ticker_class):
    """
    Full node test: 'around Q4 '24 earnings' for a fiscal-calendar company
    whose report date is Feb 2025. Must resolve to the ±window around Feb 26.
    """
    earnings_date = pd.Timestamp("2025-02-26", tz="America/New_York")
    mock_df = pd.DataFrame({"EPS Estimate": [0.89]}, index=[earnings_date])

    mock_instance = MagicMock()
    mock_instance.earnings_dates = mock_df
    mock_ticker_class.return_value = mock_instance

    result = parse_dates(_make_state("What happened around Q4 '24 earnings?", ticker="NVDA"))

    assert result["date_missing"] is False
    assert result["start_date"] == "2025-02-12"   # 14 days before Feb 26
    assert result["end_date"] == "2025-03-05"      # 7 days after Feb 26
    assert "Q4" in result["date_context"]
    assert result["date_error"] is None


# ---------------------------------------------------------------------------
# Task 1: Q{N} [of] YYYY Layer 1 regex — exact boundary assertions
# ---------------------------------------------------------------------------

BASE_STATE = {"user_message": "", "user_config": {}, "ticker": "NVDA"}

@pytest.mark.parametrize("message,expected_start,expected_end,description", [
    ("Tell me how Nvidia did Q4 2025", "2025-10-01", "2025-12-31", "Q4 YYYY"),
    ("How did TSLA do Q1 2024?", "2024-01-01", "2024-03-31", "Q1 YYYY"),
    ("Q2 2023 performance of Apple", "2023-04-01", "2023-06-30", "Q2 YYYY at start"),
    ("What happened in Q3 of 2022?", "2022-07-01", "2022-09-30", "Q3 of YYYY with 'of'"),
    ("NVDA Q4 2024 earnings results", "2024-10-01", "2024-12-31", "Q4 YYYY with earnings keyword"),
])
def test_quarter_year_regex(message, expected_start, expected_end, description):
    """Q{N} YYYY must resolve via Layer 1 regex, never reaching the LLM."""
    state = {**BASE_STATE, "user_message": message}
    with patch("agent.graph.nodes.date_parser.llm_classifier") as mock_llm:
        mock_llm.invoke.side_effect = AssertionError("Layer 3 LLM was called — Layer 1 regex missed the pattern")
        result = parse_dates(state)
    assert result["start_date"] == expected_start, f"FAILED ({description}): start {result['start_date']} != {expected_start}"
    assert result["end_date"] == expected_end, f"FAILED ({description}): end {result['end_date']} != {expected_end}"
    assert result["date_missing"] is False
    assert result["date_error"] is None
