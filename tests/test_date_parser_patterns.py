"""
Exhaustive parametrized tests for date_parser.py.

This file tests every Layer 1 pattern, Layer 2 earnings lookup, and
Layer 3 LLM fallback (mocked). Any future date expression addition must
have a test case here before the PR is merged.

Why a separate file from test_date_parser.py?
test_date_parser.py tests the node function end-to-end.
This file tests each parsing layer in isolation, making it easy to
identify exactly which layer a new pattern should live in.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from agent.graph.nodes.date_parser import (
    parse_dates,
    _parse_simple_range,
    _parse_earnings_range,
    _parse_with_llm,
)

BASE_STATE = {"user_message": "", "user_config": {}, "ticker": "NVDA"}
TODAY = datetime.now()


# ---------------------------------------------------------------------------
# Layer 1 — simple relative ranges
# ---------------------------------------------------------------------------

class TestLayer1SimpleRange:

    @pytest.mark.parametrize("n", [1, 7, 14, 30, 90])
    def test_last_n_days(self, n):
        result = _parse_simple_range(f"How did NVDA do the last {n} days?")
        assert result is not None
        start, end, ctx = result
        expected_start = (TODAY - timedelta(days=n)).strftime("%Y-%m-%d")
        assert start == expected_start
        assert end == TODAY.strftime("%Y-%m-%d")

    @pytest.mark.parametrize("n", [1, 2, 4, 12])
    def test_last_n_weeks(self, n):
        result = _parse_simple_range(f"last {n} weeks")
        assert result is not None
        start, end, ctx = result
        expected_start = (TODAY - timedelta(weeks=n)).strftime("%Y-%m-%d")
        assert start == expected_start

    @pytest.mark.parametrize("n", [1, 3, 6, 12])
    def test_last_n_months(self, n):
        result = _parse_simple_range(f"past {n} months")
        assert result is not None

    def test_last_week(self):
        result = _parse_simple_range("last week")
        assert result is not None
        start, end, ctx = result
        expected = (TODAY - timedelta(weeks=1)).strftime("%Y-%m-%d")
        assert start == expected

    def test_last_month(self):
        result = _parse_simple_range("last month")
        assert result is not None

    def test_last_quarter(self):
        result = _parse_simple_range("last quarter")
        assert result is not None
        start, _, _ = result
        expected = (TODAY - timedelta(days=90)).strftime("%Y-%m-%d")
        assert start == expected

    def test_last_year(self):
        result = _parse_simple_range("last year")
        assert result is not None

    def test_this_week(self):
        result = _parse_simple_range("this week")
        assert result is not None
        start, _, _ = result
        monday = (TODAY - timedelta(days=TODAY.weekday())).strftime("%Y-%m-%d")
        assert start == monday

    def test_this_month(self):
        result = _parse_simple_range("this month")
        assert result is not None
        start, _, _ = result
        assert start == TODAY.replace(day=1).strftime("%Y-%m-%d")

    def test_yesterday(self):
        result = _parse_simple_range("yesterday")
        assert result is not None
        start, end, _ = result
        expected = (TODAY - timedelta(days=1)).strftime("%Y-%m-%d")
        assert start == expected
        assert end == expected  # yesterday is a single-day range

    @pytest.mark.parametrize("n", [1, 3, 6])
    def test_n_months_ago(self, n):
        result = _parse_simple_range(f"{n} months ago")
        assert result is not None

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_n_weeks_ago(self, n):
        result = _parse_simple_range(f"{n} weeks ago")
        assert result is not None

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_n_days_ago(self, n):
        result = _parse_simple_range(f"{n} days ago")
        assert result is not None

    # --- Calendar quarter patterns (Bug #1 category) ---

    @pytest.mark.parametrize("quarter,year,expected_start,expected_end", [
        (1, 2024, "2024-01-01", "2024-03-31"),
        (2, 2024, "2024-04-01", "2024-06-30"),
        (3, 2024, "2024-07-01", "2024-09-30"),
        (4, 2024, "2024-10-01", "2024-12-31"),
        (4, 2025, "2025-10-01", "2025-12-31"),  # The exact bug from production
        (1, 2022, "2022-01-01", "2022-03-31"),
    ])
    def test_quarter_year(self, quarter, year, expected_start, expected_end):
        """Q{N} YYYY must resolve via Layer 1, never reaching LLM."""
        message = f"Q{quarter} {year}"
        with patch("agent.graph.nodes.date_parser.llm_classifier") as mock_llm:
            mock_llm.invoke.side_effect = AssertionError("Layer 3 LLM called unexpectedly")
            result = _parse_simple_range(message)
        assert result is not None, f"Layer 1 missed Q{quarter} {year}"
        start, end, _ = result
        assert start == expected_start, f"Q{quarter} {year}: start {start} != {expected_start}"
        assert end == expected_end, f"Q{quarter} {year}: end {end} != {expected_end}"

    @pytest.mark.parametrize("message,quarter,year", [
        ("Q3 of 2022", 3, 2022),
        ("Q1 of 2025", 1, 2025),
    ])
    def test_quarter_of_year(self, message, quarter, year):
        """'Q{N} of YYYY' variant must also resolve via Layer 1."""
        result = _parse_simple_range(message)
        assert result is not None, f"Layer 1 missed: {message!r}"

    def test_quarter_mid_sentence(self):
        """Q pattern must match anywhere in a sentence, not just at start."""
        message = "Tell me how Nvidia did Q4 2025. Show me a chart as well"
        result = _parse_simple_range(message)
        assert result is not None, "Q4 2025 not matched mid-sentence"
        start, end, _ = result
        assert start == "2025-10-01"
        assert end == "2025-12-31"

    def test_no_match_returns_none(self):
        result = _parse_simple_range("What is the stock price?")
        assert result is None


# ---------------------------------------------------------------------------
# Layer 2 — earnings-relative lookup
# ---------------------------------------------------------------------------

class TestLayer2EarningsRange:

    def test_earnings_keyword_required(self):
        """Without 'earnings' keyword, Layer 2 must return None."""
        result = _parse_earnings_range("Q2 2024 performance of NVDA", "NVDA")
        assert result is None

    def test_earnings_with_quarter_year(self):
        """With 'earnings' keyword + Q{N} YYYY, Layer 2 must attempt yfinance lookup."""
        mock_dt = datetime(2024, 5, 22)
        with patch("agent.graph.nodes.date_parser._get_earnings_date", return_value=mock_dt):
            result = _parse_earnings_range("Q2 2024 earnings", "NVDA")
        assert result is not None
        start, end, ctx = result
        # 14 days before earnings, 7 days after
        assert start == "2024-05-08"
        assert end == "2024-05-29"
        assert "Q2 2024 earnings" in ctx

    def test_earnings_yfinance_miss_returns_none(self):
        """If yfinance can't find the earnings date, Layer 2 returns None (fall through to LLM)."""
        with patch("agent.graph.nodes.date_parser._get_earnings_date", return_value=None):
            result = _parse_earnings_range("Q2 2024 earnings", "NVDA")
        assert result is None

    def test_no_ticker_returns_none(self):
        result = _parse_earnings_range("Q2 2024 earnings", "")
        assert result is None


# ---------------------------------------------------------------------------
# Layer 3 — LLM fallback (mocked)
# ---------------------------------------------------------------------------

class TestLayer3LLMFallback:

    def test_llm_called_for_complex_expression(self):
        """Layer 3 LLM must be called for expressions Layer 1 and 2 can't handle."""
        state = {**BASE_STATE, "user_message": "during the COVID crash"}
        mock_response = MagicMock()
        mock_response.content = '{"start_date": "2020-02-20", "end_date": "2020-03-23", "date_context": "COVID crash"}'

        with patch("agent.graph.nodes.date_parser.llm_classifier") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = parse_dates(state)

        mock_llm.invoke.assert_called_once()
        assert result["start_date"] == "2020-02-20"
        assert result["end_date"] == "2020-03-23"
        assert result["date_missing"] is False

    def test_llm_returns_null_sets_date_missing(self):
        """If LLM returns null dates, date_missing must be True."""
        state = {**BASE_STATE, "user_message": "some completely ambiguous message"}
        mock_response = MagicMock()
        mock_response.content = '{"start_date": null, "end_date": null, "date_context": null}'

        with patch("agent.graph.nodes.date_parser.llm_classifier") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = parse_dates(state)

        assert result["date_missing"] is True


# ---------------------------------------------------------------------------
# Node function — session context preservation
# ---------------------------------------------------------------------------

class TestSessionContext:

    def test_session_context_preserved_when_no_date_in_message(self):
        """If user sends a follow-up with no date, existing start/end must be preserved."""
        state = {
            **BASE_STATE,
            "user_message": "What about the chart?",
            "start_date": "2025-02-01",
            "end_date": "2025-02-28",
        }
        with patch("agent.graph.nodes.date_parser.llm_classifier") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(
                content='{"start_date": null, "end_date": null, "date_context": null}'
            )
            result = parse_dates(state)

        assert result["date_missing"] is False
        assert result["start_date"] == "2025-02-01"
        assert result["end_date"] == "2025-02-28"
