"""
Tests for Node 8: Options Analyzer

Strategy: mock yfinance so tests run offline.
The yfinance option_chain() call returns a namedtuple with .calls and .puts
DataFrames. We build realistic mock DataFrames that mirror that structure.

Key mocking note:
  analyze_options calls:
    yf.Ticker(ticker).options          — tuple of expiry date strings
    yf.Ticker(ticker).option_chain(d)  — namedtuple(calls=DataFrame, puts=DataFrame)
"""

import math
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.graph.nodes.options_analyzer import (
    analyze_options,
    _black_scholes_greeks,
    _calculate_max_pain,
)


# ---------------------------------------------------------------------------
# Helpers — mock data factories
# ---------------------------------------------------------------------------

OptionChain = namedtuple("OptionChain", ["calls", "puts"])


def _make_options_df(
    strikes: list,
    volumes: list,
    oi: list,
    ivs: list,
) -> pd.DataFrame:
    """Build a minimal options DataFrame matching yfinance structure."""
    return pd.DataFrame({
        "strike": [float(s) for s in strikes],
        "volume": volumes,
        "openInterest": oi,
        "impliedVolatility": ivs,
        "bid": [1.0] * len(strikes),
        "ask": [1.5] * len(strikes),
        "lastPrice": [1.2] * len(strikes),
    })


def _make_state(ticker: str = "AAPL", **extra) -> dict:
    state = {
        "user_message": f"Show me options for {ticker}",
        "user_config": {},
        "ticker": ticker,
        "intent": "options_view",
        "price_data": {"close_price": 180.0},
    }
    state.update(extra)
    return state


# ---------------------------------------------------------------------------
# Black-Scholes unit tests
# ---------------------------------------------------------------------------

def test_black_scholes_call_delta_between_0_and_1():
    """Call delta must always be between 0 and 1."""
    result = _black_scholes_greeks(S=180.0, K=180.0, T=0.25, sigma=0.3, option_type="call")
    assert 0 < result["delta"] < 1


def test_black_scholes_put_delta_between_minus1_and_0():
    """Put delta must always be between -1 and 0."""
    result = _black_scholes_greeks(S=180.0, K=180.0, T=0.25, sigma=0.3, option_type="put")
    assert -1 < result["delta"] < 0


def test_black_scholes_atm_call_delta_near_0_5():
    """ATM call delta should be approximately 0.5."""
    result = _black_scholes_greeks(S=100.0, K=100.0, T=0.5, sigma=0.2, option_type="call")
    assert 0.45 < result["delta"] < 0.60


def test_black_scholes_returns_all_greeks():
    """All four Greeks must be present and numeric."""
    result = _black_scholes_greeks(S=180.0, K=180.0, T=0.25, sigma=0.3, option_type="call")
    for key in ("delta", "gamma", "theta", "vega"):
        assert key in result
        assert result[key] is not None
        assert isinstance(result[key], float)


def test_black_scholes_invalid_inputs_return_none_greeks():
    """If T=0 or sigma=0, Greeks cannot be computed — all return None."""
    result = _black_scholes_greeks(S=180.0, K=180.0, T=0.0, sigma=0.3, option_type="call")
    assert result["delta"] is None
    assert result["gamma"] is None


# ---------------------------------------------------------------------------
# Max Pain unit tests
# ---------------------------------------------------------------------------

def test_max_pain_returns_a_strike_from_the_chain():
    """Max pain must return one of the strikes in the chain."""
    strikes = [150.0, 160.0, 170.0, 180.0, 190.0]
    calls = _make_options_df(strikes, [100] * 5, [500, 400, 300, 200, 100], [0.3] * 5)
    puts = _make_options_df(strikes, [80] * 5, [100, 200, 300, 400, 500], [0.3] * 5)

    result = _calculate_max_pain(calls, puts)

    assert result is not None
    assert result in strikes


def test_max_pain_skewed_oi_favors_expected_strike():
    """
    When all call OI is at low strikes and all put OI is at high strikes,
    max pain should be in the middle of the range.
    """
    strikes = [100.0, 150.0, 200.0]
    # Heavy call OI at 100, heavy put OI at 200 → pain for buyers minimized near 150
    calls = _make_options_df(strikes, [10, 10, 10], [1000, 10, 10], [0.3] * 3)
    puts = _make_options_df(strikes, [10, 10, 10], [10, 10, 1000], [0.3] * 3)

    result = _calculate_max_pain(calls, puts)

    assert result == pytest.approx(150.0)


def test_max_pain_returns_none_on_empty_dataframes():
    """Empty DataFrames should not crash _calculate_max_pain."""
    calls = pd.DataFrame(columns=["strike", "volume", "openInterest", "impliedVolatility"])
    puts = pd.DataFrame(columns=["strike", "volume", "openInterest", "impliedVolatility"])

    result = _calculate_max_pain(calls, puts)

    assert result is None


# ---------------------------------------------------------------------------
# Full node tests
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.options_analyzer.yf.Ticker")
def test_options_data_populated_on_success(mock_ticker_class):
    """Happy path: options_data must be returned with all required keys."""
    strikes = [160.0, 170.0, 180.0, 190.0, 200.0]
    calls = _make_options_df(strikes, [500, 400, 300, 200, 100], [1000, 800, 600, 400, 200], [0.25] * 5)
    puts = _make_options_df(strikes, [100, 200, 300, 400, 500], [200, 400, 600, 800, 1000], [0.25] * 5)
    chain = OptionChain(calls=calls, puts=puts)

    mock_instance = MagicMock()
    mock_instance.options = ("2024-06-21", "2024-07-19", "2024-09-20")
    mock_instance.option_chain.return_value = chain
    mock_ticker_class.return_value = mock_instance

    result = analyze_options(_make_state())

    assert result["options_error"] is None
    od = result["options_data"]
    assert od is not None

    required_keys = {
        "expiration_dates", "put_call_ratio",
        "highest_volume_calls", "highest_volume_puts",
        "total_call_volume", "total_put_volume",
        "average_implied_volatility", "max_pain",
    }
    assert required_keys.issubset(od.keys())


@patch("agent.graph.nodes.options_analyzer.yf.Ticker")
def test_put_call_ratio_calculation(mock_ticker_class):
    """put_call_ratio = total_put_volume / total_call_volume."""
    strikes = [170.0, 180.0, 190.0]
    calls = _make_options_df(strikes, [100, 200, 300], [500, 400, 300], [0.3] * 3)
    puts = _make_options_df(strikes, [300, 300, 300], [300, 400, 500], [0.3] * 3)
    chain = OptionChain(calls=calls, puts=puts)

    mock_instance = MagicMock()
    mock_instance.options = ("2024-06-21",)
    mock_instance.option_chain.return_value = chain
    mock_ticker_class.return_value = mock_instance

    result = analyze_options(_make_state())

    # total_call_volume = 600, total_put_volume = 900 → PCR = 1.5
    assert result["options_data"]["total_call_volume"] == 600
    assert result["options_data"]["total_put_volume"] == 900
    assert result["options_data"]["put_call_ratio"] == pytest.approx(1.5, rel=0.01)


@patch("agent.graph.nodes.options_analyzer.yf.Ticker")
def test_greeks_sample_populated_when_price_available(mock_ticker_class):
    """greeks_sample must be present when close_price is in state.price_data."""
    strikes = [170.0, 180.0, 190.0]
    calls = _make_options_df(strikes, [100, 200, 100], [500, 400, 300], [0.3, 0.25, 0.3])
    puts = _make_options_df(strikes, [100, 200, 100], [300, 400, 500], [0.3, 0.25, 0.3])
    chain = OptionChain(calls=calls, puts=puts)

    mock_instance = MagicMock()
    mock_instance.options = ("2024-06-21",)
    mock_instance.option_chain.return_value = chain
    mock_ticker_class.return_value = mock_instance

    result = analyze_options(_make_state(price_data={"close_price": 180.0}))

    greeks = result["options_data"].get("greeks_sample")
    assert greeks is not None
    assert "call" in greeks
    assert "put" in greeks
    assert greeks["call"]["delta"] is not None


@patch("agent.graph.nodes.options_analyzer.yf.Ticker")
def test_options_error_written_when_no_expiry_dates(mock_ticker_class):
    """When options tuple is empty, options_error must be set and options_data None."""
    mock_instance = MagicMock()
    mock_instance.options = ()
    mock_ticker_class.return_value = mock_instance

    result = analyze_options(_make_state())

    assert result["options_data"] is None
    assert result["options_error"] is not None


@patch("agent.graph.nodes.options_analyzer.yf.Ticker")
def test_options_error_written_on_exception(mock_ticker_class):
    """If yfinance raises during option_chain(), options_error must be set."""
    mock_instance = MagicMock()
    mock_instance.options = ("2024-06-21",)
    mock_instance.option_chain.side_effect = Exception("network timeout")
    mock_ticker_class.return_value = mock_instance

    result = analyze_options(_make_state())

    assert result["options_data"] is None
    assert result["options_error"] is not None
    assert "network timeout" in result["options_error"]


def test_options_error_written_when_ticker_missing():
    """If ticker is missing from state, options_error must be set."""
    result = analyze_options({"user_message": "show options", "user_config": {}, "ticker": ""})

    assert result["options_data"] is None
    assert result["options_error"] is not None


@patch("agent.graph.nodes.options_analyzer.yf.Ticker")
def test_options_data_preserves_existing_state(mock_ticker_class):
    """Fields written by earlier nodes must survive through options_analyzer."""
    strikes = [170.0, 180.0, 190.0]
    calls = _make_options_df(strikes, [100, 200, 100], [500, 400, 300], [0.3] * 3)
    puts = _make_options_df(strikes, [100, 200, 100], [300, 400, 500], [0.3] * 3)
    chain = OptionChain(calls=calls, puts=puts)

    mock_instance = MagicMock()
    mock_instance.options = ("2024-06-21",)
    mock_instance.option_chain.return_value = chain
    mock_ticker_class.return_value = mock_instance

    state = _make_state(intent="options_view", company_name="Apple")
    result = analyze_options(state)

    assert result["intent"] == "options_view"
    assert result["company_name"] == "Apple"


# ---------------------------------------------------------------------------
# Black-Scholes correctness tests — known analytical values
# ---------------------------------------------------------------------------

def test_black_scholes_call_delta_known_value():
    """
    Known test case: S=100, K=100, T=1yr, r=5%, sigma=20%
    Expected call delta ~0.637 (d1 = 0.35, N(0.35) = 0.6368).
    """
    result = _black_scholes_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    assert abs(result["delta"] - 0.637) < 0.005
    assert result["delta"] > 0
    assert result["gamma"] > 0


def test_black_scholes_put_delta_known_value():
    """Put delta must equal call delta - 1 (put-call parity)."""
    call = _black_scholes_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    put = _black_scholes_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
    assert abs(put["delta"] - (call["delta"] - 1)) < 0.001
