"""
Tests for Node 10: Chart Generator

No mocking needed — Plotly is a pure Python library with no network calls.
We test real chart generation using minimal synthetic daily_prices data
and inspect the serialised JSON to verify structure and content.

Why test the JSON output rather than the Figure object?
The JSON is what actually gets written to state and consumed by Chainlit.
Testing `fig.to_json()` ensures the full serialisation round-trip works,
not just that the Figure was constructed correctly in memory.
"""

import json

import pytest

from agent.graph.nodes.chart_generator import generate_chart, _build_chart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_prices(n: int = 5) -> list[dict]:
    """Build n days of synthetic OHLCV data."""
    prices = []
    for i in range(n):
        day = f"2024-05-{i + 1:02d}"
        open_ = 100.0 + i
        close = 100.0 + i + 1
        prices.append({
            "date": day,
            "open": open_,
            "high": close + 2.0,
            "low": open_ - 2.0,
            "close": close,
            "volume": 1_000_000 + i * 100_000,
        })
    return prices


def _make_state(
    ticker: str = "NVDA",
    daily_prices: list | None = None,
    volume_anomaly: dict | None = None,
    price_data: dict | None = None,
    **extra,
) -> dict:
    dp = daily_prices if daily_prices is not None else _make_daily_prices()
    pd_ = price_data if price_data is not None else {
        "ticker": ticker,
        "daily_prices": dp,
        "open_price": 100.0,
        "close_price": 105.0,
        "source": "yfinance",
    }
    state = {
        "user_message": f"Show me a chart of {ticker}",
        "user_config": {},
        "ticker": ticker,
        "price_data": pd_,
        "volume_anomaly": volume_anomaly,
    }
    state.update(extra)
    return state


def _parse_chart_json(chart_data: str) -> dict:
    return json.loads(chart_data)


# ---------------------------------------------------------------------------
# Unit tests — _build_chart helper
# ---------------------------------------------------------------------------

def test_build_chart_returns_figure():
    """_build_chart must return a Plotly Figure."""
    import plotly.graph_objects as go
    fig = _build_chart("NVDA", _make_daily_prices(), volume_anomaly=None)
    assert isinstance(fig, go.Figure)


def test_build_chart_has_candlestick_trace():
    """The figure must contain exactly one candlestick trace for normal queries."""
    fig = _build_chart("NVDA", _make_daily_prices(), volume_anomaly=None)
    trace_types = [t.type for t in fig.data]
    assert "candlestick" in trace_types


def test_build_chart_no_volume_subplot_when_not_anomalous():
    """
    When volume_anomaly.is_anomalous is False, the figure should have
    only a candlestick trace — no volume bars.
    """
    anomaly = {"is_anomalous": False, "anomaly_ratio": 1.1}
    fig = _build_chart("NVDA", _make_daily_prices(), volume_anomaly=anomaly)
    trace_types = [t.type for t in fig.data]
    assert "bar" not in trace_types


def test_build_chart_volume_subplot_added_when_anomalous():
    """
    When volume_anomaly.is_anomalous is True, a Bar trace must be added
    as the volume subplot alongside the candlestick.
    """
    anomaly = {"is_anomalous": True, "anomaly_ratio": 2.5}
    fig = _build_chart("NVDA", _make_daily_prices(), volume_anomaly=anomaly)
    trace_types = [t.type for t in fig.data]
    assert "candlestick" in trace_types
    assert "bar" in trace_types


def test_build_chart_title_includes_ticker():
    """The chart title must reference the ticker symbol."""
    fig = _build_chart("TSLA", _make_daily_prices(), volume_anomaly=None)
    assert "TSLA" in fig.layout.title.text


def test_build_chart_title_flags_anomaly():
    """When anomalous, the title should indicate unusual volume."""
    anomaly = {"is_anomalous": True, "anomaly_ratio": 3.0}
    fig = _build_chart("GME", _make_daily_prices(), volume_anomaly=anomaly)
    assert "volume" in fig.layout.title.text.lower() or "unusual" in fig.layout.title.text.lower()


# ---------------------------------------------------------------------------
# Full node tests
# ---------------------------------------------------------------------------

def test_node_chart_data_is_valid_json():
    """chart_data written to state must be valid JSON."""
    result = generate_chart(_make_state())
    assert result["chart_error"] is None
    assert result["chart_data"] is not None
    # This will raise json.JSONDecodeError if invalid
    parsed = _parse_chart_json(result["chart_data"])
    assert "data" in parsed


def test_node_chart_json_contains_candlestick():
    """The serialised figure must have a candlestick trace."""
    result = generate_chart(_make_state())
    parsed = _parse_chart_json(result["chart_data"])
    trace_types = [t.get("type") for t in parsed["data"]]
    assert "candlestick" in trace_types


def test_node_volume_subplot_present_in_json_when_anomalous():
    """When anomalous, the serialised figure must include a bar trace."""
    anomaly = {"is_anomalous": True, "anomaly_ratio": 2.8}
    result = generate_chart(_make_state(volume_anomaly=anomaly))
    parsed = _parse_chart_json(result["chart_data"])
    trace_types = [t.get("type") for t in parsed["data"]]
    assert "bar" in trace_types


def test_node_no_volume_bar_in_json_when_not_anomalous():
    """When not anomalous, bar trace must not appear in the serialised figure."""
    anomaly = {"is_anomalous": False, "anomaly_ratio": 1.1}
    result = generate_chart(_make_state(volume_anomaly=anomaly))
    parsed = _parse_chart_json(result["chart_data"])
    trace_types = [t.get("type") for t in parsed["data"]]
    assert "bar" not in trace_types


def test_node_price_data_none_sets_chart_data_none():
    """
    If price_data is None (data_fetcher failed upstream), the chart generator
    must not crash. It writes chart_data=None and chart_error with a message.
    """
    state = {
        "user_message": "Show me a chart",
        "user_config": {},
        "ticker": "NVDA",
        "price_data": None,
        "volume_anomaly": None,
    }
    result = generate_chart(state)
    assert result["chart_data"] is None
    assert result["chart_error"] is not None


def test_node_empty_daily_prices_writes_error():
    """An empty daily_prices list must not crash — write chart_error instead."""
    state = _make_state(
        price_data={
            "ticker": "NVDA",
            "daily_prices": [],   # empty
            "open_price": 100.0,
            "close_price": 105.0,
            "source": "yfinance",
        }
    )
    result = generate_chart(state)
    assert result["chart_data"] is None
    assert result["chart_error"] is not None


def test_node_preserves_existing_state_fields():
    """
    Fields written by earlier nodes must survive through chart_generator.
    Specifically: intent, company_name, response_text (from synthesizer),
    date_context all must be unchanged.
    """
    state = _make_state(
        intent="chart_request",
        company_name="NVIDIA",
        date_context="last month",
        response_text="Here is the analysis...",
        chart_requested=True,
    )
    result = generate_chart(state)

    assert result["intent"] == "chart_request"
    assert result["company_name"] == "NVIDIA"
    assert result["date_context"] == "last month"
    assert result["response_text"] == "Here is the analysis..."
    assert result["chart_requested"] is True
