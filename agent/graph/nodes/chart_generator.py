"""
Node 10: Chart Generator

Reads:  ticker, price_data (daily_prices), volume_anomaly
Writes: chart_data, chart_error

Generates a Plotly interactive candlestick chart from the daily price data
already in state. No new API calls — Node 4 (data_fetcher) fetched the
OHLCV data; this node just visualises it.

If volume_anomaly.is_anomalous is True, a volume bar subplot is added
below the candlestick to highlight the unusual trading activity.

The chart is serialised to Plotly's JSON format and stored in chart_data.
Chainlit deserialises it and renders it as an interactive chart inline.

Why JSON instead of an HTML file?
Storing the chart as JSON in state keeps the graph stateless — no file
system writes, no path management, no cleanup. Chainlit can render Plotly
JSON directly via its native Plotly element support.
"""

import json
import logging

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)


def _build_chart(
    ticker: str,
    daily_prices: list[dict],
    volume_anomaly: dict | None,
) -> go.Figure:
    """
    Build a Plotly Figure from daily_prices.

    If volume_anomaly is present and is_anomalous is True, the figure uses
    two rows: candlestick on top, volume bars on bottom. Otherwise a single
    candlestick row is returned.

    Why make_subplots even for a single row?
    Using make_subplots consistently means the downstream rendering logic
    in Chainlit doesn't need to handle two different figure structures.
    A single-row make_subplots figure is identical to a plain Figure from
    Chainlit's perspective.
    """
    dates = [d["date"] for d in daily_prices]
    opens = [d["open"] for d in daily_prices]
    highs = [d["high"] for d in daily_prices]
    lows = [d["low"] for d in daily_prices]
    closes = [d["close"] for d in daily_prices]
    volumes = [d["volume"] for d in daily_prices]

    is_anomalous = (
        volume_anomaly is not None and volume_anomaly.get("is_anomalous", False)
    )

    if is_anomalous:
        # Two-row layout: candlestick (70% height) + volume bars (30% height)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )
        row_candle = 1
        row_volume = 2
    else:
        fig = make_subplots(rows=1, cols=1)
        row_candle = 1
        row_volume = None  # not used

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=ticker,
            increasing_line_color="#26a69a",   # green for up days
            decreasing_line_color="#ef5350",   # red for down days
        ),
        row=row_candle,
        col=1,
    )

    # Volume bars (only when anomalous — per TDD)
    if is_anomalous and row_volume is not None:
        # Colour volume bars to match candle direction
        bar_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(opens, closes)
        ]
        anomaly_ratio = volume_anomaly.get("anomaly_ratio", "")
        ratio_label = f" ({anomaly_ratio:.1f}x avg)" if anomaly_ratio else ""

        fig.add_trace(
            go.Bar(
                x=dates,
                y=volumes,
                name=f"Volume{ratio_label}",
                marker_color=bar_colors,
                opacity=0.7,
            ),
            row=row_volume,
            col=1,
        )

    # Layout styling
    start = dates[0] if dates else ""
    end = dates[-1] if dates else ""
    title = f"{ticker} — {start} to {end}"
    if is_anomalous:
        title += "  ⚠ Unusual volume detected"

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,   # hide range slider for cleaner look
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
        showlegend=True,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )

    fig.update_yaxes(
        gridcolor="#313244",
        zerolinecolor="#313244",
    )
    fig.update_xaxes(
        gridcolor="#313244",
    )

    return fig


def generate_chart(state: AgentState) -> AgentState:
    """
    Generate an interactive Plotly candlestick chart from price_data in state.
    Writes chart_data (Plotly JSON string) on success, or chart_error on failure.
    """
    ticker = state.get("ticker", "")
    price_data = state.get("price_data")
    volume_anomaly = state.get("volume_anomaly")

    # Guard: if data_fetcher failed, price_data will be None
    if not price_data:
        msg = "chart_generator: price_data is missing — cannot generate chart"
        logger.warning(msg)
        return {**state, "chart_data": None, "chart_error": msg}

    daily_prices = price_data.get("daily_prices", [])

    if not daily_prices:
        msg = f"chart_generator: daily_prices list is empty for {ticker}"
        logger.warning(msg)
        return {**state, "chart_data": None, "chart_error": msg}

    try:
        fig = _build_chart(ticker, daily_prices, volume_anomaly)
        chart_json = fig.to_json()

        logger.info(
            "generate_chart → %s (%d candles, anomalous=%s)",
            ticker,
            len(daily_prices),
            bool(volume_anomaly and volume_anomaly.get("is_anomalous")),
        )

        return {**state, "chart_data": chart_json, "chart_error": None}

    except Exception as e:
        logger.error("generate_chart failed for %s: %s", ticker, e)
        return {**state, "chart_data": None, "chart_error": str(e)}
