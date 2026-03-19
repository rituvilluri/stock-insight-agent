"""
Node 10: Chart Generator

Reads:  ticker, price_data (daily_prices), volume_anomaly, date_context
Writes: chart_data, chart_error

Generates a TradingView-style Plotly interactive candlestick chart.
Always includes candlestick, volume subplot, and 20-day SMA (when >= 20 data points).

The chart is serialised to Plotly JSON and stored in chart_data.
Chainlit renders it as an interactive chart inline.
"""

import logging

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)


def _fmt_volume(vol) -> str:
    """Humanise a raw volume number into M/B/K string for hover tooltips."""
    if vol is None:
        return "N/A"
    try:
        vol = float(vol)
    except (TypeError, ValueError):
        return str(vol)
    if vol >= 1_000_000_000:
        return f"{vol / 1_000_000_000:.2f}B"
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.2f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return f"{vol:,.0f}"


def _build_chart(
    ticker: str,
    daily_prices: list[dict],
    date_context: str = "",
    volume_anomaly: dict | None = None,
) -> go.Figure:
    """
    Build a TradingView-style Plotly Figure from daily_prices.

    Always includes:
      - Candlestick trace with UI-matched green/red colors
      - Volume subplot (bottom 25% of chart height)
      - 20-day SMA overlay (amber line) when >= 20 data points available

    volume_anomaly is retained as a parameter for API compatibility
    but no longer controls volume visibility.
    """
    dates = [d["date"] for d in daily_prices]
    opens = [d["open"] for d in daily_prices]
    highs = [d["high"] for d in daily_prices]
    lows = [d["low"] for d in daily_prices]
    closes = [d["close"] for d in daily_prices]
    volumes = [d["volume"] for d in daily_prices]

    # Two-row layout: candlestick (75%) + volume (25%) — always
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Volume bar colors: green for up days, red for down days
    bar_colors = [
        "rgba(0,200,150,0.4)" if c >= o else "rgba(255,77,109,0.4)"
        for o, c in zip(opens, closes)
    ]

    # Humanised volume strings for hover tooltip
    vol_labels = [_fmt_volume(v) for v in volumes]

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=ticker,
            increasing=dict(line=dict(color="#00c896"), fillcolor="#00c896"),
            decreasing=dict(line=dict(color="#ff4d6d"), fillcolor="#ff4d6d"),
            hovertemplate=(
                f"{ticker} | %{{x|%b %d}}<br>"
                "O: $%{open:.2f}  H: $%{high:.2f}  L: $%{low:.2f}  C: $%{close:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            name="Volume",
            marker_color=bar_colors,
            text=vol_labels,
            hovertemplate="Vol: %{text}<extra></extra>",
        ),
        row=2, col=1,
    )

    # 20-day SMA overlay (only when enough data)
    if len(closes) >= 20:
        sma_series = pd.Series(closes).rolling(window=20).mean().dropna()
        sma_dates = dates[19:]  # align to the dates where SMA is valid
        fig.add_trace(
            go.Scatter(
                x=sma_dates,
                y=sma_series.tolist(),
                name="20d SMA",
                line=dict(color="#f0b429", width=1.5),
                hovertemplate="20d SMA: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Chart title
    title = f"{ticker} — {date_context}" if date_context else f"{ticker} — {dates[0]} to {dates[-1]}"

    # Layout
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16)),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#e6edf3"),
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=60, b=40),
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)",
        tickformat="$,.2f",
        row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        tickformat=",",
        row=2, col=1,
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        tickformat="%b %d",
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
    date_context = state.get("date_context", "")

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
        fig = _build_chart(ticker, daily_prices, date_context, volume_anomaly)
        chart_json = fig.to_json()

        logger.info(
            "generate_chart → %s (%d candles)",
            ticker,
            len(daily_prices),
        )

        return {**state, "chart_data": chart_json, "chart_error": None}

    except Exception as e:
        logger.error("generate_chart failed for %s: %s", ticker, e)
        return {**state, "chart_data": None, "chart_error": str(e)}
