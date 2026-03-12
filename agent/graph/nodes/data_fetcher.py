"""
Node 4: Price Data Fetcher

Reads:  ticker, start_date, end_date
Writes: price_data, volume_anomaly, price_error

Fetches daily OHLCV data for the requested ticker and date range.
Primary source: yfinance. Fallback: Alpha Vantage (only if API key is set).

Also computes volume_anomaly by comparing the period's average daily
volume against a 90-day historical baseline. A ratio > 1.5x flags
unusual trading activity — a proxy signal used by the Response
Synthesizer to add context ("volume was significantly elevated").

No LLM calls in this node — pure data retrieval and arithmetic.
"""

import logging
import os
from datetime import datetime, timedelta

import requests
import yfinance as yf

from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)

# Volume anomaly threshold from TDD section 4 Node 4.
# Configurable here so it can be adjusted without hunting through node logic.
VOLUME_ANOMALY_THRESHOLD = 1.5

# How many days of history to use for the volume baseline.
HISTORICAL_BASELINE_DAYS = 90


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _build_daily_prices(hist) -> list[dict]:
    """
    Convert a yfinance history DataFrame into the daily_prices list format
    expected by state.price_data and consumed by Node 10 (Chart Generator).

    Each entry: {date, open, high, low, close, volume}
    Why keep this separate from the summary metrics? The Chart Generator
    needs per-day OHLCV to draw a candlestick chart. If we discarded the
    DataFrame after computing summary stats, we would lose this data.
    """
    daily = []
    for ts, row in hist.iterrows():
        daily.append({
            "date": ts.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })
    return daily


def _compute_volume_anomaly(
    ticker: str, hist, start_date: str
) -> dict:
    """
    Compare the average daily volume in `hist` (the queried period) against
    a 90-day historical baseline ending at start_date.

    Returns a volume_anomaly dict. If the baseline fetch fails, is_anomalous
    is set to False and the failure is logged (not propagated — a missing
    anomaly signal should not fail the whole node).
    """
    period_avg = float(hist["Volume"].mean())

    try:
        # Fetch the 90-day baseline period ending just before the query starts.
        # We end at start_date so the baseline does not overlap the query period.
        baseline_end = datetime.strptime(start_date, "%Y-%m-%d")
        baseline_start = baseline_end - timedelta(days=HISTORICAL_BASELINE_DAYS)

        stock = yf.Ticker(ticker)
        hist_baseline = stock.history(
            start=baseline_start.strftime("%Y-%m-%d"),
            end=start_date,
        )

        if hist_baseline.empty:
            raise ValueError("Empty baseline DataFrame from yfinance")

        historical_avg = float(hist_baseline["Volume"].mean())

        if historical_avg == 0:
            raise ValueError("Historical average volume is zero — cannot compute ratio")

        anomaly_ratio = period_avg / historical_avg
        is_anomalous = anomaly_ratio > VOLUME_ANOMALY_THRESHOLD

    except Exception as e:
        logger.warning("Volume baseline fetch failed for %s: %s", ticker, e)
        # Return a partial anomaly dict; downstream node will note baseline
        # was unavailable rather than crashing.
        return {
            "average_daily_volume": round(period_avg, 0),
            "historical_average_volume": None,
            "anomaly_ratio": None,
            "is_anomalous": False,
        }

    return {
        "average_daily_volume": round(period_avg, 0),
        "historical_average_volume": round(historical_avg, 0),
        "anomaly_ratio": round(anomaly_ratio, 2),
        "is_anomalous": is_anomalous,
    }


def _fetch_yfinance(
    ticker: str, start_date: str, end_date: str
) -> tuple[dict, dict] | tuple[None, str]:
    """
    Attempt to fetch OHLCV data from yfinance.
    Returns (price_data dict, volume_anomaly dict) on success.
    Returns (None, error_message) on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if hist is None or hist.empty:
            return None, f"yfinance returned no data for {ticker} ({start_date} to {end_date})"

        open_price = round(float(hist.iloc[0]["Open"]), 2)
        close_price = round(float(hist.iloc[-1]["Close"]), 2)
        high_price = round(float(hist["High"].max()), 2)
        low_price = round(float(hist["Low"].min()), 2)
        total_volume = int(hist["Volume"].sum())
        price_change = round(close_price - open_price, 2)
        percent_change = round((price_change / open_price) * 100, 2) if open_price != 0 else 0.0

        price_data = {
            "ticker": ticker.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "open_price": open_price,
            "close_price": close_price,
            "high_price": high_price,
            "low_price": low_price,
            "total_volume": total_volume,
            "percent_change": percent_change,
            "price_change": price_change,
            "daily_prices": _build_daily_prices(hist),
            "source": "yfinance",
        }

        volume_anomaly = _compute_volume_anomaly(ticker, hist, start_date)

        return price_data, volume_anomaly

    except Exception as e:
        return None, f"yfinance error: {e}"


# ---------------------------------------------------------------------------
# Alpha Vantage fallback
# ---------------------------------------------------------------------------

def _fetch_alpha_vantage(
    ticker: str, start_date: str, end_date: str, api_key: str
) -> tuple[dict, None] | tuple[None, str]:
    """
    Fetch OHLCV data from Alpha Vantage TIME_SERIES_DAILY.
    Returns (price_data dict, None) on success — no volume anomaly because
    fetching a baseline would double the API call count against the 5/min
    rate limit on the free tier.
    Returns (None, error_message) on failure.
    """
    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY"
            f"&symbol={ticker}"
            f"&outputsize=full"
            f"&apikey={api_key}"
        )
        response = requests.get(url, timeout=10)
        data = response.json()

        time_series = data.get("Time Series (Daily)")
        if not time_series:
            return None, f"Alpha Vantage API error or rate limit: {data.get('Note') or data.get('Information') or 'unknown'}"

        import pandas as pd
        df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df_filtered = df.loc[start_date:end_date]

        if df_filtered.empty:
            return None, f"Alpha Vantage has no data for {ticker} in range {start_date} to {end_date}"

        open_price = round(float(df_filtered.iloc[0]["1. open"]), 2)
        close_price = round(float(df_filtered.iloc[-1]["4. close"]), 2)
        high_price = round(float(df_filtered["2. high"].max()), 2)
        low_price = round(float(df_filtered["3. low"].min()), 2)
        total_volume = int(df_filtered["5. volume"].sum())
        price_change = round(close_price - open_price, 2)
        percent_change = round((price_change / open_price) * 100, 2) if open_price != 0 else 0.0

        # Build daily_prices from Alpha Vantage columns
        daily_prices = []
        for ts, row in df_filtered.iterrows():
            daily_prices.append({
                "date": ts.strftime("%Y-%m-%d"),
                "open": round(float(row["1. open"]), 2),
                "high": round(float(row["2. high"]), 2),
                "low": round(float(row["3. low"]), 2),
                "close": round(float(row["4. close"]), 2),
                "volume": int(row["5. volume"]),
            })

        price_data = {
            "ticker": ticker.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "open_price": open_price,
            "close_price": close_price,
            "high_price": high_price,
            "low_price": low_price,
            "total_volume": total_volume,
            "percent_change": percent_change,
            "price_change": price_change,
            "daily_prices": daily_prices,
            "source": "alpha_vantage",
        }

        return price_data, None

    except Exception as e:
        return None, f"Alpha Vantage error: {e}"


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def fetch_price_data(state: AgentState) -> AgentState:
    """
    Fetch OHLCV price data for the ticker and date range in state.
    Tries yfinance first; falls back to Alpha Vantage if configured.
    Computes volume anomaly alongside the main data fetch.
    """
    ticker = state.get("ticker", "")
    start_date = state.get("start_date", "")
    end_date = state.get("end_date", "")

    if not ticker or not start_date or not end_date:
        msg = f"Missing required state fields: ticker={ticker!r} start={start_date!r} end={end_date!r}"
        logger.error(msg)
        return {**state, "price_data": None, "volume_anomaly": None, "price_error": msg}

    # Layer 1 — yfinance (primary)
    price_data, result = _fetch_yfinance(ticker, start_date, end_date)

    if price_data is not None:
        volume_anomaly = result  # second return value is anomaly dict on success
        logger.info(
            "fetch_price_data [yfinance] → %s %.2f%% (%d trading days)",
            ticker, price_data["percent_change"], len(price_data["daily_prices"]),
        )
        return {
            **state,
            "price_data": price_data,
            "volume_anomaly": volume_anomaly,
            "price_error": None,
        }

    yfinance_error = result  # second return value is error string on failure
    logger.warning("fetch_price_data yfinance failed: %s", yfinance_error)

    # Layer 2 — Alpha Vantage fallback (only if key is configured)
    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not alpha_key:
        logger.warning("fetch_price_data: no ALPHA_VANTAGE_API_KEY set; no fallback available")
        return {
            **state,
            "price_data": None,
            "volume_anomaly": None,
            "price_error": yfinance_error,
        }

    price_data, av_result = _fetch_alpha_vantage(ticker, start_date, end_date, alpha_key)

    if price_data is not None:
        logger.info("fetch_price_data [alpha_vantage] → %s", ticker)
        return {
            **state,
            "price_data": price_data,
            "volume_anomaly": None,   # not computed for AV fallback (rate limit concern)
            "price_error": None,
        }

    # Both sources failed
    combined_error = f"yfinance: {yfinance_error} | alpha_vantage: {av_result}"
    logger.error("fetch_price_data: all sources failed for %s: %s", ticker, combined_error)
    return {
        **state,
        "price_data": None,
        "volume_anomaly": None,
        "price_error": combined_error,
    }
