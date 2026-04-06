"""
Tests for Node 4: Price Data Fetcher

Strategy: mock yfinance and requests so tests are fast and offline.
We build realistic mock DataFrames that mirror the structure yfinance
actually returns, so our assertions catch real structural bugs.

Key mocking note:
  _fetch_yfinance calls yf.Ticker().history() TWICE:
    call 1 — the query period (returns the main data)
    call 2 — the 90-day baseline for volume anomaly
  We use side_effect lists to return different DataFrames per call.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from agent.graph.nodes.data_fetcher import (
    fetch_price_data,
    _build_daily_prices,
    _compute_volume_anomaly,
    VOLUME_ANOMALY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers — mock DataFrame factories
# ---------------------------------------------------------------------------

def _make_hist_df(
    n_days: int = 5,
    open_: float = 100.0,
    close: float = 110.0,
    high: float = 115.0,
    low: float = 95.0,
    volume: int = 1_000_000,
) -> pd.DataFrame:
    """Build a minimal yfinance-style history DataFrame."""
    dates = pd.date_range(start="2024-05-01", periods=n_days, freq="B")
    opens = [open_] + [105.0] * (n_days - 1)
    closes = [105.0] * (n_days - 1) + [close]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [high] * n_days,
            "Low": [low] * n_days,
            "Close": closes,
            "Volume": [volume] * n_days,
        },
        index=dates,
    )


def _make_state(
    ticker: str = "NVDA",
    start_date: str = "2024-05-01",
    end_date: str = "2024-05-10",
    **extra,
) -> dict:
    state = {
        "user_message": f"How did {ticker} do?",
        "user_config": {},
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
    }
    state.update(extra)
    return state


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------

def test_build_daily_prices_structure():
    """_build_daily_prices must return a list of dicts with OHLCV keys."""
    hist = _make_hist_df(n_days=3)
    result = _build_daily_prices(hist)

    assert len(result) == 3
    for entry in result:
        assert set(entry.keys()) == {"date", "open", "high", "low", "close", "volume"}
        assert isinstance(entry["date"], str)
        assert isinstance(entry["volume"], int)


def test_build_daily_prices_date_format():
    """Dates must be ISO format YYYY-MM-DD strings."""
    hist = _make_hist_df(n_days=1)
    result = _build_daily_prices(hist)
    # Validate format by parsing — will raise ValueError if wrong
    from datetime import datetime
    datetime.strptime(result[0]["date"], "%Y-%m-%d")


# ---------------------------------------------------------------------------
# Volume anomaly unit tests
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
def test_volume_anomaly_detected_when_ratio_exceeds_threshold(mock_ticker_class):
    """
    Period volume 3x the baseline should set is_anomalous=True.
    anomaly_ratio > VOLUME_ANOMALY_THRESHOLD (1.5) triggers the flag.
    """
    baseline_df = _make_hist_df(n_days=60, volume=1_000_000)
    mock_instance = MagicMock()
    mock_instance.history.return_value = baseline_df
    mock_ticker_class.return_value = mock_instance

    period_df = _make_hist_df(n_days=5, volume=3_000_000)
    result = _compute_volume_anomaly("NVDA", period_df, "2024-05-01")

    assert result["is_anomalous"] is True
    assert result["anomaly_ratio"] == pytest.approx(3.0, rel=0.01)


@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
def test_volume_anomaly_not_detected_for_normal_volume(mock_ticker_class):
    """Period volume equal to baseline → is_anomalous=False."""
    baseline_df = _make_hist_df(n_days=60, volume=1_000_000)
    mock_instance = MagicMock()
    mock_instance.history.return_value = baseline_df
    mock_ticker_class.return_value = mock_instance

    period_df = _make_hist_df(n_days=5, volume=1_000_000)
    result = _compute_volume_anomaly("NVDA", period_df, "2024-05-01")

    assert result["is_anomalous"] is False
    assert result["anomaly_ratio"] == pytest.approx(1.0, rel=0.01)


@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
def test_volume_anomaly_baseline_failure_returns_safe_default(mock_ticker_class):
    """
    If the 90-day baseline fetch fails, _compute_volume_anomaly must not
    propagate the exception. is_anomalous defaults to False.
    """
    mock_instance = MagicMock()
    mock_instance.history.side_effect = Exception("yfinance network error")
    mock_ticker_class.return_value = mock_instance

    period_df = _make_hist_df(n_days=5, volume=2_000_000)
    result = _compute_volume_anomaly("NVDA", period_df, "2024-05-01")

    assert result["is_anomalous"] is False
    assert result["historical_average_volume"] is None
    assert result["anomaly_ratio"] is None


@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
def test_volume_anomaly_exactly_at_threshold_is_not_anomalous(mock_ticker_class):
    """
    Ratio == VOLUME_ANOMALY_THRESHOLD (1.5) must NOT flag is_anomalous.
    The comparison is strictly greater-than, so the boundary itself is safe.
    This guards against a regression to >= that would produce false positives.
    """
    baseline_df = _make_hist_df(n_days=60, volume=1_000_000)
    mock_instance = MagicMock()
    mock_instance.history.return_value = baseline_df
    mock_ticker_class.return_value = mock_instance

    # Exactly 1.5x the baseline
    period_df = _make_hist_df(n_days=5, volume=1_500_000)
    result = _compute_volume_anomaly("NVDA", period_df, "2024-05-01")

    assert result["is_anomalous"] is False
    assert result["anomaly_ratio"] == pytest.approx(1.5, rel=0.01)


@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
def test_volume_anomaly_just_above_threshold_is_anomalous(mock_ticker_class):
    """
    Ratio just above VOLUME_ANOMALY_THRESHOLD (1.5) must set is_anomalous=True.
    Confirms the strict-greater-than boundary fires at the first value above 1.5.
    """
    baseline_df = _make_hist_df(n_days=60, volume=1_000_000)
    mock_instance = MagicMock()
    mock_instance.history.return_value = baseline_df
    mock_ticker_class.return_value = mock_instance

    # 1,510,000 / 1,000,000 = 1.51 — just above threshold, rounds cleanly
    period_df = _make_hist_df(n_days=1, volume=1_510_000)
    result = _compute_volume_anomaly("NVDA", period_df, "2024-05-01")

    assert result["is_anomalous"] is True
    assert result["anomaly_ratio"] == pytest.approx(1.51, rel=0.01)


# ---------------------------------------------------------------------------
# Full node tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_node_yfinance_success_populates_price_data(mock_ticker_class):
    """Happy path: yfinance returns data, all price_data fields are present."""
    query_df = _make_hist_df(n_days=5, open_=100.0, close=110.0)
    baseline_df = _make_hist_df(n_days=60, volume=1_000_000)

    mock_instance = MagicMock()
    # First call → query period; second call → baseline
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    assert result["price_error"] is None
    pd_ = result["price_data"]
    assert pd_ is not None

    required_keys = {
        "ticker", "start_date", "end_date",
        "open_price", "close_price", "high_price", "low_price",
        "total_volume", "percent_change", "price_change",
        "daily_prices", "source",
    }
    assert required_keys.issubset(pd_.keys())
    assert pd_["source"] == "yfinance"
    assert pd_["ticker"] == "NVDA"


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_node_daily_prices_list_populated(mock_ticker_class):
    """daily_prices must contain one entry per trading day."""
    query_df = _make_hist_df(n_days=5)
    baseline_df = _make_hist_df(n_days=60)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    daily = result["price_data"]["daily_prices"]
    assert len(daily) == 5
    assert all(k in daily[0] for k in ("date", "open", "high", "low", "close", "volume"))


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_node_percent_change_calculated_correctly(mock_ticker_class):
    """
    open=100 close=110 → percent_change should be 10.0.
    Tests the arithmetic in _fetch_yfinance.
    """
    query_df = _make_hist_df(n_days=3, open_=100.0, close=110.0)
    baseline_df = _make_hist_df(n_days=60)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    assert result["price_data"]["percent_change"] == pytest.approx(10.0, rel=0.01)
    assert result["price_data"]["price_change"] == pytest.approx(10.0, rel=0.01)


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_node_volume_anomaly_present_on_yfinance_success(mock_ticker_class):
    """volume_anomaly must be populated alongside price_data on success."""
    query_df = _make_hist_df(n_days=5, volume=3_000_000)
    baseline_df = _make_hist_df(n_days=60, volume=1_000_000)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    va = result["volume_anomaly"]
    assert va is not None
    assert va["is_anomalous"] is True
    assert "anomaly_ratio" in va


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.os.getenv", return_value="")
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_node_yfinance_empty_no_key_writes_price_error(mock_ticker_class, mock_getenv):
    """
    If yfinance returns empty data AND no Alpha Vantage key is set,
    price_error must be written and price_data must be None.
    """
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()  # empty
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    assert result["price_data"] is None
    assert result["price_error"] is not None


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.requests.get")
@patch("agent.graph.nodes.data_fetcher.os.getenv", return_value="fake_av_key")
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_node_falls_back_to_alpha_vantage_when_yfinance_empty(
    mock_ticker_class, mock_getenv, mock_requests_get
):
    """
    When yfinance returns empty data and an AV key exists,
    the node must call Alpha Vantage and return its data.
    """
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()  # yfinance fails
    mock_ticker_class.return_value = mock_instance

    # Build a minimal Alpha Vantage response
    av_data = {
        "Time Series (Daily)": {
            "2024-05-01": {
                "1. open": "100.0", "2. high": "115.0",
                "3. low": "95.0", "4. close": "110.0", "5. volume": "1000000",
            },
            "2024-05-02": {
                "1. open": "110.0", "2. high": "120.0",
                "3. low": "105.0", "4. close": "118.0", "5. volume": "1200000",
            },
        }
    }
    mock_requests_get.return_value = MagicMock(json=lambda: av_data)

    result = await fetch_price_data(_make_state())

    assert result["price_data"] is not None
    assert result["price_data"]["source"] == "alpha_vantage"
    assert result["price_error"] is None


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.requests.get")
@patch("agent.graph.nodes.data_fetcher.os.getenv", return_value="fake_av_key")
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_node_both_sources_fail_writes_price_error(
    mock_ticker_class, mock_getenv, mock_requests_get
):
    """If both yfinance and Alpha Vantage fail, price_error must be set."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_ticker_class.return_value = mock_instance

    mock_requests_get.return_value = MagicMock(
        json=lambda: {"Information": "Rate limit exceeded"}
    )

    result = await fetch_price_data(_make_state())

    assert result["price_data"] is None
    assert result["price_error"] is not None


@pytest.mark.asyncio
async def test_node_missing_ticker_writes_price_error():
    """If ticker is empty in state, the node must fail fast with price_error."""
    state = _make_state(ticker="")
    result = await fetch_price_data(state)

    assert result["price_data"] is None
    assert result["price_error"] is not None


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_node_preserves_existing_state_fields(mock_ticker_class):
    """Fields written by earlier nodes must survive through data_fetcher."""
    query_df = _make_hist_df(n_days=5)
    baseline_df = _make_hist_df(n_days=60)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_ticker_class.return_value = mock_instance

    state = _make_state(
        intent="stock_analysis",
        company_name="NVIDIA",
        date_context="last week",
        chart_requested=False,
    )

    result = await fetch_price_data(state)

    assert result["intent"] == "stock_analysis"
    assert result["company_name"] == "NVIDIA"
    assert result["date_context"] == "last week"
    assert result["chart_requested"] is False


# ---------------------------------------------------------------------------
# Node 4 enrichment tests — analyst data, short interest, earnings date
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_analyst_data_populated_when_info_available(mock_ticker_class):
    """
    When ticker.info contains analyst price target fields,
    analyst_data must be populated in the returned state.
    """
    query_df = _make_hist_df(n_days=5)
    baseline_df = _make_hist_df(n_days=60)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_instance.info = {
        "targetMeanPrice": 150.0,
        "targetHighPrice": 200.0,
        "targetLowPrice": 100.0,
        "numberOfAnalystOpinions": 35,
        "shortPercentOfFloat": 0.03,
        "shortRatio": 1.5,
        "sharesShort": 100_000_000,
        "sharesShortPriorMonth": 95_000_000,
    }
    mock_instance.recommendations_summary = pd.DataFrame()
    mock_instance.calendar = {}
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    assert result.get("analyst_data") is not None
    assert result["analyst_data"]["mean_target"] == 150.0
    assert result["analyst_data"]["high_target"] == 200.0
    assert result["analyst_data"]["num_analysts"] == 35


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_short_interest_populated_when_info_available(mock_ticker_class):
    """
    When ticker.info contains short interest fields,
    short_interest must be populated in the returned state.
    """
    query_df = _make_hist_df(n_days=5)
    baseline_df = _make_hist_df(n_days=60)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_instance.info = {
        "shortPercentOfFloat": 0.03,
        "shortRatio": 1.5,
        "sharesShort": 100_000_000,
        "sharesShortPriorMonth": 95_000_000,
    }
    mock_instance.recommendations_summary = pd.DataFrame()
    mock_instance.calendar = {}
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    assert result.get("short_interest") is not None
    assert result["short_interest"]["short_percent_of_float"] == 0.03
    assert result["short_interest"]["short_ratio"] == 1.5
    assert result["short_interest"]["shares_short"] == 100_000_000


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_earnings_date_populated_when_calendar_available(mock_ticker_class):
    """
    When ticker.calendar has an Earnings Date, next_earnings_date and
    days_until_earnings must be populated in the returned state.
    """
    from datetime import datetime, timedelta

    query_df = _make_hist_df(n_days=5)
    baseline_df = _make_hist_df(n_days=60)

    future_date = datetime.now() + timedelta(days=30)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_instance.info = {}
    mock_instance.recommendations_summary = pd.DataFrame()
    mock_instance.calendar = {"Earnings Date": [future_date]}
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    assert result.get("next_earnings_date") is not None
    assert result.get("days_until_earnings") is not None
    assert 28 <= result["days_until_earnings"] <= 32


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_enrichments_none_when_info_empty(mock_ticker_class):
    """
    When ticker.info is an empty dict, analyst_data and short_interest
    must be None. price_data must still be populated (enrichments are non-fatal).
    """
    query_df = _make_hist_df(n_days=5)
    baseline_df = _make_hist_df(n_days=60)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    mock_instance.info = {}
    mock_instance.recommendations_summary = pd.DataFrame()
    mock_instance.calendar = {}
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    assert result["price_data"] is not None
    assert result["price_error"] is None
    assert result.get("analyst_data") is None
    assert result.get("next_earnings_date") is None
    assert result.get("days_until_earnings") is None


@pytest.mark.asyncio
@patch("agent.graph.nodes.data_fetcher.yf.Ticker")
async def test_enrichment_exception_does_not_block_price_data(mock_ticker_class):
    """
    If the enrichment info call raises an exception, price_data must still
    be returned. Enrichment failures are non-fatal.
    """
    query_df = _make_hist_df(n_days=5)
    baseline_df = _make_hist_df(n_days=60)

    mock_instance = MagicMock()
    mock_instance.history.side_effect = [query_df, baseline_df]
    # Accessing .info raises to simulate a broken yfinance response
    type(mock_instance).info = property(lambda self: (_ for _ in ()).throw(Exception("info unavailable")))
    mock_ticker_class.return_value = mock_instance

    result = await fetch_price_data(_make_state())

    assert result["price_data"] is not None
    assert result["price_error"] is None


# ---------------------------------------------------------------------------
# Async interface test
# ---------------------------------------------------------------------------

import inspect as _inspect

def test_fetch_price_data_is_async():
    """fetch_price_data must be an async function for LangGraph async event loop compatibility."""
    assert _inspect.iscoroutinefunction(fetch_price_data), (
        "fetch_price_data must be 'async def' — sync I/O blocks LangGraph's event loop"
    )
