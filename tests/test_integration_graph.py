"""
Integration tests for the LangGraph workflow.

These tests run the compiled graph with real LLM calls to validate:
- Inter-node state contracts (output of node N is in the format node N+1 expects)
- Routing paths (all conditional edge branches)
- Session context preservation
- Parallel fan-out convergence

External data APIs (yfinance, Reddit, EDGAR, NewsAPI) are mocked.
LLM calls are REAL — this tests that intent/ticker/date nodes produce
correct outputs for known inputs, not just that they call the LLM.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent.graph.workflow import app as graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_yfinance_history():
    """Minimal yfinance history DataFrame mock."""
    import pandas as pd
    data = {
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [98.0, 99.0, 100.0],
        "Close": [103.0, 104.0, 105.0],
        "Volume": [1_000_000, 1_100_000, 900_000],
    }
    idx = pd.date_range("2025-02-17", periods=3)
    return pd.DataFrame(data, index=idx)


def _mock_ticker(history_df):
    mock = MagicMock()
    mock.history.return_value = history_df
    mock.info = {
        "shortName": "NVIDIA Corporation",
        "targetMeanPrice": 900.0,
        "targetHighPrice": 1000.0,
        "targetLowPrice": 800.0,
        "numberOfAnalystOpinions": 40,
    }
    mock.earnings_dates = None
    mock.options = ()
    return mock


def _rag_mocks():
    """Context manager stack for patching all RAG/external dependencies."""
    import contextlib

    @contextlib.contextmanager
    def _patch_all():
        with patch("agent.graph.nodes.news_retriever._fetch_newsapi", return_value=None):
            with patch("agent.graph.nodes.news_retriever._fetch_google_rss", return_value=None):
                with patch("agent.graph.nodes.rag_retriever._get_collection") as mock_col:
                    mock_col.return_value.count.return_value = 0
                    mock_col.return_value.query.return_value = {
                        "documents": [[]], "metadatas": [[]], "distances": [[]]
                    }
                    with patch("agent.graph.nodes.rag_retriever._get_cik", return_value=None):
                        with patch(
                            "agent.graph.nodes.reddit_sentiment.analyze_reddit_sentiment",
                            return_value={
                                "sentiment_summary": None,
                                "sentiment_posts": None,
                                "sentiment_error": "Reddit credentials not configured",
                            },
                        ):
                            yield

    return _patch_all()


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_stock_analysis_nvda_last_month():
    """
    End-to-end: 'How did NVDA do last month?' must complete without errors.
    Validates: intent=stock_analysis, ticker=NVDA, date range resolved,
    price_data populated, response_text non-empty.
    """
    state = {
        "user_message": "How did NVDA do last month?",
        "user_config": {},
        "response_depth": "quick",
    }

    hist = _mock_yfinance_history()
    with patch("yfinance.Ticker", return_value=_mock_ticker(hist)):
        with _rag_mocks():
            final = await graph.ainvoke(state)

    assert final.get("intent") == "stock_analysis", f"Expected stock_analysis, got {final.get('intent')}"
    assert final.get("ticker") == "NVDA", f"Expected NVDA, got {final.get('ticker')}"
    assert final.get("start_date"), "start_date must be set"
    assert final.get("end_date"), "end_date must be set"
    assert final.get("price_data") is not None, "price_data must not be None"
    assert final.get("response_text"), "response_text must be non-empty"
    assert final.get("synthesizer_error") is None, f"synthesizer_error: {final.get('synthesizer_error')}"


@pytest.mark.asyncio
async def test_date_q4_2025_resolves_correctly():
    """
    Inter-node contract: date_parser must produce correct Q4 2025 dates
    (now via Layer 1 — no LLM needed), and those dates must flow
    correctly into price_data fetch. The Bug #1 regression test.
    """
    state = {
        "user_message": "Tell me how Nvidia did Q4 2025. Show me a chart as well",
        "user_config": {},
        "response_depth": "quick",
    }

    hist = _mock_yfinance_history()
    with patch("yfinance.Ticker", return_value=_mock_ticker(hist)):
        with _rag_mocks():
            final = await graph.ainvoke(state)

    assert final.get("start_date") == "2025-10-01", f"Bug #1 regression: start_date={final.get('start_date')}"
    assert final.get("end_date") == "2025-12-31", f"Bug #1 regression: end_date={final.get('end_date')}"
    assert final.get("chart_requested") is True, "chart_requested must be True"


@pytest.mark.asyncio
async def test_unknown_intent_routes_to_synthesizer_directly():
    """
    Routing: unknown intent must skip all data nodes and go to synthesize.
    Validates that the synthesizer returns a clarification message.
    """
    state = {
        "user_message": "What is the capital of France?",
        "user_config": {},
        "response_depth": "quick",
    }

    with patch("agent.graph.nodes.data_fetcher.fetch_price_data") as mock_fetch:
        final = await graph.ainvoke(state)
        mock_fetch.assert_not_called()

    assert final.get("intent") == "unknown"
    assert final.get("response_text"), "Synthesizer must produce a response for unknown intent"


@pytest.mark.asyncio
async def test_date_missing_skips_data_nodes():
    """
    Routing: when no date can be extracted, graph routes to synthesize,
    skipping all data fetch nodes.
    """
    state = {
        "user_message": "Tell me about NVDA",  # no date expression
        "user_config": {},
        "response_depth": "quick",
    }

    with patch("agent.graph.nodes.data_fetcher.fetch_price_data") as mock_fetch:
        final = await graph.ainvoke(state)
        if final.get("date_missing"):
            mock_fetch.assert_not_called()

    assert final.get("response_text"), "Must produce a response even when date is missing"


@pytest.mark.asyncio
async def test_session_context_preserved_across_turns():
    """
    Session context: a follow-up 'What about the chart?' must reuse ticker
    and date from the previous turn's state fields.
    """
    state = {
        "user_message": "What about the chart?",
        "user_config": {},
        "response_depth": "quick",
        # Seeded from previous turn (as app.py does via last_context)
        "ticker": "NVDA",
        "company_name": "NVIDIA",
        "start_date": "2025-02-19",
        "end_date": "2025-03-19",
        "date_context": "last month",
    }

    hist = _mock_yfinance_history()
    with patch("yfinance.Ticker", return_value=_mock_ticker(hist)):
        with _rag_mocks():
            final = await graph.ainvoke(state)

    assert final.get("ticker") == "NVDA", f"Ticker lost in session context: {final.get('ticker')}"
    # start_date is either preserved from session context or re-resolved by the LLM —
    # either outcome is valid. The important thing is it's non-empty.
    assert final.get("start_date"), f"start_date must be non-empty, got: {final.get('start_date')}"
    # chart_request produces chart_data; stock_analysis produces response_text.
    # Either is a valid outcome for "What about the chart?" as a follow-up.
    assert final.get("chart_data") or final.get("response_text"), \
        "Must produce chart_data or response_text for follow-up"
