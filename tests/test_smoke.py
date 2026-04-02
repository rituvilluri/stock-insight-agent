"""
Smoke tests — real API calls, no mocks.

These tests verify the full pipeline against live data.
They are slow and require all API keys to be set.
DO NOT run in CI. Run manually with:

    PYTHONPATH=. pytest tests/test_smoke.py -v -s --no-header

Each test asserts only on final state fields, not on response prose.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip all smoke tests if GROQ_API_KEY is not set (required for LLM calls)
pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — smoke tests require live API keys"
)

from agent.graph.workflow import app as graph


@pytest.mark.asyncio
async def test_smoke_nvda_last_month():
    """Full pipeline for NVDA last month — price, news, response."""
    state = {"user_message": "How did NVDA do last month?", "user_config": {}}
    final = await graph.ainvoke(state)

    assert final.get("ticker") == "NVDA"
    assert final.get("start_date"), "start_date must be set"
    assert final.get("price_data") is not None, "price_data must not be None"
    assert final.get("response_text"), "response_text must be non-empty"
    assert final.get("synthesizer_error") is None

    print(f"\n[SMOKE] ticker={final['ticker']} start={final['start_date']} end={final['end_date']}")
    print(f"[SMOKE] price_error={final.get('price_error')}")
    print(f"[SMOKE] news articles={len(final.get('news_articles') or [])}")
    print(f"[SMOKE] filing chunks={len(final.get('filing_chunks') or [])}")


@pytest.mark.asyncio
async def test_smoke_q4_2025_date_accuracy():
    """Bug #1 regression at live API level: Q4 2025 must produce correct date range."""
    state = {
        "user_message": "Tell me how Nvidia did Q4 2025. Show me a chart as well",
        "user_config": {},
    }
    final = await graph.ainvoke(state)

    assert final.get("start_date") == "2025-10-01", f"Q4 2025 start wrong: {final.get('start_date')}"
    assert final.get("end_date") == "2025-12-31", f"Q4 2025 end wrong: {final.get('end_date')}"
    assert final.get("chart_data") is not None, "chart_data must be set for chart request"

    print(f"\n[SMOKE] start={final['start_date']} end={final['end_date']} chart={'yes' if final.get('chart_data') else 'no'}")


@pytest.mark.asyncio
async def test_smoke_analyst_brief_has_all_sections():
    """Analyst brief response must contain all required markdown sections."""
    state = {"user_message": "How did NVDA do last month?", "user_config": {}}
    final = await graph.ainvoke(state)

    response = final.get("response_text") or ""
    required_sections = [
        "## Price Action",
        "## News & Catalysts",
    ]
    for section in required_sections:
        assert section in response, f"Analyst brief missing section: {section!r}"

    print(f"\n[SMOKE] analyst brief response length={len(response)} chars")


@pytest.mark.asyncio
async def test_smoke_unknown_intent_produces_clarification():
    """Non-stock query must produce a clarification message, not a crash."""
    state = {"user_message": "What is the capital of France?", "user_config": {}}
    final = await graph.ainvoke(state)

    assert final.get("intent") == "unknown"
    assert final.get("response_text"), "Must produce a response for unknown intent"
    assert final.get("synthesizer_error") is None
