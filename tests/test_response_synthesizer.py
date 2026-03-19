"""
Tests for Node 9: Response Synthesizer

Strategy: mock llm_synthesizer.ainvoke() so tests are fast and offline.
We build synthetic state dicts and assert on the structure of the
returned state, not on the exact LLM text (which would be brittle).

Key paths tested:
  1.  Clarification path — intent="unknown"
  2.  Clarification path — date_missing=True
  3.  Normal synthesis — happy path (price data, LLM succeeds)
  4.  Partial data — news_error set, price data present
  5.  sources_cited — built from news articles, posts, filing chunks
  6.  include_current_snapshot flag passed to prompt
  7.  LLM failure -> synthesizer_error written, response_text=None
  8.  State fields from earlier nodes are preserved
  9.  general_lookup path (price data only, no news/sentiment)
  10. synthesize_response is async (ainvoke)
  11. Deep mode routes to llm_synthesizer_deep
"""

from unittest.mock import MagicMock, patch, AsyncMock
import inspect

import pytest

from agent.graph.nodes.response_synthesizer import (
    synthesize_response,
    _build_clarification_prompt,
    _build_sources_cited,
    _build_synthesis_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> dict:
    base = {
        "user_message": "How did NVDA do last week?",
        "user_config": {},
        "ticker": "NVDA",
        "company_name": "NVIDIA",
        "intent": "stock_analysis",
        "chart_requested": False,
        "date_missing": False,
        "date_context": "last week",
        "include_current_snapshot": False,
        "price_data": {
            "ticker": "NVDA",
            "start_date": "2024-05-01",
            "end_date": "2024-05-07",
            "open_price": 800.0,
            "close_price": 850.0,
            "high_price": 860.0,
            "low_price": 790.0,
            "price_change": 50.0,
            "percent_change": 6.25,
            "total_volume": 5_000_000,
            "daily_prices": [],
            "source": "yfinance",
        },
        "price_error": None,
        "volume_anomaly": None,
        "news_articles": None,
        "news_source_used": None,
        "news_error": None,
        "sentiment_summary": None,
        "sentiment_posts": None,
        "sentiment_error": None,
        "filing_chunks": None,
        "filing_ingested": None,
        "filing_error": None,
        "options_data": None,
        "options_error": None,
        "response_depth": "quick",
    }
    base.update(kwargs)
    return base


def _mock_ainvoke(text: str = "Mocked analysis response.") -> AsyncMock:
    """Return an AsyncMock whose ainvoke returns a response with .content."""
    mock_llm = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.content = text
    mock_llm.ainvoke.return_value = mock_resp
    return mock_llm


# ---------------------------------------------------------------------------
# Clarification path tests (no LLM mock needed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clarification_for_unknown_intent():
    """intent='unknown' must return a clarification message, no LLM call needed."""
    state = _make_state(intent="unknown", price_data=None)
    result = await synthesize_response(state)

    assert result["synthesizer_error"] is None
    assert result["response_text"] is not None
    assert len(result["response_text"]) > 0
    assert result["sources_cited"] == []


@pytest.mark.asyncio
async def test_clarification_for_date_missing():
    """date_missing=True must return a clarification message asking for a time period."""
    state = _make_state(date_missing=True, price_data=None)
    result = await synthesize_response(state)

    assert result["synthesizer_error"] is None
    assert result["response_text"] is not None
    text = result["response_text"].lower()
    assert "time" in text or "period" in text or "date" in text


@pytest.mark.asyncio
async def test_clarification_mentions_company_name():
    """When company_name is resolved, the clarification message should reference it."""
    state = _make_state(
        intent="unknown",
        company_name="NVIDIA",
        date_missing=True,
        price_data=None,
    )
    result = await synthesize_response(state)
    assert "NVIDIA" in result["response_text"]


# ---------------------------------------------------------------------------
# Normal synthesis path tests (LLM mocked via AsyncMock)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normal_synthesis_happy_path():
    """Happy path: price_data present, LLM returns text -> response_text populated."""
    mock_llm = _mock_ainvoke("NVDA gained 6.25% last week.")
    state = _make_state()
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)

    assert result["synthesizer_error"] is None
    assert result["response_text"] == "NVDA gained 6.25% last week."
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_synthesis_prompt_includes_price_data():
    """The prompt sent to the LLM must include the price data figures."""
    mock_llm = _mock_ainvoke()
    state = _make_state()
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        await synthesize_response(state)

    prompt_text = mock_llm.ainvoke.call_args[0][0]
    assert "800.0" in prompt_text    # open price
    assert "850.0" in prompt_text    # close price
    assert "6.25" in prompt_text     # percent change


@pytest.mark.asyncio
async def test_synthesis_discloses_news_error():
    """When news_error is set, the prompt must include the unavailability disclosure."""
    mock_llm = _mock_ainvoke()
    state = _make_state(news_error="NewsAPI rate limit exceeded")
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        await synthesize_response(state)

    prompt_text = mock_llm.ainvoke.call_args[0][0]
    assert "Unavailable" in prompt_text or "NewsAPI rate limit" in prompt_text


@pytest.mark.asyncio
async def test_synthesis_includes_volume_anomaly_when_present():
    """When volume_anomaly is anomalous, the prompt should include volume data."""
    mock_llm = _mock_ainvoke()
    state = _make_state(
        volume_anomaly={
            "is_anomalous": True,
            "anomaly_ratio": 2.5,
            "average_daily_volume": 2_500_000,
            "historical_average_volume": 1_000_000,
        }
    )
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        await synthesize_response(state)

    prompt_text = mock_llm.ainvoke.call_args[0][0]
    assert "2.5" in prompt_text or "Unusual" in prompt_text


@pytest.mark.asyncio
async def test_synthesis_current_snapshot_flag_in_prompt():
    """include_current_snapshot=True must add snapshot instructions to the prompt."""
    mock_llm = _mock_ainvoke()
    state = _make_state(include_current_snapshot=True)
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        await synthesize_response(state)

    prompt_text = mock_llm.ainvoke.call_args[0][0]
    assert "Historical" in prompt_text or "Current" in prompt_text or "snapshot" in prompt_text.lower()


# ---------------------------------------------------------------------------
# sources_cited tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sources_cited_includes_news_articles():
    """sources_cited must contain one entry per news article."""
    mock_llm = _mock_ainvoke()
    state = _make_state(
        news_articles=[
            {"title": "NVIDIA Soars", "source_name": "Reuters", "published_date": "2024-05-01",
             "url": "https://reuters.com/1", "snippet": "..."},
            {"title": "AI Chip Demand", "source_name": "Bloomberg", "published_date": "2024-05-02",
             "url": "https://bloomberg.com/2", "snippet": "..."},
        ],
        news_source_used="newsapi",
    )
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)

    news_sources = [s for s in result["sources_cited"] if s["type"] == "news"]
    assert len(news_sources) == 2
    assert news_sources[0]["title"] == "NVIDIA Soars"
    assert news_sources[0]["url"] == "https://reuters.com/1"


@pytest.mark.asyncio
async def test_sources_cited_deduplicates_filing_chunks():
    """Multiple chunks from the same filing should appear only once in sources_cited."""
    mock_llm = _mock_ainvoke()
    state = _make_state(
        filing_chunks=[
            {"text": "Revenue increased...", "filing_type": "10-Q",
             "filing_quarter": "Q2 2024", "filing_date": "2024-08-01",
             "chunk_relevance_score": 0.9},
            {"text": "Operating expenses...", "filing_type": "10-Q",
             "filing_quarter": "Q2 2024", "filing_date": "2024-08-01",
             "chunk_relevance_score": 0.8},
        ]
    )
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)

    filing_sources = [s for s in result["sources_cited"] if s["type"] == "filing"]
    assert len(filing_sources) == 1


@pytest.mark.asyncio
async def test_sources_cited_empty_when_no_data():
    """sources_cited must be an empty list when no data was retrieved."""
    mock_llm = _mock_ainvoke()
    state = _make_state(price_data=None)
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)

    assert result["sources_cited"] == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_writes_synthesizer_error():
    """If the LLM raises, synthesizer_error must be written and response_text=None."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = Exception("Groq 503 Service Unavailable")
    state = _make_state()
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)

    assert result["response_text"] is None
    assert result["synthesizer_error"] is not None
    assert "503" in result["synthesizer_error"] or "Groq" in result["synthesizer_error"]


# ---------------------------------------------------------------------------
# State preservation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_preserves_existing_state_fields():
    """Fields written by earlier nodes must survive through response_synthesizer."""
    mock_llm = _mock_ainvoke()
    state = _make_state(
        start_date="2024-05-01",
        end_date="2024-05-07",
        chart_requested=True,
        chart_data=None,
    )
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)

    assert result["start_date"] == "2024-05-01"
    assert result["end_date"] == "2024-05-07"
    assert result["chart_requested"] is True
    assert result["ticker"] == "NVDA"
    assert result["company_name"] == "NVIDIA"


# ---------------------------------------------------------------------------
# Depth routing tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quick_depth_uses_llm_synthesizer():
    """response_depth='quick' must call llm_synthesizer, not the deep variant."""
    mock_llm = _mock_ainvoke("Quick analysis result")
    state = _make_state(response_depth="quick")
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)
    mock_llm.ainvoke.assert_called_once()
    assert result["synthesizer_error"] is None


@pytest.mark.asyncio
async def test_deep_depth_uses_llm_synthesizer_deep():
    """response_depth='deep' must call llm_synthesizer_deep."""
    mock_deep_llm = _mock_ainvoke("Deep analysis result")
    state = _make_state(response_depth="deep")
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer_deep", mock_deep_llm):
        with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer") as mock_quick:
            result = await synthesize_response(state)
    mock_deep_llm.ainvoke.assert_called_once()
    mock_quick.ainvoke.assert_not_called()
    assert result["synthesizer_error"] is None


@pytest.mark.asyncio
async def test_unknown_depth_defaults_to_quick():
    """Any value other than 'deep' must fall back to quick path."""
    mock_llm = _mock_ainvoke("Quick result")
    state = _make_state(response_depth="invalid_value")
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)
    mock_llm.ainvoke.assert_called_once()
    assert result["synthesizer_error"] is None


@pytest.mark.asyncio
async def test_missing_depth_defaults_to_quick():
    """If response_depth is absent from state, default to quick path."""
    mock_llm = _mock_ainvoke("Quick result")
    state = _make_state()
    state.pop("response_depth", None)
    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)
    mock_llm.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# Prompt content tests (synchronous — test the builder directly)
# ---------------------------------------------------------------------------

def test_deep_prompt_contains_section_headers():
    """Deep Dive prompt must contain the required markdown section headers."""
    state = _make_state(response_depth="deep")
    prompt = _build_synthesis_prompt(state)
    for section in ["Price Action", "News & Catalysts", "Market Sentiment", "SEC Filings", "Options Activity"]:
        assert section in prompt, f"Deep Dive prompt missing section: {section}"


def test_grounding_instruction_in_prompt():
    """Both quick and deep prompts must contain the grounding instruction."""
    grounding = "do not fill gaps from your training knowledge"
    for depth in ("quick", "deep"):
        state = _make_state(response_depth=depth)
        prompt = _build_synthesis_prompt(state)
        assert grounding in prompt, f"Grounding instruction missing from {depth} prompt"


def test_strict_grounding_rules_in_prompt():
    """Both quick and deep prompts must contain the CRITICAL GROUNDING RULES block."""
    for depth in ("quick", "deep"):
        state = _make_state(response_depth=depth)
        prompt = _build_synthesis_prompt(state)
        assert "CRITICAL GROUNDING RULES" in prompt, (
            f"CRITICAL GROUNDING RULES missing from {depth} prompt"
        )


# ---------------------------------------------------------------------------
# Async conversion tests
# ---------------------------------------------------------------------------

def test_synthesize_response_is_async():
    """synthesize_response must be an async function (uses ainvoke)."""
    assert inspect.iscoroutinefunction(synthesize_response), (
        "synthesize_response must be async def"
    )


@pytest.mark.asyncio
async def test_synthesize_calls_ainvoke():
    """synthesize_response must use ainvoke, not invoke."""
    state = _make_state(
        price_data={
            "ticker": "NVDA",
            "start_date": "2025-02-19",
            "end_date": "2025-03-19",
            "open_price": 800.0,
            "close_price": 900.0,
            "high_price": 920.0,
            "low_price": 790.0,
            "price_change": 100.0,
            "percent_change": 5.2,
            "total_volume": 5_000_000,
            "daily_prices": [],
            "source": "yfinance",
        },
        response_depth="quick",
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="NVDA rose 5.2% last month.")

    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer", mock_llm):
        result = await synthesize_response(state)

    mock_llm.ainvoke.assert_called_once()
    mock_llm.invoke.assert_not_called()
    assert result["synthesizer_error"] is None


@pytest.mark.asyncio
async def test_synthesize_deep_mode_uses_deep_llm():
    """Deep mode must use llm_synthesizer_deep, not llm_synthesizer."""
    state = _make_state(response_depth="deep")
    mock_deep_llm = AsyncMock()
    mock_deep_llm.ainvoke.return_value = MagicMock(content="## Price Action\nNVDA rose...")

    with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer_deep", mock_deep_llm):
        with patch("agent.graph.nodes.response_synthesizer.llm_synthesizer") as mock_quick_llm:
            result = await synthesize_response(state)

    mock_deep_llm.ainvoke.assert_called_once()
    mock_quick_llm.ainvoke.assert_not_called()
    assert result["synthesizer_error"] is None
