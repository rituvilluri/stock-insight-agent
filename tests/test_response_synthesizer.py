"""
Tests for Node 9: Response Synthesizer

Strategy: mock llm_synthesizer.invoke() so tests are fast and offline.
We build synthetic state dicts and assert on the structure of the
returned state, not on the exact LLM text (which would be brittle).

Key paths tested:
  1. Clarification path — intent="unknown"
  2. Clarification path — date_missing=True
  3. Normal synthesis — happy path (price data, LLM succeeds)
  4. Partial data — news_error set, price data present
  5. sources_cited — built from news articles, posts, filing chunks
  6. include_current_snapshot flag passed to prompt
  7. LLM failure → synthesizer_error written, response_text=None
  8. State fields from earlier nodes are preserved
  9. Prompt structure and grounding instruction present
"""

from unittest.mock import MagicMock, patch

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
        "analyst_data": None,
        "short_interest": None,
        "next_earnings_date": None,
        "days_until_earnings": None,
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
    }
    base.update(kwargs)
    return base


def _mock_llm_response(text: str = "Mocked analysis response.") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.content = text
    return mock_resp


# ---------------------------------------------------------------------------
# Clarification path tests (no LLM mock needed)
# ---------------------------------------------------------------------------

def test_clarification_for_unknown_intent():
    """intent='unknown' must return a clarification message, no LLM call needed."""
    state = _make_state(intent="unknown", price_data=None)
    result = synthesize_response(state)

    assert result["synthesizer_error"] is None
    assert result["response_text"] is not None
    assert len(result["response_text"]) > 0
    assert result["sources_cited"] == []


def test_clarification_for_date_missing():
    """date_missing=True must return a clarification message asking for a time period."""
    state = _make_state(date_missing=True, price_data=None)
    result = synthesize_response(state)

    assert result["synthesizer_error"] is None
    assert result["response_text"] is not None
    # Should ask about a date/time period
    text = result["response_text"].lower()
    assert "time" in text or "period" in text or "date" in text


def test_clarification_mentions_company_name():
    """When company_name is resolved, the clarification message should reference it."""
    state = _make_state(
        intent="unknown",
        company_name="NVIDIA",
        date_missing=True,
        price_data=None,
    )
    result = synthesize_response(state)
    assert "NVIDIA" in result["response_text"]


# ---------------------------------------------------------------------------
# Normal synthesis path tests (LLM mocked)
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_normal_synthesis_happy_path(mock_llm):
    """Happy path: price_data present, LLM returns text → response_text populated."""
    mock_llm.invoke.return_value = _mock_llm_response("NVDA gained 6.25% last week.")
    state = _make_state()
    result = synthesize_response(state)

    assert result["synthesizer_error"] is None
    assert result["response_text"] == "NVDA gained 6.25% last week."
    mock_llm.invoke.assert_called_once()


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_synthesis_prompt_includes_price_data(mock_llm):
    """The prompt sent to the LLM must include the price data figures."""
    mock_llm.invoke.return_value = _mock_llm_response()
    state = _make_state()
    synthesize_response(state)

    prompt_text = mock_llm.invoke.call_args[0][0]
    assert "800.0" in prompt_text    # open price
    assert "850.0" in prompt_text    # close price
    assert "6.25" in prompt_text     # percent change


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_synthesis_discloses_news_error(mock_llm):
    """When news_error is set, the prompt must include the unavailability disclosure."""
    mock_llm.invoke.return_value = _mock_llm_response()
    state = _make_state(news_error="NewsAPI rate limit exceeded")
    synthesize_response(state)

    prompt_text = mock_llm.invoke.call_args[0][0]
    assert "Unavailable" in prompt_text or "NewsAPI rate limit" in prompt_text


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_synthesis_includes_volume_anomaly_when_present(mock_llm):
    """When volume_anomaly is anomalous, the prompt should include volume data."""
    mock_llm.invoke.return_value = _mock_llm_response()
    state = _make_state(
        volume_anomaly={
            "is_anomalous": True,
            "anomaly_ratio": 2.5,
            "average_daily_volume": 2_500_000,
            "historical_average_volume": 1_000_000,
        }
    )
    synthesize_response(state)

    prompt_text = mock_llm.invoke.call_args[0][0]
    assert "2.5" in prompt_text or "Unusual" in prompt_text


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_synthesis_current_snapshot_flag_in_prompt(mock_llm):
    """include_current_snapshot=True must add snapshot instructions to the prompt."""
    mock_llm.invoke.return_value = _mock_llm_response()
    state = _make_state(include_current_snapshot=True)
    synthesize_response(state)

    prompt_text = mock_llm.invoke.call_args[0][0]
    assert "historical" in prompt_text.lower() or "current" in prompt_text.lower() or "snapshot" in prompt_text.lower()


# ---------------------------------------------------------------------------
# sources_cited tests
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_sources_cited_includes_news_articles(mock_llm):
    """sources_cited must contain one entry per news article."""
    mock_llm.invoke.return_value = _mock_llm_response()
    state = _make_state(
        news_articles=[
            {"title": "NVIDIA Soars", "source_name": "Reuters", "published_date": "2024-05-01",
             "url": "https://reuters.com/1", "snippet": "..."},
            {"title": "AI Chip Demand", "source_name": "Bloomberg", "published_date": "2024-05-02",
             "url": "https://bloomberg.com/2", "snippet": "..."},
        ],
        news_source_used="finnhub",
    )
    result = synthesize_response(state)

    news_sources = [s for s in result["sources_cited"] if s["type"] == "news"]
    assert len(news_sources) == 2
    assert news_sources[0]["title"] == "NVIDIA Soars"
    assert news_sources[0]["url"] == "https://reuters.com/1"


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_sources_cited_deduplicates_filing_chunks(mock_llm):
    """Multiple chunks from the same filing should appear only once in sources_cited."""
    mock_llm.invoke.return_value = _mock_llm_response()
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
    result = synthesize_response(state)

    filing_sources = [s for s in result["sources_cited"] if s["type"] == "filing"]
    assert len(filing_sources) == 1


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_sources_cited_empty_when_no_data(mock_llm):
    """sources_cited must be an empty list when no data was retrieved."""
    mock_llm.invoke.return_value = _mock_llm_response()
    state = _make_state()  # all data fields are None
    result = synthesize_response(state)

    assert result["sources_cited"] == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_llm_failure_writes_synthesizer_error(mock_llm):
    """If the LLM raises, synthesizer_error must be written and response_text=None."""
    mock_llm.invoke.side_effect = Exception("Groq 503 Service Unavailable")
    state = _make_state()
    result = synthesize_response(state)

    assert result["response_text"] is None
    assert result["synthesizer_error"] is not None
    assert "503" in result["synthesizer_error"] or "Groq" in result["synthesizer_error"]


# ---------------------------------------------------------------------------
# State preservation
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_node_preserves_existing_state_fields(mock_llm):
    """Fields written by earlier nodes must survive through response_synthesizer."""
    mock_llm.invoke.return_value = _mock_llm_response()
    state = _make_state(
        start_date="2024-05-01",
        end_date="2024-05-07",
        chart_requested=True,
        chart_data=None,
    )
    result = synthesize_response(state)

    assert result["start_date"] == "2024-05-01"
    assert result["end_date"] == "2024-05-07"
    assert result["chart_requested"] is True
    assert result["ticker"] == "NVDA"
    assert result["company_name"] == "NVIDIA"


# ---------------------------------------------------------------------------
# Prompt structure tests
# ---------------------------------------------------------------------------

def test_prompt_contains_section_headers():
    """Synthesis prompt must list the standard markdown section headers."""
    state = _make_state()
    prompt = _build_synthesis_prompt(state)
    for section in ["Price Action", "News & Catalysts", "Market Sentiment", "SEC Filings", "Options Activity"]:
        assert section in prompt, f"Synthesis prompt missing section: {section}"


def test_grounding_instruction_in_prompt():
    """Prompt must contain the grounding instruction."""
    state = _make_state()
    prompt = _build_synthesis_prompt(state)
    assert "do not fill gaps from training knowledge" in prompt.lower()


def test_prompt_includes_analyst_data():
    """When analyst_data is present, prompt must include target price and consensus."""
    state = _make_state(
        analyst_data={
            "mean_target": 300.0,
            "high_target": 400.0,
            "low_target": 200.0,
            "num_analysts": 50,
            "strong_buy": 10,
            "buy": 35,
            "hold": 5,
            "sell": 0,
            "strong_sell": 0,
        }
    )
    prompt = _build_synthesis_prompt(state)
    assert "300.0" in prompt
    assert "Analyst" in prompt


def test_prompt_includes_short_interest():
    """When short_interest is present, prompt must include short data."""
    state = _make_state(
        short_interest={
            "short_percent_of_float": 0.02,
            "short_ratio": 1.5,
            "shares_short": 50_000_000,
            "shares_short_prior_month": 48_000_000,
        }
    )
    prompt = _build_synthesis_prompt(state)
    assert "Short interest" in prompt


def test_prompt_includes_earnings_date():
    """When next_earnings_date is set, prompt must include it."""
    state = _make_state(next_earnings_date="2026-05-20", days_until_earnings=57)
    prompt = _build_synthesis_prompt(state)
    assert "2026-05-20" in prompt
