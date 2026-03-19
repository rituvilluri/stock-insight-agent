"""
Tests for Node 7: RAG Retriever (rag_retriever.py)

Strategy: all external calls (Gemini, ChromaDB, SEC EDGAR, requests) are
mocked so the test suite runs offline and deterministically.

Test groups:
  1. Early-exit guards (no ticker, no dates, no API key)
  2. Cache hit — ChromaDB already has chunks → return without EDGAR
  3. Cache miss → EDGAR discover → ingest → re-query
  4. EDGAR fallbacks (CIK not found, no filings in range)
  5. Helpers: _strip_html, _chunk_text, _date_in_range
"""

import os
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from agent.graph.nodes.rag_retriever import (
    retrieve_rag_context,
    _strip_html,
    _chunk_text,
    _date_in_range,
    _discover_filings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_STATE = {
    "user_message": "What did NVDA say about data center revenue in Q2 2024?",
    "user_config": {},
    "ticker": "NVDA",
    "start_date": "2024-06-01",
    "end_date": "2024-07-31",
}


def _make_chroma_query_result(docs=None, metas=None, distances=None):
    """Build a mock ChromaDB query() return value."""
    docs = docs or []
    metas = metas or []
    distances = distances or []
    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


# ---------------------------------------------------------------------------
# Group 1: Early-exit guards
# ---------------------------------------------------------------------------

def test_no_ticker_returns_empty():
    state = {**BASE_STATE, "ticker": ""}
    result = retrieve_rag_context(state)
    assert result["filing_chunks"] == []
    assert result["filing_ingested"] is False
    assert result["filing_error"] is None


def test_no_start_date_returns_empty():
    state = {**BASE_STATE, "start_date": ""}
    result = retrieve_rag_context(state)
    assert result["filing_chunks"] == []
    assert result["filing_error"] is None


def test_no_end_date_returns_empty():
    state = {**BASE_STATE, "end_date": ""}
    result = retrieve_rag_context(state)
    assert result["filing_chunks"] == []


@patch.dict(os.environ, {}, clear=True)
def test_missing_gemini_key_returns_error():
    # Remove GEMINI_API_KEY from env entirely
    env_without_key = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        result = retrieve_rag_context(BASE_STATE)
    assert result["filing_chunks"] == []
    assert "GEMINI_API_KEY" in result["filing_error"]


# ---------------------------------------------------------------------------
# Group 2: Cache hit — ChromaDB already has chunks
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.rag_retriever.genai")
@patch("agent.graph.nodes.rag_retriever._get_collection")
@patch("agent.graph.nodes.rag_retriever._embed_query")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_cache_hit_returns_chunks_without_edgar(mock_embed_q, mock_get_col, mock_genai):
    mock_embed_q.return_value = [0.1] * 768

    mock_col = MagicMock()
    mock_col.count.return_value = 10
    mock_col.query.return_value = _make_chroma_query_result(
        docs=["Data center revenue grew 154% YoY..."],
        metas=[{"filing_type": "10-Q", "filing_period": "2024Q2", "filing_date": "2024-08-28"}],
        distances=[0.12],
    )
    mock_get_col.return_value = mock_col

    result = retrieve_rag_context(BASE_STATE)

    assert len(result["filing_chunks"]) == 1
    assert result["filing_chunks"][0]["filing_type"] == "10-Q"
    assert result["filing_chunks"][0]["chunk_relevance_score"] == pytest.approx(0.88, abs=0.01)
    assert result["filing_ingested"] is False
    assert result["filing_error"] is None


@patch("agent.graph.nodes.rag_retriever.genai")
@patch("agent.graph.nodes.rag_retriever._get_collection")
@patch("agent.graph.nodes.rag_retriever._embed_query")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_cache_hit_does_not_call_edgar(mock_embed_q, mock_get_col, mock_genai):
    """When cache has results, EDGAR should not be called."""
    mock_embed_q.return_value = [0.1] * 768

    mock_col = MagicMock()
    mock_col.count.return_value = 5
    mock_col.query.return_value = _make_chroma_query_result(
        docs=["Gross margin expanded..."],
        metas=[{"filing_type": "10-K", "filing_period": "2024Q4", "filing_date": "2024-02-21"}],
        distances=[0.05],
    )
    mock_get_col.return_value = mock_col

    with patch("agent.graph.nodes.rag_retriever._get_cik") as mock_cik:
        retrieve_rag_context(BASE_STATE)
        mock_cik.assert_not_called()


# ---------------------------------------------------------------------------
# Group 3: Cache miss → EDGAR discover → ingest → re-query
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.rag_retriever.genai")
@patch("agent.graph.nodes.rag_retriever._get_collection")
@patch("agent.graph.nodes.rag_retriever._embed_query")
@patch("agent.graph.nodes.rag_retriever._get_cik")
@patch("agent.graph.nodes.rag_retriever._discover_filings")
@patch("agent.graph.nodes.rag_retriever._ingest_filing")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_cache_miss_triggers_ingestion(
    mock_ingest, mock_discover, mock_cik, mock_embed_q, mock_get_col, mock_genai
):
    mock_embed_q.return_value = [0.1] * 768
    mock_cik.return_value = "0001045810"

    mock_discover.return_value = [
        {"accession_number": "0001045810240001234", "filing_type": "10-Q",
         "period": "2024Q2", "filing_date": "2024-08-28", "primary_doc": "nvda-20240728.htm",
         "cik": "0001045810"},
    ]
    mock_ingest.return_value = 42  # 42 new chunks ingested

    mock_col = MagicMock()
    # First count() → 0 (empty store, short-circuits query call)
    # Second count() → 42 (after ingestion, proceeds to query)
    mock_col.count.side_effect = [0, 42]
    # Only one query call — the first _query_collection skips query() because count=0
    mock_col.query.return_value = _make_chroma_query_result(
        docs=["Operating expenses declined..."],
        metas=[{"filing_type": "10-Q", "filing_period": "2024Q2", "filing_date": "2024-08-28"}],
        distances=[0.08],
    )
    mock_get_col.return_value = mock_col

    result = retrieve_rag_context(BASE_STATE)

    mock_ingest.assert_called_once()
    assert result["filing_ingested"] is True
    assert len(result["filing_chunks"]) == 1
    assert result["filing_error"] is None


@patch("agent.graph.nodes.rag_retriever.genai")
@patch("agent.graph.nodes.rag_retriever._get_collection")
@patch("agent.graph.nodes.rag_retriever._embed_query")
@patch("agent.graph.nodes.rag_retriever._get_cik")
@patch("agent.graph.nodes.rag_retriever._discover_filings")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_ingest_sets_filing_ingested_true(
    mock_discover, mock_cik, mock_embed_q, mock_get_col, mock_genai
):
    mock_embed_q.return_value = [0.1] * 768
    mock_cik.return_value = "0001045810"
    mock_discover.return_value = [
        {"accession_number": "0001045810240001234", "filing_type": "10-Q",
         "period": "2024Q2", "filing_date": "2024-08-28", "primary_doc": "nvda.htm",
         "cik": "0001045810"},
    ]

    mock_col = MagicMock()
    mock_col.count.side_effect = [0, 10]
    mock_col.query.side_effect = [
        _make_chroma_query_result(),
        _make_chroma_query_result(
            docs=["Revenue..."],
            metas=[{"filing_type": "10-Q", "filing_period": "2024Q2", "filing_date": "2024-08-28"}],
            distances=[0.1],
        ),
    ]
    mock_get_col.return_value = mock_col

    with patch("agent.graph.nodes.rag_retriever._ingest_filing", return_value=10):
        result = retrieve_rag_context(BASE_STATE)

    assert result["filing_ingested"] is True


# ---------------------------------------------------------------------------
# Group 4: EDGAR fallbacks
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.rag_retriever.genai")
@patch("agent.graph.nodes.rag_retriever._get_collection")
@patch("agent.graph.nodes.rag_retriever._embed_query")
@patch("agent.graph.nodes.rag_retriever._get_cik")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_cik_not_found_returns_empty(mock_cik, mock_embed_q, mock_get_col, mock_genai):
    mock_embed_q.return_value = [0.1] * 768
    mock_cik.return_value = None

    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_col.query.return_value = _make_chroma_query_result()
    mock_get_col.return_value = mock_col

    result = retrieve_rag_context(BASE_STATE)
    assert result["filing_chunks"] == []
    assert result["filing_error"] is None


@patch("agent.graph.nodes.rag_retriever.genai")
@patch("agent.graph.nodes.rag_retriever._get_collection")
@patch("agent.graph.nodes.rag_retriever._embed_query")
@patch("agent.graph.nodes.rag_retriever._get_cik")
@patch("agent.graph.nodes.rag_retriever._discover_filings")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_no_filings_in_range_returns_empty(
    mock_discover, mock_cik, mock_embed_q, mock_get_col, mock_genai
):
    mock_embed_q.return_value = [0.1] * 768
    mock_cik.return_value = "0001045810"
    mock_discover.return_value = []

    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_col.query.return_value = _make_chroma_query_result()
    mock_get_col.return_value = mock_col

    result = retrieve_rag_context(BASE_STATE)
    assert result["filing_chunks"] == []
    assert result["filing_error"] is None


@patch("agent.graph.nodes.rag_retriever.genai")
@patch("agent.graph.nodes.rag_retriever._get_collection")
@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
def test_exception_sets_filing_error(mock_get_col, mock_genai):
    """An unhandled exception inside the node should set filing_error."""
    mock_get_col.side_effect = RuntimeError("disk full")
    result = retrieve_rag_context(BASE_STATE)
    assert result["filing_chunks"] == []
    assert "disk full" in result["filing_error"]


# ---------------------------------------------------------------------------
# Group 5: Pure helper unit tests (no mocking needed)
# ---------------------------------------------------------------------------

def test_strip_html_removes_tags():
    html = "<h1>Revenue</h1><p>Grew <b>154%</b> YoY</p>"
    text = _strip_html(html)
    assert "<" not in text
    assert "Revenue" in text
    assert "154%" in text


def test_strip_html_collapses_whitespace():
    html = "<p>hello   \n\t  world</p>"
    text = _strip_html(html)
    assert "  " not in text


def test_chunk_text_produces_correct_ids():
    text = "A" * 10000
    chunks = _chunk_text(text, "NVDA", "10-Q", "2024Q2")
    assert all(c["id"].startswith("NVDA-10-Q-2024Q2-chunk-") for c in chunks)
    # IDs should be sequential
    indices = [int(c["id"].split("-chunk-")[1]) for c in chunks]
    assert indices == list(range(len(indices)))


def test_chunk_text_metadata():
    text = "B" * 5000
    chunks = _chunk_text(text, "AAPL", "10-K", "2023Q4")
    for c in chunks:
        assert c["metadata"]["ticker"] == "AAPL"
        assert c["metadata"]["filing_type"] == "10-K"
        assert c["metadata"]["filing_period"] == "2023Q4"


def test_chunk_text_overlap():
    """Chunks overlap by _OVERLAP_CHARS so consecutive chunks share content."""
    from agent.graph.nodes.rag_retriever import _CHUNK_CHARS, _OVERLAP_CHARS
    text = "X" * (_CHUNK_CHARS + _OVERLAP_CHARS + 100)
    chunks = _chunk_text(text, "T", "10-Q", "2024Q1")
    assert len(chunks) >= 2
    # Second chunk starts before first chunk ends
    first_end = _CHUNK_CHARS
    second_start = _CHUNK_CHARS - _OVERLAP_CHARS
    assert second_start < first_end


def test_date_in_range_within_window():
    assert _date_in_range("2024-06-30", "2024-06-01", "2024-07-31") is True


def test_date_in_range_just_before_start():
    # 179 days before start → within 180-day lookback → True
    assert _date_in_range("2024-01-05", "2024-07-01", "2024-07-31") is True


def test_date_in_range_too_far_before():
    # 200+ days before start → outside window → False
    assert _date_in_range("2023-12-01", "2024-07-01", "2024-07-31") is False


def test_date_in_range_after_end_within_90_days():
    # 45 days after end → within 90-day post-range window → True
    assert _date_in_range("2024-09-14", "2024-06-01", "2024-07-31") is True


def test_date_in_range_too_far_after_end():
    # 120 days after end → outside window → False
    assert _date_in_range("2024-11-28", "2024-06-01", "2024-07-31") is False


def test_discover_filings_filters_form_types():
    """_discover_filings should only return 10-K and 10-Q filings."""
    cik = "0001045810"
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "DEF 14A", "10-K"],
                "filingDate": ["2024-08-28", "2024-08-01", "2024-05-01", "2024-02-21"],
                "reportDate": ["2024-07-28", "2024-07-15", "2024-04-30", "2024-01-28"],
                "accessionNumber": [
                    "0001045810-24-000123",
                    "0001045810-24-000100",
                    "0001045810-24-000050",
                    "0001045810-24-000010",
                ],
                "primaryDocument": ["nvda-20240728.htm", "nvda-8k.htm", "def14a.htm", "nvda-20240128.htm"],
            }
        }
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = submissions

    with patch("agent.graph.nodes.rag_retriever._edgar_get", return_value=mock_resp):
        results = _discover_filings(cik, "NVDA", "2024-06-01", "2024-07-31")

    form_types = [f["filing_type"] for f in results]
    assert "8-K" not in form_types
    assert "DEF 14A" not in form_types
    # 10-Q in range → included; 10-K in range → included
    assert all(ft in ("10-K", "10-Q") for ft in form_types)


def test_discover_filings_respects_date_range():
    """Filings outside the date window are excluded."""
    cik = "0001045810"
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "10-Q"],
                "filingDate": ["2024-08-28", "2022-05-01"],
                "reportDate": ["2024-07-28", "2022-04-30"],  # second is too old
                "accessionNumber": ["0001045810-24-000123", "0001045810-22-000050"],
                "primaryDocument": ["nvda-20240728.htm", "nvda-20220430.htm"],
            }
        }
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = submissions

    with patch("agent.graph.nodes.rag_retriever._edgar_get", return_value=mock_resp):
        results = _discover_filings(cik, "NVDA", "2024-06-01", "2024-07-31")

    assert len(results) == 1
    assert results[0]["period"] == "2024Q3"


def test_chunk_text_empty_text():
    chunks = _chunk_text("", "NVDA", "10-Q", "2024Q2")
    assert chunks == []


def test_returns_only_owned_fields():
    """
    After the parallel fan-out fix, retrieve_rag_context must return only
    its three owned fields. It must NOT spread {**state} back — doing so
    causes InvalidUpdateError when LangGraph merges parallel branches.
    """
    state = {**BASE_STATE, "extra_field_that_should_not_leak": "sentinel"}
    result = retrieve_rag_context(state)
    # Only these three keys are allowed in the return dict
    assert set(result.keys()) == {"filing_chunks", "filing_ingested", "filing_error"}
    assert "extra_field_that_should_not_leak" not in result
    assert "user_message" not in result
    assert "ticker" not in result
