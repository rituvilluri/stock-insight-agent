"""
Node 7: RAG Retriever (SEC Filings via ChromaDB + Google Gemini Embeddings)

Reads:  ticker, start_date, end_date, user_message
Writes: filing_chunks, filing_ingested, filing_error

Retrieval workflow:
  1. Embed user_message with Google Gemini text-embedding-004.
  2. Query ChromaDB with ticker metadata filter + semantic search (top 5).
  3. If results → return them (filing_ingested=False).
  4. If empty → call SEC EDGAR to find 10-K/10-Q filings that cover the date
     range.  If filings exist, download and ingest them, then re-query.
  5. If no EDGAR filings match → return empty list (not an error).

Ingestion workflow (on-demand, first-time only):
  Download filing HTML → strip tags → chunk (≈600 tokens, 100-token overlap)
  → batch-embed with Gemini → store in ChromaDB with metadata for deduplication.

Chunk IDs: {ticker}-{filing_type}-{period}-chunk-{N:03d}
  e.g. NVDA-10Q-2024Q2-chunk-014 — ChromaDB treats duplicate IDs as no-ops.

External dependencies:
  - GEMINI_API_KEY env var (required; node returns error if absent)
  - CHROMA_PERSIST_DIR env var (default: data/vector_store)
  - SEC EDGAR public API (no auth; User-Agent header required)
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

import chromadb
from google import genai
from google.genai import types as genai_types
import requests

from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/vector_store")
COLLECTION_NAME = "sec_filings_gemini_text_embedding_004"
EMBEDDING_MODEL = "models/text-embedding-004"
EDGAR_BASE = "https://data.sec.gov"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar/data"
EDGAR_USER_AGENT = "StockInsightAgent admin@stockinsight.dev"

# Approximate chars per token: 1 token ≈ 4 chars
_CHUNK_CHARS = 2400   # ≈ 600 tokens
_OVERLAP_CHARS = 400  # ≈ 100 tokens
_TOP_K = 5
_MAX_FILINGS_TO_INGEST = 3   # cap ingestion per query to stay within rate limits


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """
    Extract plain text from SEC filing HTML using BeautifulSoup4.

    Why BeautifulSoup over stdlib HTMLParser?
    SEC 10-K/10-Q filings contain <style>, <script>, and inline XBRL tags
    that HTMLParser includes verbatim. BeautifulSoup lets us decompose
    unwanted tags before extracting text, producing clean prose.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["style", "script", "meta", "link", "head"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logger.warning("_strip_html failed: %s — returning empty string", e)
        return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, ticker: str, filing_type: str, period: str) -> list[dict]:
    """
    Split text into overlapping chunks and assign stable unique IDs.

    Returns list of dicts:
      id, text, metadata (ticker, filing_type, filing_period)
    """
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + _CHUNK_CHARS
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{ticker}-{filing_type}-{period}-chunk-{idx:03d}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "filing_period": period,
                },
            })
            idx += 1
        start += _CHUNK_CHARS - _OVERLAP_CHARS
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _get_genai_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed texts using Gemini text-embedding-004.
    Processes in batches of 100 (API limit).
    """
    client = _get_genai_client()
    all_vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        all_vectors.extend([e.values for e in response.embeddings])
    return all_vectors


def _embed_query(text: str) -> list[float]:
    client = _get_genai_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def _edgar_get(url: str) -> Optional[requests.Response]:
    """GET a SEC EDGAR URL with required User-Agent header and gentle rate limit."""
    try:
        resp = requests.get(url, headers={"User-Agent": EDGAR_USER_AGENT}, timeout=30)
        time.sleep(0.15)  # stay well under 10 req/sec courtesy limit
        return resp if resp.ok else None
    except requests.RequestException as e:
        logger.warning("EDGAR request failed for %s: %s", url, e)
        return None


def _get_cik(ticker: str) -> Optional[str]:
    """
    Resolve stock ticker to zero-padded 10-digit CIK string.
    Uses SEC's company_tickers.json mapping file.
    """
    resp = _edgar_get(EDGAR_TICKERS_URL)
    if not resp:
        return None
    try:
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
    except Exception as e:
        logger.warning("CIK lookup failed for %s: %s", ticker, e)
    return None


def _date_in_range(filing_date: str, start_date: str, end_date: str) -> bool:
    """
    True if a filing's reportDate falls within or near the query date range.
    "Near" = within 90 days after end_date to capture post-period filings.
    """
    try:
        fd = datetime.fromisoformat(filing_date)
        sd = datetime.fromisoformat(start_date)
        ed = datetime.fromisoformat(end_date)
        # The filing's reporting period can precede or overlap the query range.
        # Accept if filing's period falls within [start - 180d, end + 90d].
        from datetime import timedelta
        return (sd - timedelta(days=180)) <= fd <= (ed + timedelta(days=90))
    except ValueError:
        return False


def _discover_filings(cik: str, ticker: str, start_date: str, end_date: str) -> list[dict]:
    """
    Return a list of relevant 10-K/10-Q filing dicts for the date range.
    Each dict: accession_number, filing_type, period, filing_date, primary_doc
    """
    url = EDGAR_SUBMISSIONS.format(cik=cik)
    resp = _edgar_get(url)
    if not resp:
        return []

    try:
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
    except Exception as e:
        logger.warning("EDGAR submissions parse failed for %s: %s", ticker, e)
        return []

    results = []
    for i, form in enumerate(forms):
        if form not in ("10-K", "10-Q"):
            continue
        report_date = report_dates[i] if i < len(report_dates) else ""
        if not report_date or not _date_in_range(report_date, start_date, end_date):
            continue
        acc = accessions[i] if i < len(accessions) else ""
        doc = primary_docs[i] if i < len(primary_docs) else ""
        if not acc or not doc:
            continue

        # Derive quarter label from report date, e.g. 2024Q2
        try:
            rd = datetime.fromisoformat(report_date)
            quarter = f"{rd.year}Q{((rd.month - 1) // 3) + 1}"
        except ValueError:
            quarter = report_date[:7].replace("-", "")

        results.append({
            "accession_number": acc.replace("-", ""),
            "filing_type": form,
            "period": quarter,
            "filing_date": dates[i] if i < len(dates) else report_date,
            "primary_doc": doc,
            "cik": cik,
        })

        if len(results) >= _MAX_FILINGS_TO_INGEST:
            break

    return results


def _download_filing(cik: str, accession_no: str, primary_doc: str) -> Optional[str]:
    """Download a filing document from EDGAR and return clean plain text."""
    # EDGAR Archive URLs use the accession number WITHOUT dashes in the path segment.
    # e.g. /Archives/edgar/data/{cik}/{acc_no_dashes}/{primary_doc}
    # accession_no is already stored without dashes by _discover_filings.
    acc_no_dashes = accession_no.replace("-", "")
    url = f"{EDGAR_FILING_BASE}/{int(cik)}/{acc_no_dashes}/{primary_doc}"
    resp = _edgar_get(url)
    if not resp:
        return None
    return _strip_html(resp.text)


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _ingest_filing(collection: chromadb.Collection, filing: dict, ticker: str) -> int:
    """
    Download, chunk, embed, and store a single filing.
    Returns number of new chunks stored (0 if all already present).
    """
    text = _download_filing(filing["cik"], filing["accession_number"], filing["primary_doc"])
    if not text or len(text) < 200:
        logger.warning("Empty or too-short filing text for %s %s", ticker, filing["period"])
        return 0

    chunks = _chunk_text(text, ticker, filing["filing_type"], filing["period"])
    if not chunks:
        return 0

    # Check which chunk IDs are already stored (deduplication)
    existing_ids = set(collection.get(ids=[c["id"] for c in chunks])["ids"])
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    if not new_chunks:
        logger.info("All %d chunks already in ChromaDB for %s %s", len(chunks), ticker, filing["period"])
        return 0

    texts = [c["text"] for c in new_chunks]
    vectors = _embed_texts(texts)

    collection.add(
        ids=[c["id"] for c in new_chunks],
        documents=texts,
        embeddings=vectors,
        metadatas=[c["metadata"] for c in new_chunks],
    )
    logger.info("Ingested %d new chunks for %s %s", len(new_chunks), ticker, filing["period"])
    return len(new_chunks)


def _query_collection(
    collection: chromadb.Collection,
    ticker: str,
    user_message: str,
) -> list[dict]:
    """
    Semantic search in ChromaDB filtered to ticker, returns top-K chunks.
    Returns list of filing_chunk dicts matching the state schema.
    """
    try:
        count = collection.count()
        if count == 0:
            return []
    except Exception:
        return []

    query_vec = _embed_query(user_message or ticker)

    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(_TOP_K, count),
            where={"ticker": ticker},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning("ChromaDB query failed: %s", e)
        return []

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        # Convert cosine distance → similarity score (0–1)
        score = round(1 - dist, 4)
        chunks.append({
            "text": doc,
            "filing_type": meta.get("filing_type", ""),
            "filing_quarter": meta.get("filing_period", ""),
            "filing_date": meta.get("filing_date", ""),
            "chunk_relevance_score": score,
        })

    return chunks


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def retrieve_rag_context(state: AgentState) -> AgentState:
    """
    Node 7: RAG Retriever.

    Retrieves relevant SEC filing chunks from ChromaDB.  Triggers on-demand
    ingestion from SEC EDGAR when no cached chunks exist for the ticker.
    """
    ticker = state.get("ticker", "")
    start_date = state.get("start_date", "")
    end_date = state.get("end_date", "")
    user_message = state.get("user_message", "")

    if not ticker:
        logger.debug("retrieve_rag_context: no ticker, skipping")
        return {"filing_chunks": [], "filing_ingested": False, "filing_error": None}

    if not start_date or not end_date:
        logger.debug("retrieve_rag_context: no date range for %s, skipping", ticker)
        return {"filing_chunks": [], "filing_ingested": False, "filing_error": None}

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.warning("retrieve_rag_context: GEMINI_API_KEY not set")
        return {"filing_chunks": [], "filing_ingested": False, "filing_error": "GEMINI_API_KEY not configured"}

    try:
        collection = _get_collection()

        # Step 1: try retrieval from existing vector store
        chunks = _query_collection(collection, ticker, user_message)
        if chunks:
            logger.info("retrieve_rag_context: %d chunks retrieved from cache for %s", len(chunks), ticker)
            return {"filing_chunks": chunks, "filing_ingested": False, "filing_error": None}

        # Step 2: no cached chunks → discover and ingest from EDGAR
        cik = _get_cik(ticker)
        if not cik:
            logger.info("retrieve_rag_context: CIK not found for %s", ticker)
            return {"filing_chunks": [], "filing_ingested": False, "filing_error": None}

        filings = _discover_filings(cik, ticker, start_date, end_date)
        if not filings:
            logger.info("retrieve_rag_context: no EDGAR filings found for %s in range", ticker)
            return {"filing_chunks": [], "filing_ingested": False, "filing_error": None}

        total_new = 0
        for filing in filings:
            total_new += _ingest_filing(collection, filing, ticker)

        if total_new == 0:
            logger.info("retrieve_rag_context: filings already ingested or empty for %s", ticker)

        # Step 3: re-query after ingestion
        chunks = _query_collection(collection, ticker, user_message)
        logger.info(
            "retrieve_rag_context: ingested %d new chunks, retrieved %d for %s",
            total_new, len(chunks), ticker,
        )
        return {"filing_chunks": chunks, "filing_ingested": total_new > 0, "filing_error": None}

    except Exception as e:
        logger.error("retrieve_rag_context failed: %s", e)
        return {"filing_chunks": [], "filing_ingested": False, "filing_error": str(e)}
