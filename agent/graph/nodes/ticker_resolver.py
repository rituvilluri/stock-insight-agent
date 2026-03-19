"""
Node 2: Ticker Resolver

Reads:  user_message
Writes: ticker, company_name, ticker_error

Resolves the stock ticker symbol from the user's message using three
layers in priority order:
  1. Direct ticker detection  — all-caps 1-5 letter word in the message
  2. Lookup table             — hardcoded company name → (ticker, name) map
  3. LLM fallback             — only when layers 1 and 2 both miss

Why three layers instead of always using the LLM?
The LLM adds ~1 second of latency and consumes tokens for every call.
For the vast majority of queries, a fast regex + lookup table is sufficient.
The LLM is kept as a fallback for genuinely unknown companies, not as the
default path.
"""

import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_classifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup table
# Maps lowercase company name variants → (ticker, canonical_company_name).
# Add entries here as new stocks become commonly requested.
# Why hardcode instead of using a database? Zero latency, zero dependencies,
# and this set covers ~90% of retail investor queries.
# ---------------------------------------------------------------------------
TICKER_LOOKUP: dict[str, tuple[str, str]] = {
    # NVIDIA
    "nvidia": ("NVDA", "NVIDIA"),
    "nvda": ("NVDA", "NVIDIA"),
    # Apple
    "apple": ("AAPL", "Apple"),
    "aapl": ("AAPL", "Apple"),
    # Microsoft
    "microsoft": ("MSFT", "Microsoft"),
    "msft": ("MSFT", "Microsoft"),
    # Google / Alphabet
    "google": ("GOOGL", "Alphabet (Google)"),
    "alphabet": ("GOOGL", "Alphabet (Google)"),
    "googl": ("GOOGL", "Alphabet (Google)"),
    "goog": ("GOOGL", "Alphabet (Google)"),
    # Amazon
    "amazon": ("AMZN", "Amazon"),
    "amzn": ("AMZN", "Amazon"),
    # Meta
    "meta": ("META", "Meta"),
    "facebook": ("META", "Meta"),
    # Tesla
    "tesla": ("TSLA", "Tesla"),
    "tsla": ("TSLA", "Tesla"),
    # GameStop
    "gamestop": ("GME", "GameStop"),
    "game stop": ("GME", "GameStop"),
    "gme": ("GME", "GameStop"),
    # AMD
    "amd": ("AMD", "AMD"),
    "advanced micro devices": ("AMD", "AMD"),
    # Netflix
    "netflix": ("NFLX", "Netflix"),
    "nflx": ("NFLX", "Netflix"),
    # Palantir
    "palantir": ("PLTR", "Palantir"),
    "pltr": ("PLTR", "Palantir"),
    # Broadcom
    "broadcom": ("AVGO", "Broadcom"),
    "avgo": ("AVGO", "Broadcom"),
    # Spotify
    "spotify": ("SPOT", "Spotify"),
    "spot": ("SPOT", "Spotify"),
    # Uber
    "uber": ("UBER", "Uber"),
    # Airbnb
    "airbnb": ("ABNB", "Airbnb"),
    "abnb": ("ABNB", "Airbnb"),
    # Coinbase
    "coinbase": ("COIN", "Coinbase"),
    "coin": ("COIN", "Coinbase"),
    # Snowflake
    "snowflake": ("SNOW", "Snowflake"),
    "snow": ("SNOW", "Snowflake"),
    # Intel
    "intel": ("INTC", "Intel"),
    "intc": ("INTC", "Intel"),
    # JPMorgan
    "jpmorgan": ("JPM", "JPMorgan Chase"),
    "jp morgan": ("JPM", "JPMorgan Chase"),
    "jpm": ("JPM", "JPMorgan Chase"),
    # Berkshire
    "berkshire": ("BRK-B", "Berkshire Hathaway"),
}

# Words that match [A-Z]{2-5} but are never ticker symbols.
# Layer 1 skips these to avoid false positives.
_TICKER_BLOCKLIST = frozenset({
    "CEO", "CFO", "COO", "IPO", "ETF", "SEC", "FED", "GDP", "CPI", "PPI",
    "EPS", "PE", "AI", "ML", "API", "AR", "VR", "PR", "US", "EU",
    "UK", "THE", "FOR", "AND", "OR", "BUT", "IN", "ON", "AT",
    # Single-letter and common English words
    "I", "A", "AN", "MY", "IF", "SO", "UP", "DO", "BE", "IS", "IT", "AS",
    "TO",
    # Quarter labels — date references, not tickers
    "Q1", "Q2", "Q3", "Q4",
})

# Regex to detect a standalone ticker: 2-5 uppercase letters, word boundary.
# Why word boundary (\b)? Without it, "PLANS" inside "EXPLAINS" would match.
_DIRECT_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")

# LLM prompt for fallback resolution
_SYSTEM_PROMPT = """\
You are a stock ticker resolver. Extract the stock ticker symbol and company
name from the user's message.

Respond with ONLY a JSON object — no explanation, no markdown:
{"ticker": "NVDA", "company_name": "NVIDIA"}

If you cannot identify a stock in the message, respond:
{"ticker": null, "company_name": null}
"""


# ---------------------------------------------------------------------------
# Layer 1: direct ticker detection
# ---------------------------------------------------------------------------

def _detect_direct_ticker(message: str) -> tuple[str, str] | None:
    """
    Return (ticker, company_name) if the message contains an all-caps ticker
    symbol typed directly by the user. Returns None if not found.

    Applies _TICKER_BLOCKLIST to filter common all-caps words that are not
    ticker symbols (CEO, AI, GDP, etc.) before returning a match.
    """
    matches = _DIRECT_TICKER_RE.findall(message)
    for match in matches:
        if match not in _TICKER_BLOCKLIST:
            return (match, match)  # use ticker as company_name placeholder
    return None


# ---------------------------------------------------------------------------
# Layer 2: lookup table
# ---------------------------------------------------------------------------

def _lookup_table(message: str) -> tuple[str, str] | None:
    """
    Scan the lowercased message for known company name substrings.
    Return (ticker, company_name) on first match, None if no match.

    Longer keys are checked first to prevent "meta" matching before
    "meta platforms" if both were in the table.
    """
    lower_msg = message.lower()
    # Sort by key length descending so longer/more-specific names win
    for key in sorted(TICKER_LOOKUP.keys(), key=len, reverse=True):
        if key in lower_msg:
            return TICKER_LOOKUP[key]
    return None


# ---------------------------------------------------------------------------
# Layer 3: LLM fallback
# ---------------------------------------------------------------------------

def _llm_resolve(message: str) -> tuple[str, str] | None:
    """
    Ask the LLM to extract ticker and company name. Returns (ticker,
    company_name) on success, None if the LLM cannot identify a stock.
    Raises on network/parse errors so the caller can write to ticker_error.
    """
    response = llm_classifier.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=message),
    ])

    raw = response.content.strip()

    # Strip markdown fences defensively (same issue as intent classifier)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    ticker = parsed.get("ticker")
    company_name = parsed.get("company_name")

    if not ticker:
        return None

    return (ticker, company_name or ticker)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def resolve_ticker(state: AgentState) -> AgentState:
    """
    Resolve the stock ticker and company name from the user's message.
    Tries three layers in order: direct detection → lookup table → LLM.
    """
    user_message = state["user_message"]

    try:
        # Layer 1 — direct ticker typed by user (e.g. "how did NVDA do?")
        result = _detect_direct_ticker(user_message)
        if result:
            ticker, company_name = result
            logger.info("resolve_ticker [direct] → ticker=%r", ticker)
            return {**state, "ticker": ticker, "company_name": company_name, "ticker_error": None}

        # Layer 2 — company name in lookup table (e.g. "what happened with nvidia")
        result = _lookup_table(user_message)
        if result:
            ticker, company_name = result
            logger.info("resolve_ticker [lookup] → ticker=%r company=%r", ticker, company_name)
            return {**state, "ticker": ticker, "company_name": company_name, "ticker_error": None}

        # Before falling back to the LLM, check if session context already has
        # a valid ticker. If layers 1 and 2 both missed, the message likely
        # contains no explicit stock mention (e.g. "does it have anything to do
        # with the war?" or "what about earnings?"). In that case, preserve the
        # session ticker rather than asking the LLM — which would hallucinate a
        # stock from whatever word sounds most "stock-like" in the message.
        seeded_ticker = state.get("ticker", "")
        if seeded_ticker:
            logger.info(
                "resolve_ticker: no ticker in message, preserving session ticker %r",
                seeded_ticker,
            )
            return {**state, "ticker_error": None}

        # Layer 3 — LLM fallback: only reached when there is no session context
        # AND the user typed something the lookup table doesn't recognise.
        # Example: "how did Scorpio Tankers do?" on a fresh session.
        logger.info("resolve_ticker: no match in direct/lookup, falling back to LLM")
        result = _llm_resolve(user_message)
        if result:
            ticker, company_name = result
            logger.info("resolve_ticker [llm] → ticker=%r company=%r", ticker, company_name)
            return {**state, "ticker": ticker, "company_name": company_name, "ticker_error": None}

        logger.warning("resolve_ticker: could not identify a ticker in message")
        return {
            **state,
            "ticker": "",
            "company_name": "",
            "ticker_error": "Could not identify a stock ticker in the message.",
        }

    except Exception as e:
        logger.error("resolve_ticker failed: %s", e)
        return {
            **state,
            "ticker": "",
            "company_name": "",
            "ticker_error": str(e),
        }
