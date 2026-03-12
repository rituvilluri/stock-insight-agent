"""
Node 3: Date Parser

Reads:  user_message, ticker
Writes: start_date, end_date, date_context, date_missing,
        include_current_snapshot, date_error

Three-layer architecture, tried in order:
  1. Simple relative ranges  — regex + date arithmetic, no LLM, no API
  2. Earnings-relative lookup — regex extracts quarter/year, yfinance
                                provides the actual earnings date
  3. LLM fallback            — for complex/ambiguous expressions

Also detects:
  include_current_snapshot: user wants both historical AND current data
  date_missing: all layers failed; graph routes to clarification prompt

Why not always use the LLM for date parsing?
The LLM adds latency and consumes rate-limit tokens. Regex handles ~80%
of real queries instantly. yfinance covers the earnings use-case that
is central to this product. The LLM is reserved for genuinely hard cases
like "during the COVID crash" or "when tariffs were announced".
"""

import json
import logging
import re
from datetime import datetime, timedelta

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.nodes.state import AgentState
from llm.llm_setup import llm_classifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Days before/after an earnings date when building the earnings window.
# Values come from TDD section 4 Node 3.
EARNINGS_WINDOW_BEFORE = 14
EARNINGS_WINDOW_AFTER = 7

# Quarter-to-calendar-month boundaries (inclusive)
_QUARTER_MONTHS: dict[int, tuple[int, int]] = {
    1: (1, 3),
    2: (4, 6),
    3: (7, 9),
    4: (10, 12),
}

# Phrases that signal the user also wants current market data alongside
# whatever historical period they asked about.
_CURRENT_SNAPSHOT_PHRASES = [
    "right now",
    "currently",
    "what's happening",
    "what is happening",
    "at the moment",
    "these days",
    "happening now",
    "current market",
    "current conditions",
    "right now",
]

# ---------------------------------------------------------------------------
# LLM prompt for Layer 3 fallback
# ---------------------------------------------------------------------------

_LLM_DATE_PROMPT = """\
You are a date range extractor for a stock analysis assistant.

Extract the date range implied by the user's message and return it as JSON.
Use ISO format (YYYY-MM-DD) for dates. Today's date is {today}.

Respond with ONLY a JSON object — no explanation, no markdown:
{{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "date_context": "brief description"}}

If no date range can be determined, respond:
{{"start_date": null, "end_date": null, "date_context": null}}
"""


# ---------------------------------------------------------------------------
# Layer 1: simple relative range patterns
# ---------------------------------------------------------------------------

def _parse_simple_range(message: str) -> tuple[str, str, str] | None:
    """
    Match common relative date expressions via regex.
    Returns (start_date_iso, end_date_iso, date_context) or None.

    Uses re.search() — not re.match() — so the pattern can appear
    anywhere in the sentence, not just at the start.
    (The existing tools/date/date_parser_tool.py used re.match(), which
    caused silent failures for messages like "How did NVDA do last week?")
    """
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    def _fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    # Each entry: (compiled_regex, handler that returns (start, end, context))
    # Patterns are ordered from most-specific to least-specific to avoid
    # "last month" matching before "last 3 months".
    patterns = [
        # "last N days" / "past N days"
        (
            re.compile(r"(?:last|past)\s+(\d+)\s+days?", re.IGNORECASE),
            lambda m: (
                _fmt(now - timedelta(days=int(m.group(1)))),
                today,
                f"last {m.group(1)} days",
            ),
        ),
        # "last N weeks" / "past N weeks"
        (
            re.compile(r"(?:last|past)\s+(\d+)\s+weeks?", re.IGNORECASE),
            lambda m: (
                _fmt(now - timedelta(weeks=int(m.group(1)))),
                today,
                f"last {m.group(1)} weeks",
            ),
        ),
        # "last N months" / "past N months"
        (
            re.compile(r"(?:last|past)\s+(\d+)\s+months?", re.IGNORECASE),
            lambda m: (
                _fmt(now - timedelta(days=30 * int(m.group(1)))),
                today,
                f"last {m.group(1)} months",
            ),
        ),
        # "last week" / "past week"
        (
            re.compile(r"(?:last|past)\s+week\b", re.IGNORECASE),
            lambda m: (_fmt(now - timedelta(weeks=1)), today, "last week"),
        ),
        # "last month" / "past month"
        (
            re.compile(r"(?:last|past)\s+month\b", re.IGNORECASE),
            lambda m: (_fmt(now - timedelta(days=30)), today, "last month"),
        ),
        # "last quarter" / "past quarter"
        (
            re.compile(r"(?:last|past)\s+quarter\b", re.IGNORECASE),
            lambda m: (_fmt(now - timedelta(days=90)), today, "last quarter"),
        ),
        # "last year" / "past year"
        (
            re.compile(r"(?:last|past)\s+year\b", re.IGNORECASE),
            lambda m: (_fmt(now - timedelta(days=365)), today, "last year"),
        ),
        # "this week" — from the most recent Monday
        (
            re.compile(r"\bthis\s+week\b", re.IGNORECASE),
            lambda m: (
                _fmt(now - timedelta(days=now.weekday())),
                today,
                "this week",
            ),
        ),
        # "this month" — from the 1st of the current month
        (
            re.compile(r"\bthis\s+month\b", re.IGNORECASE),
            lambda m: (_fmt(now.replace(day=1)), today, "this month"),
        ),
        # "yesterday"
        (
            re.compile(r"\byesterday\b", re.IGNORECASE),
            lambda m: (
                _fmt(now - timedelta(days=1)),
                _fmt(now - timedelta(days=1)),
                "yesterday",
            ),
        ),
        # "3 months ago" style
        (
            re.compile(r"(\d+)\s+months?\s+ago\b", re.IGNORECASE),
            lambda m: (
                _fmt(now - timedelta(days=30 * int(m.group(1)))),
                today,
                f"{m.group(1)} months ago",
            ),
        ),
        # "3 weeks ago" style
        (
            re.compile(r"(\d+)\s+weeks?\s+ago\b", re.IGNORECASE),
            lambda m: (
                _fmt(now - timedelta(weeks=int(m.group(1)))),
                today,
                f"{m.group(1)} weeks ago",
            ),
        ),
        # "3 days ago" style
        (
            re.compile(r"(\d+)\s+days?\s+ago\b", re.IGNORECASE),
            lambda m: (
                _fmt(now - timedelta(days=int(m.group(1)))),
                today,
                f"{m.group(1)} days ago",
            ),
        ),
    ]

    for pattern, handler in patterns:
        match = pattern.search(message)
        if match:
            start, end, ctx = handler(match)
            return start, end, ctx

    return None


# ---------------------------------------------------------------------------
# Layer 2: earnings-relative lookup
# ---------------------------------------------------------------------------

def _extract_earnings_quarter_year(message: str) -> tuple[int, int] | None:
    """
    Extract (quarter, year) from messages like:
      "around Q2 2024 earnings", "Q3 '23 earnings", "earnings Q1 2025"
    Returns None if no earnings quarter/year is found.
    """
    if not re.search(r"earnings?", message, re.IGNORECASE):
        return None

    # Match "Q{1-4}" followed by a 2- or 4-digit year.
    # The '? handles shorthand like "Q1 '23" (apostrophe before 2-digit year).
    match = re.search(r"[Qq]([1-4])\s*'?(?:20)?(\d{2})\b", message)
    if not match:
        return None

    quarter = int(match.group(1))
    year_part = int(match.group(2))
    # Interpret 2-digit years as 2000s (e.g. '24 → 2024)
    year = 2000 + year_part if year_part < 100 else year_part

    if not (2000 <= year <= 2035):
        return None

    return quarter, year


def _get_earnings_date(ticker: str, quarter: int, year: int) -> datetime | None:
    """
    Look up the actual earnings date for ticker/quarter/year via yfinance.
    Returns a naive datetime on success, None on failure or no data.

    Why a separate function?
    Makes it easy to mock in tests without mocking the whole node, and
    keeps the yfinance-specific logic isolated from the parsing logic.
    """
    if not ticker:
        return None

    try:
        stock = yf.Ticker(ticker)
        df = stock.earnings_dates

        if df is None or df.empty:
            return None

        start_month, end_month = _QUARTER_MONTHS[quarter]

        for ts in df.index:
            # earnings_dates index is timezone-aware; strip tz for arithmetic
            dt = ts.to_pydatetime()
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)

            if dt.year == year and start_month <= dt.month <= end_month:
                return dt

        return None

    except Exception as e:
        logger.warning("yfinance earnings lookup failed for %s Q%d %d: %s", ticker, quarter, year, e)
        return None


def _parse_earnings_range(
    message: str, ticker: str
) -> tuple[str, str, str] | None:
    """
    If the message references a specific earnings quarter/year, look up the
    actual earnings date and build a ±window around it.
    Returns (start_date_iso, end_date_iso, date_context) or None.
    """
    result = _extract_earnings_quarter_year(message)
    if not result:
        return None

    quarter, year = result
    earnings_dt = _get_earnings_date(ticker, quarter, year)

    if not earnings_dt:
        logger.info(
            "earnings date not found in yfinance for Q%d %d %s; "
            "will fall through to LLM",
            quarter, year, ticker,
        )
        return None

    start = earnings_dt - timedelta(days=EARNINGS_WINDOW_BEFORE)
    end = earnings_dt + timedelta(days=EARNINGS_WINDOW_AFTER)

    context = (
        f"around Q{quarter} {year} earnings "
        f"({EARNINGS_WINDOW_BEFORE} days before through "
        f"{EARNINGS_WINDOW_AFTER} days after {earnings_dt.strftime('%Y-%m-%d')})"
    )

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), context


# ---------------------------------------------------------------------------
# Layer 3: LLM fallback
# ---------------------------------------------------------------------------

def _parse_with_llm(message: str) -> tuple[str, str, str] | None:
    """
    Ask the LLM to extract a date range from a complex/ambiguous message.
    Returns (start_date_iso, end_date_iso, date_context) or None.
    Raises on LLM/network errors (caller writes to date_error).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    system_content = _LLM_DATE_PROMPT.format(today=today)

    response = llm_classifier.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=message),
    ])

    raw = response.content.strip()

    # Strip markdown fences defensively
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)

    start = parsed.get("start_date")
    end = parsed.get("end_date")
    ctx = parsed.get("date_context") or "date range from context"

    if not start or not end:
        return None

    return start, end, ctx


# ---------------------------------------------------------------------------
# Current snapshot detection
# ---------------------------------------------------------------------------

def _has_current_snapshot_request(message: str) -> bool:
    """
    Return True if the message contains language suggesting the user also
    wants current market data alongside whatever historical period they
    specified. E.g. "...around Q2 earnings? And what's happening right now?"
    """
    lower = message.lower()
    return any(phrase in lower for phrase in _CURRENT_SNAPSHOT_PHRASES)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def parse_dates(state: AgentState) -> AgentState:
    """
    Resolve a date range from the user's message.
    Tries three layers in order: simple regex → earnings lookup → LLM.
    Sets date_missing=True if all layers fail.
    """
    user_message = state["user_message"]
    # ticker may be empty string if ticker_resolver failed — that's fine,
    # we just skip the earnings lookup layer in that case
    ticker = state.get("ticker", "")

    include_current = _has_current_snapshot_request(user_message)

    try:
        # Layer 1 — simple relative range
        result = _parse_simple_range(user_message)
        if result:
            start, end, ctx = result
            logger.info("parse_dates [simple] → %s to %s (%s)", start, end, ctx)
            return {
                **state,
                "start_date": start,
                "end_date": end,
                "date_context": ctx,
                "date_missing": False,
                "include_current_snapshot": include_current,
                "date_error": None,
            }

        # Layer 2 — earnings-relative lookup (requires ticker + yfinance)
        result = _parse_earnings_range(user_message, ticker)
        if result:
            start, end, ctx = result
            logger.info("parse_dates [earnings] → %s to %s (%s)", start, end, ctx)
            return {
                **state,
                "start_date": start,
                "end_date": end,
                "date_context": ctx,
                "date_missing": False,
                "include_current_snapshot": include_current,
                "date_error": None,
            }

        # Layer 3 — LLM fallback for complex/ambiguous expressions
        logger.info("parse_dates: no simple/earnings match, falling back to LLM")
        result = _parse_with_llm(user_message)
        if result:
            start, end, ctx = result
            logger.info("parse_dates [llm] → %s to %s (%s)", start, end, ctx)
            return {
                **state,
                "start_date": start,
                "end_date": end,
                "date_context": ctx,
                "date_missing": False,
                "include_current_snapshot": include_current,
                "date_error": None,
            }

        # All layers failed — no date range could be determined
        logger.warning("parse_dates: could not determine a date range")
        return {
            **state,
            "start_date": "",
            "end_date": "",
            "date_context": "",
            "date_missing": True,
            "include_current_snapshot": False,
            "date_error": None,
        }

    except Exception as e:
        logger.error("parse_dates failed: %s", e)
        return {
            **state,
            "start_date": "",
            "end_date": "",
            "date_context": "",
            "date_missing": True,
            "include_current_snapshot": False,
            "date_error": str(e),
        }
