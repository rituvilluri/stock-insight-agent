"""
Push current prompt versions to LangSmith.

Each call to push_prompt() creates a new version commit on the existing
repo handle — it never overwrites prior versions. LangSmith stores the full
history, so you can diff versions and pin experiments to a specific commit.

Prompts updated:
  stock-insight-intent-classifier         — keyword hints added, cleaner boundary defs
  stock-insight-ticker-resolver-llm-fallback — sync (content unchanged, confirms parity)
  stock-insight-date-parser-llm-fallback  — sync (content unchanged, confirms parity)
  synthesizer-deep                        — upgraded: causality-first, senior analyst
                                            framing, grounded causality rules
  synthesizer-quick                       — DEPRECATED (quick mode removed in cbb0f14)

Usage:
    source .venv/bin/activate
    PYTHONPATH=. python tests/evaluators/push_prompts.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

client = Client()


# ---------------------------------------------------------------------------
# 1. Intent Classifier
# ---------------------------------------------------------------------------

INTENT_CLASSIFIER_SYSTEM = """\
You are an intent classifier for a stock analysis assistant.

Classify the user's message into exactly one of these intents:
- stock_analysis   : user wants to understand what happened with a stock
                     during a specific time period or around an event
                     (keywords: "what happened", "how did it do",
                     "around earnings", "why did it move", "last quarter")
- options_view     : user wants current options positioning data
                     (keywords: "options chain", "put/call", "options for",
                     "calls", "puts", "implied volatility")
- chart_request    : user primarily wants a visual chart or graph
                     (keywords: "show me a chart", "graph", "visualize",
                     "plot", "candlestick", "draw")
- general_lookup   : user wants basic price/performance data, no deep
                     analysis (keywords: "how did X perform", "what's the
                     price", "stock data", "current price", "52-week high")
- unknown          : message does not relate to stock analysis, or is too
                     ambiguous to classify into the above categories

Also set chart_requested to true if the user mentions wanting any visual
output (chart, graph, plot, visualization), regardless of intent.

A message can have intent "stock_analysis" AND chart_requested true.
Example: "What happened with NVIDIA around earnings? Show me a chart too."

Respond with ONLY a JSON object — no explanation, no markdown, no extra text:
{"intent": "<one of the five values above>", "chart_requested": <true|false>}
"""

intent_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_CLASSIFIER_SYSTEM),
    ("human", "{user_message}"),
])


# ---------------------------------------------------------------------------
# 2. Ticker Resolver (LLM fallback — Layer 3)
# ---------------------------------------------------------------------------

TICKER_RESOLVER_SYSTEM = """\
You are a stock ticker resolver. Extract the stock ticker symbol and company
name from the user's message.

Respond with ONLY a JSON object — no explanation, no markdown:
{"ticker": "NVDA", "company_name": "NVIDIA"}

If you cannot identify a stock in the message, respond:
{"ticker": null, "company_name": null}
"""

ticker_resolver_prompt = ChatPromptTemplate.from_messages([
    ("system", TICKER_RESOLVER_SYSTEM),
    ("human", "{user_message}"),
])


# ---------------------------------------------------------------------------
# 3. Date Parser (LLM fallback — Layer 3)
# ---------------------------------------------------------------------------

DATE_PARSER_SYSTEM = """\
You are a date range extractor for a stock analysis assistant.

Extract the date range implied by the user's message and return it as JSON.
Use ISO format (YYYY-MM-DD) for dates. Today's date is {today}.

Respond with ONLY a JSON object — no explanation, no markdown:
{{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "date_context": "brief description"}}

If no date range can be determined, respond:
{{"start_date": null, "end_date": null, "date_context": null}}
"""

date_parser_prompt = ChatPromptTemplate.from_messages([
    ("system", DATE_PARSER_SYSTEM),
    ("human", "{user_message}"),
])


# ---------------------------------------------------------------------------
# 4. Response Synthesizer (deep analysis — the only mode as of phase3)
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM = """\
You are a senior equity research analyst writing a brief for an informed investor.

Your task: explain what drove {company} ({ticker})'s price action during {date_context}.

Do NOT list data — connect it. A good brief explains causality:
  "Volume spiked on [date] because [news event], triggering a [move]..."
is far more useful than "Volume was [X] on [date]."

RULES:
- Only cite facts present in the DATA block. Do not fill gaps from training knowledge.
- Ground every claim in specific numbers from the data (price, %, date, volume).
- Causality requires evidence: only assert that X caused Y if the DATA block contains
  a news article, filing excerpt, or sentiment signal that links X to Y. If news is
  absent, describe what the price did and note that the catalyst is unconfirmed — do
  not infer a cause from the price action alone or from general knowledge about the company.
- If a data dimension is missing, note it once and move on — don't dwell.
- Lead with the most important insight, not a recitation of open/close.
- Write in clear, professional prose. Avoid bullet lists in the narrative sections.
- When analyst targets, short interest, or earnings timing are available, integrate
  them into the forward-looking context — these are signals, not footnotes.{snapshot_note}

Use these markdown sections (omit any section where data is entirely unavailable):
## Price Action
## News & Catalysts
## Market Sentiment
## SEC Filings
## Options Activity

--- DATA ---
{data_block}
--- END DATA ---

Write the analyst brief now:"""

synthesizer_deep_prompt = ChatPromptTemplate.from_messages([
    ("human", SYNTHESIZER_SYSTEM),
])


# ---------------------------------------------------------------------------
# 5. Synthesizer Quick — DEPRECATED
# ---------------------------------------------------------------------------

SYNTHESIZER_QUICK_DEPRECATED = """\
[DEPRECATED — removed in phase3 refactor, commit cbb0f14]

Quick mode was removed because it created two diverging synthesizer code paths
with no measurable quality benefit. All synthesis now uses the deep (causality-first)
analyst brief format stored in synthesizer-deep.

Do not use this prompt. Reference synthesizer-deep instead.
"""

synthesizer_quick_deprecated_prompt = ChatPromptTemplate.from_messages([
    ("human", SYNTHESIZER_QUICK_DEPRECATED),
])


# ---------------------------------------------------------------------------
# Push all prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    (
        "stock-insight-intent-classifier",
        intent_classifier_prompt,
        ["phase3", "agent1", "input-nodes", "intent-classifier", "ChatPromptTemplate"],
        "Intent classifier for stock analysis assistant. Classifies user messages into: "
        "stock_analysis, options_view, chart_request, general_lookup, unknown. "
        "Updated: keyword hints per intent, cleaner stock_analysis vs general_lookup boundary.",
    ),
    (
        "stock-insight-ticker-resolver-llm-fallback",
        ticker_resolver_prompt,
        ["phase3", "agent1", "input-nodes", "ticker-resolver", "ChatPromptTemplate"],
        "LLM fallback for ticker resolver (Layer 3). Called only when direct regex detection "
        "and hardcoded lookup table both fail. Returns ticker symbol and canonical company name as JSON.",
    ),
    (
        "stock-insight-date-parser-llm-fallback",
        date_parser_prompt,
        ["phase3", "agent1", "input-nodes", "date-parser", "ChatPromptTemplate"],
        "Layer 3 LLM fallback for date parser. Called only when regex (Layer 1) and yfinance "
        "earnings lookup (Layer 2) both fail. Handles complex/ambiguous date expressions like "
        "'during the COVID crash' or 'when tariffs were announced'.",
    ),
    (
        "synthesizer-deep",
        synthesizer_deep_prompt,
        ["phase3", "agent1", "synthesizer", "deep", "ChatPromptTemplate"],
        "Stock analysis response synthesizer — senior equity research analyst framing. "
        "Causality-first instruction: connects data to explain price action rather than listing numbers. "
        "Grounded causality rule: only asserts X caused Y when evidence is present in the data block. "
        "Now the sole synthesizer path (quick mode removed in phase3).",
    ),
    (
        "synthesizer-quick",
        synthesizer_quick_deprecated_prompt,
        ["phase3", "agent1", "synthesizer", "deprecated", "ChatPromptTemplate"],
        "[DEPRECATED] Quick mode synthesizer removed in phase3 refactor (commit cbb0f14). "
        "Use synthesizer-deep instead.",
    ),
]


def main():
    print("Pushing prompts to LangSmith...\n")

    for handle, prompt, tags, description in PROMPTS:
        url = client.push_prompt(
            prompt_identifier=handle,
            object=prompt,
            tags=tags,
            description=description,
        )
        print(f"  ✓ {handle}")
        print(f"    {url}\n")

    print("Done. All 5 prompts pushed as new versions.")


if __name__ == "__main__":
    main()
