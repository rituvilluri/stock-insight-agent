"""
One-time dataset cleanup script: stock-insight-agent-baseline

Problems being fixed:
1. 10 original examples use `expected_*` key format — evaluators read `intent`, `ticker`,
   `start_date`, `end_date` directly, so those examples produce None scores for all evaluators.
2. 5 newer examples use the correct format but are missing `ticker` in reference outputs.
3. chart_request example (`f0d71b50`) is broken — seeded input + "What about the chart?"
   is ambiguous and not representative of a real user query.
4. Dataset covers too few paths: missing fixed-date queries with verifiable date assertions,
   comparative snapshot path, historical news-absent case, multi-source synthesis case.

After this script:
- All 15 existing examples have evaluator-compatible reference outputs
- Broken chart example replaced with a clean standalone chart_request
- 8 new examples cover the identified path gaps
- Total: ~22 examples with full coverage across intents and evaluator dimensions

Usage:
    source .venv/bin/activate
    PYTHONPATH=. python tests/evaluators/dataset_cleanup.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

client = Client()
DATASET_ID = "4c3965c4-4cd7-4e9d-a6ea-52234b7f65ee"


# ---------------------------------------------------------------------------
# Step 1: Update existing examples to evaluator-compatible reference outputs
# ---------------------------------------------------------------------------

# Format: (example_id, new_outputs, optional_new_inputs, optional_new_metadata)
# Evaluator keys:
#   intent_accuracy     → reference_outputs["intent"]
#   date_range_accuracy → reference_outputs["start_date"] + ["end_date"]
#   (ticker is informational — no evaluator checks it, but good for debugging)

EXISTING_UPDATES = [
    # ── original 10 (had `expected_*` format — not read by any evaluator) ──

    (
        "b5237eff-cbf9-4f81-b2dc-d41c99d25441",  # simple_stock_lookup_nvda
        {
            "intent": "general_lookup",
            "ticker": "NVDA",
        },
        None,
        {"name": "simple_stock_lookup_nvda"},
    ),
    (
        "18b0ff71-4e63-4e0e-be92-599d575c3313",  # company_name_resolution_apple
        {
            "intent": "general_lookup",
            "ticker": "AAPL",
        },
        None,
        {"name": "company_name_resolution_apple"},
    ),
    (
        "187a6edd-3588-44c4-8e6c-3a80bb17f279",  # earnings_period_stock_analysis
        # Q2 2024 earnings window for NVDA (reported Aug 28 2024; Q2 = Apr-Jun)
        {
            "intent": "stock_analysis",
            "ticker": "NVDA",
            "start_date": "2024-04-01",
            "end_date": "2024-06-30",
        },
        None,
        {"name": "earnings_period_stock_analysis"},
    ),
    (
        "630cdada-6ed7-41e6-b2b9-ee7370448a59",  # sector_specific_tanker_stock
        # Relative date — no fixed dates assertable; intent + ticker is enough
        {
            "intent": "general_lookup",
            "ticker": "STNG",
        },
        None,
        {"name": "sector_specific_tanker_stock"},
    ),
    (
        "0508a704-48c4-4f5a-a1f7-6a54c8511d22",  # options_view_intent
        {
            "intent": "options_view",
            "ticker": "AMD",
        },
        None,
        {"name": "options_view_intent"},
    ),
    (
        "85deba62-770d-4380-9550-ac3909a747e4",  # unknown_intent_off_topic
        {
            "intent": "unknown",
        },
        None,
        {"name": "unknown_intent_off_topic"},
    ),
    (
        "0a66c3e2-950d-493b-b64d-d6b9e27cda28",  # date_missing_triggers_clarification
        {
            "intent": "stock_analysis",
            "ticker": "NVDA",
            "date_missing": True,
        },
        None,
        {"name": "date_missing_triggers_clarification"},
    ),
    (
        "a1829066-e027-4f87-acfb-1918d8329f72",  # stock_analysis_with_news_attribution
        # Relative date — not assertable
        {
            "intent": "stock_analysis",
            "ticker": "MSFT",
        },
        None,
        {"name": "stock_analysis_with_news_attribution"},
    ),
    (
        "5bc53151-8068-41d8-9a58-0136e26a0357",  # llm_ticker_fallback_unknown_company
        # Relative date — not assertable
        {
            "intent": "general_lookup",
            "ticker": "MSTR",
        },
        None,
        {"name": "llm_ticker_fallback_unknown_company"},
    ),
    (
        "4d870433-7230-4984-b989-53e3edc60a0d",  # chart_request_intent
        {
            "intent": "chart_request",
            "ticker": "TSLA",
            "chart_requested": True,
        },
        None,
        {"name": "chart_request_intent"},
    ),

    # ── newer 5 (correct format but missing ticker / incomplete) ──

    (
        "5b774f41-ed8f-4de7-99e8-82649a40b364",  # TSLA Q1 2024
        {
            "intent": "stock_analysis",
            "ticker": "TSLA",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
        },
        None,
        {"name": "tsla_q1_2024_fixed_dates"},
    ),
    (
        "73820653-b094-4cdc-869a-f074b0e0e6e3",  # Nvidia Q4 2025
        {
            "intent": "stock_analysis",
            "ticker": "NVDA",
            "start_date": "2025-10-01",
            "end_date": "2025-12-31",
        },
        None,
        {"name": "nvda_q4_2025_fixed_dates"},
    ),
    (
        "ce106a72-a5e9-4ce8-8e93-c746ec3ead42",  # NVDA last month (relative)
        {
            "intent": "stock_analysis",
            "ticker": "NVDA",
        },
        None,
        {"name": "nvda_deep_dive_relative_date"},
    ),
    (
        "e71e063f-0998-4602-bb52-72330473cd71",  # Apple Q2 2024 earnings
        # Apple Q2 fiscal = Jan-Mar 2024 (reported May 2); user says "Q2 2024 earnings"
        # which the date_parser maps to Apr-Jun 2024 (calendar Q2) with earnings context
        {
            "intent": "stock_analysis",
            "ticker": "AAPL",
            "start_date": "2024-04-01",
            "end_date": "2024-06-30",
        },
        None,
        {"name": "aapl_q2_2024_earnings_period"},
    ),
    (
        "f0d71b50-fd2e-4322-abf9-cb3b9d41fe5f",  # broken chart_request with seeded state
        # Replace with a clean standalone chart query — no seeded inputs
        {
            "intent": "chart_request",
            "ticker": "NVDA",
            "chart_requested": True,
        },
        {
            "user_config": {},
            "user_message": "Show me a chart for NVDA over the last month",
        },
        {"name": "nvda_chart_standalone"},
    ),
]


# ---------------------------------------------------------------------------
# Step 2: New examples — covering path gaps
# ---------------------------------------------------------------------------

NEW_EXAMPLES = [
    # ── Fixed-date queries (date_range_accuracy evaluator gets real signal) ──
    {
        "inputs": {
            "user_config": {},
            "user_message": "How did Apple do in January 2025?",
        },
        "outputs": {
            "intent": "stock_analysis",
            "ticker": "AAPL",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        },
        "metadata": {"name": "aapl_january_2025_fixed_dates"},
    },
    {
        "inputs": {
            "user_config": {},
            "user_message": "How did SPY do in Q3 2024?",
        },
        "outputs": {
            "intent": "stock_analysis",
            "ticker": "SPY",
            "start_date": "2024-07-01",
            "end_date": "2024-09-30",
        },
        "metadata": {"name": "spy_q3_2024_fixed_dates"},
    },
    {
        "inputs": {
            "user_config": {},
            "user_message": "Walk me through Microsoft from January to March 2025",
        },
        "outputs": {
            "intent": "stock_analysis",
            "ticker": "MSFT",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
        },
        "metadata": {"name": "msft_jan_mar_2025_explicit_range"},
    },

    # ── Comparative / include_current_snapshot path ──
    {
        "inputs": {
            "user_config": {},
            "user_message": "How is TSLA doing now compared to last month?",
        },
        "outputs": {
            "intent": "stock_analysis",
            "ticker": "TSLA",
            "include_current_snapshot": True,
        },
        "metadata": {"name": "tsla_current_vs_historical_snapshot"},
    },

    # ── Historical with known news absence (tests hallucination guard) ──
    {
        "inputs": {
            "user_config": {},
            "user_message": "What happened with GameStop in January 2021?",
        },
        "outputs": {
            "intent": "stock_analysis",
            "ticker": "GME",
            "start_date": "2021-01-01",
            "end_date": "2021-01-31",
        },
        "metadata": {"name": "gme_jan_2021_historical_squeeze"},
    },

    # ── Stress ticker — less liquid, tests ticker resolver + RAG path ──
    {
        "inputs": {
            "user_config": {},
            "user_message": "How did Palantir do in Q1 2025?",
        },
        "outputs": {
            "intent": "stock_analysis",
            "ticker": "PLTR",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
        },
        "metadata": {"name": "pltr_q1_2025_stress_ticker"},
    },

    # ── general_lookup completion path ──
    {
        "inputs": {
            "user_config": {},
            "user_message": "What's Meta's options activity looking like this week?",
        },
        "outputs": {
            "intent": "options_view",
            "ticker": "META",
        },
        "metadata": {"name": "meta_options_view_general_lookup"},
    },

    # ── Second unknown intent — different surface (not off-topic, ambiguous stock Q) ──
    {
        "inputs": {
            "user_config": {},
            "user_message": "What should I buy right now?",
        },
        "outputs": {
            "intent": "unknown",
        },
        "metadata": {"name": "unknown_intent_investment_advice"},
    },

    # ── date_missing path for a company name (not ticker) ──
    {
        "inputs": {
            "user_config": {},
            "user_message": "Tell me about Amazon stock",
        },
        "outputs": {
            "intent": "stock_analysis",
            "ticker": "AMZN",
            "date_missing": True,
        },
        "metadata": {"name": "amzn_date_missing_company_name"},
    },
]


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

def main():
    print(f"Dataset: {DATASET_ID}")
    print()

    # Update existing
    print(f"Updating {len(EXISTING_UPDATES)} existing examples...")
    for i, (example_id, new_outputs, new_inputs, new_metadata) in enumerate(EXISTING_UPDATES, 1):
        kwargs = {"outputs": new_outputs}
        if new_inputs is not None:
            kwargs["inputs"] = new_inputs
        if new_metadata is not None:
            kwargs["metadata"] = new_metadata
        client.update_example(example_id=example_id, **kwargs)
        name = (new_metadata or {}).get("name", example_id[:8])
        print(f"  [{i:02d}] updated: {name}")

    print()

    # Create new
    print(f"Creating {len(NEW_EXAMPLES)} new examples...")
    for i, ex in enumerate(NEW_EXAMPLES, 1):
        name = ex["metadata"].get("name", f"new_{i}")
        result = client.create_example(
            dataset_id=DATASET_ID,
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            metadata=ex["metadata"],
        )
        print(f"  [{i:02d}] created: {name}  (id={str(result.id)[:8]})")

    print()
    total = 15 + len(NEW_EXAMPLES)
    print(f"Done. Dataset now has {total} examples.")
    print(f"View at: https://smith.langchain.com/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
