"""
Fix mislabeled examples in stock-insight-agent-baseline dataset.

Two categories of fixes:

1. Intent labels (3 examples) — labeled general_lookup but agent correctly
   classifies them as stock_analysis (all have explicit time references like
   "last month", "last 3 weeks", "last 2 weeks").

2. Date range labels (2 examples) — labeled with full calendar quarter
   (2024-04-01 → 2024-06-30) but queries say "around Q2 earnings", which
   correctly triggers the agent's earnings-window mode (14 days before through
   7 days after the actual report date). Expected outputs updated to match
   the agent's intended behavior:
     - AAPL Q2 2024: reported 2024-05-02 → window 2024-04-18 → 2024-05-09
     - NVDA Q2 2024: reported 2024-05-22 → window 2024-05-08 → 2024-05-29

Usage:
    source .venv/bin/activate
    PYTHONPATH=. python tests/evaluators/fix_dataset_labels.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

client = Client()
DATASET_ID = "4c3965c4-4cd7-4e9d-a6ea-52234b7f65ee"

FIXES = [
    # ── Intent fixes: general_lookup → stock_analysis ──────────────────────
    (
        "b5237eff-cbf9-4f81-b2dc-d41c99d25441",  # simple_stock_lookup_nvda
        # Query: "How did NVIDIA do last month?" — time-bounded → stock_analysis
        {"intent": "stock_analysis", "ticker": "NVDA"},
        "intent: general_lookup → stock_analysis",
    ),
    (
        "18b0ff71-4e63-4e0e-be92-599d575c3313",  # company_name_resolution_apple
        # Query: "How did Apple perform over the last 3 weeks?" → stock_analysis
        {"intent": "stock_analysis", "ticker": "AAPL"},
        "intent: general_lookup → stock_analysis",
    ),
    (
        "630cdada-6ed7-41e6-b2b9-ee7370448a59",  # sector_specific_tanker_stock
        # Query: "How did Scorpio Tankers do over the last 2 weeks?" → stock_analysis
        {"intent": "stock_analysis", "ticker": "STNG"},
        "intent: general_lookup → stock_analysis",
    ),

    # ── Date range fixes: full quarter → actual earnings window ────────────
    (
        "e71e063f-0998-4602-bb52-72330473cd71",  # aapl_q2_2024_earnings_period
        # Query: "What happened with Apple around Q2 2024 earnings?"
        # AAPL reported 2024-05-02; window = 14d before → 7d after
        {
            "intent": "stock_analysis",
            "ticker": "AAPL",
            "start_date": "2024-04-18",
            "end_date": "2024-05-09",
        },
        "date range: full Q2 → earnings window (2024-04-18 → 2024-05-09)",
    ),
    (
        "187a6edd-3588-44c4-8e6c-3a80bb17f279",  # earnings_period_stock_analysis
        # Query: "What happened with NVIDIA around Q2 2024 earnings?"
        # NVDA reported 2024-05-22; window = 14d before → 7d after
        {
            "intent": "stock_analysis",
            "ticker": "NVDA",
            "start_date": "2024-05-08",
            "end_date": "2024-05-29",
        },
        "date range: full Q2 → earnings window (2024-05-08 → 2024-05-29)",
    ),
]


def main():
    print(f"Dataset: {DATASET_ID}")
    print(f"Applying {len(FIXES)} fixes...\n")

    for example_id, new_outputs, description in FIXES:
        client.update_example(example_id=example_id, outputs=new_outputs)
        print(f"  ✓ {example_id[:8]}  {description}")

    print(f"\nDone. View at: https://smith.langchain.com/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
