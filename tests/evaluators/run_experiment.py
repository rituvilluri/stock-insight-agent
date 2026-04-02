"""
Run a LangSmith evaluation experiment against the stock-insight-agent-baseline dataset.

Usage:
    source .venv/bin/activate
    PYTHONPATH=. python tests/evaluators/run_experiment.py

This script:
  1. Runs the LangGraph graph against all examples in the dataset
  2. Scores each run with the 6 custom evaluators
  3. Uploads results to LangSmith under the experiment name 'baseline-post-review'

Compare 'baseline-post-review' against 'baseline-4d729529' in the LangSmith UI
to see the before/after quality improvement.

Requirements: GROQ_API_KEY and LANGSMITH_API_KEY must be set in .env
"""

import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from agent.graph.workflow import app as graph
from tests.evaluators.custom_evaluators import ALL_EVALUATORS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATASET_NAME = "stock-insight-agent-baseline"
EXPERIMENT_NAME = "all-nodes-wired"


def run_graph(inputs: dict) -> dict:
    """
    Synchronous wrapper around the async LangGraph graph.
    LangSmith's client.evaluate() calls this for each dataset example.
    """
    state = {
        "user_message": inputs.get("user_message", ""),
        "user_config": inputs.get("user_config", {}),
        # Pass through any seeded context fields (ticker, dates) from dataset examples
        **{k: v for k, v in inputs.items() if k in ("ticker", "company_name", "start_date", "end_date", "date_context")},
    }
    return asyncio.run(graph.ainvoke(state))


if __name__ == "__main__":
    client = Client()

    print(f"Running experiment '{EXPERIMENT_NAME}' against dataset '{DATASET_NAME}'...")
    print(f"Evaluators: {[fn.__name__ for fn in ALL_EVALUATORS]}")
    print()

    results = client.evaluate(
        run_graph,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=EXPERIMENT_NAME,
        description=(
            "Full pipeline: all 8 nodes wired (intent, ticker, date, price, news, "
            "reddit, RAG, options). Hallucination evaluator upgraded to llama-3.3-70b "
            "with graded 1-5 rubric. Compare against baseline-4d729529 for "
            "pre/post Phase 2-3 quality delta."
        ),
        max_concurrency=1,   # sequential — avoids Groq rate limits
    )

    print(f"\nExperiment complete.")
    print(f"View results at: https://smith.langchain.com")
    print(f"Filter by experiment prefix: {EXPERIMENT_NAME}")
