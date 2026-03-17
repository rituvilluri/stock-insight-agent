"""
LangSmith eval runner for the Stock Insight Agent.

Usage:
    source .venv/bin/activate
    PYTHONPATH=. python evals/run_evals.py

What this does:
1. Creates (or updates) a LangSmith dataset named "stock-insight-agent-baseline"
2. Upserts the 10 ground-truth examples
3. Runs an experiment: invokes the graph for each example
4. Scores each run with 5 evaluators
5. Uploads results — view at https://smith.langchain.com

Evaluators (all return a score of 0 or 1):
  - intent_accuracy:     classified intent matches expected_intent
  - ticker_accuracy:     resolved ticker matches expected_ticker
  - has_price_data:      price_data is non-null when must_have_price_data=True
  - source_attribution:  sources_cited is non-empty when must_cite_sources=True
  - no_ticker_error:     ticker_error is None when must_not_have_ticker_error=True
"""

import logging
import os

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from agent.graph.workflow import app as graph
from evals.dataset import EXAMPLES

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "stock-insight-agent-baseline"


# ---------------------------------------------------------------------------
# Graph target function
# ---------------------------------------------------------------------------

def run_graph(inputs: dict) -> dict:
    """Invoke the graph and return the full state as the output."""
    return graph.invoke(inputs)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def eval_intent_accuracy(run, example) -> dict:
    expected = example.outputs.get("expected_intent")
    if expected is None:
        return {"key": "intent_accuracy", "score": None, "comment": "no expected intent"}
    actual = (run.outputs or {}).get("intent", "")
    score = 1 if actual == expected else 0
    return {
        "key": "intent_accuracy",
        "score": score,
        "comment": f"expected={expected}, got={actual}",
    }


def eval_ticker_accuracy(run, example) -> dict:
    expected = example.outputs.get("expected_ticker")
    if expected is None:
        return {"key": "ticker_accuracy", "score": None, "comment": "no expected ticker"}
    actual = (run.outputs or {}).get("ticker", "")
    score = 1 if actual == expected else 0
    return {
        "key": "ticker_accuracy",
        "score": score,
        "comment": f"expected={expected}, got={actual}",
    }


def eval_has_price_data(run, example) -> dict:
    if not example.outputs.get("must_have_price_data"):
        return {"key": "has_price_data", "score": None, "comment": "not required"}
    price_data = (run.outputs or {}).get("price_data")
    score = 1 if price_data is not None else 0
    return {"key": "has_price_data", "score": score}


def eval_source_attribution(run, example) -> dict:
    if not example.outputs.get("must_cite_sources"):
        return {"key": "source_attribution", "score": None, "comment": "not required"}
    sources = (run.outputs or {}).get("sources_cited") or []
    score = 1 if len(sources) > 0 else 0
    return {
        "key": "source_attribution",
        "score": score,
        "comment": f"{len(sources)} sources cited",
    }


def eval_no_ticker_error(run, example) -> dict:
    if not example.outputs.get("must_not_have_ticker_error"):
        return {"key": "no_ticker_error", "score": None, "comment": "not required"}
    ticker_error = (run.outputs or {}).get("ticker_error")
    score = 1 if ticker_error is None else 0
    return {
        "key": "no_ticker_error",
        "score": score,
        "comment": f"ticker_error={ticker_error}",
    }


# ---------------------------------------------------------------------------
# Dataset setup
# ---------------------------------------------------------------------------

def upsert_dataset(client: Client) -> str:
    """Create the dataset if it doesn't exist, then upsert all examples."""
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        dataset = datasets[0]
        logger.info("Using existing dataset: %s (%s)", DATASET_NAME, dataset.id)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Baseline eval dataset for the Stock Insight Agent. "
                "Covers all intent types, ticker resolution layers, "
                "date parsing, and synthesis quality."
            ),
        )
        logger.info("Created dataset: %s (%s)", DATASET_NAME, dataset.id)

    existing = {
        ex.metadata.get("name"): ex
        for ex in client.list_examples(dataset_id=dataset.id)
    }

    for ex in EXAMPLES:
        name = ex["name"]
        if name in existing:
            client.update_example(
                example_id=existing[name].id,
                inputs=ex["inputs"],
                outputs=ex["reference_outputs"],
                metadata={"name": name},
            )
            logger.info("Updated example: %s", name)
        else:
            client.create_example(
                inputs=ex["inputs"],
                outputs=ex["reference_outputs"],
                dataset_id=dataset.id,
                metadata={"name": name},
            )
            logger.info("Created example: %s", name)

    return dataset.id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        raise EnvironmentError("LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) not set in .env")

    client = Client()
    dataset_id = upsert_dataset(client)

    logger.info("Running evaluation experiment...")
    results = evaluate(
        run_graph,
        data=DATASET_NAME,
        evaluators=[
            eval_intent_accuracy,
            eval_ticker_accuracy,
            eval_has_price_data,
            eval_source_attribution,
            eval_no_ticker_error,
        ],
        experiment_prefix="baseline",
        metadata={
            "phase": "2",
            "classifier_model": "llama-3.1-8b-instant",
            "synthesizer_model": "llama-3.3-70b-versatile",
        },
    )

    logger.info("Evaluation complete. View results at https://smith.langchain.com")
    logger.info("Experiment: %s", results.experiment_name)
