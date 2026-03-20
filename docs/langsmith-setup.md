# LangSmith Setup Guide

One-time setup steps per project. These are UI actions that cannot be automated.

## Project Setup

1. Go to LangSmith → Projects → `stock-insight-agent`
2. Confirm `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT=stock-insight-agent` are set in `.env`

## Running Experiments (fully automated)

The preferred approach — no UI setup required:

```bash
source .venv/bin/activate
PYTHONPATH=. python tests/evaluators/run_experiment.py
```

This runs all 6 evaluators against the dataset and uploads results to LangSmith
under the experiment name `baseline-post-review`. View results at smith.langchain.com.

## Custom Evaluators (UI approach — optional)

If you want to save evaluators for reuse from the LangSmith UI:

1. LangSmith → Experiments → Evaluators → **+ New Evaluator**
2. Name: (use the key from the function, e.g. `date_range_accuracy`)
3. Type: **Python Code**
4. Adapt the function from `tests/evaluators/custom_evaluators.py` to use
   the UI signature: `def fn(run, example)` where `run.outputs` and `example.outputs`
   replace the `outputs` and `reference_outputs` parameters
5. Save

Evaluators to create (6 total):
- `date_range_accuracy`
- `chart_generated_when_requested`
- `rag_chunks_retrieved`
- `response_depth_respected`
- `intent_accuracy`
- `source_attribution`

## Prebuilt Evaluators (2 clicks each)

1. LangSmith → Projects → `stock-insight-agent` → Online Evaluation → **+ Add Evaluator**
2. Select **Hallucination** (LLM as judge) → Enable
3. Select **Conciseness** (LLM as judge) → Enable

## Dataset: Add 5 New Examples

Go to LangSmith → Datasets → `stock-insight-agent-evals` (or create it).

Add these 5 examples:

| Input query | Expected outputs |
|-------------|-----------------|
| "Tell me how Nvidia did Q4 2025" | `{"intent": "stock_analysis", "start_date": "2025-10-01", "end_date": "2025-12-31"}` |
| "How did TSLA do Q1 2024?" | `{"intent": "stock_analysis", "start_date": "2024-01-01", "end_date": "2024-03-31"}` |
| "Deep dive on NVDA last month" | `{"intent": "stock_analysis", "response_depth": "deep"}` |
| "What happened with Apple around Q2 2024 earnings?" | `{"intent": "stock_analysis"}` |
| "What about the chart?" (follow-up — test session context) | `{"intent": "chart_request", "chart_requested": true}` |

## Running an Experiment

After setup, run the `baseline-post-review` experiment:

```
From the LangSmith UI: Experiments → Run Experiment
Dataset: stock-insight-agent-evals
Project: stock-insight-agent
Name: baseline-post-review
Evaluators: select all 6 custom + Hallucination + Conciseness
```

Compare `baseline-post-review` against `baseline-4d729529` to produce before/after quality report.
