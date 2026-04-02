"""
LangSmith custom evaluators for the Stock Insight Agent.

Two usage modes:

1. SDK / run_experiment.py (signature: outputs, reference_outputs):
   Used by client.evaluate() in run_experiment.py. Each function receives the
   graph's output dict and the dataset example's expected outputs.

2. LangSmith UI (signature: run, example):
   If you want to save an evaluator in the LangSmith UI (Experiments → Evaluators
   → + New Evaluator → Python Code), see docs/langsmith-setup.md for the UI
   variant of each function. The logic is identical; only the signature differs.
"""


# ---------------------------------------------------------------------------
# SDK evaluators  (outputs: dict, reference_outputs: dict) -> dict
# ---------------------------------------------------------------------------

def date_range_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    """
    Assert start_date and end_date match the expected values in the dataset example.
    Score: 1.0 if both match, 0.0 otherwise. None if no reference dates provided.
    """
    actual_start = outputs.get("start_date", "")
    actual_end = outputs.get("end_date", "")
    expected_start = reference_outputs.get("start_date", "")
    expected_end = reference_outputs.get("end_date", "")

    if not expected_start or not expected_end:
        return {"key": "date_range_accuracy", "score": None, "comment": "No reference dates in example"}

    match = (actual_start == expected_start) and (actual_end == expected_end)
    return {
        "key": "date_range_accuracy",
        "score": 1.0 if match else 0.0,
        "comment": f"actual=[{actual_start} → {actual_end}] expected=[{expected_start} → {expected_end}]",
    }


def chart_generated_when_requested(outputs: dict, reference_outputs: dict) -> dict:
    """
    If chart_requested=True in output, assert chart_data is non-null.
    Score: 1.0 if chart was not requested, or if chart_data is present.
            0.0 if chart was requested but chart_data is None.
    """
    chart_requested = outputs.get("chart_requested", False)
    chart_data = outputs.get("chart_data")

    if not chart_requested:
        return {"key": "chart_generated_when_requested", "score": 1.0, "comment": "Chart not requested — N/A"}

    score = 1.0 if chart_data else 0.0
    return {
        "key": "chart_generated_when_requested",
        "score": score,
        "comment": "chart_data present" if chart_data else "chart_requested=True but chart_data is None",
    }


def rag_chunks_retrieved(outputs: dict, reference_outputs: dict) -> dict:
    """
    If intent is stock_analysis or general_lookup, assert filing_chunks is non-empty.
    Score: 1.0 if chunks present or intent does not require filings.
            0.0 if intent requires filings and chunks is empty.
    """
    intent = outputs.get("intent", "")
    filing_chunks = outputs.get("filing_chunks") or []

    if intent not in ("stock_analysis", "general_lookup"):
        return {"key": "rag_chunks_retrieved", "score": 1.0, "comment": f"Intent={intent} — RAG not required"}

    score = 1.0 if len(filing_chunks) > 0 else 0.0
    return {
        "key": "rag_chunks_retrieved",
        "score": score,
        "comment": f"{len(filing_chunks)} chunks retrieved",
    }


def intent_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    """
    Assert output intent matches the expected intent in the dataset example.
    Score: 1.0 if match, 0.0 otherwise. None if no reference intent provided.
    """
    actual = outputs.get("intent", "")
    expected = reference_outputs.get("intent", "")

    if not expected:
        return {"key": "intent_accuracy", "score": None, "comment": "No reference intent in example"}

    match = actual == expected
    return {
        "key": "intent_accuracy",
        "score": 1.0 if match else 0.0,
        "comment": f"actual={actual!r} expected={expected!r}",
    }


def source_attribution(outputs: dict, reference_outputs: dict) -> dict:
    """
    Assert sources_cited is non-empty when news_articles or filing_chunks are available.
    Score: 1.0 if sources cited when data available, or no data available.
            0.0 if data was available but no sources were cited.
    """
    news_articles = outputs.get("news_articles") or []
    filing_chunks = outputs.get("filing_chunks") or []
    sources_cited = outputs.get("sources_cited") or []

    data_available = len(news_articles) > 0 or len(filing_chunks) > 0

    if not data_available:
        return {"key": "source_attribution", "score": 1.0, "comment": "No data available — N/A"}

    score = 1.0 if len(sources_cited) > 0 else 0.0
    return {
        "key": "source_attribution",
        "score": score,
        "comment": f"{len(sources_cited)} sources cited, {len(news_articles)} news + {len(filing_chunks)} filing chunks available",
    }


def hallucination(outputs: dict, reference_outputs: dict) -> dict:
    """
    LLM-as-judge: checks whether response_text makes claims not supported
    by synthesizer_context (the full synthesis prompt including the data block
    the model was given).

    Uses llama-3.3-70b-versatile as judge — a different model family from the
    synthesizer (Gemini 2.5 Flash), ensuring independent evaluation.

    Score scale: judge outputs integer 1–5, normalized to 0.0–1.0 in Python.
      5 → 1.0  — Clean: every claim traceable to source data
      4 → 0.75 — Minor: slight paraphrasing, no invented facts
      3 → 0.5  — Moderate: one unverifiable claim or causal assertion
      2 → 0.25 — Significant: multiple unsupported claims or one fabrication
      1 → 0.0  — Critical: invented numbers, non-existent quotes, training-data gap-filling

    Returns None if synthesizer_context or response_text is missing.
    """
    import json
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    response_text = outputs.get("response_text")
    context = outputs.get("synthesizer_context")

    if not response_text or not context:
        return {"key": "hallucination", "score": None, "comment": "Missing response_text or synthesizer_context"}

    judge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024)

    prompt = f"""You are an expert financial AI evaluator auditing a stock analysis agent for hallucinations.

## Background

This agent retrieves financial data — price metrics, news articles, Reddit sentiment, SEC filing excerpts, and options data — then passes it to an LLM with an explicit rule: "Only cite facts present in the DATA block. Do not fill gaps from training knowledge."

You will receive the SOURCE DATA (the exact data block the model was given) and the RESPONSE it produced. Your job is to identify any specific factual claims in the RESPONSE that are not supported by the SOURCE DATA.

## High-Risk Claim Types to Check

**Financial figures** — prices, percentages, volumes, dates, earnings numbers. A close-but-wrong number is still a fabrication (e.g., response says "fell 8.3%" but data says 7.1%).

**Causal claims** — "X happened because of Y." The data must explicitly connect X and Y. Temporal proximity in the data does not justify causal language in the response.

**Quotes and headlines** — if a specific news headline or management quote is cited, it must appear verbatim or near-verbatim in the news articles or SEC filing excerpts in the data.

**Training-data gap-filling** — if the data shows a source was unavailable (e.g., "News Articles: Unavailable — ..."), the response must not reference specific events from that dimension. This is the most serious failure mode.

**Attribution errors** — a claim attributed to a named source (e.g., "Reuters reported...", "the 10-Q states...") must be traceable to that source in the data.

## What Is NOT a Hallucination

- Rounding for readability (7.8% → "roughly 8%"), provided it is not presented as exact
- Reasonable characterization of a trend the data clearly supports
- Standard financial framing and analytical language
- Disclosing that a data source was unavailable (this is correct behavior, not a hallucination)
- Synthesizing multiple data points into a single coherent observation, when the underlying data supports it

## Scoring Rubric

Score the response on a scale of 1 to 5:

5 — Clean. Every specific claim is directly traceable to the source data. Nothing was invented.
4 — Minor. Slight rounding or paraphrasing that is not presented as exact, but no invented facts.
3 — Moderate. One specific claim or causal assertion that cannot be verified against the data.
2 — Significant. Multiple unverifiable claims, or one clear fabrication (wrong figure, non-existent event).
1 — Critical. Invented financial figures presented as fact, quotes absent from the data, or gap-filling from training knowledge where the data explicitly marked a source as unavailable.

## Output Format

Respond with JSON only — no explanation outside the JSON:
{{
  "score": <integer 1 to 5>,
  "verdict": <"clean" | "minor" | "moderate" | "significant" | "critical">,
  "unsupported_claims": [
    {{"claim": "<exact quote from response>", "issue": "<why this is unsupported by the data>"}}
  ],
  "reasoning": "<2-3 sentences summarizing your overall assessment>"
}}

If no hallucinations are found, unsupported_claims must be an empty list [].

--- SOURCE DATA ---
{context[:8000]}
--- END SOURCE DATA ---

--- RESPONSE ---
{response_text[:3000]}
--- END RESPONSE ---"""

    try:
        result = judge.invoke([HumanMessage(content=prompt)])
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        raw_score = int(parsed.get("score", 1))
        score = round((raw_score - 1) / 4, 2)  # normalize 1–5 → 0.0–1.0
        verdict = parsed.get("verdict", "")
        claims = parsed.get("unsupported_claims", [])
        reasoning = parsed.get("reasoning", "")

        claim_summary = f" | {len(claims)} unsupported claim(s): {claims[0]['claim'][:80]}..." if claims else ""
        comment = f"verdict={verdict}{claim_summary} | {reasoning}"

        return {
            "key": "hallucination",
            "score": score,
            "comment": comment,
        }
    except Exception as e:
        return {"key": "hallucination", "score": None, "comment": f"Judge failed: {e}"}


ALL_EVALUATORS = [
    date_range_accuracy,
    chart_generated_when_requested,
    rag_chunks_retrieved,
    intent_accuracy,
    source_attribution,
    hallucination,
]
