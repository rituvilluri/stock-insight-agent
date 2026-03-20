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


def response_depth_respected(outputs: dict, reference_outputs: dict) -> dict:
    """
    If response_depth=deep, assert all 5 markdown sections are present in response_text.
    Score: 1.0 if depth=quick or all 5 sections present. 0.0 if any section missing.

    Section names match the response_synthesizer deep prompt:
      ## Price Action, ## News & Catalysts, ## Market Sentiment,
      ## SEC Filings, ## Options Activity
    """
    response_depth = outputs.get("response_depth", "quick")
    response_text = outputs.get("response_text") or ""

    if response_depth != "deep":
        return {"key": "response_depth_respected", "score": 1.0, "comment": "Quick mode — sections not required"}

    required_sections = [
        "## Price Action",
        "## News & Catalysts",
        "## Market Sentiment",
        "## SEC Filings",
        "## Options Activity",
    ]
    missing = [s for s in required_sections if s not in response_text]
    score = 1.0 if not missing else 0.0
    return {
        "key": "response_depth_respected",
        "score": score,
        "comment": f"Missing sections: {missing}" if missing else "All 5 sections present",
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
    by synthesizer_context (the actual data the model was given).

    Score: 1.0 = no hallucination detected, 0.0 = hallucination detected.
    Returns None if synthesizer_context or response_text is missing.
    """
    import json
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    response_text = outputs.get("response_text")
    context = outputs.get("synthesizer_context")

    if not response_text or not context:
        return {"key": "hallucination", "score": None, "comment": "Missing response_text or synthesizer_context"}

    judge = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    prompt = f"""You are evaluating whether an AI response hallucinates facts not present in the source data.

<source_data>
{context[:3000]}
</source_data>

<response>
{response_text[:2000]}
</response>

Does the response make any specific factual claims (numbers, dates, events, quotes) that are NOT supported by the source data above?

Reply with JSON only: {{"hallucination": true/false, "reason": "one sentence"}}"""

    try:
        result = judge.invoke([HumanMessage(content=prompt)])
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        detected = parsed.get("hallucination", False)
        return {
            "key": "hallucination",
            "score": 0.0 if detected else 1.0,
            "comment": parsed.get("reason", ""),
        }
    except Exception as e:
        return {"key": "hallucination", "score": None, "comment": f"Judge failed: {e}"}


ALL_EVALUATORS = [
    date_range_accuracy,
    chart_generated_when_requested,
    rag_chunks_retrieved,
    response_depth_respected,
    intent_accuracy,
    source_attribution,
    hallucination,
]
