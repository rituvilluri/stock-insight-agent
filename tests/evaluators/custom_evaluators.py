"""
LangSmith custom evaluators for the Stock Insight Agent.

These functions are used as custom code evaluators in LangSmith experiments.
To use: copy each function into a LangSmith evaluator via the UI
(Experiments → Evaluators → + New Evaluator → Python Code).

Each function signature: evaluate(run, example) -> dict
  run:     the LangSmith run object (contains inputs and outputs)
  example: the dataset example (contains reference outputs)
  return:  {"key": "evaluator_name", "score": 0.0–1.0}
"""


def date_range_accuracy(run, example) -> dict:
    """
    Assert start_date and end_date match the quarter/period in the query.
    Score: 1.0 if both dates match expected, 0.0 otherwise.

    Expected outputs in dataset example:
      {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}
    """
    outputs = run.outputs or {}
    reference = example.outputs or {}

    actual_start = outputs.get("start_date", "")
    actual_end = outputs.get("end_date", "")
    expected_start = reference.get("start_date", "")
    expected_end = reference.get("end_date", "")

    if not expected_start or not expected_end:
        return {"key": "date_range_accuracy", "score": None, "comment": "No reference dates in example"}

    match = (actual_start == expected_start) and (actual_end == expected_end)
    return {
        "key": "date_range_accuracy",
        "score": 1.0 if match else 0.0,
        "comment": f"actual=[{actual_start} → {actual_end}] expected=[{expected_start} → {expected_end}]",
    }


def chart_generated_when_requested(run, example) -> dict:
    """
    If chart_requested=True in output, assert chart_data is non-null.
    Score: 1.0 if chart_requested=False (nothing to check) or chart_data is present.
            0.0 if chart_requested=True but chart_data is None.
    """
    outputs = run.outputs or {}
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


def rag_chunks_retrieved(run, example) -> dict:
    """
    If intent includes filings (stock_analysis or general_lookup),
    assert len(filing_chunks) > 0.
    Score: 1.0 if chunks present or intent does not require filings.
            0.0 if intent requires filings and chunks is empty.
    """
    outputs = run.outputs or {}
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


def response_depth_respected(run, example) -> dict:
    """
    If response_depth=deep, assert all 5 markdown sections are present in response_text.
    Score: 1.0 if depth=quick (no sections required) or all 5 sections present.
            0.0 if depth=deep and any section is missing.

    Section names match the response_synthesizer deep prompt:
      ## Price Action, ## News & Catalysts, ## Market Sentiment,
      ## SEC Filings, ## Options Activity
    """
    outputs = run.outputs or {}
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


def intent_accuracy(run, example) -> dict:
    """
    Assert output intent matches expected_intent in dataset example.
    Score: 1.0 if match, 0.0 otherwise.

    Expected outputs in dataset example:
      {"intent": "stock_analysis"}  (or whichever intent is expected)
    """
    outputs = run.outputs or {}
    reference = example.outputs or {}

    actual = outputs.get("intent", "")
    expected = reference.get("intent", "")

    if not expected:
        return {"key": "intent_accuracy", "score": None, "comment": "No reference intent in example"}

    match = actual == expected
    return {
        "key": "intent_accuracy",
        "score": 1.0 if match else 0.0,
        "comment": f"actual={actual!r} expected={expected!r}",
    }


def source_attribution(run, example) -> dict:
    """
    Assert len(sources_cited) > 0 when news_articles or filing_chunks are non-empty.
    Score: 1.0 if sources_cited is populated when data is available, or if no data available.
            0.0 if data was available but no sources were cited.
    """
    outputs = run.outputs or {}
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
