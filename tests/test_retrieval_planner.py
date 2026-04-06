"""
Tests for the Retrieval Planner node (Phase 5).

Strategy:
- plan_retrieval() node: mock llm_planner to control JSON output and
  verify flag extraction, markdown-fence stripping, and fallback on failure.
- route_after_plan_retrieval() router: pure Python — no mocking needed.
  Verify Send() targets and the all-False safety-net fallback.
"""

from unittest.mock import MagicMock, patch

import pytest
from langgraph.types import Send

from agent.graph.nodes.retrieval_planner import plan_retrieval
from agent.graph.workflow import route_after_plan_retrieval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> dict:
    base = {
        "user_message": "How did NVDA perform last quarter?",
        "user_config": {},
        "ticker": "NVDA",
        "intent": "stock_analysis",
        "date_context": "last quarter",
    }
    base.update(kwargs)
    return base


def _mock_llm_response(content: str):
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# plan_retrieval node — happy path
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_plan_all_true(mock_llm):
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"fetch_news": true, "fetch_sentiment": true, "fetch_rag": true}'
    )
    result = plan_retrieval(_make_state())
    assert result["retrieval_plan"] == {
        "fetch_news": True, "fetch_sentiment": True, "fetch_rag": True
    }
    assert result["planner_error"] is None


@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_plan_rag_disabled_for_simple_query(mock_llm):
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"fetch_news": true, "fetch_sentiment": true, "fetch_rag": false}'
    )
    result = plan_retrieval(_make_state())
    assert result["retrieval_plan"]["fetch_rag"] is False
    assert result["retrieval_plan"]["fetch_news"] is True
    assert result["planner_error"] is None


@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_plan_sentiment_disabled(mock_llm):
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"fetch_news": true, "fetch_sentiment": false, "fetch_rag": true}'
    )
    result = plan_retrieval(_make_state())
    assert result["retrieval_plan"]["fetch_sentiment"] is False
    assert result["retrieval_plan"]["fetch_rag"] is True


@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_plan_strips_markdown_fences(mock_llm):
    mock_llm.invoke.return_value = _mock_llm_response(
        "```json\n{\"fetch_news\": true, \"fetch_sentiment\": false, \"fetch_rag\": false}\n```"
    )
    result = plan_retrieval(_make_state())
    assert result["retrieval_plan"]["fetch_news"] is True
    assert result["retrieval_plan"]["fetch_sentiment"] is False
    assert result["retrieval_plan"]["fetch_rag"] is False
    assert result["planner_error"] is None


@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_plan_missing_keys_default_to_true(mock_llm):
    """Partial JSON — missing keys should default to True (safe fallback)."""
    mock_llm.invoke.return_value = _mock_llm_response('{"fetch_news": true}')
    result = plan_retrieval(_make_state())
    assert result["retrieval_plan"]["fetch_news"] is True
    assert result["retrieval_plan"]["fetch_sentiment"] is True
    assert result["retrieval_plan"]["fetch_rag"] is True


# ---------------------------------------------------------------------------
# plan_retrieval node — failure / fallback
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_plan_llm_exception_activates_all(mock_llm):
    """LLM failure must not crash — all nodes activated as fallback."""
    mock_llm.invoke.side_effect = Exception("Groq quota exceeded")
    result = plan_retrieval(_make_state())
    assert result["retrieval_plan"] == {
        "fetch_news": True, "fetch_sentiment": True, "fetch_rag": True
    }
    assert result["planner_error"] is not None
    assert "quota" in result["planner_error"]


@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_plan_invalid_json_activates_all(mock_llm):
    """Malformed JSON must fall back to all-active plan."""
    mock_llm.invoke.return_value = _mock_llm_response("not valid json {{")
    result = plan_retrieval(_make_state())
    assert result["retrieval_plan"] == {
        "fetch_news": True, "fetch_sentiment": True, "fetch_rag": True
    }
    assert result["planner_error"] is not None


# ---------------------------------------------------------------------------
# plan_retrieval — state passthrough
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_plan_preserves_existing_state(mock_llm):
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"fetch_news": true, "fetch_sentiment": true, "fetch_rag": false}'
    )
    state = _make_state(ticker="AAPL", intent="stock_analysis", company_name="Apple")
    result = plan_retrieval(state)
    assert result["ticker"] == "AAPL"
    assert result["intent"] == "stock_analysis"
    assert result["company_name"] == "Apple"


# ---------------------------------------------------------------------------
# route_after_plan_retrieval — pure Python, no mocking needed
# ---------------------------------------------------------------------------

def test_router_all_true_returns_three_sends():
    state = _make_state(retrieval_plan={"fetch_news": True, "fetch_sentiment": True, "fetch_rag": True})
    result = route_after_plan_retrieval(state)
    assert len(result) == 3
    targets = {s.node for s in result}
    assert targets == {"retrieve_news", "reddit_sentiment", "retrieve_rag"}


def test_router_rag_false_omits_rag():
    state = _make_state(retrieval_plan={"fetch_news": True, "fetch_sentiment": True, "fetch_rag": False})
    result = route_after_plan_retrieval(state)
    targets = {s.node for s in result}
    assert "retrieve_rag" not in targets
    assert "retrieve_news" in targets
    assert "reddit_sentiment" in targets


def test_router_sentiment_false_omits_sentiment():
    state = _make_state(retrieval_plan={"fetch_news": True, "fetch_sentiment": False, "fetch_rag": True})
    result = route_after_plan_retrieval(state)
    targets = {s.node for s in result}
    assert "reddit_sentiment" not in targets
    assert {"retrieve_news", "retrieve_rag"}.issubset(targets)


def test_router_only_news():
    state = _make_state(retrieval_plan={"fetch_news": True, "fetch_sentiment": False, "fetch_rag": False})
    result = route_after_plan_retrieval(state)
    assert len(result) == 1
    assert result[0].node == "retrieve_news"


def test_router_all_false_safety_net_activates_all():
    """All flags False would deadlock synthesize — safety net must fire."""
    state = _make_state(retrieval_plan={"fetch_news": False, "fetch_sentiment": False, "fetch_rag": False})
    result = route_after_plan_retrieval(state)
    assert len(result) == 3
    targets = {s.node for s in result}
    assert targets == {"retrieve_news", "reddit_sentiment", "retrieve_rag"}


def test_router_missing_plan_defaults_all_active():
    """No retrieval_plan in state — should behave as all True."""
    state = _make_state()  # no retrieval_plan key
    result = route_after_plan_retrieval(state)
    assert len(result) == 3


def test_router_returns_send_objects():
    """Verify the router returns proper Send() instances, not strings."""
    state = _make_state(retrieval_plan={"fetch_news": True, "fetch_sentiment": True, "fetch_rag": True})
    result = route_after_plan_retrieval(state)
    for item in result:
        assert isinstance(item, Send)


# ---------------------------------------------------------------------------
# Prompt content and intent forwarding
# ---------------------------------------------------------------------------

def test_planner_prompt_contains_general_lookup_rag_rule():
    """
    The system prompt must contain the decision rule that disables fetch_rag
    for general_lookup intent. If this rule is removed, the planner will fetch
    RAG on every general_lookup query, wasting ChromaDB API budget.
    """
    from agent.graph.nodes.retrieval_planner import _PLANNER_SYSTEM_PROMPT

    assert "general_lookup" in _PLANNER_SYSTEM_PROMPT, (
        "System prompt must reference general_lookup in the fetch_rag decision rule"
    )
    assert "fetch_rag" in _PLANNER_SYSTEM_PROMPT, (
        "System prompt must contain fetch_rag decision rule"
    )


@patch("agent.graph.nodes.retrieval_planner.llm_planner")
def test_planner_forwards_intent_to_llm(mock_llm):
    """
    The human message sent to the LLM must include the intent field so the
    LLM can apply the correct fetch_rag decision for general_lookup queries.
    If intent is not forwarded, the LLM cannot distinguish query types.
    """
    mock_llm.invoke.return_value = _mock_llm_response(
        '{"fetch_news": true, "fetch_sentiment": false, "fetch_rag": false}'
    )
    state = _make_state(intent="general_lookup")
    plan_retrieval(state)

    call_args = mock_llm.invoke.call_args[0][0]
    human_message_content = call_args[1].content
    assert "general_lookup" in human_message_content, (
        f"LLM human message must include 'general_lookup'. Got: {human_message_content!r}"
    )
