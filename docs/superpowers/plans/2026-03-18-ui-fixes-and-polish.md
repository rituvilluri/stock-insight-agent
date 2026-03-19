# UI Fixes and Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `InvalidUpdateError` crash, add Quick/Deep Dive chat profiles, and polish the Chainlit UI with a dark finance theme and TradingView-style charts.

**Architecture:** Fix the parallel fan-out bug in `rag_retriever.py` first (everything else is broken until this lands), then add depth routing through state → synthesizer, then visual polish in chart generator, app.py, config, and CSS.

**Tech Stack:** Python 3.11, LangGraph, Chainlit 2.x, Plotly, pandas (already installed), Groq (llama-3.3-70b-versatile)

**Spec:** `docs/superpowers/specs/2026-03-18-ui-fixes-and-polish-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `agent/graph/nodes/rag_retriever.py` | Modify | Fix all 7 return paths to return only owned fields |
| `agent/graph/nodes/state.py` | Modify | Add `response_depth` field |
| `llm/llm_setup.py` | Modify | Add `llm_synthesizer_deep` |
| `agent/graph/nodes/response_synthesizer.py` | Modify | Depth routing, Deep Dive prompt, grounding |
| `agent/graph/nodes/chart_generator.py` | Modify | TradingView theme, volume always-on, SMA, hover |
| `app/chainlit/app.py` | Modify | Chat profiles, loading indicator, collapsible sources, chart label |
| `.chainlit/config.toml` | Modify | Dark theme, hide CoT |
| `public/stylesheet.css` | Modify | Full rewrite — dark green finance theme |
| `tests/test_rag_retriever.py` | Modify | Verify existing tests still pass (no state spread to check) |
| `tests/test_response_synthesizer.py` | Modify | Add tests for depth routing and grounding |
| `tests/test_chart_generator.py` | Modify | Update tests for volume-always-on, add SMA/theme tests |

---

## Task 1: Fix `InvalidUpdateError` in `rag_retriever.py`

**Files:**
- Modify: `agent/graph/nodes/rag_retriever.py`
- Test: `tests/test_rag_retriever.py`

**Context:** The existing tests only assert on `filing_chunks`, `filing_ingested`, and `filing_error` — they never check for state passthrough fields. After the fix, those tests will still pass unchanged. We add one new test to explicitly verify that state fields are NOT leaked back.

- [ ] **Step 1: Run existing rag_retriever tests to establish baseline**

```bash
cd /path/to/repo && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_rag_retriever.py -v
```

Expected: All tests pass (baseline before changes).

- [ ] **Step 2: Add a test that verifies no state leakage after the fix**

In `tests/test_rag_retriever.py`, add at the end of the file:

```python
def test_returns_only_owned_fields():
    """
    After the parallel fan-out fix, retrieve_rag_context must return only
    its three owned fields. It must NOT spread {**state} back — doing so
    causes InvalidUpdateError when LangGraph merges parallel branches.
    """
    state = {**BASE_STATE, "extra_field_that_should_not_leak": "sentinel"}
    result = retrieve_rag_context(state)
    # Only these three keys are allowed in the return dict
    assert set(result.keys()) == {"filing_chunks", "filing_ingested", "filing_error"}
    assert "extra_field_that_should_not_leak" not in result
    assert "user_message" not in result
    assert "ticker" not in result
```

- [ ] **Step 3: Run the new test to confirm it fails**

```bash
PYTHONPATH=. pytest tests/test_rag_retriever.py::test_returns_only_owned_fields -v
```

Expected: FAIL — currently returns `{**state, ...}` so `user_message` IS in result.

- [ ] **Step 4: Fix all 7 return paths in `retrieve_rag_context`**

In `agent/graph/nodes/rag_retriever.py`, find every `return {**state, ...}` call inside `retrieve_rag_context` (lines ~396–454) and replace each with a return containing only the three owned fields.

There are 7 return paths. Each must become one of these two forms:

**Success form:**
```python
return {"filing_chunks": chunks, "filing_ingested": False, "filing_error": None}
```

**Error/empty form:**
```python
return {"filing_chunks": [], "filing_ingested": False, "filing_error": None}
# or with error:
return {"filing_chunks": [], "filing_ingested": False, "filing_error": str(e)}
```

Map all 7 existing returns:
1. No ticker guard → `{"filing_chunks": [], "filing_ingested": False, "filing_error": None}`
2. No date guard → `{"filing_chunks": [], "filing_ingested": False, "filing_error": None}`
3. No GEMINI_API_KEY → `{"filing_chunks": [], "filing_ingested": False, "filing_error": "GEMINI_API_KEY not configured"}`
4. Cache hit → `{"filing_chunks": chunks, "filing_ingested": False, "filing_error": None}`
5. CIK not found → `{"filing_chunks": [], "filing_ingested": False, "filing_error": None}`
6. No filings found → `{"filing_chunks": [], "filing_ingested": False, "filing_error": None}`
7. Post-ingest return → `{"filing_chunks": chunks, "filing_ingested": total_new > 0, "filing_error": None}`
8. Exception handler → `{"filing_chunks": [], "filing_ingested": False, "filing_error": str(e)}`

(Note: there are 8 paths including the exception handler — update all of them.)

- [ ] **Step 5: Run all rag_retriever tests**

```bash
PYTHONPATH=. pytest tests/test_rag_retriever.py -v
```

Expected: All tests pass, including `test_returns_only_owned_fields`.

- [ ] **Step 6: Run the full test suite to verify no regressions**

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add agent/graph/nodes/rag_retriever.py tests/test_rag_retriever.py
git commit -m "fix: return only owned fields from retrieve_rag_context to resolve InvalidUpdateError"
```

---

## Task 2: Add `response_depth` field to state and `llm_synthesizer_deep`

**Files:**
- Modify: `agent/graph/nodes/state.py`
- Modify: `llm/llm_setup.py`

No new tests needed — `state.py` has no test file (it's a TypedDict definition), and `llm_setup.py` is configuration. Both changes are purely additive.

- [ ] **Step 1: Add `response_depth` to `AgentState` in `state.py`**

In `agent/graph/nodes/state.py`, add the following field after `synthesizer_error` at the bottom of the class (before the closing of `AgentState`):

```python
# -------------------------------------------------------------------------
# Chat profile — set by app.py from the Chainlit chat profile selection
# -------------------------------------------------------------------------

response_depth: str
# "quick" or "deep". Controls synthesizer prompt style and token budget.
# "quick" → current concise prompt, max_tokens=1024 (default)
# "deep"  → structured analyst brief, max_tokens=2048
# Not marked Required — nodes use state.get("response_depth", "quick").
# Set by app.py before the graph starts running.
```

- [ ] **Step 2: Add `llm_synthesizer_deep` to `llm_setup.py`**

In `llm/llm_setup.py`, add the following after the existing `llm_synthesizer` block:

```python
# Used by the Response Synthesizer on the Deep Dive path.
# Same model as llm_synthesizer; higher token budget for structured briefs.
# streaming=True is required — app.py streams tokens via on_chat_model_stream events.
llm_synthesizer_deep = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2048,
    groq_api_key=_groq_key,
    streaming=True,
)
```

- [ ] **Step 3: Run full test suite to verify no regressions**

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: All tests pass (additive changes only).

- [ ] **Step 4: Commit**

```bash
git add agent/graph/nodes/state.py llm/llm_setup.py
git commit -m "feat: add response_depth state field and llm_synthesizer_deep config"
```

---

## Task 3: Update `response_synthesizer.py` — depth routing and grounding

**Files:**
- Modify: `agent/graph/nodes/response_synthesizer.py`
- Test: `tests/test_response_synthesizer.py`

- [ ] **Step 1: Add tests for the new Deep Dive path and grounding**

In `tests/test_response_synthesizer.py`, add the following test cases. Import `llm_synthesizer_deep` at the top alongside the existing import of `llm_synthesizer`:

```python
# At the top of existing imports, verify llm_synthesizer_deep is importable:
from llm.llm_setup import llm_synthesizer, llm_synthesizer_deep
```

Add these tests at the end of the file:

```python
# ---------------------------------------------------------------------------
# Depth routing tests
# ---------------------------------------------------------------------------

@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_quick_depth_uses_llm_synthesizer(mock_llm):
    """response_depth='quick' must call llm_synthesizer, not the deep variant."""
    mock_llm.invoke.return_value = MagicMock(content="Quick analysis result")
    state = _make_state(response_depth="quick")
    result = synthesize_response(state)
    assert mock_llm.invoke.called
    assert result["synthesizer_error"] is None


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer_deep")
def test_deep_depth_uses_llm_synthesizer_deep(mock_deep_llm):
    """response_depth='deep' must call llm_synthesizer_deep."""
    mock_deep_llm.invoke.return_value = MagicMock(content="Deep analysis result")
    state = _make_state(response_depth="deep")
    result = synthesize_response(state)
    assert mock_deep_llm.invoke.called
    assert result["synthesizer_error"] is None


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_unknown_depth_defaults_to_quick(mock_llm):
    """Any value other than 'deep' must fall back to quick path."""
    mock_llm.invoke.return_value = MagicMock(content="Quick result")
    state = _make_state(response_depth="invalid_value")
    result = synthesize_response(state)
    assert mock_llm.invoke.called
    assert result["synthesizer_error"] is None


@patch("agent.graph.nodes.response_synthesizer.llm_synthesizer")
def test_missing_depth_defaults_to_quick(mock_llm):
    """If response_depth is absent from state, default to quick path."""
    mock_llm.invoke.return_value = MagicMock(content="Quick result")
    state = _make_state()  # no response_depth key
    state.pop("response_depth", None)
    result = synthesize_response(state)
    assert mock_llm.invoke.called


def test_deep_prompt_contains_section_headers():
    """Deep Dive prompt must contain the required markdown section headers."""
    from agent.graph.nodes.response_synthesizer import _build_synthesis_prompt
    state = _make_state(response_depth="deep")
    prompt = _build_synthesis_prompt(state)
    for section in ["Price Action", "News & Catalysts", "Market Sentiment", "SEC Filings", "Options Activity"]:
        assert section in prompt, f"Deep Dive prompt missing section: {section}"


def test_grounding_instruction_in_prompt():
    """Both quick and deep prompts must contain the grounding instruction."""
    from agent.graph.nodes.response_synthesizer import _build_synthesis_prompt
    grounding = "do not fill gaps from your training knowledge"
    for depth in ("quick", "deep"):
        state = _make_state(response_depth=depth)
        prompt = _build_synthesis_prompt(state)
        assert grounding in prompt, f"Grounding instruction missing from {depth} prompt"
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
PYTHONPATH=. pytest tests/test_response_synthesizer.py::test_deep_depth_uses_llm_synthesizer_deep tests/test_response_synthesizer.py::test_deep_prompt_contains_section_headers tests/test_response_synthesizer.py::test_grounding_instruction_in_prompt -v
```

Expected: FAIL — these features don't exist yet.

- [ ] **Step 3: Implement depth routing and grounding in `response_synthesizer.py`**

**3a. Add import at top of file:**
```python
from llm.llm_setup import llm_synthesizer, llm_synthesizer_deep
```

**3b. In `_build_synthesis_prompt`, add `response_depth` check at the very top of the function (before `sections = []`):**
```python
response_depth = state.get("response_depth", "quick")
```

**3c. Replace the existing `prompt = (...)` block at the bottom of `_build_synthesis_prompt` with depth-aware construction:**

Note: `structure_instruction` is already defined as a local variable earlier in `_build_synthesis_prompt` (it exists in the current code). Do not remove or redefine it — just use it in the quick-path prompt as shown below.

```python
grounding_instruction = (
    "Only reference dates, prices, and events that appear in the DATA block below. "
    "If a fact is not in the data, say it is unavailable — do not fill gaps from "
    "your training knowledge."
)

if response_depth == "deep":
    prompt = (
        f"You are a stock analysis assistant. Generate a comprehensive analyst brief "
        f"for {company} ({ticker}) covering {date_context}.\n\n"
        f"Rules:\n"
        f"- {grounding_instruction}\n"
        f"- Reference specific data points (prices, percentages, dates) from the data below\n"
        f"- If a section's data is unavailable, state this explicitly under that heading\n"
        f"{snapshot_instruction}\n\n"
        f"Structure your response with these exact markdown sections:\n"
        f"## Price Action\n## News & Catalysts\n## Market Sentiment\n"
        f"## SEC Filings\n## Options Activity\n\n"
        f"--- DATA ---\n{data_block}\n--- END DATA ---\n\n"
        f"Generate the analyst brief now:"
    )
else:
    # quick (default) — any value other than "deep" uses this path
    prompt = (
        f"You are a stock analysis assistant. Generate a concise, factual response "
        f"about {company} ({ticker}) for {date_context}.\n\n"
        f"Rules:\n"
        f"- {grounding_instruction}\n"
        f"- Reference specific data points (prices, percentages, dates) from the data below\n"
        f"- Cite the source for every factual claim\n"
        f"- {structure_instruction}"
        f"{snapshot_instruction}\n\n"
        f"--- DATA ---\n{data_block}\n--- END DATA ---\n\n"
        f"Generate the analysis response now:"
    )

return prompt
```

**3d. In `synthesize_response` (Path 2), route between LLM instances based on depth:**

Replace:
```python
response = llm_synthesizer.invoke(prompt)
```

With:
```python
response_depth = state.get("response_depth", "quick")
llm = llm_synthesizer_deep if response_depth == "deep" else llm_synthesizer
response = llm.invoke(prompt)
```

- [ ] **Step 4: Run all synthesizer tests**

```bash
PYTHONPATH=. pytest tests/test_response_synthesizer.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Run full test suite**

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/graph/nodes/response_synthesizer.py tests/test_response_synthesizer.py
git commit -m "feat: add Deep Dive mode and prompt grounding to response synthesizer"
```

---

## Task 4: Update `chart_generator.py` — TradingView-style chart

**Files:**
- Modify: `agent/graph/nodes/chart_generator.py`
- Test: `tests/test_chart_generator.py`

**Context:** Three existing tests will fail after the volume-always-on change:
- `test_build_chart_no_volume_subplot_when_not_anomalous` → update to assert volume IS always present
- `test_node_no_volume_bar_in_json_when_not_anomalous` → same update
- `test_build_chart_title_flags_anomaly` → anomaly warning is removed from title

- [ ] **Step 1: Update failing tests to reflect the new behavior**

In `tests/test_chart_generator.py`, update these three tests:

```python
def test_build_chart_volume_always_shown():
    """
    Volume subplot is now always shown regardless of anomaly status.
    (Behavioral change from spec: previously only on anomaly detection.)
    """
    # Not anomalous — volume still shown
    anomaly = {"is_anomalous": False, "anomaly_ratio": 1.1}
    fig = _build_chart("NVDA", _make_daily_prices(), volume_anomaly=anomaly)
    trace_types = [t.type for t in fig.data]
    assert "bar" in trace_types

    # No anomaly object at all — volume still shown
    fig2 = _build_chart("NVDA", _make_daily_prices(), volume_anomaly=None)
    trace_types2 = [t.type for t in fig2.data]
    assert "bar" in trace_types2


def test_build_chart_title_uses_ticker_only():
    """
    Title format is '{TICKER} — {date_context}'. Anomaly warning is no
    longer appended to the title (volume is always shown now).
    """
    anomaly = {"is_anomalous": True, "anomaly_ratio": 3.0}
    fig = _build_chart("GME", _make_daily_prices(), volume_anomaly=anomaly)
    # Title must contain ticker but NOT the old anomaly warning text
    assert "GME" in fig.layout.title.text
    assert "unusual" not in fig.layout.title.text.lower()
    assert "volume" not in fig.layout.title.text.lower()


def test_node_volume_bar_always_present_in_json():
    """Volume bar trace must appear in serialised JSON for all queries."""
    # With no anomaly
    result = generate_chart(_make_state(volume_anomaly=None))
    parsed = _parse_chart_json(result["chart_data"])
    trace_types = [t.get("type") for t in parsed["data"]]
    assert "bar" in trace_types

    # With non-anomalous
    anomaly = {"is_anomalous": False, "anomaly_ratio": 1.1}
    result2 = generate_chart(_make_state(volume_anomaly=anomaly))
    parsed2 = _parse_chart_json(result2["chart_data"])
    trace_types2 = [t.get("type") for t in parsed2["data"]]
    assert "bar" in trace_types2
```

Also add new tests for the SMA and theme:

```python
def test_build_chart_sma_trace_present_with_enough_data():
    """20-day SMA line must appear when daily_prices has >= 20 candles."""
    fig = _build_chart("NVDA", _make_daily_prices(n=25), volume_anomaly=None)
    trace_names = [t.name for t in fig.data]
    assert "20d SMA" in trace_names


def test_build_chart_sma_absent_with_fewer_than_20_candles():
    """SMA must be omitted when daily_prices has < 20 candles."""
    fig = _build_chart("NVDA", _make_daily_prices(n=10), volume_anomaly=None)
    trace_names = [t.name for t in fig.data]
    assert "20d SMA" not in trace_names


def test_build_chart_dark_background():
    """Chart background must match the UI dark theme (#0f1117)."""
    fig = _build_chart("NVDA", _make_daily_prices(), volume_anomaly=None)
    assert fig.layout.plot_bgcolor == "#0f1117"
    assert fig.layout.paper_bgcolor == "#0f1117"


def test_build_chart_green_up_candles():
    """Up candle color must be the UI green accent #00c896."""
    fig = _build_chart("NVDA", _make_daily_prices(), volume_anomaly=None)
    candle_trace = next(t for t in fig.data if t.type == "candlestick")
    assert candle_trace.increasing.line.color == "#00c896"


def test_generate_chart_reads_date_context_for_title():
    """Chart title must use date_context from state when available."""
    state = _make_state(date_context="Q2 2024 earnings")
    result = generate_chart(state)
    assert result["chart_error"] is None
    parsed = _parse_chart_json(result["chart_data"])
    assert "Q2 2024 earnings" in parsed["layout"]["title"]["text"]
```

- [ ] **Step 2: Run new/updated tests to confirm they fail**

```bash
PYTHONPATH=. pytest tests/test_chart_generator.py -v
```

Expected: Several failures — the old volume-conditional logic and colors don't match new spec.

- [ ] **Step 3: Rewrite `_build_chart` in `chart_generator.py`**

Replace the entire `_build_chart` function and update `generate_chart` to pass `date_context`:

```python
import pandas as pd

def _fmt_volume(vol) -> str:
    """Humanise a raw volume number into M/B/K string for hover tooltips."""
    if vol is None:
        return "N/A"
    try:
        vol = float(vol)
    except (TypeError, ValueError):
        return str(vol)
    if vol >= 1_000_000_000:
        return f"{vol / 1_000_000_000:.2f}B"
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.2f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return f"{vol:,.0f}"


def _build_chart(
    ticker: str,
    daily_prices: list[dict],
    date_context: str = "",
    volume_anomaly: dict | None = None,
) -> go.Figure:
    """
    Build a TradingView-style Plotly Figure from daily_prices.

    Always includes:
      - Candlestick trace with UI-matched green/red colors
      - Volume subplot (bottom 25% of chart height)
      - 20-day SMA overlay (amber line) when >= 20 data points available

    volume_anomaly is retained as a parameter for API compatibility
    but no longer controls volume visibility.
    """
    dates = [d["date"] for d in daily_prices]
    opens = [d["open"] for d in daily_prices]
    highs = [d["high"] for d in daily_prices]
    lows = [d["low"] for d in daily_prices]
    closes = [d["close"] for d in daily_prices]
    volumes = [d["volume"] for d in daily_prices]

    # Two-row layout: candlestick (75%) + volume (25%) — always
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Volume bar colors: green for up days, red for down days
    bar_colors = [
        "rgba(0,200,150,0.4)" if c >= o else "rgba(255,77,109,0.4)"
        for o, c in zip(opens, closes)
    ]

    # Humanised volume strings for hover tooltip
    vol_labels = [_fmt_volume(v) for v in volumes]

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=ticker,
            increasing=dict(line=dict(color="#00c896"), fillcolor="#00c896"),
            decreasing=dict(line=dict(color="#ff4d6d"), fillcolor="#ff4d6d"),
            hovertemplate=(
                f"{ticker} | %{{x|%b %d}}<br>"
                "O: $%{open:.2f}  H: $%{high:.2f}  L: $%{low:.2f}  C: $%{close:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            name="Volume",
            marker_color=bar_colors,
            text=vol_labels,
            hovertemplate="Vol: %{text}<extra></extra>",
        ),
        row=2, col=1,
    )

    # 20-day SMA overlay (only when enough data)
    if len(closes) >= 20:
        sma_series = pd.Series(closes).rolling(window=20).mean().dropna()
        sma_dates = dates[19:]  # align to the dates where SMA is valid
        fig.add_trace(
            go.Scatter(
                x=sma_dates,
                y=sma_series.tolist(),
                name="20d SMA",
                line=dict(color="#f0b429", width=1.5),
                hovertemplate="20d SMA: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Chart title
    title = f"{ticker} — {date_context}" if date_context else f"{ticker} — {dates[0]} to {dates[-1]}"

    # Layout
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16)),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#e6edf3"),
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=60, b=40),
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)",
        tickformat="$,.2f",
        row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        tickformat=",",
        row=2, col=1,
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        tickformat="%b %d",
    )

    return fig
```

Also update `generate_chart` to pass `date_context` to `_build_chart`:

```python
def generate_chart(state: AgentState) -> AgentState:
    ticker = state.get("ticker", "")
    price_data = state.get("price_data")
    volume_anomaly = state.get("volume_anomaly")
    date_context = state.get("date_context", "")  # NEW — read for chart title

    if not price_data:
        msg = "chart_generator: price_data is missing — cannot generate chart"
        logger.warning(msg)
        return {**state, "chart_data": None, "chart_error": msg}

    daily_prices = price_data.get("daily_prices", [])

    if not daily_prices:
        msg = f"chart_generator: daily_prices list is empty for {ticker}"
        logger.warning(msg)
        return {**state, "chart_data": None, "chart_error": msg}

    try:
        fig = _build_chart(ticker, daily_prices, date_context, volume_anomaly)  # pass date_context
        chart_json = fig.to_json()

        logger.info(
            "generate_chart → %s (%d candles)",
            ticker,
            len(daily_prices),
        )

        return {**state, "chart_data": chart_json, "chart_error": None}

    except Exception as e:
        logger.error("generate_chart failed for %s: %s", ticker, e)
        return {**state, "chart_data": None, "chart_error": str(e)}
```

Add `import pandas as pd` at the top of `chart_generator.py`.

- [ ] **Step 4: Run all chart tests**

```bash
PYTHONPATH=. pytest tests/test_chart_generator.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Run full test suite**

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/graph/nodes/chart_generator.py tests/test_chart_generator.py
git commit -m "feat: TradingView-style dark chart with volume subplot and 20d SMA"
```

---

## Task 5: Update `app.py` — chat profiles, loading indicator, collapsible sources, chart label

**Files:**
- Modify: `app/chainlit/app.py`

No unit tests for `app.py` (Chainlit UI layer — test manually). Verification is done by running the app.

- [ ] **Step 1: Add chat profiles**

Add the following decorator before `on_chat_start`:

```python
@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Quick Analysis",
            markdown_description="Concise summary with key price, news, and sentiment data.",
            icon="⚡",
        ),
        cl.ChatProfile(
            name="Deep Dive",
            markdown_description="Comprehensive analyst brief with structured sections.",
            icon="🔬",
        ),
    ]
```

- [ ] **Step 2: Update `on_chat_start` to reference the active profile**

Replace the existing `on_chat_start` body with:

```python
@cl.on_chat_start
async def start():
    profile = cl.user_session.get("chat_profile") or "Quick Analysis"
    mode_note = "Deep Dive mode — structured analyst briefs." if profile == "Deep Dive" else "Quick Analysis mode — concise summaries."
    await cl.Message(
        content=(
            f"**Welcome to the Stock Insight Agent** · {mode_note}\n\n"
            "Ask me about any stock's performance over a time period. Examples:\n\n"
            "- *How did NVIDIA do last month?*\n"
            "- *Show me a chart of Tesla from Q1 2024*\n"
            "- *What happened with Apple around Q2 2024 earnings?*\n\n"
            "I can retrieve price data, generate interactive charts, "
            "and provide a narrative analysis with source citations."
        ),
        author=AUTHOR,
    ).send()
```

- [ ] **Step 3: Inject `response_depth` into initial state in `on_message`**

Replace the `initial_state` construction:

```python
profile = cl.user_session.get("chat_profile") or "Quick Analysis"
response_depth = "deep" if profile == "Deep Dive" else "quick"
initial_state = {
    "user_message": message.content,
    "user_config": {},
    "response_depth": response_depth,
    **last_context,
}
```

- [ ] **Step 4: Update the loading indicator (pre-stream message)**

Replace:
```python
streaming_msg = cl.Message(content="", author=AUTHOR)
```

With:
```python
streaming_msg = cl.Message(content="Analyzing your query...", author=AUTHOR)
```

The `stream_token()` calls will overwrite this content as tokens arrive.

- [ ] **Step 5: Replace sources message with collapsible `cl.Text` element**

Replace the entire sources block (lines ~128–142):

```python
sources_cited = final_state.get("sources_cited") or []
if sources_cited:
    source_lines = []
    for s in sources_cited:
        label = s.get("title", "Source")
        url = s.get("url", "")
        source_type = s.get("type", "")
        icon = {"news": "📰", "reddit": "💬", "filing": "📄"}.get(source_type, "🔗")
        if url:
            source_lines.append(f"{icon} [{label}]({url})")
        else:
            source_lines.append(f"{icon} {label}")

    sources_content = "\n\n".join(source_lines)
    await cl.Message(
        content="",
        author=AUTHOR,
        elements=[
            cl.Text(
                name="📎 View Sources",
                content=sources_content,
                display="side",
            )
        ],
    ).send()
```

- [ ] **Step 6: Suppress Plotly watermark in chart**

In `chart_generator.py`, `cl.Plotly` does not accept a `config` parameter directly — the `displaylogo` setting is applied on the Plotly Figure object itself before serialisation. Add this line inside `generate_chart` after `fig = _build_chart(...)`:

```python
fig.update_layout(showlegend=True)  # already set, but also add:
# Suppress Plotly logo — set via config in to_json is not supported;
# instead disable via layout modebar
fig.layout.update({"modebar": {"remove": ["toImage", "sendDataToCloud"]}})
```

Actually the simplest approach: pass `config={"displaylogo": False}` when creating the `cl.Plotly` element in `app.py`. Update the chart message in the next step.

- [ ] **Step 7: Add label to chart message and suppress Plotly logo**

In the chart rendering block, replace:
```python
await cl.Message(
    content="",
    author=AUTHOR,
    elements=[cl.Plotly(name="Stock Chart", figure=fig, display="inline")],
).send()
```

With:
```python
await cl.Message(
    content="📊 Interactive Chart",
    author=AUTHOR,
    elements=[cl.Plotly(name="Stock Chart", figure=fig, display="inline", config={"displaylogo": False})],
).send()
```

- [ ] **Step 7: Run full test suite to verify no regressions**

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: All tests pass (app.py has no unit tests).

- [ ] **Step 8: Commit**

```bash
git add app/chainlit/app.py
git commit -m "feat: add Quick/Deep Dive chat profiles and UI improvements to Chainlit app"
```

---

## Task 6: Config and CSS — dark theme + green accent

**Files:**
- Modify: `.chainlit/config.toml`
- Modify: `public/stylesheet.css`

No tests — verify manually by running the app.

- [ ] **Step 1: Update `.chainlit/config.toml`**

Two changes:
1. Uncomment `# default_theme = "dark"` → `default_theme = "dark"` (do NOT add a duplicate key)
2. Change `cot = "full"` → `cot = "hidden"`

- [ ] **Step 2: Rewrite `public/stylesheet.css`**

Replace the entire file contents with:

```css
/* Stock Insight Agent — Dark Finance Theme
   Targets Chainlit 2.x-compatible selectors only.
   Class names like .cl-message do not exist in Chainlit 2.x — do not add them.
*/

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0f1117;
    color: #e6edf3;
}

/* ── Message step containers ──────────────────────────────────────────── */

.step {
    background-color: #161b22;
    border: 1px solid rgba(0, 200, 150, 0.12);
    border-radius: 10px;
    box-shadow: 0 0 12px rgba(0, 200, 150, 0.04);
    margin: 6px 0;
    padding: 2px 0;
    transition: border-color 0.2s ease;
}

.step:hover {
    border-color: rgba(0, 200, 150, 0.25);
}

/* ── Chat input ───────────────────────────────────────────────────────── */

[data-testid="chat-input"] {
    background-color: #161b22 !important;
    border: 1px solid rgba(0, 200, 150, 0.2) !important;
    border-radius: 12px !important;
    color: #e6edf3 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="chat-input"]:focus-within {
    border-color: #00c896 !important;
    box-shadow: 0 0 0 3px rgba(0, 200, 150, 0.12) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: #0f1117;
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 200, 150, 0.4);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 200, 150, 0.7);
}

/* ── Links ────────────────────────────────────────────────────────────── */

a {
    color: #00c896;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
    color: #00e6b0;
}

/* ── Code blocks ─────────────────────────────────────────────────────── */

code {
    background-color: #1c2128;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.88em;
    color: #00c896;
}

pre code {
    background: transparent;
    border: none;
    padding: 0;
    color: inherit;
}

/* ── Accent elements (buttons, active states) ─────────────────────────── */

button[data-testid="send-button"] {
    background-color: #00c896 !important;
    border-radius: 8px !important;
    transition: background-color 0.2s ease, transform 0.1s ease;
}

button[data-testid="send-button"]:hover {
    background-color: #00e6b0 !important;
    transform: scale(1.04);
}
```

- [ ] **Step 3: Verify the app runs and looks correct**

```bash
PYTHONPATH=. chainlit run app/chainlit/app.py
```

Manual checks:
- [ ] Dark background loads immediately
- [ ] Chat profile picker appears (Quick Analysis / Deep Dive)
- [ ] Profile-aware welcome message shown on start
- [ ] Query runs without crashing (no InvalidUpdateError)
- [ ] Response streams token-by-token
- [ ] "📎 View Sources" collapsible panel appears (if sources present)
- [ ] Chart renders with dark theme, green/red candles, volume bars
- [ ] No CoT step indicators visible during graph execution

- [ ] **Step 4: Commit**

```bash
git add .chainlit/config.toml public/stylesheet.css
git commit -m "feat: dark finance theme with green accent — Chainlit config and CSS"
```

---

## Final Verification

- [ ] **Run full test suite one last time**

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Smoke test the app end-to-end**

Ask: *"How did NVDA do last month?"* in Quick Analysis mode.
Ask: *"Deep dive on TSLA Q2 2024"* in Deep Dive mode.

Verify:
- Quick mode: concise paragraph response, streaming, sources panel, chart
- Deep mode: structured markdown sections, longer response, same visual polish
