# Design Spec: Bug Fixes + UI Polish
**Date:** 2026-03-18
**Status:** Approved

---

## Problem Statement

The Chainlit app is currently broken and visually unpolished:

1. Every query crashes with `InvalidUpdateError` on `user_message` — nothing works end-to-end
2. Streaming never fires because the graph crashes before reaching the synthesizer
3. Chart generation never runs for the same reason
4. The LLM hallucinates dates by pulling from training memory instead of injected data
5. The UI has no theme, no visual hierarchy, and dead CSS that targets non-existent class names
6. No way for users to choose response depth (quick summary vs deep analysis)

---

## Goals

- Fix all crashes so the full graph runs end-to-end
- Make the UI dark, clean, and professionally styled (finance/trading terminal aesthetic)
- Add Quick / Deep Dive chat profiles
- Improve response quality for Deep Dive mode
- Ground the synthesizer against date hallucination
- Style the Plotly chart to match the UI and add analytical value (volume, SMA)

---

## Section 1: Bug Fixes

### 1a. `InvalidUpdateError` — Root Cause and Fix

**Root cause:** Only `retrieve_rag_context` (Node 7) has the bug. `retrieve_news` (Node 5) and `analyze_reddit_sentiment` (Node 6) already return only their owned fields — confirmed by code inspection.

`retrieve_rag_context` returns `{**state, ...owned_fields}` in all 7 of its return paths. When it converges with Nodes 5 and 6 at `synthesize`, LangGraph sees multiple values for `user_message` and other shared fields, throwing `InvalidUpdateError`.

**Fix:** Change all return statements in `retrieve_rag_context` to return only the three fields it owns:
```python
return {"filing_chunks": ..., "filing_ingested": ..., "filing_error": ...}
```

All 7 return paths in the function must be updated. No changes needed to `news_retriever.py` or `reddit_sentiment.py`.

### 1b. Date Hallucination — Synthesizer Prompt Grounding

Add the following grounding instruction to `_build_synthesis_prompt` in `response_synthesizer.py`, immediately before the DATA block:

> "Only reference dates, prices, and events that appear in the DATA block below. If a fact is not in the data, say it is unavailable — do not fill gaps from your training knowledge."

---

## Section 2: Chat Profiles — Quick vs Deep Dive

### `llm/llm_setup.py`
Add `llm_synthesizer_deep` below the existing `llm_synthesizer`:
```python
llm_synthesizer_deep = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2048,
    groq_api_key=_groq_key,
    streaming=True,  # required for token streaming in Deep Dive mode
)
```

### `agent/graph/nodes/state.py`
Add one field to `AgentState` (not `Required` — use `state.get("response_depth", "quick")` pattern throughout):
```python
response_depth: str
# "quick" or "deep". Set by app.py from the Chainlit chat profile.
# Read by Node 9 to select prompt style and token budget.
# Not marked Required — defaults to "quick" if absent.
```

### `app/chainlit/app.py`

**Add chat profiles** (before `on_chat_start`):
```python
@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(name="Quick Analysis", markdown_description="Concise summary with key data points."),
        cl.ChatProfile(name="Deep Dive", markdown_description="Comprehensive analyst brief with structured sections."),
    ]
```

**Update `on_chat_start`** to personalise the welcome message based on selected profile:
```python
profile = cl.user_session.get("chat_profile")
# Adjust welcome message to mention the active mode
```

**Update `on_message`** to inject `response_depth` into initial state:
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

### `agent/graph/nodes/response_synthesizer.py`

In `_build_synthesis_prompt`, add depth routing at the top:
```python
response_depth = state.get("response_depth", "quick")
```

- `"quick"` (default): current prompt unchanged, uses existing `llm_synthesizer` (max_tokens=1024). Any value other than `"deep"` is treated as `"quick"`.
- `"deep"`: replace prompt with structured analyst brief (see below), call `llm_synthesizer_deep` instead of `llm_synthesizer`. The `company`, `ticker`, and `date_context` variables already exist as locals in `_build_synthesis_prompt` — use them directly.

Deep Dive prompt instruction:
> "Generate a comprehensive analyst brief for {company} ({ticker}) covering {date_context}. Structure your response with the following markdown sections: ## Price Action, ## News & Catalysts, ## Market Sentiment, ## SEC Filings, ## Options Activity. Include specific numbers, percentages, and dates from the data. If a section's data is unavailable, state this explicitly under that heading. Only reference dates, prices, and events that appear in the DATA block below."

The LLM call must also switch objects:
```python
llm = llm_synthesizer_deep if response_depth == "deep" else llm_synthesizer
response = llm.invoke(prompt)
```

---

## Section 3: UI Polish

### `.chainlit/config.toml`
Two changes:
- Uncomment the existing `# default_theme = "dark"` line (do not add a duplicate key)
- Change `cot = "full"` to `cot = "hidden"`

### `public/stylesheet.css` — Full rewrite

The existing file uses Chainlit 1.x class names (`.cl-message`, `.cl-chat-container`, etc.) that no longer exist in Chainlit 2.x. Full rewrite targeting only reliable selectors.

**Reliable selectors in Chainlit 2.x:**
- `body` — page background and font
- `@import` — Google Fonts (Inter)
- `:root` — CSS custom properties if Chainlit exposes them
- `.step` — message step containers (stable in Chainlit 2.x)
- `[data-testid="chat-input"]` — input area
- `::-webkit-scrollbar` family — scrollbar styling

**Visual spec:**
- Background: `#0f1117` (off-black)
- Green accent: `#00c896`
- Font: Inter (import from Google Fonts)
- Message containers: `#161b22` background, `2px solid rgba(0,200,150,0.15)` left border, `box-shadow: 0 1px 8px rgba(0,200,150,0.05)` inner glow
- User message containers: `#1c2128` background
- Input area: dark background, `#00c896` focus ring
- Scrollbar: 6px width, `#0f1117` track, `#00c896` thumb at 60% opacity

### `app/chainlit/app.py` — UX improvements

**Loading indicator:**
The existing `streaming_msg = cl.Message(content="", author=AUTHOR)` is sent immediately as an empty message. Change its initial content to `"Analyzing your query..."` (ticker is not available before graph runs — use generic fallback). Once the first streaming token arrives, `stream_token()` overwrites this content naturally. No second message object needed — `streaming_msg` is both the loading indicator and the streamed response.

```python
streaming_msg = cl.Message(content="Analyzing your query...", author=AUTHOR)
await streaming_msg.send()
```

On the first `stream_token()` call, Chainlit replaces the initial content with the streamed tokens.

**Collapsible sources:**
`cl.Text` with `display="side"` renders as a collapsible side panel in Chainlit 2.x — no `unsafe_allow_html` required. Replace the current separate sources message with:
```python
if sources_cited:
    sources_text = "\n\n".join(
        f"{icon} [{label}]({url})" if url else f"{icon} {label}"
        for s in sources_cited
        for label, url, icon in [(
            s.get("title", "Source"),
            s.get("url", ""),
            {"news": "📰", "reddit": "💬", "filing": "📄"}.get(s.get("type", ""), "🔗")
        )]
    )
    await cl.Message(
        content="",
        author=AUTHOR,
        elements=[cl.Text(name="📎 View Sources", content=sources_text, display="side")],
    ).send()
```

If `cl.Text(display="side")` does not render as collapsible in the installed Chainlit version, fall back to a plain message with a `---` divider separating it visually. Do not enable `unsafe_allow_html`.

**Chart label:** Add `"📊 Interactive Chart"` as the `content` of the chart message so it does not appear as a blank bubble:
```python
await cl.Message(
    content="📊 Interactive Chart",
    author=AUTHOR,
    elements=[cl.Plotly(name="Stock Chart", figure=fig, display="inline")],
).send()
```

---

## Section 4: Chart Styling (`agent/graph/nodes/chart_generator.py`)

Apply TradingView-style dark chart aesthetic, consistent with the UI theme.

**Template and background:**
- Use `plotly_dark` template as base
- Override `plot_bgcolor` and `paper_bgcolor` to `#0f1117`

**Candlestick colors:**
- Up candles (increasing): `#00c896` (green accent, fill and line)
- Down candles (decreasing): `#ff4d6d` (muted red, fill and line)

**Volume subplot:**
Volume bars are always shown (behavioral change: previously only rendered when `volume_anomaly.is_anomalous` — that condition is removed and volume is always displayed). Use `make_subplots(rows=2, specs=[[{"secondary_y": False}], [{"secondary_y": False}]])` with row heights `[0.75, 0.25]`. Bar colors match candle direction: green for up days, red for down days, at 40% opacity. Requires `daily_prices` list from `price_data` — if absent, render chart without volume subplot (guard with `if daily_prices`).

**20-day SMA overlay:**
- Compute using pandas: `pd.Series(close_prices).rolling(window=20).mean()`
- Plot only days where the full 20-day window exists (drop leading NaN values with `.dropna()`)
- If fewer than 20 data points exist in `daily_prices`, skip the SMA entirely (no partial line)
- Line color: `#f0b429` (amber), width: 1.5px, name: "20d SMA" in legend

**Grid and axes:**
- Grid lines: `rgba(255,255,255,0.04)` on both axes
- Y-axis tick format: `"$,.2f"`
- X-axis tick format: `"%b %d"`
- Zero line: disabled

**Hover template:**
```
"{ticker} | %{x|%b %d}<br>O: $%{open:.2f}  H: $%{high:.2f}  L: $%{low:.2f}  C: $%{close:.2f}<br>Vol: %{text}<extra></extra>"
```
Pass humanized volume as `text` parameter (use existing `_fmt_volume` helper from `response_synthesizer.py` or inline the same logic).

**Chart title:** `f"{ticker} — {date_context}"` where `date_context` is read from `state.get("date_context", f"{start_date} to {end_date}")`. The `generate_chart` function must read `date_context` from state in addition to `price_data`.

**No watermark:** Set `config={"displaylogo": False}` on the Plotly figure (passed through `cl.Plotly`).

---

## Files Changed

| File | Change |
|---|---|
| `agent/graph/nodes/rag_retriever.py` | Fix all 7 return paths to return only `filing_chunks`, `filing_ingested`, `filing_error` |
| `agent/graph/nodes/response_synthesizer.py` | Deep Dive prompt + grounding instruction + depth routing between two LLM instances |
| `agent/graph/nodes/chart_generator.py` | TradingView-style chart: dark theme, colored candles, volume subplot (always-on), 20d SMA, hover template, date_context title |
| `llm/llm_setup.py` | Add `llm_synthesizer_deep` with `streaming=True`, `max_tokens=2048` |
| `agent/graph/nodes/state.py` | Add `response_depth: str` field (not Required) |
| `app/chainlit/app.py` | Chat profiles, profile-aware welcome, `response_depth` in state, loading indicator, collapsible sources, chart label |
| `.chainlit/config.toml` | Uncomment `default_theme = "dark"`, change `cot` to `"hidden"` |
| `public/stylesheet.css` | Full rewrite — dark green finance theme using Chainlit 2.x-compatible selectors |

---

## Out of Scope

- Custom frontend (deferred to later phase)
- RAG ingestion improvements (Phase 5)
- Additional chart types (OHLC, area)
- Mobile responsiveness beyond Chainlit defaults
- `unsafe_allow_html` — must remain `false`
