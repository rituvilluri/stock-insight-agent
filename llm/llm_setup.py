import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_groq_key = os.getenv("GROQ_API_KEY")
# Used by classifier/extractor nodes (Intent, Ticker, Date).
# temperature=0 → deterministic JSON output; no creativity needed here.
# max_tokens=256 → these nodes return small JSON objects, 512 was wasteful.
llm_classifier = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=256,
    groq_api_key=_groq_key,
)

# Model selection rationale (2026-03-19 review):
# llm_classifier uses llama-3.1-8b-instant for structured JSON extraction
# (intent, ticker, date nodes). Experiment baseline-4d729529 shows 56% intent
# accuracy — root cause is prompt quality (no few-shot examples), not model
# capability. Keeping 8B for now; Agent 1 improves the prompt with few-shot
# examples targeting 85%+. If accuracy remains below 85% after Agent 1,
# escalate llm_classifier to llama-3.3-70b-versatile.

# Used by the Response Synthesizer (Node 9).
# temperature=0.3 → slight variation for natural-sounding prose.
# max_tokens=1024 → narrative responses need more room than JSON classifiers.
# 70b model for better multi-source reasoning quality (Decision 12).
# streaming=True → enables token-by-token delivery via astream_events.
llm_synthesizer = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    groq_api_key=_groq_key,
    streaming=True,
)

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
