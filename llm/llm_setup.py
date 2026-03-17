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

# Legacy alias kept so the existing tool_caller.py and app.py don't break
# while the Phase 1 rebuild is in progress. Remove once workflow.py is rebuilt.
llm = llm_classifier
