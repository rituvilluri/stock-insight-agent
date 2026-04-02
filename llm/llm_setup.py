import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_groq_key = os.getenv("GROQ_API_KEY")
_gemini_key = os.getenv("GEMINI_API_KEY")

# Used by classifier/extractor nodes (Intent, Ticker, Date).
# temperature=0 → deterministic JSON output; no creativity needed here.
# max_tokens=256 → these nodes return small JSON objects, 512 was wasteful.
# Groq llama-3.1-8b: near-zero latency for structured JSON extraction.
llm_classifier = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=256,
    groq_api_key=_groq_key,
)

# Used by the Response Synthesizer (Node 9).
# Gemini 2.5 Flash: reasoning model for narrative synthesis across multi-source data.
# temperature=0.3 → slight variation for natural-sounding prose.
# max_output_tokens=4096 → analyst briefs need room for substantive analysis.
# thinking_budget=1024 → enables internal reasoning so the model connects data
#   points (price action ↔ news ↔ catalysts) rather than just reciting them.
# streaming=True → enables token-by-token delivery via astream_events.
llm_synthesizer = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_output_tokens=4096,
    thinking_budget=1024,
    google_api_key=_gemini_key,
    streaming=True,
)

