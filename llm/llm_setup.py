import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_groq_key = os.getenv("GROQ_API_KEY")
_gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
_gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

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
# Gemini 2.5 Pro: most capable Gemini model; stronger reasoning for multi-source synthesis.
# temperature=0.3 → slight variation for natural-sounding prose.
# max_output_tokens=4096 → analyst briefs need room for substantive analysis.
# thinking_budget=2048 → doubled from Flash baseline; Pro benefits from deeper reasoning
#   passes when connecting price action ↔ news ↔ catalysts ↔ SEC filings.
# vertexai=True → routes through Vertex AI (ADC auth) instead of AI Studio API key.
# streaming=True → enables token-by-token delivery via astream_events.
llm_synthesizer = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.3,
    max_output_tokens=4096,
    thinking_budget=2048,
    vertexai=True,
    project=_gcp_project,
    location=_gcp_location,
    streaming=True,
)

# Used by the Retrieval Planner node (Phase 5).
# Same model as llm_classifier but with more tokens — the planner prompt
# is longer (it receives query context) and the JSON output is 3 fields.
# temperature=0 → deterministic routing decisions.
llm_planner = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    groq_api_key=_groq_key,
)
