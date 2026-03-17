"""
Ground-truth eval dataset for the Stock Insight Agent.

Each example defines:
  - inputs: the AgentState fields passed to the graph (user_message + user_config)
  - reference_outputs: expected values used by evaluators to score the run

Evaluators do NOT do string matching on response_text. They check structural
properties: was the right intent classified? was a ticker resolved? does the
response cite sources? These are stable across LLM response variation.
"""

EXAMPLES = [
    {
        "name": "simple_stock_lookup_nvda",
        "inputs": {
            "user_message": "How did NVIDIA do last month?",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "general_lookup",
            "expected_ticker": "NVDA",
            "expected_company": "NVIDIA",
            "must_have_price_data": True,
            "must_cite_sources": False,  # general_lookup may not always have news
            "must_not_have_ticker_error": True,
        },
    },
    {
        "name": "company_name_resolution_apple",
        "inputs": {
            "user_message": "How did Apple perform over the last 3 weeks?",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "general_lookup",
            "expected_ticker": "AAPL",
            "expected_company": "Apple",
            "must_have_price_data": True,
            "must_not_have_ticker_error": True,
        },
    },
    {
        "name": "earnings_period_stock_analysis",
        "inputs": {
            "user_message": "What happened with NVIDIA around Q2 2024 earnings?",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "stock_analysis",
            "expected_ticker": "NVDA",
            "must_have_price_data": True,
            "must_cite_sources": True,
            "must_not_have_ticker_error": True,
        },
    },
    {
        "name": "sector_specific_tanker_stock",
        "inputs": {
            "user_message": "How did Scorpio Tankers do over the last 2 weeks?",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "general_lookup",
            "expected_ticker": "STNG",
            "must_have_price_data": True,
            "must_not_have_ticker_error": True,
        },
    },
    {
        "name": "chart_request_intent",
        "inputs": {
            "user_message": "Show me a chart of Tesla over the last month",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "chart_request",
            "expected_ticker": "TSLA",
            "must_have_chart_data": True,
            "must_not_have_ticker_error": True,
        },
    },
    {
        "name": "options_view_intent",
        "inputs": {
            "user_message": "What does the options chain look like for AMD right now?",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "options_view",
            "expected_ticker": "AMD",
            "must_not_have_ticker_error": True,
        },
    },
    {
        "name": "unknown_intent_off_topic",
        "inputs": {
            "user_message": "What's the weather like in New York today?",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "unknown",
            "must_have_response_text": True,
            "response_must_not_contain_price": True,
        },
    },
    {
        "name": "date_missing_triggers_clarification",
        "inputs": {
            "user_message": "Tell me about NVIDIA",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_ticker": "NVDA",
            "must_have_response_text": True,
            "date_missing_expected": True,
        },
    },
    {
        "name": "stock_analysis_with_news_attribution",
        "inputs": {
            "user_message": "What happened with Microsoft last month?",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "stock_analysis",
            "expected_ticker": "MSFT",
            "must_have_price_data": True,
            "must_cite_sources": True,
            "must_not_have_ticker_error": True,
        },
    },
    {
        "name": "llm_ticker_fallback_unknown_company",
        "inputs": {
            "user_message": "How did MicroStrategy do last week?",
            "user_config": {},
        },
        "reference_outputs": {
            "expected_intent": "general_lookup",
            "expected_ticker": "MSTR",
            "must_have_price_data": True,
            "must_not_have_ticker_error": True,
        },
    },
]
