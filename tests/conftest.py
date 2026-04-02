"""
Pytest configuration: disable LangSmith tracing during test runs.

Tests use mocked LLMs and fake price data. Tracing them to the production
LangSmith project pollutes real-run visibility with zero-token mock traces.
"""
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
