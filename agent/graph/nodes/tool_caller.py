"""
Tool calling node for the Stock Insight Agent.
This node handles the main tool execution logic.
"""

from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm.llm_setup import llm
import yfinance as yf
import plotly.graph_objects as go
import os
import json

# Import the simplified tools
from tools.date.date_parser_tool import parse_relative_range
from tools.stockprice.stock_analyzer import get_stock_data

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    next: Annotated[str, "The next action to take"]

def get_stock_chart(symbol: str, period: str = "3mo") -> str:
    """Generate an interactive stock chart HTML and return its file path."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist is None or hist.empty:
            return f"No historical data available for {symbol} with period {period}."

        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close']
        )])

        fig.update_layout(
            title=f"{symbol} Stock Price ({period})",
            yaxis_title="Price (USD)",
            xaxis_title="Date"
        )

        charts_dir = os.path.join("app", "chainlit", "static", "charts")
        os.makedirs(charts_dir, exist_ok=True)
        chart_path = os.path.join(charts_dir, f"{symbol}_{period}.html")
        fig.write_html(chart_path)
        # Also return an inline-renderable payload for Chainlit (handled in app/chainlit/app.py)
        return "PLOTLY_JSON:" + fig.to_json()
    except Exception as e:
        return f"Error generating chart for {symbol}: {str(e)}"

# Tool registry
tool_registry = {
    "parse_date_range": parse_relative_range,
    "get_stock_data": get_stock_data,
    "get_stock_chart": get_stock_chart,
}

# LLM prompt for tool selection
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful stock market analysis assistant. Your task is to:
    1. Understand the user's request about stocks
    2. Extract the company name/ticker symbol from the user's message
    3. Determine which tool to use and return a SINGLE JSON response
    
    Available tools:
    - parse_date_range: Parse natural language date ranges like "last 3 weeks" into ISO dates
    - get_stock_data: Get historical stock data (requires ticker, start_date, end_date)
    - get_stock_chart: Generate stock price chart HTML (requires symbol and optional period)
    
    IMPORTANT: Return ONLY ONE tool call at a time. For requests like "how did NVIDIA do over the last week?":
    1. First call parse_date_range to get the date range
    2. Then call get_stock_data with the ticker and parsed dates
    
    Extract the ticker symbol from the company name:
    - "NVIDIA" or "nvidia" → "NVDA"
    - "Apple" or "apple" → "AAPL" 
    - "Microsoft" or "microsoft" → "MSFT"
    - "Google" or "Alphabet" → "GOOGL"
    - "Amazon" or "amazon" → "AMZN"
    - "Tesla" or "tesla" → "TSLA"
    - "Meta" or "Facebook" → "META"
    - If user provides a ticker symbol directly (like "AAPL"), use it as-is
    
    Return ONLY ONE of these formats:
    {{
        "tool": "parse_date_range",
        "parameters": {{"date_range": "last week"}}
    }}
    
    OR
    
    {{
        "tool": "get_stock_data", 
        "parameters": {{"ticker": "NVDA", "start_date": "2024-01-01", "end_date": "2024-01-31"}}
    }}
    
    OR
    
    {{
        "tool": "get_stock_chart",
        "parameters": {{"symbol": "NVDA", "period": "3mo"}}
    }}
    
    Do NOT return multiple tool calls or explanatory text. Return ONLY the JSON object."""),
    MessagesPlaceholder(variable_name="messages"),
])

def call_tool(state: AgentState) -> AgentState:
    """
    Main tool calling node that processes user requests and executes appropriate tools.
    """
    last_message = state["messages"][-1]
    response = llm.invoke(prompt.format(messages=state["messages"]))
    
    try:
        content = response.content
        if isinstance(content, str):
            parsed = json.loads(content)
        elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], str):
            parsed = json.loads(content[0])
        else:
            raise ValueError(f"Unexpected response content type: {type(content)}")
        
        tool_name = parsed.get("tool")
        parameters = parsed.get("parameters", {})
        tool_func = tool_registry.get(tool_name)
        
        if tool_func:
            # Handle different tool signatures
            if tool_name == "parse_date_range":
                # parse_date_range expects a single string parameter
                date_range = parameters.get("date_range", "")
                result = tool_func(date_range)
                
                # After parsing dates, we need to get stock data
                try:
                    # Handle both tuple result and JSON string result
                    if isinstance(result, tuple):
                        # Direct tuple result from parse_relative_range
                        start_date, end_date = result
                        date_result = {"start_date": start_date, "end_date": end_date, "success": True}
                    else:
                        # JSON string result
                        date_result = json.loads(result)
                    
                    if date_result.get("success"):
                        # Extract ticker from the original message using LLM
                        original_msg = state["messages"][0].content
                        ticker_prompt = f"""Extract the stock ticker symbol from this message: "{original_msg}"

Return ONLY a JSON object with the ticker symbol:
{{
    "ticker": "NVDA"
}}

Common mappings:
- "NVIDIA" or "nvidia" → "NVDA"
- "Apple" or "apple" → "AAPL" 
- "Microsoft" or "microsoft" → "MSFT"
- "Google" or "Alphabet" → "GOOGL"
- "Amazon" or "amazon" → "AMZN"
- "Tesla" or "tesla" → "TSLA"
- "Meta" or "Facebook" → "META"
- If a ticker symbol is provided directly, use it as-is"""

                        ticker_response = llm.invoke(ticker_prompt)
                        ticker_content = ticker_response.content
                        if isinstance(ticker_content, str):
                            ticker_parsed = json.loads(ticker_content)
                            ticker = ticker_parsed.get("ticker")
                            
                            if ticker:
                                # Get stock data with the parsed dates using the new tool
                                stock_result = get_stock_data(
                                    ticker, 
                                    date_result["start_date"], 
                                    date_result["end_date"]
                                )
                                result = f"Date range: {date_result['start_date']} to {date_result['end_date']}\n\n{stock_result}"
                except Exception as e:
                    # If anything fails, return the original date parsing result
                    pass
                    
            elif tool_name == "get_stock_data":
                # get_stock_data expects keyword arguments
                result = tool_func(**parameters)
            else:
                result = tool_func(**parameters)
        else:
            result = f"Unknown tool: {tool_name}"
    except Exception as e:
        result = f"Error processing request: {str(e)}\nRaw response: {response.content}"
    
    state["messages"].append(AIMessage(content=result))
    return state
