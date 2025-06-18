import chainlit as cl
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from llm.llm_setup import llm
import json

# Import the new simplified tools
from tools.date.date_parser_tool import parse_relative_range
from tools.stockprice.stock_analyzer import get_stock_data

# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    next: Annotated[str, "The next action to take"]

# Define tools for stock analysis using new simplified tools
@tool
def parse_date_range(date_range: str) -> str:
    """Parse natural language date ranges into ISO format dates."""
    try:
        # Use the simplified date parser
        result = parse_relative_range(date_range)
        if result:
            start_date, end_date = result
            return json.dumps({
                "start_date": start_date,
                "end_date": end_date,
                "success": True
            })
        else:
            return json.dumps({
                "error": f"Could not parse date range: {date_range}",
                "success": False
            })
    except Exception as e:
        return json.dumps({
            "error": f"Date parsing failed: {str(e)}",
            "success": False
        })

@tool
def get_stock_data_tool(ticker: str, start_date: str, end_date: str) -> str:
    """Get historical stock data for a given ticker and date range."""
    try:
        # Use the new consolidated stock analyzer
        result = get_stock_data(ticker, start_date, end_date)
        return result
    except Exception as e:
        return f"Error retrieving stock data: {str(e)}"

@tool
def get_stock_chart(symbol: str, period: str = "1mo") -> str:
    """Generate a stock price chart for a given symbol and period."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close']
        )])
        
        fig.update_layout(
            title=f"{symbol} Stock Price",
            yaxis_title="Price (USD)",
            xaxis_title="Date"
        )
        
        # Save the chart as an HTML file
        chart_path = f"app/chainlit/static/charts/{symbol}_{period}.html"
        fig.write_html(chart_path)
        return chart_path
    except Exception as e:
        return f"Error generating chart for {symbol}: {str(e)}"

# Tool registry for dispatch
tool_registry = {
    "parse_date_range": parse_date_range,
    "get_stock_data": get_stock_data_tool,
    "get_stock_chart": get_stock_chart,
}

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful stock market analysis assistant. Your task is to:
    1. Understand the user's request about stocks
    2. Extract the company name/ticker symbol from the user's message
    3. Determine which tool to use and return a SINGLE JSON response
    
    Available tools:
    - parse_date_range: Parse natural language date ranges like "last 3 weeks" into ISO dates
    - get_stock_data: Get historical stock data (requires ticker, start_date, end_date)
    - get_stock_chart: Generate stock price chart (requires symbol and optional period)
    
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
    
    Do NOT return multiple tool calls or explanatory text. Return ONLY the JSON object."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Define the nodes
def should_continue(state: AgentState) -> str:
    # If the last message is from the AI, we're done
    if isinstance(state["messages"][-1], AIMessage):
        return "end"
    return "continue"

def call_tool(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    response = llm.invoke(prompt.format(messages=state["messages"]))
    try:
        import json
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
            elif tool_name == "get_stock_chart":
                # get_stock_chart expects keyword arguments
                result = tool_func(**parameters)
            else:
                result = tool_func(**parameters)
        else:
            result = f"Unknown tool: {tool_name}"
    except Exception as e:
        result = f"Error processing request: {str(e)}\nRaw response: {response.content}"
    
    state["messages"].append(AIMessage(content=result))
    return state

workflow = StateGraph(AgentState)
workflow.add_node("call_tool", call_tool)
workflow.add_conditional_edges(
    "call_tool",
    should_continue,
    {
        "continue": "call_tool",
        "end": END
    }
)
workflow.set_entry_point("call_tool")
app = workflow.compile()

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Stock Insight Agent! How can I help you analyze stocks today?").send()

@cl.on_message
async def main(message: cl.Message):
    state = {
        "messages": [HumanMessage(content=message.content)],
        "next": "call_tool"
    }
    result = app.invoke(state)
    last_message = result["messages"][-1]
    if isinstance(last_message.content, str) and last_message.content.endswith('.html'):
        await cl.Message(content="Here's the stock chart:").send()
        await cl.Message(content=last_message.content).send()
    else:
        await cl.Message(content=last_message.content).send() 