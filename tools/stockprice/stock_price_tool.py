import json
from langchain.tools import Tool
from tools.stockprice.stock_price_retriever import StockPriceRetriever

stock_price_retriever = StockPriceRetriever()

def stock_price_tool_func(input: str) -> str:
    try:
        args = json.loads(input)
        ticker = args["ticker"]
        start_date = args["start_date"]
        end_date = args["end_date"]
        result = stock_price_retriever.run(ticker=ticker, start_date=start_date, end_date=end_date)
        return json.dumps(result)  # Convert dict result to JSON string
    except Exception as e:
        return json.dumps({"error": f"Error parsing input or retrieving stock price: {e}"})

description = (
    "Use this tool to get historical stock price data for a given stock ticker symbol. "
    "Input must include 'ticker', 'start_date', and 'end_date' fields in ISO date format. "
    "Provide input as a JSON string. Use the DateParserTool first if you need to convert natural language date ranges into valid ISO dates."
)

stock_price_tool = Tool(
    name="StockPriceTool",
    func=stock_price_tool_func,
    description=description
)
