from llm.llm_setup import llm
from tools.stockprice.stock_price_tool import stock_price_tool, StockPriceArgs
from tools.date.date_parser_tool import date_parser_tool

def test_llm():
    print("Testing LLM invoke with simple prompt...")
    response = llm.invoke("Hello, how are you?")
    print("LLM response:", response)

def test_stock_price_tool():
    print("Testing StockPriceTool with sample input...")
    args = StockPriceArgs(ticker="AAPL", start_date="2024-06-01", end_date="2024-06-10")
    result = stock_price_tool.func(args)
    print("StockPriceTool result:", result)

def test_date_parser_tool():
    print("\nTesting DateParserTool with sample input...")
    result = date_parser_tool.func(start_date="2 weeks ago", end_date="today")
    print("DateParserTool response:", result)

if __name__ == "__main__":
    test_llm()
    test_stock_price_tool()
    test_date_parser_tool()
