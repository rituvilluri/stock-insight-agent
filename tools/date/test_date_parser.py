# test_date_parser.py
from tools.date.date_parser_tool import parse_market_aware_range

def test_input(start: str, end: str):
    result = parse_market_aware_range(start_date=start, end_date=end)
    print(f"\n==== Input: {start=} to {end=} ====")
    print(result)

if __name__ == "__main__":
    print("Testing DateParserTool with natural language input...")
    test_input("2 weeks ago", "today")
    test_input("last Monday", "today")
    test_input("last Monday", "next thursday")
