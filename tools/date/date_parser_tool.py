import json
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

def parse_relative_range(text: str) -> Optional[Tuple[str, str]]:
    """Parse relative date ranges like 'last week', '3 months ago'."""
    now = datetime.now()
    
    # Simple patterns for common stock analysis queries
    patterns = {
        r'last (\d+) week': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
        r'last week': lambda m: (now - timedelta(weeks=1), now),
        r'last (\d+) month': lambda m: (now - timedelta(days=30*int(m.group(1))), now),
        r'last month': lambda m: (now - timedelta(days=30), now),
        r'last (\d+) day': lambda m: (now - timedelta(days=int(m.group(1))), now),
        r'yesterday': lambda m: (now - timedelta(days=1), now),
        r'today': lambda m: (now, now),
        r'this week': lambda m: (now - timedelta(days=now.weekday()), now),
        r'this month': lambda m: (now.replace(day=1), now),
    }
    
    for pattern, handler in patterns.items():
        match = re.match(pattern, text.lower())
        if match:
            start, end = handler(match)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    
    return None

def parse_market_aware_range_input(input: str) -> str:
    """Parse date range input and return ISO format dates."""
    try:
        args = json.loads(input)
        date_range = args.get("date_range", "")
        
        # Try to parse as relative range first
        result = parse_relative_range(date_range)
        if result:
            start_date, end_date = result
            return json.dumps({
                "start_date": start_date,
                "end_date": end_date,
                "success": True
            })
        
        # If not a relative range, try to parse individual dates
        start_input = args.get("start_date", "")
        end_input = args.get("end_date", "")
        
        if start_input and end_input:
            # Simple date parsing - you can enhance this if needed
            try:
                start_date = datetime.strptime(start_input, "%Y-%m-%d").strftime("%Y-%m-%d")
                end_date = datetime.strptime(end_input, "%Y-%m-%d").strftime("%Y-%m-%d")
                return json.dumps({
                    "start_date": start_date,
                    "end_date": end_date,
                    "success": True
                })
            except ValueError:
                pass
        
        return json.dumps({
            "error": f"Could not parse date range: {date_range}",
            "success": False
        })

    except Exception as e:
        return json.dumps({
            "error": f"Date parsing failed: {str(e)}",
            "success": False
        })

# Keep the LangChain Tool for backward compatibility, but simplified
from langchain.tools import Tool

date_parser_tool = Tool(
    name="DateParserTool",
    func=parse_market_aware_range_input,
    description=(
        "Convert natural language date ranges like 'last week' or '3 months ago' "
        "into ISO-formatted dates. Input should be a JSON string with 'date_range' key."
    )
)
