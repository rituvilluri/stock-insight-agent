import json
import re
import pandas as pd
from datetime import datetime, timedelta
from langchain.tools import Tool
import dateparser
import pandas_market_calendars as mcal

nyse = mcal.get_calendar("XNYS")

def get_previous_market_day(dt: datetime) -> datetime:
    schedule = nyse.schedule(start_date=dt - timedelta(days=7), end_date=dt)
    past_days = schedule.index[schedule.index <= dt]
    return past_days[-1].to_pydatetime() if len(past_days) > 0 else dt

def get_next_market_day(dt: datetime) -> datetime:
    schedule = nyse.schedule(start_date=dt, end_date=dt + timedelta(days=7))
    future_days = schedule.index[schedule.index >= dt]
    return future_days[0].to_pydatetime() if len(future_days) > 0 else dt

def resolve_relative_weekday(text: str, base_date: datetime) -> datetime | None:
    match = re.match(r"(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text.lower())
    if not match:
        return None
    direction, weekday_str = match.groups()
    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }
    target = weekday_map[weekday_str]
    current = base_date.weekday()

    if direction == "last":
        return base_date - timedelta(days=(current - target + 7) % 7 or 7)
    if direction == "this":
        return base_date - timedelta(days=current) + timedelta(days=target)
    if direction == "next":
        return base_date + timedelta(days=(target - current + 7) % 7 or 7)
    return None

def parse_market_aware_range_input(input: str) -> str:
    now = datetime.now()
    try:
        args = json.loads(input)
        start_input = args.get("start_date")
        end_input = args.get("end_date")

        parsed_start = resolve_relative_weekday(start_input, now) or dateparser.parse(
            start_input, settings={"RELATIVE_BASE": now}
        )
        parsed_end = resolve_relative_weekday(end_input, now) or dateparser.parse(
            end_input, settings={"RELATIVE_BASE": now}
        )

        if not parsed_start or not parsed_end:
            return json.dumps({"error": "Could not parse one or both of the dates."})

        def is_explicit_year(text: str) -> bool:
            return bool(re.search(r"\b(19|20)\d{2}\b", text))

        MAX_YEAR_ALLOWED = now.year + 1
        if parsed_start.year >= MAX_YEAR_ALLOWED and not is_explicit_year(start_input):
            parsed_start = parsed_start.replace(year=now.year)
        if parsed_end.year >= MAX_YEAR_ALLOWED and not is_explicit_year(end_input):
            parsed_end = parsed_end.replace(year=now.year)

        market_days = set(day.date() for day in nyse.valid_days(
            start_date=now - timedelta(days=730),
            end_date=now + timedelta(days=730)
        ))

        adjusted_start = parsed_start if parsed_start.date() in market_days else get_previous_market_day(parsed_start)
        adjusted_end = parsed_end if parsed_end.date() in market_days else get_next_market_day(parsed_end)

        return json.dumps({
            "start_date": adjusted_start.date().isoformat(),
            "end_date": adjusted_end.date().isoformat()
        })

    except Exception as e:
        return json.dumps({"error": f"Date parsing failed: {str(e)}"})

date_parser_tool = Tool(
    name="DateParserTool",
    func=parse_market_aware_range_input,
    description=(
        "Convert natural language date ranges like 'last Monday to this Friday' or "
        "'2 weeks ago to today' into ISO-formatted NYSE market days. "
        "Input must be a JSON string with keys: 'start_date' and 'end_date'."
    )
)
