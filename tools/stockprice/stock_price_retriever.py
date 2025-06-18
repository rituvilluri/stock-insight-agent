import yfinance as yf
import requests
import pandas as pd
from typing import Dict, Union
from datetime import datetime
import dateparser
import pytz

class StockPriceRetriever:
    def __init__(self):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.eastern = pytz.timezone("US/Eastern")

    def run(self, ticker: str, start_date: str, end_date: str) -> str:
        # Parse natural language dates
        parsed_start = dateparser.parse(
            start_date,
            settings={'TIMEZONE': 'US/Eastern', 'RETURN_AS_TIMEZONE_AWARE': True}
        )
        parsed_end = dateparser.parse(
            end_date,
            settings={'TIMEZONE': 'US/Eastern', 'RETURN_AS_TIMEZONE_AWARE': True}
        )

        if not parsed_start or not parsed_end:
            return f"Sorry, I couldn't understand the dates '{start_date}' and '{end_date}'. Please try again."

        # Convert to formatted strings for consistency
        start_str = parsed_start.astimezone(self.eastern).strftime("%Y-%m-%d")
        end_str = parsed_end.astimezone(self.eastern).strftime("%Y-%m-%d")

        result = self._get_from_yfinance(ticker, start_str, end_str)
        if "error" in result:
            print("Yahoo Finance failed, trying Alpha Vantage...")
            result = self._get_from_alpha_vantage(ticker, start_str, end_str)

        if "error" in result:
            return (
                f"Sorry, I couldn't retrieve stock data for {ticker.upper()} "
                f"between {start_str} and {end_str}. Reason: {result['error']}"
            )

        return (
            f"Between {start_str} and {end_str}, the stock {ticker.upper()} opened at "
            f"${result['open_price']} and closed at ${result['close_price']}. "
            f"That's a change of {result['percent_change']}%. "
            f"(Source: {result['source']})"
        )

    def _get_from_yfinance(self, ticker: str, start_date: str, end_date: str) -> dict:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                return {"error": f"No data from Yahoo Finance for {ticker}"}

            # Convert index from UTC to US/Eastern
            df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")

            open_price = df.iloc[0]["Open"]
            close_price = df.iloc[-1]["Close"]
            change = ((close_price - open_price) / open_price) * 100

            return {
                "source": "yfinance",
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "open_price": round(open_price, 2),
                "close_price": round(close_price, 2),
                "percent_change": round(change, 2),
                "ohlc_data": df.reset_index().to_dict(orient="records")
            }

        except Exception as e:
            return {"error": f"yfinance failed: {str(e)}"}

    def _get_from_alpha_vantage(self, ticker: str, start_date: str, end_date: str) -> dict:
        try:
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}"
                f"&apikey={self.alpha_vantage_api_key}&outputsize=full"
            )
            response = requests.get(url)
            data = response.json()

            if "Time Series (Daily)" not in data:
                return {"error": "Alpha Vantage API error or limit exceeded"}

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            df_filtered = df.loc[start_date:end_date]
            if df_filtered.empty:
                return {"error": "No Alpha Vantage data in selected range"}

            open_price = df_filtered.iloc[0]["1. open"]
            close_price = df_filtered.iloc[-1]["4. close"]
            change = ((close_price - open_price) / open_price) * 100

            return {
                "source": "alpha_vantage",
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "open_price": round(open_price, 2),
                "close_price": round(close_price, 2),
                "percent_change": round(change, 2),
                "ohlc_data": df_filtered.reset_index().to_dict(orient="records")
            }

        except Exception as e:
            return {"error": f"Alpha Vantage failed: {str(e)}"}
