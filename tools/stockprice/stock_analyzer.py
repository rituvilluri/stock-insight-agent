import yfinance as yf
import requests
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StockAnalyzer:
    """Consolidated stock data retrieval and analysis with fallback support."""
    
    def __init__(self):
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.use_alpha_vantage_fallback = bool(self.alpha_vantage_api_key)
    
    def get_stock_data(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """Get stock data with yfinance primary and Alpha Vantage fallback."""
        try:
            # Try yfinance first
            result = self._get_from_yfinance(ticker, start_date, end_date)
            
            # If yfinance fails and we have Alpha Vantage configured, try fallback
            if not result.get("success") and self.use_alpha_vantage_fallback:
                print(f"Yahoo Finance failed for {ticker}, trying Alpha Vantage...")
                result = self._get_from_alpha_vantage(ticker, start_date, end_date)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to retrieve data for {ticker}: {str(e)}"
            }
    
    def _get_from_yfinance(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """Get stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return {
                    "success": False,
                    "error": f"No data available for {ticker} between {start_date} and {end_date}"
                }
            
            # Calculate key metrics
            open_price = float(hist.iloc[0]['Open'])
            close_price = float(hist.iloc[-1]['Close'])
            high_price = float(hist['High'].max())
            low_price = float(hist['Low'].min())
            volume = int(hist['Volume'].sum())
            percent_change = ((close_price - open_price) / open_price) * 100
            price_change = close_price - open_price
            
            return {
                "success": True,
                "ticker": ticker.upper(),
                "start_date": start_date,
                "end_date": end_date,
                "open_price": round(open_price, 2),
                "close_price": round(close_price, 2),
                "high_price": round(high_price, 2),
                "low_price": round(low_price, 2),
                "volume": volume,
                "percent_change": round(percent_change, 2),
                "price_change": round(price_change, 2),
                "source": "yfinance"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Yahoo Finance failed: {str(e)}"
            }
    
    def _get_from_alpha_vantage(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """Get stock data from Alpha Vantage (fallback)."""
        try:
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}"
                f"&apikey={self.alpha_vantage_api_key}&outputsize=full"
            )
            response = requests.get(url)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                return {
                    "success": False,
                    "error": "Alpha Vantage API error or limit exceeded"
                }
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Filter by date range
            df_filtered = df.loc[start_date:end_date]
            if df_filtered.empty:
                return {
                    "success": False,
                    "error": "No Alpha Vantage data in selected range"
                }
            
            # Calculate metrics
            open_price = df_filtered.iloc[0]["1. open"]
            close_price = df_filtered.iloc[-1]["4. close"]
            high_price = df_filtered["2. high"].max()
            low_price = df_filtered["3. low"].min()
            volume = int(df_filtered["6. volume"].sum())
            percent_change = ((close_price - open_price) / open_price) * 100
            price_change = close_price - open_price
            
            return {
                "success": True,
                "ticker": ticker.upper(),
                "start_date": start_date,
                "end_date": end_date,
                "open_price": round(open_price, 2),
                "close_price": round(close_price, 2),
                "high_price": round(high_price, 2),
                "low_price": round(low_price, 2),
                "volume": volume,
                "percent_change": round(percent_change, 2),
                "price_change": round(price_change, 2),
                "source": "alpha_vantage"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Alpha Vantage failed: {str(e)}"
            }
    
    def get_current_price(self, ticker: str) -> Dict:
        """Get current stock price."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('regularMarketPrice', 0)
            
            return {
                "success": True,
                "ticker": ticker.upper(),
                "current_price": round(current_price, 2),
                "currency": info.get('currency', 'USD'),
                "market_cap": info.get('marketCap'),
                "pe_ratio": info.get('trailingPE'),
                "source": "yfinance"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get current price for {ticker}: {str(e)}"
            }

# Create a global instance
stock_analyzer = StockAnalyzer()

# LangGraph-compatible function (no LangChain Tool wrapper needed)
def get_stock_data(ticker: str, start_date: str, end_date: str) -> str:
    """Get historical stock data for a given ticker and date range."""
    result = stock_analyzer.get_stock_data(ticker, start_date, end_date)
    
    if result.get("success"):
        # Create a human-readable summary
        summary = (
            f"Between {start_date} and {end_date}, {result['ticker']} opened at "
            f"${result['open_price']} and closed at ${result['close_price']}. "
            f"That's a change of {result['percent_change']}% "
            f"(${result['price_change']}). "
            f"High: ${result['high_price']}, Low: ${result['low_price']}, "
            f"Volume: {result['volume']:,} shares. "
            f"(Source: {result['source']})"
        )
        return summary
    else:
        return result.get("error", "Unknown error occurred")

# Keep LangChain Tool for backward compatibility
from langchain.tools import Tool

def stock_price_tool_func(input: str) -> str:
    """LangChain Tool wrapper for backward compatibility."""
    try:
        args = json.loads(input)
        ticker = args["ticker"]
        start_date = args["start_date"]
        end_date = args["end_date"]
        return get_stock_data(ticker, start_date, end_date)
    except Exception as e:
        return f"Error parsing input or retrieving stock price: {e}"

stock_price_tool = Tool(
    name="StockPriceTool",
    func=stock_price_tool_func,
    description=(
        "Get historical stock price data for a given stock ticker symbol. "
        "Input must include 'ticker', 'start_date', and 'end_date' fields in ISO date format. "
        "Provide input as a JSON string. Use the DateParserTool first if you need to convert natural language date ranges into valid ISO dates."
    )
) 