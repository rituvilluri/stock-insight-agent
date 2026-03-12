"""
Free News Scraper for Stock Analysis
Uses RSS feeds from major financial news sources - completely free!
"""

import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict
import re
from bs4 import BeautifulSoup
import ssl
import urllib.request

class NewsScraper:
    def __init__(self):
        # Free RSS feeds from major financial news sources
        self.rss_feeds = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'cnbc': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            # Alternative free sources
            'investing_news': 'https://www.investing.com/rss/news.rss',
            'financial_times': 'https://www.ft.com/rss/home'
        }
        
        # Create SSL context that doesn't verify certificates (for development)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
    def fetch_news(self, days_back: int = 7) -> List[Dict]:
        """
        Fetch news articles from RSS feeds
        
        Args:
            days_back: How many days back to fetch news
            
        Returns:
            List of news articles with title, summary, link, date
        """
        all_articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for source, url in self.rss_feeds.items():
            try:
                print(f"Fetching news from {source}...")
                # Use urllib with SSL context
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, context=self.ssl_context) as response:
                    feed_data = response.read()
                feed = feedparser.parse(feed_data)
                
                if feed.bozo:
                    print(f"Warning: {source} feed has issues: {feed.bozo_exception}")
                
                print(f"Found {len(feed.entries)} entries from {source}")
                
                for entry in feed.entries:
                    # Parse the date
                    article_date = self._parse_date(entry.get('published', ''))
                    
                    # Only include recent articles
                    if article_date and article_date >= cutoff_date:
                        article = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'date': article_date,
                            'source': source
                        }
                        all_articles.append(article)
                    elif not article_date:
                        print(f"No valid date for article: {entry.get('title', 'No title')}")
                        
            except Exception as e:
                print(f"Error fetching from {source}: {e}")
                continue
                
        # Sort by date (newest first)
        all_articles.sort(key=lambda x: x['date'], reverse=True)
        return all_articles
    
    def search_news_for_ticker(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """
        Search for news articles related to a specific stock ticker
        
        Args:
            ticker: Stock symbol (e.g., 'NVDA', 'AAPL')
            days_back: How many days back to search
            
        Returns:
            List of relevant news articles
        """
        all_news = self.fetch_news(days_back)
        relevant_articles = []
        
        # Search terms to look for
        search_terms = [
            ticker.upper(),
            ticker.lower(),
            # Add company name variations if needed
        ]
        
        for article in all_news:
            # Check if ticker appears in title or summary
            text_to_search = f"{article['title']} {article['summary']}".lower()
            
            for term in search_terms:
                if term.lower() in text_to_search:
                    relevant_articles.append(article)
                    break
                    
        return relevant_articles
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats from RSS feeds"""
        if not date_str:
            return None
            
        # Common RSS date formats
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        return None

# Test function
def test_news_scraper():
    """Test the news scraper"""
    scraper = NewsScraper()
    
    print("Testing news scraper...")
    
    # Test fetching general news
    print("\n1. Fetching general financial news...")
    general_news = scraper.fetch_news(days_back=3)
    print(f"Found {len(general_news)} articles")
    
    if general_news:
        print(f"Latest article: {general_news[0]['title']}")
        print(f"From: {general_news[0]['source']}")
    
    # Test searching for specific ticker
    print("\n2. Searching for NVDA news...")
    nvda_news = scraper.search_news_for_ticker('NVDA', days_back=7)
    print(f"Found {len(nvda_news)} NVDA-related articles")
    
    if nvda_news:
        print(f"Latest NVDA article: {nvda_news[0]['title']}")
        print(f"Date: {nvda_news[0]['date']}")

if __name__ == "__main__":
    test_news_scraper()
