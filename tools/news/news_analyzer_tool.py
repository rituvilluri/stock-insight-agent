"""
News Analysis Tool for Stock Insight Agent
Integrates with vector store for semantic search and RAG capabilities
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

# Import our vector store (we'll create this next)
from data.vector_store.news_vector_store import NewsVectorStore

class NewsAnalyzerTool:
    """
    Tool for analyzing news articles with RAG capabilities
    """

    def __init__(self):
        """Initialize the news analyzer with vector store"""
        self.vector_store = NewsVectorStore()
        self.news_api_key = os.getenv('NEWS_API_KEY')  # Optional for free tier

    def fetch_news_from_api(self, query: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch news articles from free news APIs or web scraping

        Args:
            query: Search query (stock symbol or company name)
            days_back: Number of days to look back

        Returns:
            List of news articles
        """
        articles = []

        # Try free news API (NewsAPI.org has free tier)
        if self.news_api_key:
            try:
                # This would use a real news API - simplified for demo
                print(f"Fetching news for: {query}")
                # In real implementation, you'd make API calls here
                articles = self._mock_news_fetch(query, days_back)
            except Exception as e:
                print(f"API fetch failed: {e}, using mock data")
                articles = self._mock_news_fetch(query, days_back)
        else:
            # Mock data for demo purposes
            articles = self._mock_news_fetch(query, days_back)

        return articles

    def _mock_news_fetch(self, query: str, days_back: int) -> List[Dict[str, Any]]:
        """Mock news fetching for demonstration"""
        # This would be replaced with actual API calls
        mock_articles = [
            {
                "title": f"{query} Reports Strong Quarterly Results",
                "content": f"{query} exceeded analyst expectations with robust revenue growth...",
                "url": f"https://example.com/{query.lower()}-earnings",
                "published_date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                "source": "Financial News Daily",
                "symbols": [query]
            }
            for i in range(min(days_back, 5))
        ]
        return mock_articles

    def analyze_news_sentiment(self, query: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze news sentiment for a given query using RAG

        Args:
            query: Stock symbol or company name to analyze
            days_back: Number of days to look back

        Returns:
            Analysis results with sentiment and key insights
        """
        # Fetch relevant news
        news_articles = self.fetch_news_from_api(query, days_back)

        # Add to vector store for future searches
        if news_articles:
            self.vector_store.add_news_articles(news_articles)

        # Search vector store for relevant context
        similar_news = self.vector_store.search_similar_news(
            query=f"news about {query} stock performance and outlook",
            n_results=10,
            symbols=[query]
        )

        # Perform sentiment analysis (simplified)
        sentiment_summary = self._analyze_sentiment(similar_news)

        return {
            "query": query,
            "analysis_date": datetime.now().isoformat(),
            "time_period_days": days_back,
            "articles_analyzed": len(similar_news),
            "sentiment_summary": sentiment_summary,
            "top_articles": similar_news[:3],  # Return top 3 most relevant
            "recommendations": self._generate_recommendations(sentiment_summary, query)
        }

    def _analyze_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple sentiment analysis based on keywords"""
        positive_words = ['strong', 'growth', 'profit', 'beat', 'exceed', 'positive', 'bullish']
        negative_words = ['decline', 'loss', 'miss', 'weak', 'concern', 'bearish', 'drop']

        total_articles = len(articles)
        if total_articles == 0:
            return {"overall": "neutral", "confidence": 0, "positive_count": 0, "negative_count": 0}

        positive_count = 0
        negative_count = 0

        for article in articles:
            content = (article.get('title', '') + ' ' + article.get('content', '')).lower()

            pos_matches = sum(1 for word in positive_words if word in content)
            neg_matches = sum(1 for word in negative_words if word in content)

            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1

        # Determine overall sentiment
        if positive_count > negative_count:
            overall = "positive"
        elif negative_count > positive_count:
            overall = "negative"
        else:
            overall = "neutral"

        confidence = abs(positive_count - negative_count) / max(total_articles, 1)

        return {
            "overall": overall,
            "confidence": round(confidence, 2),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_count": total_articles
        }

    def _generate_recommendations(self, sentiment: Dict[str, Any], symbol: str) -> List[str]:
        """Generate trading recommendations based on sentiment"""
        recommendations = []

        if sentiment["overall"] == "positive" and sentiment["confidence"] > 0.3:
            recommendations.append(f"Consider bullish positions in {symbol}")
            recommendations.append("Positive news sentiment suggests potential upside")

        elif sentiment["overall"] == "negative" and sentiment["confidence"] > 0.3:
            recommendations.append(f"Consider bearish positions or reducing exposure to {symbol}")
            recommendations.append("Negative news sentiment indicates potential downside risk")

        else:
            recommendations.append(f"Mixed or neutral sentiment for {symbol}")
            recommendations.append("Monitor for clearer trend development")

        if sentiment["confidence"] < 0.2:
            recommendations.append("Low confidence analysis - gather more data")

        return recommendations

    def search_news_by_symbol(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for all news related to a specific symbol

        Args:
            symbol: Stock symbol to search for
            limit: Maximum number of results to return

        Returns:
            List of relevant news articles
        """
        return self.vector_store.search_similar_news(
            query=f"news and analysis about {symbol}",
            n_results=limit,
            symbols=[symbol]
        )

# Example usage
if __name__ == "__main__":
    tool = NewsAnalyzerTool()

    # Analyze sentiment for a stock
    analysis = tool.analyze_news_sentiment("TSLA", days_back=7)
    print(f"Analysis for TSLA: {json.dumps(analysis, indent=2)}")

    # Search news by symbol
    news = tool.search_news_by_symbol("NVDA", limit=5)
    print(f"\nNews for NVDA: {len(news)} articles found")