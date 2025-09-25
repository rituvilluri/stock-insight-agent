#!/usr/bin/env python3
"""
Test script for the RAG system with vector database
Tests the news analysis and semantic search capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.news.news_analyzer_tool import NewsAnalyzerTool
from data.vector_store.news_vector_store import NewsVectorStore

def test_vector_store():
    """Test basic vector store functionality"""
    print("🧪 Testing Vector Store...")

    # Initialize vector store
    vector_store = NewsVectorStore()

    # Add some sample news articles
    sample_articles = [
        {
            "title": "Tesla Reports Record Q3 Earnings",
            "content": "Tesla exceeded expectations with strong electric vehicle deliveries and energy storage growth.",
            "url": "https://example.com/tesla-q3-earnings",
            "published_date": "2024-09-25",
            "source": "Reuters",
            "symbols": ["TSLA"]
        },
        {
            "title": "NVIDIA AI Chip Sales Surge",
            "content": "NVIDIA's data center revenue grew significantly due to high demand for AI training chips.",
            "url": "https://example.com/nvidia-ai-sales",
            "published_date": "2024-09-24",
            "source": "CNBC",
            "symbols": ["NVDA"]
        },
        {
            "title": "Apple iPhone Sales Decline in China",
            "content": "Apple faces competition in China market with declining iPhone sales amid economic pressures.",
            "url": "https://example.com/apple-china-sales",
            "published_date": "2024-09-23",
            "source": "Wall Street Journal",
            "symbols": ["AAPL"]
        }
    ]

    print(f"Adding {len(sample_articles)} sample articles to vector store...")
    vector_store.add_news_articles(sample_articles)

    # Test search functionality
    print("\n🔍 Testing semantic search...")
    search_results = vector_store.search_similar_news(
        query="Tesla stock performance and earnings",
        n_results=3
    )

    print(f"Found {len(search_results)} relevant articles:")
    for i, result in enumerate(search_results, 1):
        score = f"{result['similarity_score']:.3f}"
        print(f"{i}. {result['title']} (Score: {score})")
        print(f"   Source: {result['source']} | Symbols: {result['symbols']}")

    # Test symbol-specific search
    print("\n📊 Testing symbol-specific search...")
    tesla_news = vector_store.search_similar_news(
        query="Tesla news",
        n_results=5,
        symbols=["TSLA"]
    )

    print(f"Found {len(tesla_news)} Tesla-specific articles:")
    for result in tesla_news:
        score = f"{result['similarity_score']:.3f}"
    print(f"- {result['title']} (Score: {score})")

    # Get collection stats
    stats = vector_store.get_collection_stats()
    print(f"\n📈 Vector Store Stats: {stats}")

    return True

def test_news_analyzer():
    """Test the news analyzer tool"""
    print("\n📰 Testing News Analyzer Tool...")

    analyzer = NewsAnalyzerTool()

    # Test sentiment analysis
    print("Analyzing sentiment for TSLA...")
    analysis = analyzer.analyze_news_sentiment("TSLA", days_back=7)

    print("Analysis Results:")
    print(f"- Articles analyzed: {analysis['articles_analyzed']}")
    print(f"- Overall sentiment: {analysis['sentiment_summary']['overall']}")
    print(f"- Confidence: {analysis['sentiment_summary']['confidence']}")
    print(f"- Recommendations: {len(analysis['recommendations'])} suggestions")

    for rec in analysis['recommendations'][:2]:  # Show first 2 recommendations
        print(f"  • {rec}")

    # Test news search by symbol
    print(f"\nSearching news for NVDA...")
    news_results = analyzer.search_news_by_symbol("NVDA", limit=3)

    print(f"Found {len(news_results)} articles:")
    for result in news_results:
        print(f"- {result['title']}")
        print(f"  Source: {result['source']} | Date: {result['published_date']}")

    return True

def test_integration():
    """Test integration with the existing workflow"""
    print("\n🔗 Testing Integration...")

    try:
        from agent.graph.nodes.tool_caller import tool_registry

        # Check if news tools are registered
        news_tools = [tool for tool in tool_registry.keys() if 'news' in tool]
        print(f"News tools registered: {news_tools}")

        if 'analyze_news_sentiment' in tool_registry and 'search_news_by_symbol' in tool_registry:
            print("✅ News analysis tools are properly integrated!")
        else:
            print("❌ Some news tools are missing from integration")
            return False

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting RAG System Tests...\n")

    try:
        # Test vector store
        if not test_vector_store():
            print("❌ Vector store test failed")
            return False

        # Test news analyzer
        if not test_news_analyzer():
            print("❌ News analyzer test failed")
            return False

        # Test integration
        if not test_integration():
            print("❌ Integration test failed")
            return False

        print("\n🎉 All tests passed! Your RAG system is ready to use!")
        print("\n📝 Next steps:")
        print("1. Start your Chainlit app: cd /workspace && python -m chainlit run app/chainlit/app.py")
        print("2. Try asking: 'Analyze news sentiment for Tesla'")
        print("3. Or try: 'What news do you have about NVIDIA?'")
        print("4. Or try: 'Show me Tesla stock chart for the last 6 months'")

        return True

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)