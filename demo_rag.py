#!/usr/bin/env python3
"""
Demo script showing the RAG system in action
This demonstrates the vector database and news analysis capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.news.news_analyzer_tool import NewsAnalyzerTool
from data.vector_store.news_vector_store import NewsVectorStore

def demo_rag_system():
    """Demonstrate the RAG system capabilities"""
    print("🚀 Stock Insight Agent - RAG System Demo")
    print("=" * 50)

    # Initialize the system
    print("1. Initializing Vector Store and News Analyzer...")
    vector_store = NewsVectorStore()
    news_analyzer = NewsAnalyzerTool()

    # Add sample news articles
    print("\n2. Adding Sample News Articles to Vector Database...")
    sample_articles = [
        {
            "title": "Tesla Reports Strong Q4 Earnings",
            "content": "Tesla exceeded expectations with record electric vehicle deliveries. Revenue grew 30% year-over-year to $25.2 billion.",
            "url": "https://example.com/tesla-q4-earnings",
            "published_date": "2024-01-25",
            "source": "Financial Times",
            "symbols": ["TSLA"]
        },
        {
            "title": "NVIDIA AI Chip Demand Surges",
            "content": "NVIDIA's data center revenue jumped 40% as demand for AI training chips continues to grow rapidly.",
            "url": "https://example.com/nvidia-ai-demand",
            "published_date": "2024-01-24",
            "source": "Reuters",
            "symbols": ["NVDA"]
        },
        {
            "title": "Apple Faces Supply Chain Challenges",
            "content": "Apple's iPhone production impacted by component shortages in Asia, affecting Q1 sales projections.",
            "url": "https://example.com/apple-supply-chain",
            "published_date": "2024-01-23",
            "source": "Wall Street Journal",
            "symbols": ["AAPL"]
        },
        {
            "title": "Meta Platforms Beats Earnings Expectations",
            "content": "Meta's advertising revenue grew strongly with improved user engagement across platforms.",
            "url": "https://example.com/meta-earnings",
            "published_date": "2024-01-22",
            "source": "CNBC",
            "symbols": ["META"]
        }
    ]

    vector_store.add_news_articles(sample_articles)
    print(f"✅ Added {len(sample_articles)} articles to the database")

    # Demonstrate semantic search
    print("\n3. Testing Semantic Search Capabilities...")
    print("🔍 Searching for 'Tesla earnings news':")
    tesla_results = vector_store.search_similar_news(
        query="Tesla earnings news",
        n_results=3
    )

    for i, result in enumerate(tesla_results, 1):
        print(f"   {i}. {result['title']}")
        score = f"{result['similarity_score']:.3f}"
        print(f"      Score: {score} | Source: {result['source']}")

    print("\n🔍 Searching for 'AI chip stocks':")
    ai_results = vector_store.search_similar_news(
        query="AI chip stocks",
        n_results=3
    )

    for i, result in enumerate(ai_results, 1):
        print(f"   {i}. {result['title']}")
        score = f"{result['similarity_score']:.3f}"
        print(f"      Score: {score} | Source: {result['source']}")

    # Demonstrate symbol-specific search
    print("\n4. Testing Symbol-Specific Search...")
    print("📈 Searching for all Tesla-related news:")
    tesla_news = vector_store.search_similar_news(
        query="Tesla stock news",
        n_results=5,
        symbols=["TSLA"]
    )

    print(f"   Found {len(tesla_news)} Tesla articles:")
    for result in tesla_news:
        print(f"   - {result['title']}")

    print("\n📈 Searching for all NVIDIA-related news:")
    nvidia_news = vector_store.search_similar_news(
        query="NVIDIA stock news",
        n_results=5,
        symbols=["NVDA"]
    )

    print(f"   Found {len(nvidia_news)} NVIDIA articles:")
    for result in nvidia_news:
        print(f"   - {result['title']}")

    # Demonstrate news sentiment analysis
    print("\n5. Testing News Sentiment Analysis...")
    print("📰 Analyzing sentiment for Tesla (TSLA):")
    tesla_analysis = news_analyzer.analyze_news_sentiment("TSLA", days_back=7)

    print(f"   Articles analyzed: {tesla_analysis['articles_analyzed']}")
    print(f"   Overall sentiment: {tesla_analysis['sentiment_summary']['overall']}")
    print(f"   Confidence: {tesla_analysis['sentiment_summary']['confidence']}")
    print(f"   Positive articles: {tesla_analysis['sentiment_summary']['positive_count']}")
    print(f"   Negative articles: {tesla_analysis['sentiment_summary']['negative_count']}")

    print("\n   Recommendations:")
    for rec in tesla_analysis['recommendations']:
        print(f"   • {rec}")

    # Show database statistics
    print("\n6. Database Statistics:")
    stats = vector_store.get_collection_stats()
    print(f"   Total articles: {stats['total_articles']}")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Embedding model: {stats['embedding_model']}")

    print("\n" + "=" * 50)
    print("✅ RAG System Demo Complete!")
    print("\n📝 What this means for your Stock Insight Agent:")
    print("• Semantic search across news articles")
    print("• Symbol-specific filtering")
    print("• Sentiment analysis capabilities")
    print("• Integration with your existing LangGraph workflow")
    print("• Ready for real news API integration")
    print("\n🚀 Next steps:")
    print("1. Set your GROQ_API_KEY environment variable")
    print("2. Start Chainlit: python -m chainlit run app/chainlit/app.py")
    print("3. Try asking: 'Analyze news sentiment for Tesla'")
    print("4. Or try: 'What news do you have about NVIDIA?'")

if __name__ == "__main__":
    demo_rag_system()