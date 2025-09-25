"""
News Article Vector Store Setup for Stock Insight Agent
Provides semantic search capabilities for news analysis using ChromaDB
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Any
from datetime import datetime

class NewsVectorStore:
    def __init__(self, persist_directory: str = "./data/vector_store/chroma_news"):
        """
        Initialize the ChromaDB vector store for news articles

        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding model (free and local)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create collection for news articles
        self.collection = self.client.get_or_create_collection(
            name="news_articles",
            metadata={"description": "News articles for stock analysis"}
        )

    def add_news_articles(self, articles: List[Dict[str, Any]]):
        """
        Add news articles to the vector store

        Args:
            articles: List of article dictionaries with keys:
                - title: Article title
                - content: Article content/summary
                - url: Article URL
                - published_date: Publication date
                - source: News source
                - symbols: List of related stock symbols
        """
        if not articles:
            return

        # Prepare data for insertion
        ids = []
        documents = []
        metadatas = []

        for i, article in enumerate(articles):
            article_id = f"news_{i}_{int(datetime.now().timestamp())}"

            # Combine title and content for embedding
            text = f"{article.get('title', '')} {article.get('content', '')}"

            ids.append(article_id)
            documents.append(text)
            metadatas.append({
                "title": article.get('title', ''),
                "url": article.get('url', ''),
                "published_date": article.get('published_date', ''),
                "source": article.get('source', ''),
                "symbols": json.dumps(article.get('symbols', [])),
                "added_date": datetime.now().isoformat()
            })

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Added {len(articles)} articles to vector store")

    def search_similar_news(self, query: str, n_results: int = 5, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for news articles similar to the query

        Args:
            query: Search query
            n_results: Number of results to return
            symbols: Filter by specific stock symbols

        Returns:
            List of relevant articles with similarity scores
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        # Prepare where clause for symbol filtering
        where_clause = None
        if symbols:
            where_clause = {"symbols": {"$in": symbols}}

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "title": results['metadatas'][0][i].get('title', ''),
                "content": results['documents'][0][i],
                "url": results['metadatas'][0][i].get('url', ''),
                "published_date": results['metadatas'][0][i].get('published_date', ''),
                "source": results['metadatas'][0][i].get('source', ''),
                "symbols": json.loads(results['metadatas'][0][i].get('symbols', '[]')),
                "similarity_score": results['distances'][0][i]
            })

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        count = self.collection.count()
        return {
            "total_articles": count,
            "collection_name": "news_articles",
            "embedding_model": "all-MiniLM-L6-v2"
        }

# Example usage for testing
if __name__ == "__main__":
    # Initialize vector store
    vector_store = NewsVectorStore()

    # Example: Add some sample news articles
    sample_articles = [
        {
            "title": "Tesla Stock Surges on Strong Q3 Earnings",
            "content": "Tesla reported better than expected earnings with record deliveries...",
            "url": "https://example.com/tesla-earnings",
            "published_date": "2024-09-25",
            "source": "Financial Times",
            "symbols": ["TSLA"]
        },
        {
            "title": "NVIDIA AI Chip Demand Continues to Grow",
            "content": "NVIDIA's latest GPU sales exceed expectations...",
            "url": "https://example.com/nvidia-ai",
            "published_date": "2024-09-24",
            "source": "TechCrunch",
            "symbols": ["NVDA"]
        }
    ]

    # Add articles
    vector_store.add_news_articles(sample_articles)

    # Search for relevant news
    results = vector_store.search_similar_news("Tesla earnings report", n_results=3)

    print("Search Results:")
    for result in results:
        score = f"{result['similarity_score']:.3f}"
    print(f"- {result['title']} (Score: {score})")

    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"\nCollection Stats: {stats}")