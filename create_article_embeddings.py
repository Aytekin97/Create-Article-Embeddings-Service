from pinecone import Pinecone
from config import settings
from loguru import logger
from vector_manager import VectorManager
from openai_client import OpenAiClient
from datetime import datetime
import json
import requests
import os


def create_embeddings():
    logger.info("Method called")
    client = OpenAiClient()
    vector_manager = VectorManager()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    # Access the Pinecone indexes
    article_index = pc.Index("article-embeddings")

    # Send request to get news articles
    try:       
        news = requests.get("https://fetch-news-to-display-production.up.railway.app/api/news/Tesla").json()
        article_summaries = [item['summary'] for item in news if 'summary' in item]
    except Exception as e:
        logger.error(f"An error occured while sending a request to the DB to receive news articles: {e}")

    # Generate article embeddings
    article_embeddings = vector_manager.vectorize(client, article_summaries)

    # Get today's date
    today_date = datetime.now().strftime("%Y/%m/%d")

    # Write text embeddings to the text index
    article_vectors = [
        {
            "id": article['title'],
            "values": embedding,
            "metadata": {
                "title": article['title'],
                "company_name": article['company_name'].lower(),
                "type": "article",
                "summary": article['summary'],
                "published_date": article['published_date'],
                "link": article['link'],
                "classification_score": article.get('classification_score', 0),  # Default to 0
                "upload_date": today_date,
            }
        }
        for (article, embedding) in zip(news, article_embeddings)
    ]
    try:
        article_index.upsert(vectors=article_vectors)
        logger.success(f"{len(article_vectors)} article embeddings written to the article index.")
    except Exception as e:
        logger.error(f"Failed to upsert text embeddings: {str(e)}")


    logger.success("Embeddings created successfully for articles")

create_embeddings()