"""
RAG Module for ByNoemie Chatbot

Provides:
- ChromaDB database with two collections (products, vibes)
- Data processor for CSV import and incremental vibe generation
- Search functionality across products and vibes

Usage:
    from src.rag import ProductDatabase, DataProcessor
    
    # Initialize database
    db = ProductDatabase()
    
    # Process CSV (only new products get vibes)
    processor = DataProcessor(database=db)
    stats = processor.process_csv("products.csv")
    
    # Search
    results = db.search("romantic dinner dress")
"""

from .database import (
    Product,
    ProductVibe,
    ProductDatabase,
    get_database
)

from .data_processor import (
    ProcessingStats,
    DataProcessor
)


__all__ = [
    # Database
    "Product",
    "ProductVibe",
    "ProductDatabase",
    "get_database",
    
    # Data processor
    "ProcessingStats",
    "DataProcessor",
]
