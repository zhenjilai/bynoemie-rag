"""
ChromaDB Database Manager

Manages two collections:
1. products - Original product data with embeddings
2. product_vibes - Generated vibe tags with embeddings

Usage:
    from src.rag.database import ProductDatabase
    
    db = ProductDatabase()
    
    # Add products
    db.add_products(products_list)
    
    # Add vibes for products
    db.add_vibes(product_id, vibe_tags)
    
    # Search by query
    results = db.search_products("romantic dinner dress")
    results = db.search_by_vibe("main character energy")
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Product:
    """Product data model"""
    product_id: str
    product_name: str
    product_type: str
    product_description: str
    colors_available: str
    material: str
    price_min: float
    price_max: float
    price_currency: str = "MYR"
    product_url: str = ""
    image_url: str = ""
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_text(self) -> str:
        """Convert to searchable text"""
        return f"{self.product_name}. {self.product_type}. {self.product_description}. Colors: {self.colors_available}. Material: {self.material}."
    
    def content_hash(self) -> str:
        """Generate hash for change detection"""
        content = f"{self.product_name}{self.product_description}{self.colors_available}{self.material}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class ProductVibe:
    """Product vibe tags model"""
    product_id: str
    vibe_tags: List[str]
    mood_summary: str = ""
    ideal_for: str = ""
    styling_tip: str = ""
    generation_method: str = "rule_based"  # rule_based, llm, hybrid
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_text(self) -> str:
        """Convert to searchable text"""
        vibes_text = ", ".join(self.vibe_tags)
        return f"{vibes_text}. {self.mood_summary}. {self.ideal_for}."


class ProductDatabase:
    """
    ChromaDB-based product database with two collections.
    
    Collections:
    - products: Original product data
    - product_vibes: Generated vibe tags
    
    Both collections use embeddings for semantic search.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/embeddings/chroma_db",
        collection_prefix: str = "bynoemie",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_prefix = collection_prefix
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self._embedding_fn = self._create_embedding_function()
        
        # Get or create collections
        self._products_collection = self._get_or_create_collection("products")
        self._vibes_collection = self._get_or_create_collection("product_vibes")
        
        logger.info(f"Initialized ProductDatabase at {self.persist_directory}")
        logger.info(f"Products collection: {self._products_collection.count()} items")
        logger.info(f"Vibes collection: {self._vibes_collection.count()} items")
    
    def _create_embedding_function(self):
        """Create embedding function using sentence-transformers"""
        try:
            from chromadb.utils import embedding_functions
            
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
        except Exception as e:
            logger.warning(f"Failed to create embedding function: {e}")
            logger.info("Using default embeddings")
            return None
    
    def _get_or_create_collection(self, name: str):
        """Get or create a collection"""
        collection_name = f"{self.collection_prefix}_{name}"
        
        return self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"description": f"ByNoemie {name} collection"}
        )
    
    # =========================================================================
    # PRODUCTS COLLECTION
    # =========================================================================
    
    def add_product(self, product: Product) -> bool:
        """Add a single product to the database"""
        try:
            self._products_collection.upsert(
                ids=[product.product_id],
                documents=[product.to_text()],
                metadatas=[{
                    "product_id": product.product_id,
                    "product_name": product.product_name,
                    "product_type": product.product_type,
                    "colors": product.colors_available,
                    "material": product.material,
                    "price_min": product.price_min,
                    "price_max": product.price_max,
                    "price_currency": product.price_currency,
                    "product_url": product.product_url,
                    "content_hash": product.content_hash(),
                    "updated_at": product.updated_at
                }]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add product {product.product_id}: {e}")
            return False
    
    def add_products(self, products: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Add multiple products to the database.
        
        Returns:
            Tuple of (added_count, skipped_count)
        """
        added = 0
        skipped = 0
        
        for p in products:
            try:
                product = Product(
                    product_id=str(p.get("product_id", "")),
                    product_name=p.get("product_name", ""),
                    product_type=p.get("product_type", ""),
                    product_description=p.get("product_description", ""),
                    colors_available=p.get("colors_available", ""),
                    material=p.get("material", ""),
                    price_min=float(p.get("price_min", 0)),
                    price_max=float(p.get("price_max", 0)),
                    price_currency=p.get("price_currency", "MYR"),
                    product_url=p.get("product_url", ""),
                    image_url=p.get("image_url", "")
                )
                
                if self.add_product(product):
                    added += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process product: {e}")
                skipped += 1
        
        logger.info(f"Added {added} products, skipped {skipped}")
        return added, skipped
    
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a product by ID"""
        try:
            result = self._products_collection.get(
                ids=[product_id],
                include=["metadatas", "documents"]
            )
            
            if result["ids"]:
                return {
                    "product_id": product_id,
                    "document": result["documents"][0] if result["documents"] else "",
                    **result["metadatas"][0]
                }
            return None
        except Exception:
            return None
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products"""
        result = self._products_collection.get(
            include=["metadatas", "documents"]
        )
        
        products = []
        for i, pid in enumerate(result["ids"]):
            products.append({
                "product_id": pid,
                "document": result["documents"][i] if result["documents"] else "",
                **result["metadatas"][i]
            })
        
        return products
    
    def search_products(
        self,
        query: str,
        n_results: int = 5,
        filter_type: str = None
    ) -> List[Dict[str, Any]]:
        """Search products by semantic similarity"""
        where_filter = None
        if filter_type:
            where_filter = {"product_type": filter_type}
        
        results = self._products_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )
        
        products = []
        for i, pid in enumerate(results["ids"][0]):
            products.append({
                "product_id": pid,
                "document": results["documents"][0][i],
                "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                **results["metadatas"][0][i]
            })
        
        return products
    
    def product_exists(self, product_id: str) -> bool:
        """Check if a product exists"""
        result = self._products_collection.get(ids=[product_id])
        return len(result["ids"]) > 0
    
    def get_product_hash(self, product_id: str) -> Optional[str]:
        """Get content hash for change detection"""
        product = self.get_product(product_id)
        if product:
            return product.get("content_hash")
        return None
    
    # =========================================================================
    # VIBES COLLECTION
    # =========================================================================
    
    def add_vibes(self, vibe: ProductVibe) -> bool:
        """Add vibe tags for a product"""
        try:
            self._vibes_collection.upsert(
                ids=[vibe.product_id],
                documents=[vibe.to_text()],
                metadatas=[{
                    "product_id": vibe.product_id,
                    "vibe_tags": json.dumps(vibe.vibe_tags),
                    "mood_summary": vibe.mood_summary,
                    "ideal_for": vibe.ideal_for,
                    "styling_tip": vibe.styling_tip,
                    "generation_method": vibe.generation_method,
                    "created_at": vibe.created_at
                }]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add vibes for {vibe.product_id}: {e}")
            return False
    
    def add_vibes_batch(
        self,
        vibes_list: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """Add vibes for multiple products"""
        added = 0
        skipped = 0
        
        for v in vibes_list:
            try:
                vibe = ProductVibe(
                    product_id=str(v.get("product_id", "")),
                    vibe_tags=v.get("vibe_tags", []),
                    mood_summary=v.get("mood_summary", ""),
                    ideal_for=v.get("ideal_for", ""),
                    styling_tip=v.get("styling_tip", ""),
                    generation_method=v.get("generation_method", "rule_based")
                )
                
                if self.add_vibes(vibe):
                    added += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                logger.warning(f"Failed to add vibes: {e}")
                skipped += 1
        
        logger.info(f"Added vibes for {added} products, skipped {skipped}")
        return added, skipped
    
    def get_vibes(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get vibes for a product"""
        try:
            result = self._vibes_collection.get(
                ids=[product_id],
                include=["metadatas", "documents"]
            )
            
            if result["ids"]:
                metadata = result["metadatas"][0]
                return {
                    "product_id": product_id,
                    "document": result["documents"][0] if result["documents"] else "",
                    "vibe_tags": json.loads(metadata.get("vibe_tags", "[]")),
                    "mood_summary": metadata.get("mood_summary", ""),
                    "ideal_for": metadata.get("ideal_for", ""),
                    "styling_tip": metadata.get("styling_tip", ""),
                    "generation_method": metadata.get("generation_method", ""),
                    "created_at": metadata.get("created_at", "")
                }
            return None
        except Exception:
            return None
    
    def search_by_vibe(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search products by vibe similarity"""
        results = self._vibes_collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        vibes = []
        for i, pid in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            vibes.append({
                "product_id": pid,
                "vibe_tags": json.loads(metadata.get("vibe_tags", "[]")),
                "mood_summary": metadata.get("mood_summary", ""),
                "similarity": 1 - results["distances"][0][i]
            })
        
        return vibes
    
    def has_vibes(self, product_id: str) -> bool:
        """Check if vibes exist for a product"""
        result = self._vibes_collection.get(ids=[product_id])
        return len(result["ids"]) > 0
    
    def get_products_without_vibes(self) -> List[str]:
        """Get product IDs that don't have vibes generated"""
        all_products = set(self._products_collection.get()["ids"])
        products_with_vibes = set(self._vibes_collection.get()["ids"])
        
        return list(all_products - products_with_vibes)
    
    # =========================================================================
    # COMBINED SEARCH
    # =========================================================================
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        search_products: bool = True,
        search_vibes: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Combined search across products and vibes.
        Returns products with their vibes, ranked by combined similarity.
        """
        results = {}
        
        # Search products
        if search_products:
            product_results = self.search_products(query, n_results * 2)
            for p in product_results:
                pid = p["product_id"]
                results[pid] = {
                    **p,
                    "product_similarity": p["similarity"],
                    "vibe_similarity": 0,
                    "vibe_tags": []
                }
        
        # Search vibes
        if search_vibes:
            vibe_results = self.search_by_vibe(query, n_results * 2)
            for v in vibe_results:
                pid = v["product_id"]
                if pid in results:
                    results[pid]["vibe_similarity"] = v["similarity"]
                    results[pid]["vibe_tags"] = v["vibe_tags"]
                    results[pid]["mood_summary"] = v.get("mood_summary", "")
                else:
                    # Get product info
                    product = self.get_product(pid)
                    if product:
                        results[pid] = {
                            **product,
                            "product_similarity": 0,
                            "vibe_similarity": v["similarity"],
                            "vibe_tags": v["vibe_tags"],
                            "mood_summary": v.get("mood_summary", "")
                        }
        
        # Calculate combined score and sort
        for pid, data in results.items():
            data["combined_score"] = (
                data.get("product_similarity", 0) * 0.4 +
                data.get("vibe_similarity", 0) * 0.6  # Vibes weighted more
            )
        
        sorted_results = sorted(
            results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        return sorted_results[:n_results]
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "products_count": self._products_collection.count(),
            "vibes_count": self._vibes_collection.count(),
            "products_without_vibes": len(self.get_products_without_vibes()),
            "persist_directory": str(self.persist_directory)
        }
    
    def clear_all(self):
        """Clear all collections (use with caution!)"""
        self._client.delete_collection(f"{self.collection_prefix}_products")
        self._client.delete_collection(f"{self.collection_prefix}_product_vibes")
        
        # Recreate collections
        self._products_collection = self._get_or_create_collection("products")
        self._vibes_collection = self._get_or_create_collection("product_vibes")
        
        logger.warning("Cleared all collections")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_db_instance: Optional[ProductDatabase] = None


def get_database(
    persist_directory: str = "./data/embeddings/chroma_db",
    **kwargs
) -> ProductDatabase:
    """Get singleton database instance"""
    global _db_instance
    
    if _db_instance is None:
        _db_instance = ProductDatabase(
            persist_directory=persist_directory,
            **kwargs
        )
    
    return _db_instance
