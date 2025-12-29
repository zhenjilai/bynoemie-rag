"""
Data Processor for ByNoemie RAG Chatbot

Handles:
1. Loading products from CSV
2. Detecting new/changed products
3. Generating vibes only for new products (incremental)
4. Saving to ChromaDB

Usage:
    from src.rag.data_processor import DataProcessor
    
    processor = DataProcessor()
    
    # Process CSV file (only new products get vibes generated)
    stats = processor.process_csv("products.csv")
    
    # Force regenerate vibes for all products
    stats = processor.process_csv("products.csv", force_regenerate=True)
"""

import os
import csv
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from .database import ProductDatabase, Product, ProductVibe, get_database

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics from processing run"""
    total_products: int = 0
    new_products: int = 0
    updated_products: int = 0
    unchanged_products: int = 0
    vibes_generated: int = 0
    vibes_skipped: int = 0
    errors: int = 0
    processing_time_seconds: float = 0


class DataProcessor:
    """
    Processes product data from CSV and generates vibes.
    
    Key Features:
    - Incremental processing: Only generates vibes for new/changed products
    - Change detection: Uses content hash to detect modifications
    - Multiple generation methods: rule-based, LLM, or hybrid
    """
    
    def __init__(
        self,
        database: ProductDatabase = None,
        vibe_method: str = "hybrid",  # rule_based, llm, hybrid
        llm_provider: str = None
    ):
        self.db = database or get_database()
        self.vibe_method = vibe_method
        self.llm_provider = llm_provider
        
        # Initialize vibe generator based on method
        self._vibe_generator = None
        self._rule_generator = None
    
    def _get_rule_generator(self):
        """Get rule-based vibe generator"""
        if self._rule_generator is None:
            from src.vibe_generator import extract_vibes_from_product
            self._rule_generator = extract_vibes_from_product
        return self._rule_generator
    
    def _get_llm_generator(self):
        """Get LLM-based vibe generator"""
        if self._vibe_generator is None:
            try:
                from src.vibe_generator import create_vibe_generator
                self._vibe_generator = create_vibe_generator()
            except Exception as e:
                logger.warning(f"LLM generator not available: {e}")
                return None
        return self._vibe_generator
    
    def load_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load products from CSV file"""
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        products = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                # Generate product_id if not present
                if not row.get("product_id"):
                    row["product_id"] = f"prod_{i+1:04d}"
                
                # Clean and normalize fields
                product = {
                    "product_id": str(row.get("product_id", "")),
                    "product_name": row.get("product_name", "").strip(),
                    "product_type": row.get("product_type", "").strip(),
                    "product_description": row.get("product_description", "").strip(),
                    "colors_available": row.get("colors_available", "").strip(),
                    "material": row.get("material", "").strip(),
                    "price_min": float(row.get("price_min", 0) or 0),
                    "price_max": float(row.get("price_max", 0) or 0),
                    "price_currency": row.get("price_currency", "MYR").strip(),
                    "product_url": row.get("product_url", "").strip(),
                    "image_url": row.get("image_url", "").strip(),
                }
                
                products.append(product)
        
        logger.info(f"Loaded {len(products)} products from {csv_path}")
        return products
    
    def detect_changes(
        self,
        products: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Detect new, updated, and unchanged products.
        
        Returns:
            Tuple of (new_products, updated_products, unchanged_products)
        """
        new_products = []
        updated_products = []
        unchanged_products = []
        
        for p in products:
            product_id = p["product_id"]
            
            # Create Product object to get hash
            product = Product(
                product_id=product_id,
                product_name=p["product_name"],
                product_type=p["product_type"],
                product_description=p["product_description"],
                colors_available=p["colors_available"],
                material=p["material"],
                price_min=p["price_min"],
                price_max=p["price_max"],
                price_currency=p["price_currency"]
            )
            
            current_hash = product.content_hash()
            
            # Check if product exists
            existing_hash = self.db.get_product_hash(product_id)
            
            if existing_hash is None:
                # New product
                new_products.append(p)
            elif existing_hash != current_hash:
                # Content changed
                updated_products.append(p)
            else:
                # No change
                unchanged_products.append(p)
        
        logger.info(
            f"Change detection: {len(new_products)} new, "
            f"{len(updated_products)} updated, "
            f"{len(unchanged_products)} unchanged"
        )
        
        return new_products, updated_products, unchanged_products
    
    def generate_vibes_for_product(
        self,
        product: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate vibes and visual attributes for a single product"""
        
        result = {
            "product_id": product["product_id"],
            "vibe_tags": [],
            "mood_summary": "",
            "ideal_for": "",
            "styling_tip": "",
            "occasions": [],
            # NEW fields
            "category": "",
            "subcategory": "",
            "materials": [],
            "has_embellishment": False,
            "style_attributes": [],
            "silhouette": "",
            "generation_method": self.vibe_method
        }
        
        try:
            if self.vibe_method == "rule_based":
                # Rule-based only
                generator = self._get_rule_generator()
                result["vibe_tags"] = generator(product)
                result["generation_method"] = "rule_based"
                
            elif self.vibe_method == "llm":
                # LLM only
                generator = self._get_llm_generator()
                if generator:
                    llm_result = generator.generate(
                        product_id=product["product_id"],
                        product_name=product["product_name"],
                        product_type=product["product_type"],
                        description=product["product_description"],
                        colors=product["colors_available"],
                        material=product["material"]
                    )
                    result["vibe_tags"] = llm_result.get("vibe_tags", [])
                    result["mood_summary"] = llm_result.get("mood_summary", "")
                    result["ideal_for"] = llm_result.get("ideal_for", "")
                    result["styling_tip"] = llm_result.get("styling_tip", "")
                    result["occasions"] = llm_result.get("occasions", [])
                    # NEW fields from enhanced workflow
                    result["category"] = llm_result.get("category", "")
                    result["subcategory"] = llm_result.get("subcategory", "")
                    result["materials"] = llm_result.get("materials", [])
                    result["has_embellishment"] = llm_result.get("has_embellishment", False)
                    result["style_attributes"] = llm_result.get("style_attributes", [])
                    result["silhouette"] = llm_result.get("silhouette", "")
                    result["generation_method"] = "llm"
                else:
                    # Fallback to rule-based
                    generator = self._get_rule_generator()
                    result["vibe_tags"] = generator(product)
                    result["generation_method"] = "rule_based_fallback"
                    
            elif self.vibe_method == "hybrid":
                # Hybrid: rule-based + LLM enhancement
                rule_generator = self._get_rule_generator()
                rule_vibes = rule_generator(product)
                
                llm_generator = self._get_llm_generator()
                if llm_generator:
                    llm_result = llm_generator.generate(
                        product_id=product["product_id"],
                        product_name=product["product_name"],
                        product_type=product["product_type"],
                        description=product["product_description"],
                        colors=product["colors_available"],
                        material=product["material"]
                    )
                    
                    # Merge vibes (LLM first, then rule-based unique ones)
                    llm_vibes = llm_result.get("vibe_tags", [])
                    all_vibes = list(llm_vibes)
                    
                    for vibe in rule_vibes:
                        if vibe.lower() not in [v.lower() for v in all_vibes]:
                            all_vibes.append(vibe)
                    
                    result["vibe_tags"] = all_vibes[:12]  # Cap at 12
                    result["mood_summary"] = llm_result.get("mood_summary", "")
                    result["ideal_for"] = llm_result.get("ideal_for", "")
                    result["styling_tip"] = llm_result.get("styling_tip", "")
                    result["occasions"] = llm_result.get("occasions", [])
                    # NEW fields from enhanced workflow
                    result["category"] = llm_result.get("category", "")
                    result["subcategory"] = llm_result.get("subcategory", "")
                    result["materials"] = llm_result.get("materials", [])
                    result["has_embellishment"] = llm_result.get("has_embellishment", False)
                    result["style_attributes"] = llm_result.get("style_attributes", [])
                    result["silhouette"] = llm_result.get("silhouette", "")
                    result["generation_method"] = "hybrid"
                else:
                    result["vibe_tags"] = rule_vibes
                    result["generation_method"] = "rule_based_fallback"
                    
        except Exception as e:
            logger.error(f"Vibe generation failed for {product['product_id']}: {e}")
            # Fallback to rule-based
            try:
                generator = self._get_rule_generator()
                result["vibe_tags"] = generator(product)
                result["generation_method"] = "rule_based_error_fallback"
            except:
                result["vibe_tags"] = ["stylish", "versatile", "elegant"]
                result["generation_method"] = "default_fallback"
        
        return result
    
    def process_csv(
        self,
        csv_path: str,
        force_regenerate: bool = False,
        batch_size: int = 10
    ) -> ProcessingStats:
        """
        Process CSV file and generate vibes for new/changed products.
        
        Args:
            csv_path: Path to CSV file
            force_regenerate: If True, regenerate vibes for all products
            batch_size: Number of products to process at once
            
        Returns:
            ProcessingStats with details
        """
        import time
        start_time = time.time()
        
        stats = ProcessingStats()
        
        # Load products
        products = self.load_csv(csv_path)
        stats.total_products = len(products)
        
        # Detect changes
        new_products, updated_products, unchanged_products = self.detect_changes(products)
        
        stats.new_products = len(new_products)
        stats.updated_products = len(updated_products)
        stats.unchanged_products = len(unchanged_products)
        
        # Determine which products need vibes
        if force_regenerate:
            products_to_process = products
            logger.info(f"Force regenerating vibes for all {len(products)} products")
        else:
            products_to_process = new_products + updated_products
            logger.info(f"Processing {len(products_to_process)} new/updated products")
        
        # Add all products to database (upsert)
        self.db.add_products(products)
        
        # Generate vibes for products that need it
        total_to_process = len(products_to_process)
        for i, product in enumerate(products_to_process):
            try:
                # Progress indicator
                progress = f"[{i+1}/{total_to_process}]"
                print(f"\n   {progress} Processing: {product.get('product_name', 'Unknown')[:40]}")
                
                # Check if vibes already exist (for updated products)
                if not force_regenerate and self.db.has_vibes(product["product_id"]):
                    # Product updated but vibes exist - only regenerate if content changed
                    if product in unchanged_products:
                        print(f"   {progress} ⏭️  Skipped (unchanged)")
                        stats.vibes_skipped += 1
                        continue
                
                # Generate vibes
                print(f"   {progress} 🔄 Generating vibes ({self.vibe_method})...")
                
                vibe_result = self.generate_vibes_for_product(product)
                
                # Save to database
                vibe = ProductVibe(
                    product_id=vibe_result["product_id"],
                    vibe_tags=vibe_result["vibe_tags"],
                    mood_summary=vibe_result.get("mood_summary", ""),
                    ideal_for=vibe_result.get("ideal_for", ""),
                    styling_tip=vibe_result.get("styling_tip", ""),
                    occasions=vibe_result.get("occasions", []),
                    # NEW fields
                    category=vibe_result.get("category", ""),
                    subcategory=vibe_result.get("subcategory", ""),
                    materials=vibe_result.get("materials", []),
                    has_embellishment=vibe_result.get("has_embellishment", False),
                    style_attributes=vibe_result.get("style_attributes", []),
                    silhouette=vibe_result.get("silhouette", ""),
                    generation_method=vibe_result.get("generation_method", "")
                )
                
                if self.db.add_vibes(vibe):
                    stats.vibes_generated += 1
                    print(f"   {progress} ✅ Generated: {', '.join(vibe_result['vibe_tags'][:3])}...")
                else:
                    stats.errors += 1
                    print(f"   {progress} ❌ Failed to save")
                    
            except Exception as e:
                logger.error(f"Error processing {product.get('product_id')}: {e}")
                print(f"   {progress} ❌ Error: {e}")
                stats.errors += 1
        
        # Handle products without vibes (from unchanged list)
        products_without_vibes = self.db.get_products_without_vibes()
        
        for product_id in products_without_vibes:
            # Find product in unchanged list
            product = next(
                (p for p in unchanged_products if p["product_id"] == product_id),
                None
            )
            
            if product:
                logger.info(f"Generating missing vibes for {product['product_name']}")
                
                vibe_result = self.generate_vibes_for_product(product)
                
                vibe = ProductVibe(
                    product_id=vibe_result["product_id"],
                    vibe_tags=vibe_result["vibe_tags"],
                    mood_summary=vibe_result.get("mood_summary", ""),
                    ideal_for=vibe_result.get("ideal_for", ""),
                    styling_tip=vibe_result.get("styling_tip", ""),
                    occasions=vibe_result.get("occasions", []),
                    # NEW fields
                    category=vibe_result.get("category", ""),
                    subcategory=vibe_result.get("subcategory", ""),
                    materials=vibe_result.get("materials", []),
                    has_embellishment=vibe_result.get("has_embellishment", False),
                    style_attributes=vibe_result.get("style_attributes", []),
                    silhouette=vibe_result.get("silhouette", ""),
                    generation_method=vibe_result.get("generation_method", "")
                )
                
                if self.db.add_vibes(vibe):
                    stats.vibes_generated += 1
        
        stats.processing_time_seconds = time.time() - start_time
        
        # Log summary
        logger.info("=" * 50)
        logger.info("Processing Complete!")
        logger.info(f"  Total products: {stats.total_products}")
        logger.info(f"  New products: {stats.new_products}")
        logger.info(f"  Updated products: {stats.updated_products}")
        logger.info(f"  Unchanged products: {stats.unchanged_products}")
        logger.info(f"  Vibes generated: {stats.vibes_generated}")
        logger.info(f"  Vibes skipped: {stats.vibes_skipped}")
        logger.info(f"  Errors: {stats.errors}")
        logger.info(f"  Time: {stats.processing_time_seconds:.2f}s")
        logger.info("=" * 50)
        
        return stats
    
    def export_to_json(self, output_path: str):
        """Export all products with vibes and metadata to JSON"""
        products = self.db.get_all_products()
        
        export_data = []
        for p in products:
            vibes = self.db.get_vibes(p["product_id"])
            
            export_data.append({
                **p,
                # Existing vibe fields
                "vibe_tags": vibes.get("vibe_tags", []) if vibes else [],
                "mood_summary": vibes.get("mood_summary", "") if vibes else "",
                "ideal_for": vibes.get("ideal_for", "") if vibes else "",
                "styling_tip": vibes.get("styling_tip", "") if vibes else "",
                "occasions": vibes.get("occasions", []) if vibes else [],
                # NEW visual/structural fields
                "category": vibes.get("category", "") if vibes else "",
                "subcategory": vibes.get("subcategory", "") if vibes else "",
                "materials": vibes.get("materials", []) if vibes else [],
                "has_embellishment": vibes.get("has_embellishment", False) if vibes else False,
                "style_attributes": vibes.get("style_attributes", []) if vibes else [],
                "silhouette": vibes.get("silhouette", "") if vibes else "",
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(export_data)} products to {output_path}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for data processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process product data and generate vibes")
    
    parser.add_argument(
        "csv_file",
        help="Path to products CSV file"
    )
    parser.add_argument(
        "--method",
        choices=["rule_based", "llm", "hybrid"],
        default="hybrid",
        help="Vibe generation method"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate vibes for all products"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./data/embeddings/chroma_db",
        help="ChromaDB persist directory"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create processor
    db = get_database(persist_directory=args.db_path)
    processor = DataProcessor(database=db, vibe_method=args.method)
    
    # Process CSV
    stats = processor.process_csv(args.csv_file, force_regenerate=args.force)
    
    # Export if requested
    if args.export:
        processor.export_to_json(args.export)
    
    # Print stats
    print(f"\n✅ Processing complete!")
    print(f"   Vibes generated: {stats.vibes_generated}")
    print(f"   Database stats: {db.get_stats()}")


if __name__ == "__main__":
    main()