#!/usr/bin/env python3
"""
Fully Automated Scraper for bynoemie.com.my
Automatically discovers and scrapes all products from all collections
"""

import requests
import json
import csv
import time
from typing import List, Dict, Any, Set
import re
from datetime import datetime


class FullAutoScraper:
    def __init__(self):
        self.base_url = "https://bynoemie.com.my"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.all_product_handles = set()
        
    def discover_collections(self) -> List[str]:
        """
        Automatically discover all collections from the store
        """
        collections = ['frontpage']  # Default collection
        
        try:
            # Try to get collections list
            url = f"{self.base_url}/collections.json"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                discovered = [c.get('handle') for c in data.get('collections', []) if c.get('handle')]
                if discovered:
                    collections = discovered
                    print(f"✓ Discovered {len(collections)} collections")
                else:
                    print("⚠ Using default 'frontpage' collection")
            else:
                print("⚠ Could not fetch collections list, using 'frontpage'")
                
        except Exception as e:
            print(f"⚠ Error discovering collections: {e}")
            print("  Using default 'frontpage' collection")
        
        return collections
    
    def get_all_products_from_collection(self, collection: str) -> List[Dict[str, Any]]:
        """
        Fetch all products from a specific collection
        """
        products = []
        page = 1
        
        while True:
            url = f"{self.base_url}/collections/{collection}/products.json?page={page}&limit=250"
            
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('products'):
                    break
                
                batch = data['products']
                products.extend(batch)
                
                # Track unique handles
                for p in batch:
                    self.all_product_handles.add(p.get('handle'))
                
                print(f"  └─ Page {page}: {len(batch)} products")
                
                if len(batch) < 250:
                    break
                
                page += 1
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  └─ Error on page {page}: {e}")
                break
        
        return products
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """
        Automatically discover and fetch ALL products from ALL collections
        """
        print("\n" + "=" * 70)
        print("DISCOVERING COLLECTIONS")
        print("=" * 70)
        
        collections = self.discover_collections()
        
        print("\n" + "=" * 70)
        print("FETCHING PRODUCTS FROM ALL COLLECTIONS")
        print("=" * 70)
        
        all_products = []
        products_by_collection = {}
        
        for i, collection in enumerate(collections, 1):
            print(f"\n[{i}/{len(collections)}] Collection: '{collection}'")
            products = self.get_all_products_from_collection(collection)
            
            if products:
                products_by_collection[collection] = len(products)
                # Tag products with their collection
                for p in products:
                    p['_source_collection'] = collection
                all_products.extend(products)
        
        # Remove duplicates based on product handle
        unique_products = {}
        for product in all_products:
            handle = product.get('handle')
            if handle not in unique_products:
                unique_products[handle] = product
            else:
                # Merge collections if product appears in multiple
                existing = unique_products[handle]
                if '_source_collection' in existing and '_source_collection' in product:
                    collections_set = set(existing.get('_collections', existing['_source_collection']).split(', '))
                    collections_set.add(product['_source_collection'])
                    existing['_collections'] = ', '.join(sorted(collections_set))
        
        unique_list = list(unique_products.values())
        
        print("\n" + "=" * 70)
        print("COLLECTION SUMMARY")
        print("=" * 70)
        for collection, count in sorted(products_by_collection.items()):
            print(f"  {collection}: {count} products")
        print(f"\n  Total unique products: {len(unique_list)}")
        print(f"  Total product handles discovered: {len(self.all_product_handles)}")
        
        return unique_list
    
    def extract_product_info(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format product information
        """
        variants = product.get('variants', [])
        
        # Extract colors
        colors = set()
        sizes = set()
        weights = set()
        
        for option in product.get('options', []):
            option_name = option.get('name', '').lower()
            if 'color' in option_name or 'colour' in option_name:
                colors.update(option.get('values', []))
            elif 'size' in option_name:
                sizes.update(option.get('values', []))
        
        # Get variant-specific information
        for variant in variants:
            if variant.get('weight'):
                weights.add(f"{variant['weight']} {variant.get('weight_unit', 'kg')}")
        
        # Check stock availability
        in_stock = any(variant.get('available', False) for variant in variants)
        total_inventory = sum(
            variant.get('inventory_quantity', 0) 
            for variant in variants 
            if variant.get('inventory_quantity') is not None
        )
        
        # Get price information
        prices = [float(variant.get('price', 0)) for variant in variants if variant.get('price')]
        compare_prices = [
            float(variant.get('compare_at_price', 0)) 
            for variant in variants 
            if variant.get('compare_at_price')
        ]
        
        min_price = min(prices) if prices else 0
        max_price = max(prices) if prices else 0
        
        # Check if on sale
        on_sale = any(
            variant.get('compare_at_price') and 
            float(variant.get('compare_at_price', 0)) > float(variant.get('price', 0))
            for variant in variants
        )
        
        # Extract material from description
        material = self._extract_material(product.get('body_html', ''))
        
        # Get product images
        images = [img.get('src') for img in product.get('images', [])]
        
        # Get collection info
        collection = product.get('_collections', product.get('_source_collection', 'N/A'))
        
        # Build the structured data
        product_info = {
            'product_id': product.get('id'),
            'product_name': product.get('title'),
            'product_handle': product.get('handle'),
            'product_collection': collection,
            'product_type': product.get('product_type'),
            'vendor': product.get('vendor'),
            'colors_available': ', '.join(sorted(colors)) if colors else 'N/A',
            'stock_availability': 'In Stock' if in_stock else 'Out of Stock',
            'total_inventory': total_inventory,
            'size_options': ', '.join(sorted(sizes)) if sizes else 'N/A',
            'weights': ', '.join(sorted(weights)) if weights else 'N/A',
            'material': material,
            'shipping_fee': 'Contact store for shipping details',
            'price_min': min_price,
            'price_max': max_price,
            'on_sale': 'Yes' if on_sale else 'No',
            'original_price_min': min(compare_prices) if compare_prices else None,
            'original_price_max': max(compare_prices) if compare_prices else None,
            'price_currency': 'MYR',
            'product_link': f"{self.base_url}/products/{product.get('handle')}",
            'product_description': self._clean_html(product.get('body_html', '')),
            'image_1': images[0] if len(images) > 0 else '',
            'image_2': images[1] if len(images) > 1 else '',
            'image_3': images[2] if len(images) > 2 else '',
            'total_images': len(images),
            'tags': ', '.join(product.get('tags', [])),
            'created_at': product.get('created_at'),
            'updated_at': product.get('updated_at'),
            'published_at': product.get('published_at'),
            'variants_count': len(variants)
        }
        
        return product_info
    
    def _extract_material(self, html_description: str) -> str:
        """
        Try to extract material information from product description
        """
        if not html_description:
            return 'N/A'
        
        material_keywords = [
            'cotton', 'polyester', 'silk', 'linen', 'wool', 'leather',
            'satin', 'chiffon', 'velvet', 'denim', 'lace', 'nylon',
            'spandex', 'elastane', 'rayon', 'viscose', 'acrylic',
            'crepe', 'georgette', 'organza', 'tulle', 'jersey'
        ]
        
        text = self._clean_html(html_description).lower()
        
        found_materials = []
        for material in material_keywords:
            if material in text and material.capitalize() not in found_materials:
                found_materials.append(material.capitalize())
        
        return ', '.join(found_materials) if found_materials else 'N/A'
    
    def _clean_html(self, html_text: str) -> str:
        """
        Remove HTML tags and clean text
        """
        if not html_text:
            return ''
        
        clean = re.sub(r'<[^>]+>', '', html_text)
        clean = clean.replace('&nbsp;', ' ')
        clean = clean.replace('&amp;', '&')
        clean = clean.replace('&lt;', '<')
        clean = clean.replace('&gt;', '>')
        clean = clean.replace('&#39;', "'")
        clean = clean.replace('&quot;', '"')
        clean = ' '.join(clean.split())
        
        return clean
    
    def save_to_csv(self, products: List[Dict[str, Any]], filename: str = None):
        """
        Save products to CSV file in data folder
        """
        if not products:
            print("No products to save")
            return
        
        # Create data folder if it doesn't exist
        import os
        os.makedirs('data', exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'data/bynoemie_all_products_{timestamp}.csv'
        elif not filename.startswith('data/'):
            filename = f'data/{filename}'
        
        fieldnames = list(products[0].keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(products)
        
        print(f"✓ Saved to: {filename}")
        return filename
    
    def save_to_json(self, products: List[Dict[str, Any]], filename: str = None):
        """
        Save products to JSON file in data folder
        """
        # Create data folder if it doesn't exist
        import os
        os.makedirs('data', exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'data/bynoemie_all_products_{timestamp}.json'
        elif not filename.startswith('data/'):
            filename = f'data/{filename}'
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(products, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {filename}")
        return filename
    
    def scrape_all(self):
        """
        Main scraping function - fully automated
        """
        print("\n" + "=" * 70)
        print("BYNOEMIE.COM.MY - FULLY AUTOMATED SCRAPER")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get all products from all collections
        raw_products = self.get_all_products()
        
        print("\n" + "=" * 70)
        print("PROCESSING PRODUCTS")
        print("=" * 70)
        
        formatted_products = []
        
        for i, product in enumerate(raw_products, 1):
            print(f"[{i}/{len(raw_products)}] Processing: {product.get('title')}")
            product_info = self.extract_product_info(product)
            formatted_products.append(product_info)
            time.sleep(0.3)
        
        # Save data
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        csv_file = self.save_to_csv(formatted_products)
        json_file = self.save_to_json(formatted_products)
        
        # Print summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total products scraped: {len(formatted_products)}")
        print(f"Unique product handles: {len(self.all_product_handles)}")
        print(f"Products in stock: {sum(1 for p in formatted_products if p['stock_availability'] == 'In Stock')}")
        print(f"Products out of stock: {sum(1 for p in formatted_products if p['stock_availability'] == 'Out of Stock')}")
        print(f"Products on sale: {sum(1 for p in formatted_products if p['on_sale'] == 'Yes')}")
        
        if formatted_products:
            prices = [p['price_min'] for p in formatted_products if p['price_min'] > 0]
            if prices:
                print(f"Price range: RM {min(prices):.2f} - RM {max(prices):.2f}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "=" * 70)
        
        return formatted_products, csv_file, json_file


def main():
    """
    Main execution function
    """
    scraper = FullAutoScraper()
    products, csv_file, json_file = scraper.scrape_all()
    
    print("\n✅ SCRAPING COMPLETED SUCCESSFULLY!")
    print(f"\nYour files are ready:")
    print(f"  📄 CSV: {csv_file}")
    print(f"  📄 JSON: {json_file}")
    print(f"\nTotal products: {len(products)}")


if __name__ == "__main__":
    main()