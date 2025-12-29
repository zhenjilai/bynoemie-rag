"""
ByNoemie Policy Scraper

Scrapes and stores:
- Terms of Service
- Refund Policy  
- Shipping Policy

Saves to JSON and ChromaDB for RAG retrieval.

Usage:
    python scripts/scrape_policies.py
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup

# Policy URLs
POLICY_URLS = {
    "terms_of_service": "https://nfryvz-my.bynoemie.com/policies/terms-of-service",
    "refund_policy": "https://nfryvz-my.bynoemie.com/policies/refund-policy",
    "shipping_policy": "https://nfryvz-my.bynoemie.com/policies/shipping-policy"
}

# Alternative URLs (main site)
ALT_POLICY_URLS = {
    "terms_of_service": "https://bynoemie.com.my/policies/terms-of-service",
    "refund_policy": "https://bynoemie.com.my/policies/refund-policy",
    "shipping_policy": "https://bynoemie.com.my/policies/shipping-policy"
}


def scrape_policy(url: str, policy_name: str) -> Optional[Dict]:
    """Scrape a single policy page"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        print(f"  Fetching {policy_name} from {url}...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try different selectors for Shopify policy pages
        content = None
        
        # Try main content area
        selectors = [
            'div.shopify-policy__body',
            'div.policy-content',
            'div.rte',
            'article',
            'main',
            'div.page-content'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(separator='\n', strip=True)
                if len(content) > 100:  # Valid content found
                    break
        
        if not content:
            # Fallback: get all text from body
            body = soup.find('body')
            if body:
                # Remove script and style elements
                for script in body(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                content = body.get_text(separator='\n', strip=True)
        
        if not content or len(content) < 50:
            print(f"  ⚠️ Could not extract content from {policy_name}")
            return None
        
        # Clean up content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        clean_content = '\n'.join(lines)
        
        # Create policy document
        policy_doc = {
            "policy_id": policy_name,
            "policy_name": policy_name.replace('_', ' ').title(),
            "url": url,
            "content": clean_content,
            "content_hash": hashlib.md5(clean_content.encode()).hexdigest(),
            "scraped_at": datetime.now().isoformat(),
            "word_count": len(clean_content.split()),
            "sections": extract_sections(clean_content)
        }
        
        print(f"  ✅ Scraped {policy_name}: {policy_doc['word_count']} words")
        return policy_doc
        
    except requests.RequestException as e:
        print(f"  ❌ Failed to fetch {policy_name}: {e}")
        return None


def extract_sections(content: str) -> List[Dict]:
    """Extract sections from policy content"""
    sections = []
    current_section = {"title": "Introduction", "content": []}
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this looks like a section header (short, possibly uppercase or numbered)
        is_header = (
            len(line) < 100 and 
            (line.isupper() or 
             line.endswith(':') or 
             line.startswith(('1.', '2.', '3.', '4.', '5.', 'Section', 'Article', 'SECTION')))
        )
        
        if is_header and current_section["content"]:
            # Save current section
            sections.append({
                "title": current_section["title"],
                "content": '\n'.join(current_section["content"])
            })
            current_section = {"title": line.rstrip(':'), "content": []}
        else:
            current_section["content"].append(line)
    
    # Add last section
    if current_section["content"]:
        sections.append({
            "title": current_section["title"],
            "content": '\n'.join(current_section["content"])
        })
    
    return sections


def save_to_json(policies: List[Dict], output_path: str):
    """Save policies to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(policies, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(policies)} policies to {output_path}")


def save_to_chromadb(policies: List[Dict], db_path: str = "data/embeddings/chroma_db"):
    """Save policies to ChromaDB for RAG retrieval with embeddings"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Ensure directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Delete existing collection if it exists (to refresh)
        try:
            client.delete_collection("policies")
            print("  → Deleted existing policies collection")
        except:
            pass
        
        # Create policies collection with default embedding function
        collection = client.create_collection(
            name="policies",
            metadata={"description": "ByNoemie store policies for RAG"}
        )
        
        # Prepare documents
        ids = []
        documents = []
        metadatas = []
        
        for policy in policies:
            policy_id = policy["policy_id"]
            policy_name = policy["policy_name"]
            
            # Add full policy document
            full_id = f"{policy_id}_full"
            ids.append(full_id)
            documents.append(policy["content"])
            metadatas.append({
                "policy_id": policy_id,
                "policy_name": policy_name,
                "url": policy.get("url", ""),
                "type": "full_policy",
                "word_count": str(policy.get("word_count", 0)),
                "scraped_at": policy.get("scraped_at", "")
            })
            
            # Add individual sections for better retrieval
            for i, section in enumerate(policy.get("sections", [])):
                section_id = f"{policy_id}_section_{i}"
                section_content = f"{section['title']}\n\n{section['content']}"
                
                ids.append(section_id)
                documents.append(section_content)
                metadatas.append({
                    "policy_id": policy_id,
                    "policy_name": policy_name,
                    "section_title": section["title"],
                    "type": "policy_section",
                    "parent_policy": policy_id
                })
        
        # Add all documents to collection (will be auto-embedded)
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ Saved {len(ids)} documents to ChromaDB")
        print(f"   → Full policies: {len(policies)}")
        print(f"   → Sections: {len(ids) - len(policies)}")
        print(f"   → Path: {db_path}")
        
    except ImportError:
        print("⚠️ ChromaDB not installed. Run: pip install chromadb")
        print("   Skipping ChromaDB storage.")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        import traceback
        traceback.print_exc()


def create_sample_policies() -> List[Dict]:
    """Create sample policies if scraping fails"""
    print("Creating sample policies...")
    
    policies = [
        {
            "policy_id": "terms_of_service",
            "policy_name": "Terms of Service",
            "url": "https://www.bynoemie.com/policies/terms-of-service",
            "content": """TERMS OF SERVICE

Welcome to ByNoemie. By accessing or using our website, you agree to be bound by these Terms of Service.

1. GENERAL CONDITIONS
We reserve the right to refuse service to anyone for any reason at any time. You understand that your content may be transferred unencrypted.

2. ACCURACY OF INFORMATION
We are not responsible if information made available on this site is not accurate, complete or current. The material on this site is provided for general information only.

3. MODIFICATIONS TO SERVICE AND PRICES
Prices for our products are subject to change without notice. We reserve the right to modify or discontinue the Service without notice at any time.

4. PRODUCTS AND SERVICES
Certain products or services may be available exclusively online through the website. These products or services may have limited quantities.

5. PAYMENT
We accept various payment methods including credit cards and online banking. All payments are processed securely.

6. INTELLECTUAL PROPERTY
All content on this website including images, text, and designs are the property of ByNoemie and protected by copyright laws.

Contact us at hello@bynoemie.com for any questions regarding these terms.""",
            "content_hash": "sample_tos_hash",
            "scraped_at": datetime.now().isoformat(),
            "word_count": 200,
            "sections": [
                {"title": "General Conditions", "content": "We reserve the right to refuse service to anyone."},
                {"title": "Accuracy of Information", "content": "Information is provided for general purposes only."},
                {"title": "Modifications", "content": "Prices and services may change without notice."},
                {"title": "Payment", "content": "We accept credit cards and online banking."}
            ]
        },
        {
            "policy_id": "refund_policy",
            "policy_name": "Refund Policy",
            "url": "https://www.bynoemie.com/policies/refund-policy",
            "content": """REFUND POLICY

At ByNoemie, we want you to be completely satisfied with your purchase.

RETURNS
- Items must be returned within 14 days of delivery
- Items must be unworn, unwashed, and with original tags attached
- Items must be in original packaging

NON-RETURNABLE ITEMS
- Sale items marked as final sale
- Intimates and swimwear
- Customized or personalized items
- Items worn, washed, or altered

REFUND PROCESS
1. Contact us at hello@bynoemie.com with your order number
2. We will provide return instructions
3. Ship the item back to us
4. Refund will be processed within 5-7 business days after we receive the item

EXCHANGES
We offer exchanges for different sizes or colors, subject to availability. Contact us to arrange an exchange.

DAMAGED OR DEFECTIVE ITEMS
If you receive a damaged or defective item, please contact us within 48 hours with photos of the damage.

For any questions, email us at hello@bynoemie.com""",
            "content_hash": "sample_refund_hash",
            "scraped_at": datetime.now().isoformat(),
            "word_count": 180,
            "sections": [
                {"title": "Returns", "content": "Items must be returned within 14 days, unworn with tags."},
                {"title": "Non-Returnable Items", "content": "Sale items, intimates, and customized items cannot be returned."},
                {"title": "Refund Process", "content": "Contact us, ship item back, refund processed in 5-7 days."},
                {"title": "Exchanges", "content": "We offer exchanges subject to availability."},
                {"title": "Damaged Items", "content": "Contact us within 48 hours with photos."}
            ]
        },
        {
            "policy_id": "shipping_policy",
            "policy_name": "Shipping Policy",
            "url": "https://www.bynoemie.com/policies/shipping-policy",
            "content": """SHIPPING POLICY

ByNoemie ships within Malaysia and internationally.

DOMESTIC SHIPPING (MALAYSIA)
- Standard Shipping: 3-5 business days - RM10 (Free for orders above RM200)
- Express Shipping: 1-2 business days - RM20

INTERNATIONAL SHIPPING
- Southeast Asia: 5-10 business days - RM30
- Rest of World: 10-20 business days - RM50

PROCESSING TIME
Orders are processed within 1-2 business days. You will receive a tracking number once your order ships.

TRACKING
All orders include tracking. You will receive tracking information via email once your order is dispatched.

CUSTOMS AND DUTIES
International orders may be subject to customs duties and taxes. These are the responsibility of the customer.

SHIPPING DELAYS
We are not responsible for delays caused by customs, weather, or carrier issues. Please allow extra time during peak seasons.

LOST PACKAGES
If your package is lost, please contact us. We will work with the carrier to locate your package or provide a replacement/refund.

Contact: hello@bynoemie.com""",
            "content_hash": "sample_shipping_hash",
            "scraped_at": datetime.now().isoformat(),
            "word_count": 190,
            "sections": [
                {"title": "Domestic Shipping", "content": "Standard 3-5 days RM10, Express 1-2 days RM20. Free shipping above RM200."},
                {"title": "International Shipping", "content": "SEA 5-10 days RM30, Rest of World 10-20 days RM50."},
                {"title": "Processing Time", "content": "Orders processed within 1-2 business days."},
                {"title": "Tracking", "content": "All orders include tracking via email."},
                {"title": "Customs", "content": "International duties are customer responsibility."}
            ]
        }
    ]
    
    return policies


def main():
    print("=" * 60)
    print("ByNoemie Policy Scraper")
    print("=" * 60)
    
    policies = []
    
    # Try to scrape from main URLs
    for policy_name, url in ALT_POLICY_URLS.items():
        policy = scrape_policy(url, policy_name)
        if policy:
            policies.append(policy)
    
    # If scraping failed, try alternative URLs
    if len(policies) < 3:
        print("\nTrying alternative URLs...")
        for policy_name, url in POLICY_URLS.items():
            if not any(p["policy_id"] == policy_name for p in policies):
                policy = scrape_policy(url, policy_name)
                if policy:
                    policies.append(policy)
    
    # If still no policies, use samples
    if not policies:
        print("\n⚠️ Scraping failed. Using sample policies...")
        policies = create_sample_policies()
    
    # Save to JSON
    output_path = "data/policies/policies.json"
    save_to_json(policies, output_path)
    
    # Save to ChromaDB
    save_to_chromadb(policies)
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! {len(policies)} policies processed")
    print("=" * 60)
    
    return policies


if __name__ == "__main__":
    main()
