"""
ByNoemie Database Initializer - Create Demo Data in ChromaDB

Run this script to initialize/reset the database with sample users and orders.

Usage:
    python init_database.py
    
This will create:
- 5 sample users with complete profiles
- 8 sample orders with various statuses
- All data stored in ChromaDB at data/chromadb/
"""

import os
import json
import chromadb
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "data/chromadb")

# Real Product IDs from bynoemie_products.json
PRODUCTS = {
    "Coco Dress": {"id": 9763570811170, "price": 300.00, "sizes": ["Free Size"], "colors": ["Black", "Gold"]},
    "Ella Dress": {"id": 9773307363618, "price": 288.00, "sizes": ["M", "S"], "colors": ["White"]},
    "Chantelle Heels": {"id": 9763112714530, "price": 178.00, "sizes": ["36", "37", "38", "39", "40"], "colors": ["Black", "Nude", "Pink", "White"]},
    "Valeria Bodycon Dress": {"id": 9699997516066, "price": 238.00, "sizes": ["M", "S"], "colors": ["Black"]},
    "Kylie Jumpsuit": {"id": 9800639152418, "price": 268.00, "sizes": ["M", "S"], "colors": ["Beige", "Black"]},
    "Tiara Satin Dress": {"id": 9773314081058, "price": 268.00, "sizes": ["M", "S"], "colors": ["Champagne", "White"]},
    "Luna Dress": {"id": 9763569828130, "price": 258.00, "sizes": ["M", "S"], "colors": ["Champagne", "White"]},
    "Sierra Satin Maxi Dress": {"id": 9763568451874, "price": 278.00, "sizes": ["M", "S"], "colors": ["Black", "Maroon"]},
    "Ravalle Heels": {"id": 9762693218594, "price": 158.00, "sizes": ["36", "37", "38", "39"], "colors": ["Black", "Nude"]},
    "Vera": {"id": 9769077768482, "price": 148.00, "sizes": ["One Size"], "colors": ["Black", "White"]},
    "Dahlia": {"id": 9769071804706, "price": 158.00, "sizes": ["One Size"], "colors": ["Black", "White"]},
    "The Sienna": {"id": 9762646360354, "price": 168.00, "sizes": ["One Size"], "colors": ["Brown", "Black"]},
}


# =============================================================================
# SAMPLE USERS DATA
# =============================================================================
def get_sample_users():
    now = datetime.now()
    
    return [
        {
            "user_id": "USR-001",
            "name": "Sarah Chen",
            "email": "sarah.chen@email.com",
            "phone": "+60 12-345 6789",
            "gender": "Female",
            "birthday": "1995-03-15",
            "age": 29,
            "address": {
                "street": "123 Fashion Avenue",
                "city": "Kuala Lumpur",
                "state": "Wilayah Persekutuan",
                "postcode": "50450",
                "country": "Malaysia"
            },
            "registered_at": (now - timedelta(days=365)).isoformat(),
            "membership_tier": "Gold",
            "preferences": {
                "preferred_size": "S",
                "preferred_style": ["Elegant", "Romantic"],
                "preferred_colors": ["Black", "White", "Pink"],
                "shoe_size": "37"
            },
            "total_orders": 8,
            "total_spent": 2450.00,
            "notes": "VIP customer, prefers silk materials"
        },
        {
            "user_id": "USR-002",
            "name": "Emily Wong",
            "email": "emily.wong@email.com",
            "phone": "+60 11-234 5678",
            "gender": "Female",
            "birthday": "1990-08-22",
            "age": 34,
            "address": {
                "street": "456 Style Street",
                "city": "Petaling Jaya",
                "state": "Selangor",
                "postcode": "47300",
                "country": "Malaysia"
            },
            "registered_at": (now - timedelta(days=180)).isoformat(),
            "membership_tier": "Silver",
            "preferences": {
                "preferred_size": "M",
                "preferred_style": ["Bold", "Chic"],
                "preferred_colors": ["Red", "Gold", "Black"],
                "shoe_size": "38"
            },
            "total_orders": 4,
            "total_spent": 1120.00,
            "notes": "Loves statement pieces"
        },
        {
            "user_id": "USR-003",
            "name": "Jessica Tan",
            "email": "jessica.tan@email.com",
            "phone": "+60 16-789 0123",
            "gender": "Female",
            "birthday": "1998-12-01",
            "age": 26,
            "address": {
                "street": "789 Trendy Road",
                "city": "Georgetown",
                "state": "Penang",
                "postcode": "10200",
                "country": "Malaysia"
            },
            "registered_at": (now - timedelta(days=30)).isoformat(),
            "membership_tier": "Bronze",
            "preferences": {
                "preferred_size": "XS",
                "preferred_style": ["Minimalist", "Modern"],
                "preferred_colors": ["White", "Beige", "Navy"],
                "shoe_size": "36"
            },
            "total_orders": 1,
            "total_spent": 268.00,
            "notes": "New customer"
        },
        {
            "user_id": "USR-004",
            "name": "Michelle Lee",
            "email": "michelle.lee@email.com",
            "phone": "+60 17-456 7890",
            "gender": "Female",
            "birthday": "1988-05-10",
            "age": 36,
            "address": {
                "street": "321 Glamour Lane",
                "city": "Johor Bahru",
                "state": "Johor",
                "postcode": "80000",
                "country": "Malaysia"
            },
            "registered_at": (now - timedelta(days=730)).isoformat(),
            "membership_tier": "Platinum",
            "preferences": {
                "preferred_size": "M",
                "preferred_style": ["Glamorous", "Elegant"],
                "preferred_colors": ["Gold", "Silver", "Champagne"],
                "shoe_size": "39"
            },
            "total_orders": 25,
            "total_spent": 8750.00,
            "notes": "Top customer, always buys matching accessories"
        },
        {
            "user_id": "USR-005",
            "name": "Amanda Lim",
            "email": "amanda.lim@email.com",
            "phone": "+60 19-111 2222",
            "gender": "Female",
            "birthday": "1992-07-18",
            "age": 32,
            "address": {
                "street": "55 Chic Boulevard",
                "city": "Ipoh",
                "state": "Perak",
                "postcode": "30000",
                "country": "Malaysia"
            },
            "registered_at": (now - timedelta(days=270)).isoformat(),
            "membership_tier": "Silver",
            "preferences": {
                "preferred_size": "S",
                "preferred_style": ["Classic", "Sophisticated"],
                "preferred_colors": ["Navy", "Cream", "Burgundy"],
                "shoe_size": "37"
            },
            "total_orders": 6,
            "total_spent": 1580.00,
            "notes": "Prefers classic cuts"
        }
    ]


# =============================================================================
# SAMPLE ORDERS DATA
# =============================================================================
def get_sample_orders():
    now = datetime.now()
    
    return [
        # ORD-001: SHIPPED - Luna Dress
        {
            "order_id": "ORD-001",
            "user_id": "USR-001",
            "product_id": 9763569828130,
            "product_name": "Luna Dress",
            "size": "S",
            "color": "White",
            "quantity": 1,
            "unit_price": 258.00,
            "total_price": 258.00,
            "currency": "MYR",
            "status": "shipped",
            "order_datetime": (now - timedelta(days=5, hours=14, minutes=30)).isoformat(),
            "confirmed_datetime": (now - timedelta(days=5, hours=14, minutes=25)).isoformat(),
            "processing_datetime": (now - timedelta(days=4, hours=10)).isoformat(),
            "shipped_datetime": (now - timedelta(days=2, hours=9, minutes=15)).isoformat(),
            "delivered_datetime": None,
            "cancelled_datetime": None,
            "tracking_number": "MY123456789",
            "shipping_carrier": "J&T Express",
            "shipping_address": {
                "recipient": "Sarah Chen",
                "phone": "+60 12-345 6789",
                "street": "123 Fashion Avenue",
                "city": "Kuala Lumpur",
                "state": "Wilayah Persekutuan",
                "postcode": "50450",
                "country": "Malaysia"
            },
            "estimated_delivery": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
            "payment_method": "Credit Card",
            "payment_status": "Paid",
            "notes": "Please leave at door if not home"
        },
        
        # ORD-002: PROCESSING - Chantelle Heels (Can modify)
        {
            "order_id": "ORD-002",
            "user_id": "USR-001",
            "product_id": 9763112714530,
            "product_name": "Chantelle Heels",
            "size": "37",
            "color": "Black",
            "quantity": 1,
            "unit_price": 178.00,
            "total_price": 178.00,
            "currency": "MYR",
            "status": "processing",
            "order_datetime": (now - timedelta(days=2, hours=16, minutes=45)).isoformat(),
            "confirmed_datetime": (now - timedelta(days=2, hours=16, minutes=40)).isoformat(),
            "processing_datetime": (now - timedelta(days=1, hours=9)).isoformat(),
            "shipped_datetime": None,
            "delivered_datetime": None,
            "cancelled_datetime": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "shipping_address": {
                "recipient": "Sarah Chen",
                "phone": "+60 12-345 6789",
                "street": "123 Fashion Avenue",
                "city": "Kuala Lumpur",
                "state": "Wilayah Persekutuan",
                "postcode": "50450",
                "country": "Malaysia"
            },
            "estimated_delivery": (now + timedelta(days=5)).strftime("%Y-%m-%d"),
            "payment_method": "Online Banking",
            "payment_status": "Paid",
            "notes": ""
        },
        
        # ORD-003: DELIVERED - Kylie Jumpsuit
        {
            "order_id": "ORD-003",
            "user_id": "USR-002",
            "product_id": 9800639152418,
            "product_name": "Kylie Jumpsuit",
            "size": "M",
            "color": "Black",
            "quantity": 1,
            "unit_price": 268.00,
            "total_price": 268.00,
            "currency": "MYR",
            "status": "delivered",
            "order_datetime": (now - timedelta(days=10, hours=11, minutes=20)).isoformat(),
            "confirmed_datetime": (now - timedelta(days=10, hours=11, minutes=15)).isoformat(),
            "processing_datetime": (now - timedelta(days=9, hours=8)).isoformat(),
            "shipped_datetime": (now - timedelta(days=7, hours=14, minutes=30)).isoformat(),
            "delivered_datetime": (now - timedelta(days=3, hours=10, minutes=45)).isoformat(),
            "cancelled_datetime": None,
            "tracking_number": "MY987654321",
            "shipping_carrier": "Poslaju",
            "shipping_address": {
                "recipient": "Emily Wong",
                "phone": "+60 11-234 5678",
                "street": "456 Style Street",
                "city": "Petaling Jaya",
                "state": "Selangor",
                "postcode": "47300",
                "country": "Malaysia"
            },
            "estimated_delivery": (now - timedelta(days=4)).strftime("%Y-%m-%d"),
            "payment_method": "E-Wallet",
            "payment_status": "Paid",
            "notes": "Gift wrapping requested"
        },
        
        # ORD-004: CONFIRMED - Ella Dress (Can modify/cancel)
        {
            "order_id": "ORD-004",
            "user_id": "USR-003",
            "product_id": 9773307363618,
            "product_name": "Ella Dress",
            "size": "S",
            "color": "White",
            "quantity": 1,
            "unit_price": 288.00,
            "total_price": 288.00,
            "currency": "MYR",
            "status": "confirmed",
            "order_datetime": (now - timedelta(hours=6, minutes=30)).isoformat(),
            "confirmed_datetime": (now - timedelta(hours=6, minutes=25)).isoformat(),
            "processing_datetime": None,
            "shipped_datetime": None,
            "delivered_datetime": None,
            "cancelled_datetime": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "shipping_address": {
                "recipient": "Jessica Tan",
                "phone": "+60 16-789 0123",
                "street": "789 Trendy Road",
                "city": "Georgetown",
                "state": "Penang",
                "postcode": "10200",
                "country": "Malaysia"
            },
            "estimated_delivery": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "payment_method": "Credit Card",
            "payment_status": "Paid",
            "notes": ""
        },
        
        # ORD-005: CANCELLED - Coco Dress
        {
            "order_id": "ORD-005",
            "user_id": "USR-002",
            "product_id": 9763570811170,
            "product_name": "Coco Dress",
            "size": "Free Size",
            "color": "Black",
            "quantity": 1,
            "unit_price": 300.00,
            "total_price": 300.00,
            "currency": "MYR",
            "status": "cancelled",
            "order_datetime": (now - timedelta(days=8, hours=9)).isoformat(),
            "confirmed_datetime": (now - timedelta(days=8, hours=8, minutes=55)).isoformat(),
            "processing_datetime": None,
            "shipped_datetime": None,
            "delivered_datetime": None,
            "cancelled_datetime": (now - timedelta(days=8, hours=5)).isoformat(),
            "tracking_number": None,
            "shipping_carrier": None,
            "shipping_address": {
                "recipient": "Emily Wong",
                "phone": "+60 11-234 5678",
                "street": "456 Style Street",
                "city": "Petaling Jaya",
                "state": "Selangor",
                "postcode": "47300",
                "country": "Malaysia"
            },
            "estimated_delivery": None,
            "payment_method": "Credit Card",
            "payment_status": "Refunded",
            "notes": "Customer requested cancellation - changed mind"
        },
        
        # ORD-006: SHIPPED - Sierra Satin Maxi Dress (x2)
        {
            "order_id": "ORD-006",
            "user_id": "USR-004",
            "product_id": 9763568451874,
            "product_name": "Sierra Satin Maxi Dress",
            "size": "M",
            "color": "Maroon",
            "quantity": 2,
            "unit_price": 278.00,
            "total_price": 556.00,
            "currency": "MYR",
            "status": "shipped",
            "order_datetime": (now - timedelta(days=3, hours=20)).isoformat(),
            "confirmed_datetime": (now - timedelta(days=3, hours=19, minutes=55)).isoformat(),
            "processing_datetime": (now - timedelta(days=2, hours=8)).isoformat(),
            "shipped_datetime": (now - timedelta(days=1, hours=11)).isoformat(),
            "delivered_datetime": None,
            "cancelled_datetime": None,
            "tracking_number": "MY555666777",
            "shipping_carrier": "DHL",
            "shipping_address": {
                "recipient": "Michelle Lee",
                "phone": "+60 17-456 7890",
                "street": "321 Glamour Lane",
                "city": "Johor Bahru",
                "state": "Johor",
                "postcode": "80000",
                "country": "Malaysia"
            },
            "estimated_delivery": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
            "payment_method": "Credit Card",
            "payment_status": "Paid",
            "notes": "Express delivery requested"
        },
        
        # ORD-007: DELIVERED - Ravalle Heels
        {
            "order_id": "ORD-007",
            "user_id": "USR-005",
            "product_id": 9762693218594,
            "product_name": "Ravalle Heels",
            "size": "37",
            "color": "Nude",
            "quantity": 1,
            "unit_price": 158.00,
            "total_price": 158.00,
            "currency": "MYR",
            "status": "delivered",
            "order_datetime": (now - timedelta(days=15, hours=10)).isoformat(),
            "confirmed_datetime": (now - timedelta(days=15, hours=9, minutes=55)).isoformat(),
            "processing_datetime": (now - timedelta(days=14, hours=9)).isoformat(),
            "shipped_datetime": (now - timedelta(days=12, hours=14)).isoformat(),
            "delivered_datetime": (now - timedelta(days=8, hours=11, minutes=30)).isoformat(),
            "cancelled_datetime": None,
            "tracking_number": "MY111222333",
            "shipping_carrier": "Ninja Van",
            "shipping_address": {
                "recipient": "Amanda Lim",
                "phone": "+60 19-111 2222",
                "street": "55 Chic Boulevard",
                "city": "Ipoh",
                "state": "Perak",
                "postcode": "30000",
                "country": "Malaysia"
            },
            "estimated_delivery": (now - timedelta(days=9)).strftime("%Y-%m-%d"),
            "payment_method": "Online Banking",
            "payment_status": "Paid",
            "notes": "For wedding dinner"
        },
        
        # ORD-008: PENDING - Vera Bag (Can modify/cancel)
        {
            "order_id": "ORD-008",
            "user_id": "USR-004",
            "product_id": 9769077768482,
            "product_name": "Vera",
            "size": "One Size",
            "color": "Black",
            "quantity": 1,
            "unit_price": 148.00,
            "total_price": 148.00,
            "currency": "MYR",
            "status": "pending_confirmation",
            "order_datetime": (now - timedelta(hours=2)).isoformat(),
            "confirmed_datetime": None,
            "processing_datetime": None,
            "shipped_datetime": None,
            "delivered_datetime": None,
            "cancelled_datetime": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "shipping_address": {
                "recipient": "Michelle Lee",
                "phone": "+60 17-456 7890",
                "street": "321 Glamour Lane",
                "city": "Johor Bahru",
                "state": "Johor",
                "postcode": "80000",
                "country": "Malaysia"
            },
            "estimated_delivery": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "payment_method": "Pending",
            "payment_status": "Pending",
            "notes": "Awaiting payment confirmation"
        }
    ]


# =============================================================================
# INITIALIZE CHROMADB
# =============================================================================
def init_database(reset=False):
    """Initialize ChromaDB with sample data"""
    
    print("=" * 60)
    print("ByNoemie Database Initializer")
    print("=" * 60)
    
    # Create ChromaDB directory
    os.makedirs(CHROMADB_PATH, exist_ok=True)
    print(f"\n📁 ChromaDB path: {CHROMADB_PATH}")
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    print("✅ Connected to ChromaDB")
    
    # Reset collections if requested
    if reset:
        print("\n🗑️  Resetting existing collections...")
        try:
            client.delete_collection("users_collection")
            print("   - Deleted users_collection")
        except:
            pass
        try:
            client.delete_collection("orders_collection")
            print("   - Deleted orders_collection")
        except:
            pass
    
    # ==========================================================================
    # CREATE USERS COLLECTION
    # ==========================================================================
    print("\n👥 Creating Users Collection...")
    users_collection = client.get_or_create_collection(
        name="users_collection",
        metadata={"description": "ByNoemie customer profiles"}
    )
    
    users = get_sample_users()
    for user in users:
        doc_text = f"User {user['user_id']} {user['name']} {user['email']} {user['membership_tier']}"
        
        users_collection.upsert(
            ids=[user['user_id']],
            documents=[doc_text],
            metadatas=[{
                "user_id": user['user_id'],
                "name": user['name'],
                "email": user['email'],
                "phone": user.get('phone', ''),
                "gender": user.get('gender', 'Female'),
                "birthday": user.get('birthday', ''),
                "membership_tier": user.get('membership_tier', 'Bronze'),
                "total_orders": user.get('total_orders', 0),
                "total_spent": user.get('total_spent', 0.0),
                "data_json": json.dumps(user)
            }]
        )
        print(f"   ✅ {user['user_id']}: {user['name']} ({user['membership_tier']})")
    
    print(f"\n   Total users: {users_collection.count()}")
    
    # ==========================================================================
    # CREATE ORDERS COLLECTION
    # ==========================================================================
    print("\n📦 Creating Orders Collection...")
    orders_collection = client.get_or_create_collection(
        name="orders_collection",
        metadata={"description": "ByNoemie order history"}
    )
    
    orders = get_sample_orders()
    for order in orders:
        doc_text = f"Order {order['order_id']} {order['product_name']} {order['user_id']} {order['status']}"
        
        orders_collection.upsert(
            ids=[order['order_id']],
            documents=[doc_text],
            metadatas=[{
                "order_id": order['order_id'],
                "user_id": order['user_id'],
                "product_id": str(order['product_id']),
                "product_name": order['product_name'],
                "status": order['status'],
                "total_price": order['total_price'],
                "order_datetime": order['order_datetime'],
                "data_json": json.dumps(order)
            }]
        )
        
        status_emoji = {
            "pending_confirmation": "⏳",
            "confirmed": "✅",
            "processing": "📋",
            "shipped": "🚚",
            "delivered": "🎉",
            "cancelled": "❌"
        }
        emoji = status_emoji.get(order['status'], "📦")
        print(f"   {emoji} {order['order_id']}: {order['product_name']} - {order['status']}")
    
    print(f"\n   Total orders: {orders_collection.count()}")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 60)
    print("✅ DATABASE INITIALIZED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print(f"   - Users: {users_collection.count()}")
    print(f"   - Orders: {orders_collection.count()}")
    
    print("\n📋 Sample Users:")
    print("   | ID      | Name         | Tier     | Spent      |")
    print("   |---------|--------------|----------|------------|")
    for u in users:
        print(f"   | {u['user_id']} | {u['name']:<12} | {u['membership_tier']:<8} | MYR {u['total_spent']:>6.2f} |")
    
    print("\n📦 Sample Orders:")
    print("   | ID      | Product              | Status     | Can Modify? |")
    print("   |---------|----------------------|------------|-------------|")
    for o in orders:
        can_modify = "✅ Yes" if o['status'] in ['pending_confirmation', 'confirmed', 'processing'] else "❌ No"
        status_display = o['status'].replace('_', ' ').title()[:10]
        print(f"   | {o['order_id']} | {o['product_name']:<20} | {status_display:<10} | {can_modify:<11} |")
    
    print("\n🎯 Demo Commands to Try:")
    print("   - 'Track order ORD-001'")
    print("   - 'Cancel order ORD-004' then type 'DELETE'")
    print("   - 'Change order ORD-002 to size 38' then type 'CHANGE'")
    print("   - 'Show my profile'")
    print("   - 'Check my orders'")
    print("   - 'I want to order the Luna Dress in size S, White'")
    
    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys
    
    reset = "--reset" in sys.argv or "-r" in sys.argv
    
    if reset:
        print("⚠️  Reset mode: Will delete existing data and recreate!")
        confirm = input("Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    try:
        init_database(reset=reset)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)