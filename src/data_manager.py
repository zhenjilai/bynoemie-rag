"""
ByNoemie Data Manager - ChromaDB Storage

Tables stored in ChromaDB:
1. users_collection - Customer profiles
2. orders_collection - Order history with tracking

All data persisted to ChromaDB for vector search capability.
"""

import os
import json
import chromadb
from chromadb.config import Settings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CHROMADB CONNECTION
# =============================================================================
def get_chromadb_client():
    """Get ChromaDB client with persistent storage"""
    persist_dir = os.environ.get("CHROMADB_PATH", "data/chromadb")
    os.makedirs(persist_dir, exist_ok=True)
    
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        return client
    except Exception as e:
        print(f"ChromaDB connection error: {e}")
        return None


# =============================================================================
# PRODUCT ID MAPPING (from bynoemie_products.json)
# =============================================================================
PRODUCT_IDS = {
    "Coco Dress": 9763570811170,
    "Ella Dress": 9773307363618,
    "Chantelle Heels": 9763112714530,
    "Valeria Bodycon Dress": 9699997516066,
    "Kylie Jumpsuit": 9800639152418,
    "Tiara Satin Dress": 9773314081058,
    "Maddison Dress": 9773308477730,
    "Camilia Flora Dress": 9773303169314,
    "Alexandria Set": 9773298647330,
    "Annabelle Dress": 9763561242914,
    "The Sienna": 9762646360354,
    "Vera": 9769077768482,
    "Dahlia": 9769071804706,
    "Luna Dress": 9763569828130,
    "Leila Dress": 9763567272226,
    "Yuna Set": 9945433571618,
    "Dianna Dress": 9854147658018,
    "Monica Dress": 9773311525154,
    "Leslie Dress": 9800636465442,
    "The Classic": 9775129624866,
    "Zera Mini Dress": 9773344784674,
    "Aimme Satin Top": 9773098959138,
    "The Sparkle": 9773092208930,
    "Mimi Dress": 9763564912930,
    "Aurelia Heels": 9763086663970,
    "Gigi Set": 9854146412834,
    "Florina Dress": 9854144905506,
    "Vennesa Dress": 9854143136034,
    "Yla Dress": 9854141366562,
    "Noemie Premium Tumbler": 9817851756834,
    "Anna Floral Beach Dress": 9800635351330,
    "Aurel Beach Dress": 9800633811234,
    "Sparkle Mini Dress": 9800629813538,
    "Vela Mini Dress": 9773344096546,
    "Mini Velora": 9773309526306,
    "The Harper": 9773089653026,
    "The Elan": 9773087621410,
    "Nana Dress": 9769083273506,
    "Sierra Satin Maxi Dress": 9763568451874,
    "Emily Dress": 9763566158114,
    "Ravalle Heels": 9762693218594,
    "Angelie Dress": 9945621889314,
}

def get_product_id(product_name: str) -> int:
    """Get actual product ID from product name"""
    return PRODUCT_IDS.get(product_name, 0)


# =============================================================================
# USER MANAGER - ChromaDB Storage
# =============================================================================
class UserManager:
    """
    User management with ChromaDB storage.
    
    Collection: users_collection
    Document: JSON string of user data
    Metadata: user_id, name, email, membership_tier for filtering
    """
    
    COLLECTION_NAME = "users_collection"
    MEMBERSHIP_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
    
    def __init__(self):
        self.client = get_chromadb_client()
        self.collection = self._get_or_create_collection()
        self._initialize_sample_users()
    
    def _get_or_create_collection(self):
        """Get or create users collection"""
        if not self.client:
            return None
        try:
            return self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "ByNoemie customer profiles"}
            )
        except Exception as e:
            print(f"Error creating users collection: {e}")
            return None
    
    def _calculate_age(self, birthday: str) -> int:
        """Calculate age from birthday"""
        try:
            birth_date = datetime.strptime(birthday, "%Y-%m-%d")
            today = datetime.now()
            age = today.year - birth_date.year
            if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                age -= 1
            return age
        except:
            return 0
    
    def _initialize_sample_users(self):
        """Initialize sample users if collection is empty"""
        if not self.collection:
            return
        
        # Check if already has data
        try:
            existing = self.collection.count()
            if existing > 0:
                return
        except:
            pass
        
        now = datetime.now()
        
        sample_users = [
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
        
        # Insert into ChromaDB
        for user in sample_users:
            self._upsert_user(user)
    
    def _upsert_user(self, user: Dict):
        """Insert or update user in ChromaDB"""
        if not self.collection:
            return
        
        try:
            # Create searchable document text
            doc_text = f"User {user['user_id']} {user['name']} {user['email']} {user.get('membership_tier', 'Bronze')}"
            
            self.collection.upsert(
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
                    "data_json": json.dumps(user)  # Full data as JSON
                }]
            )
        except Exception as e:
            print(f"Error upserting user: {e}")
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        if not self.collection:
            return None
        
        try:
            result = self.collection.get(
                ids=[user_id.upper()],
                include=["metadatas"]
            )
            
            if result['metadatas'] and len(result['metadatas']) > 0:
                return json.loads(result['metadatas'][0].get('data_json', '{}'))
        except Exception as e:
            print(f"Error getting user: {e}")
        
        return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        if not self.collection:
            return None
        
        try:
            result = self.collection.get(
                where={"email": email.lower()},
                include=["metadatas"]
            )
            
            if result['metadatas'] and len(result['metadatas']) > 0:
                return json.loads(result['metadatas'][0].get('data_json', '{}'))
        except:
            pass
        return None
    
    def get_user_by_name(self, name: str) -> Optional[Dict]:
        """Search user by name"""
        if not self.collection:
            return None
        
        try:
            result = self.collection.query(
                query_texts=[name],
                n_results=1,
                include=["metadatas"]
            )
            
            if result['metadatas'] and len(result['metadatas']) > 0 and len(result['metadatas'][0]) > 0:
                return json.loads(result['metadatas'][0][0].get('data_json', '{}'))
        except:
            pass
        return None
    
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        if not self.collection:
            return []
        
        try:
            result = self.collection.get(include=["metadatas"])
            users = []
            for meta in result.get('metadatas', []):
                if meta and 'data_json' in meta:
                    users.append(json.loads(meta['data_json']))
            return users
        except:
            return []
    
    def create_user(self, name: str, email: str, phone: str = None,
                   gender: str = "Female", birthday: str = None,
                   address: Dict = None) -> Dict:
        """Create new user"""
        # Generate user ID
        all_users = self.get_all_users()
        user_num = len(all_users) + 1
        user_id = f"USR-{user_num:03d}"
        
        user = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "phone": phone or "",
            "gender": gender,
            "birthday": birthday or "",
            "age": self._calculate_age(birthday) if birthday else 0,
            "address": address or {},
            "registered_at": datetime.now().isoformat(),
            "membership_tier": "Bronze",
            "preferences": {
                "preferred_size": "M",
                "preferred_style": [],
                "preferred_colors": [],
                "shoe_size": "38"
            },
            "total_orders": 0,
            "total_spent": 0.0,
            "notes": ""
        }
        
        self._upsert_user(user)
        return user
    
    def update_user(self, user_id: str, updates: Dict) -> Tuple[bool, str, Optional[Dict]]:
        """Update user profile"""
        user = self.get_user(user_id)
        if not user:
            return False, "User not found", None
        
        for key, value in updates.items():
            if key in user and key != 'user_id':
                user[key] = value
        
        if 'birthday' in updates and updates['birthday']:
            user['age'] = self._calculate_age(updates['birthday'])
        
        self._upsert_user(user)
        return True, "User updated successfully", user
    
    def update_order_stats(self, user_id: str, order_amount: float):
        """Update user's order statistics after new order"""
        user = self.get_user(user_id)
        if user:
            user['total_orders'] = user.get('total_orders', 0) + 1
            user['total_spent'] = user.get('total_spent', 0) + order_amount
            
            # Update tier based on spending
            if user['total_spent'] >= 5000:
                user['membership_tier'] = "Platinum"
            elif user['total_spent'] >= 2000:
                user['membership_tier'] = "Gold"
            elif user['total_spent'] >= 500:
                user['membership_tier'] = "Silver"
            
            self._upsert_user(user)
    
    def format_user_profile(self, user: Dict) -> str:
        """Format user profile for display"""
        address = user.get('address', {})
        addr_str = f"{address.get('street', '')}, {address.get('city', '')}, {address.get('postcode', '')}"
        
        prefs = user.get('preferences', {})
        styles = ', '.join(prefs.get('preferred_style', [])) or 'Not set'
        colors = ', '.join(prefs.get('preferred_colors', [])) or 'Not set'
        
        return f"""👤 **Customer Profile: {user['name']}**

**Personal Info:**
• User ID: {user['user_id']}
• Email: {user.get('email', 'N/A')}
• Phone: {user.get('phone', 'N/A')}
• Gender: {user.get('gender', 'N/A')}
• Birthday: {user.get('birthday', 'N/A')} (Age: {user.get('age', 'N/A')})

**Address:**
{addr_str}

**Membership:**
• Tier: {user.get('membership_tier', 'Bronze')} ⭐
• Member Since: {user.get('registered_at', 'N/A')[:10] if user.get('registered_at') else 'N/A'}
• Total Orders: {user.get('total_orders', 0)}
• Total Spent: MYR {user.get('total_spent', 0):.2f}

**Preferences:**
• Size: {prefs.get('preferred_size', 'M')}
• Shoe Size: {prefs.get('shoe_size', '38')}
• Style: {styles}
• Colors: {colors}

**Notes:** {user.get('notes', 'None')}"""


# =============================================================================
# ORDER MANAGER - ChromaDB Storage
# =============================================================================
class OrderManager:
    """
    Order management with ChromaDB storage.
    
    Collection: orders_collection
    Uses actual product_id from bynoemie_products.json
    """
    
    COLLECTION_NAME = "orders_collection"
    
    # Order statuses
    STATUS_PENDING = "pending_confirmation"
    STATUS_CONFIRMED = "confirmed"
    STATUS_PROCESSING = "processing"
    STATUS_SHIPPED = "shipped"
    STATUS_DELIVERED = "delivered"
    STATUS_CANCELLED = "cancelled"
    
    MODIFIABLE_STATUSES = [STATUS_PENDING, STATUS_CONFIRMED, STATUS_PROCESSING]
    
    def __init__(self, user_manager: UserManager = None):
        self.client = get_chromadb_client()
        self.collection = self._get_or_create_collection()
        self.user_manager = user_manager
        self._initialize_sample_orders()
    
    def _get_or_create_collection(self):
        """Get or create orders collection"""
        if not self.client:
            return None
        try:
            return self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "ByNoemie orders history"}
            )
        except Exception as e:
            print(f"Error creating orders collection: {e}")
            return None
    
    def _initialize_sample_orders(self):
        """Initialize sample orders with real product IDs"""
        if not self.collection:
            return
        
        try:
            existing = self.collection.count()
            if existing > 0:
                return
        except:
            pass
        
        now = datetime.now()
        
        # Sample orders with REAL product IDs from bynoemie_products.json
        sample_orders = [
            {
                "order_id": "ORD-001",
                "user_id": "USR-001",
                "product_id": 9763569828130,  # Luna Dress
                "product_name": "Luna Dress",
                "size": "S",
                "color": "White",
                "quantity": 1,
                "unit_price": 258.00,
                "total_price": 258.00,
                "currency": "MYR",
                "status": self.STATUS_SHIPPED,
                "order_datetime": (now - timedelta(days=5, hours=14)).isoformat(),
                "confirmed_datetime": (now - timedelta(days=5, hours=13)).isoformat(),
                "processing_datetime": (now - timedelta(days=4, hours=10)).isoformat(),
                "shipped_datetime": (now - timedelta(days=2, hours=9)).isoformat(),
                "delivered_datetime": None,
                "cancelled_datetime": None,
                "tracking_number": "MY123456789",
                "shipping_carrier": "J&T Express",
                "shipping_address": {
                    "recipient": "Sarah Chen",
                    "phone": "+60 12-345 6789",
                    "street": "123 Fashion Avenue",
                    "city": "Kuala Lumpur",
                    "postcode": "50450"
                },
                "estimated_delivery": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
                "payment_method": "Credit Card",
                "payment_status": "Paid",
                "notes": "Leave at door"
            },
            {
                "order_id": "ORD-002",
                "user_id": "USR-001",
                "product_id": 9763112714530,  # Chantelle Heels
                "product_name": "Chantelle Heels",
                "size": "37",
                "color": "Black",
                "quantity": 1,
                "unit_price": 178.00,
                "total_price": 178.00,
                "currency": "MYR",
                "status": self.STATUS_PROCESSING,
                "order_datetime": (now - timedelta(days=2, hours=16)).isoformat(),
                "confirmed_datetime": (now - timedelta(days=2, hours=15)).isoformat(),
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
                    "postcode": "50450"
                },
                "estimated_delivery": (now + timedelta(days=5)).strftime("%Y-%m-%d"),
                "payment_method": "Online Banking",
                "payment_status": "Paid",
                "notes": ""
            },
            {
                "order_id": "ORD-003",
                "user_id": "USR-002",
                "product_id": 9800639152418,  # Kylie Jumpsuit
                "product_name": "Kylie Jumpsuit",
                "size": "M",
                "color": "Black",
                "quantity": 1,
                "unit_price": 268.00,
                "total_price": 268.00,
                "currency": "MYR",
                "status": self.STATUS_DELIVERED,
                "order_datetime": (now - timedelta(days=10, hours=11)).isoformat(),
                "confirmed_datetime": (now - timedelta(days=10, hours=10)).isoformat(),
                "processing_datetime": (now - timedelta(days=9, hours=8)).isoformat(),
                "shipped_datetime": (now - timedelta(days=7, hours=14)).isoformat(),
                "delivered_datetime": (now - timedelta(days=3, hours=10)).isoformat(),
                "cancelled_datetime": None,
                "tracking_number": "MY987654321",
                "shipping_carrier": "Poslaju",
                "shipping_address": {
                    "recipient": "Emily Wong",
                    "phone": "+60 11-234 5678",
                    "street": "456 Style Street",
                    "city": "Petaling Jaya",
                    "postcode": "47300"
                },
                "estimated_delivery": (now - timedelta(days=4)).strftime("%Y-%m-%d"),
                "payment_method": "E-Wallet",
                "payment_status": "Paid",
                "notes": "Gift wrapping"
            },
            {
                "order_id": "ORD-004",
                "user_id": "USR-003",
                "product_id": 9773307363618,  # Ella Dress
                "product_name": "Ella Dress",
                "size": "S",
                "color": "White",
                "quantity": 1,
                "unit_price": 288.00,
                "total_price": 288.00,
                "currency": "MYR",
                "status": self.STATUS_CONFIRMED,
                "order_datetime": (now - timedelta(hours=6)).isoformat(),
                "confirmed_datetime": (now - timedelta(hours=5)).isoformat(),
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
                    "postcode": "10200"
                },
                "estimated_delivery": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
                "payment_method": "Credit Card",
                "payment_status": "Paid",
                "notes": ""
            },
            {
                "order_id": "ORD-005",
                "user_id": "USR-002",
                "product_id": 9763570811170,  # Coco Dress
                "product_name": "Coco Dress",
                "size": "Free Size",
                "color": "Black",
                "quantity": 1,
                "unit_price": 300.00,
                "total_price": 300.00,
                "currency": "MYR",
                "status": self.STATUS_CANCELLED,
                "order_datetime": (now - timedelta(days=8, hours=9)).isoformat(),
                "confirmed_datetime": (now - timedelta(days=8, hours=8)).isoformat(),
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
                    "postcode": "47300"
                },
                "estimated_delivery": None,
                "payment_method": "Credit Card",
                "payment_status": "Refunded",
                "notes": "Customer cancelled"
            },
            {
                "order_id": "ORD-006",
                "user_id": "USR-004",
                "product_id": 9763568451874,  # Sierra Satin Maxi Dress
                "product_name": "Sierra Satin Maxi Dress",
                "size": "M",
                "color": "Maroon",
                "quantity": 2,
                "unit_price": 278.00,
                "total_price": 556.00,
                "currency": "MYR",
                "status": self.STATUS_SHIPPED,
                "order_datetime": (now - timedelta(days=3, hours=20)).isoformat(),
                "confirmed_datetime": (now - timedelta(days=3, hours=19)).isoformat(),
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
                    "postcode": "80000"
                },
                "estimated_delivery": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
                "payment_method": "Credit Card",
                "payment_status": "Paid",
                "notes": "Express delivery"
            },
            {
                "order_id": "ORD-007",
                "user_id": "USR-005",
                "product_id": 9762693218594,  # Ravalle Heels
                "product_name": "Ravalle Heels",
                "size": "37",
                "color": "Nude",
                "quantity": 1,
                "unit_price": 158.00,
                "total_price": 158.00,
                "currency": "MYR",
                "status": self.STATUS_DELIVERED,
                "order_datetime": (now - timedelta(days=15, hours=10)).isoformat(),
                "confirmed_datetime": (now - timedelta(days=15, hours=9)).isoformat(),
                "processing_datetime": (now - timedelta(days=14, hours=9)).isoformat(),
                "shipped_datetime": (now - timedelta(days=12, hours=14)).isoformat(),
                "delivered_datetime": (now - timedelta(days=8, hours=11)).isoformat(),
                "cancelled_datetime": None,
                "tracking_number": "MY111222333",
                "shipping_carrier": "Ninja Van",
                "shipping_address": {
                    "recipient": "Amanda Lim",
                    "phone": "+60 19-111 2222",
                    "street": "55 Chic Boulevard",
                    "city": "Ipoh",
                    "postcode": "30000"
                },
                "estimated_delivery": (now - timedelta(days=9)).strftime("%Y-%m-%d"),
                "payment_method": "Online Banking",
                "payment_status": "Paid",
                "notes": "For wedding"
            },
            {
                "order_id": "ORD-008",
                "user_id": "USR-004",
                "product_id": 9769077768482,  # Vera (bag)
                "product_name": "Vera",
                "size": "One Size",
                "color": "Black",
                "quantity": 1,
                "unit_price": 148.00,
                "total_price": 148.00,
                "currency": "MYR",
                "status": self.STATUS_PENDING,
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
                    "postcode": "80000"
                },
                "estimated_delivery": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
                "payment_method": "Pending",
                "payment_status": "Pending",
                "notes": "Awaiting payment"
            }
        ]
        
        for order in sample_orders:
            self._upsert_order(order)
    
    def _upsert_order(self, order: Dict):
        """Insert or update order in ChromaDB"""
        if not self.collection:
            return
        
        try:
            doc_text = f"Order {order['order_id']} {order['product_name']} {order['user_id']} {order['status']}"
            
            self.collection.upsert(
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
        except Exception as e:
            print(f"Error upserting order: {e}")
    
    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        all_orders = self.get_all_orders()
        order_num = len(all_orders) + 1
        return f"ORD-{order_num:03d}"
    
    def create_order(self, user_id: str, product: Dict, size: str, color: str,
                    quantity: int = 1, shipping_address: Dict = None) -> Dict:
        """Create new order with real product_id"""
        order_id = self.generate_order_id()
        now = datetime.now()
        
        # Get actual product_id from product dict or lookup
        product_name = product.get('product_name', 'Unknown')
        product_id = product.get('product_id', get_product_id(product_name))
        unit_price = product.get('price_min', 0)
        
        order = {
            "order_id": order_id,
            "user_id": user_id,
            "product_id": product_id,
            "product_name": product_name,
            "size": size,
            "color": color,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_price": unit_price * quantity,
            "currency": product.get('price_currency', 'MYR'),
            "status": self.STATUS_CONFIRMED,
            "order_datetime": now.isoformat(),
            "confirmed_datetime": now.isoformat(),
            "processing_datetime": None,
            "shipped_datetime": None,
            "delivered_datetime": None,
            "cancelled_datetime": None,
            "tracking_number": None,
            "shipping_carrier": None,
            "shipping_address": shipping_address or {},
            "estimated_delivery": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "payment_method": "Pending",
            "payment_status": "Pending",
            "notes": ""
        }
        
        self._upsert_order(order)
        
        if self.user_manager:
            self.user_manager.update_order_stats(user_id, order['total_price'])
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID"""
        if not self.collection:
            return None
        
        order_id = order_id.upper().replace(' ', '')
        if not order_id.startswith('ORD-'):
            order_id = f"ORD-{order_id}"
        
        try:
            result = self.collection.get(
                ids=[order_id],
                include=["metadatas"]
            )
            
            if result['metadatas'] and len(result['metadatas']) > 0:
                return json.loads(result['metadatas'][0].get('data_json', '{}'))
        except:
            pass
        return None
    
    def get_all_orders(self) -> List[Dict]:
        """Get all orders"""
        if not self.collection:
            return []
        
        try:
            result = self.collection.get(include=["metadatas"])
            orders = []
            for meta in result.get('metadatas', []):
                if meta and 'data_json' in meta:
                    orders.append(json.loads(meta['data_json']))
            return orders
        except:
            return []
    
    def get_orders_by_user(self, user_id: str) -> List[Dict]:
        """Get all orders for a user"""
        if not self.collection:
            return []
        
        try:
            result = self.collection.get(
                where={"user_id": user_id.upper()},
                include=["metadatas"]
            )
            orders = []
            for meta in result.get('metadatas', []):
                if meta and 'data_json' in meta:
                    orders.append(json.loads(meta['data_json']))
            return orders
        except:
            return []
    
    def get_recent_orders(self, limit: int = 5, user_id: str = None) -> List[Dict]:
        """Get recent orders sorted by date"""
        orders = self.get_orders_by_user(user_id) if user_id else self.get_all_orders()
        orders.sort(key=lambda x: x.get('order_datetime', ''), reverse=True)
        return orders[:limit]
    
    def can_modify_order(self, order_id: str) -> Tuple[bool, str]:
        """Check if order can be modified"""
        order = self.get_order(order_id)
        if not order:
            return False, "Order not found"
        if order['status'] in self.MODIFIABLE_STATUSES:
            return True, "Order can be modified"
        return False, f"Cannot modify - status is **{order['status'].replace('_', ' ').title()}**"
    
    def can_cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Check if order can be cancelled"""
        order = self.get_order(order_id)
        if not order:
            return False, "Order not found"
        if order['status'] == self.STATUS_CANCELLED:
            return False, "Order already cancelled"
        if order['status'] in self.MODIFIABLE_STATUSES:
            return True, "Order can be cancelled"
        return False, f"Cannot cancel - order has been **{order['status'].replace('_', ' ')}**"
    
    def modify_order(self, order_id: str, new_size: str = None, new_color: str = None,
                    new_quantity: int = None) -> Tuple[bool, str, Optional[Dict]]:
        """Modify order"""
        can_modify, reason = self.can_modify_order(order_id)
        if not can_modify:
            return False, reason, None
        
        order = self.get_order(order_id)
        if not order:
            return False, "Order not found", None
        
        changes = []
        if new_size and new_size != order['size']:
            changes.append(f"Size: {order['size']} → {new_size}")
            order['size'] = new_size
        if new_color and new_color != order['color']:
            changes.append(f"Color: {order['color']} → {new_color}")
            order['color'] = new_color
        if new_quantity and new_quantity != order['quantity']:
            changes.append(f"Quantity: {order['quantity']} → {new_quantity}")
            order['quantity'] = new_quantity
            order['total_price'] = order['unit_price'] * new_quantity
        
        if not changes:
            return False, "No changes specified", order
        
        self._upsert_order(order)
        return True, "• " + "\n• ".join(changes), order
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str, Optional[Dict]]:
        """Cancel order"""
        can_cancel, reason = self.can_cancel_order(order_id)
        if not can_cancel:
            return False, reason, None
        
        order = self.get_order(order_id)
        if not order:
            return False, "Order not found", None
        
        order['status'] = self.STATUS_CANCELLED
        order['cancelled_datetime'] = datetime.now().isoformat()
        order['payment_status'] = "Refund Pending"
        
        self._upsert_order(order)
        return True, f"Order {order_id} cancelled. Refund in 3-5 days.", order
    
    def track_order(self, order_id: str) -> str:
        """Get tracking info"""
        order = self.get_order(order_id)
        if not order:
            return f"❌ Order **{order_id}** not found."
        
        status_info = {
            self.STATUS_PENDING: ("⏳", "Pending"),
            self.STATUS_CONFIRMED: ("✅", "Confirmed"),
            self.STATUS_PROCESSING: ("📋", "Processing"),
            self.STATUS_SHIPPED: ("🚚", "Shipped"),
            self.STATUS_DELIVERED: ("🎉", "Delivered"),
            self.STATUS_CANCELLED: ("❌", "Cancelled")
        }
        emoji, status_display = status_info.get(order['status'], ("📦", order['status']))
        
        lines = [
            f"📦 **Order {order['order_id']}**",
            f"**Product ID:** {order['product_id']}",
            f"**Product:** {order['product_name']}",
            f"**Details:** Size {order['size']} | {order['color']} | Qty: {order['quantity']}",
            f"**Total:** {order['currency']} {order['total_price']:.2f}",
            "",
            f"**Status:** {emoji} {status_display}",
            "",
            "**Timeline:**"
        ]
        
        if order.get('order_datetime'):
            lines.append(f"• Ordered: {order['order_datetime'][:16].replace('T', ' ')}")
        if order.get('shipped_datetime'):
            lines.append(f"• Shipped: {order['shipped_datetime'][:16].replace('T', ' ')}")
        if order.get('delivered_datetime'):
            lines.append(f"• Delivered: {order['delivered_datetime'][:16].replace('T', ' ')}")
        if order.get('cancelled_datetime'):
            lines.append(f"• Cancelled: {order['cancelled_datetime'][:16].replace('T', ' ')}")
        
        if order.get('tracking_number'):
            lines.extend(["", f"**Tracking:** {order['tracking_number']}", f"**Carrier:** {order.get('shipping_carrier', 'N/A')}"])
        
        if order.get('estimated_delivery') and order['status'] not in [self.STATUS_DELIVERED, self.STATUS_CANCELLED]:
            lines.append(f"**Est. Delivery:** {order['estimated_delivery']}")
        
        if order['status'] in self.MODIFIABLE_STATUSES:
            lines.extend(["", "_✏️ Can be modified/cancelled_"])
        elif order['status'] == self.STATUS_SHIPPED:
            lines.extend(["", "_🔒 Cannot modify - already shipped_"])
        
        return '\n'.join(lines)
    
    def format_order_summary(self, order: Dict) -> str:
        """Format order summary"""
        emoji = {"pending_confirmation": "⏳", "confirmed": "✅", "processing": "📋",
                 "shipped": "🚚", "delivered": "🎉", "cancelled": "❌"}.get(order['status'], "📦")
        return f"""**{order['order_id']}** {emoji}
• Product ID: {order['product_id']}
• {order['product_name']}
• Size: {order['size']} | Color: {order['color']} | Qty: {order['quantity']}
• Total: {order['currency']} {order['total_price']:.2f}
• Status: {order['status'].replace('_', ' ').title()}"""


# =============================================================================
# DATABASE MANAGER
# =============================================================================
class DatabaseManager:
    """Combined database access"""
    def __init__(self):
        self.users = UserManager()
        self.orders = OrderManager(self.users)
    
    def get_user_with_orders(self, user_id: str) -> Optional[Dict]:
        user = self.users.get_user(user_id)
        if not user:
            return None
        orders = self.orders.get_orders_by_user(user_id)
        return {"user": user, "orders": orders}