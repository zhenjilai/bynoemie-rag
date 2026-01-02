"""
ByNoemie Order Management System

Features:
- Create, modify, delete orders
- Track order history in ChromaDB
- Update stock counts in both JSON and ChromaDB
- Order status tracking

Usage:
    from src.orders import OrderManager
    manager = OrderManager()
    order = manager.create_order(...)
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class OrderManager:
    """Manages orders with ChromaDB and JSON persistence"""
    
    def __init__(
        self,
        orders_json_path: str = "data/orders/orders.json",
        stock_json_path: str = "data/stock/stock_inventory.json",
        chroma_db_path: str = "data/embeddings/chroma_db"
    ):
        # Check for alternative order files
        if not os.path.exists(orders_json_path):
            alt_path = "data/orders/sample_orders.json"
            if os.path.exists(alt_path):
                orders_json_path = alt_path
        
        self.orders_json_path = orders_json_path
        self.stock_json_path = stock_json_path
        self.chroma_db_path = chroma_db_path
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(orders_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(stock_json_path), exist_ok=True)
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Load existing data
        self.orders = self._load_orders()
        self.stock = self._load_stock()
    
    def _init_chromadb(self):
        """Initialize ChromaDB collections"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Orders collection
            self.orders_collection = self.chroma_client.get_or_create_collection(
                name="orders",
                metadata={"description": "Customer orders"}
            )
            
            # Order history/amendments collection
            self.order_history_collection = self.chroma_client.get_or_create_collection(
                name="order_history",
                metadata={"description": "Order amendments and history"}
            )
            
            # Stock collection
            self.stock_collection = self.chroma_client.get_or_create_collection(
                name="stock",
                metadata={"description": "Product stock levels"}
            )
            
            self.chromadb_available = True
            
        except ImportError:
            print("⚠️ ChromaDB not installed. Using JSON only.")
            self.chromadb_available = False
        except Exception as e:
            print(f"⚠️ ChromaDB error: {e}. Using JSON only.")
            self.chromadb_available = False
    
    def _load_orders(self) -> Dict:
        """Load orders from JSON"""
        if os.path.exists(self.orders_json_path):
            with open(self.orders_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"orders": [], "last_updated": None}
    
    def _load_stock(self) -> Dict:
        """Load stock from JSON, keyed by product_name (lowercase) for consistency"""
        if os.path.exists(self.stock_json_path):
            with open(self.stock_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert to dict keyed by product_name (lowercase) for easy lookup
                if isinstance(data, list):
                    return {item.get('product_name', '').lower(): item for item in data}
                return data
        return {}
    
    def _save_orders(self):
        """Save orders to JSON"""
        self.orders["last_updated"] = datetime.now().isoformat()
        with open(self.orders_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.orders, f, indent=2, ensure_ascii=False)
    
    def _save_stock(self):
        """Save stock to JSON"""
        stock_list = list(self.stock.values())
        with open(self.stock_json_path, 'w', encoding='utf-8') as f:
            json.dump(stock_list, f, indent=2, ensure_ascii=False)
    
    def _save_order_to_chromadb(self, order: Dict):
        """Save order to ChromaDB"""
        if not self.chromadb_available:
            return
        
        try:
            # Build items string without nested f-strings
            items_list = []
            for item in order['items']:
                item_str = "{} ({}/{})".format(
                    item['product_name'], 
                    item['size'], 
                    item['color']
                )
                items_list.append(item_str)
            items_text = ', '.join(items_list)
            
            order_id = order['order_id']
            customer_name = order['customer_name']
            total_amount = order['total_amount']
            currency = order['currency']
            status = order['status']
            
            order_text = "Order {} for {}. Items: {}. Total: {} {}. Status: {}".format(
                order_id, customer_name, items_text, total_amount, currency, status
            )
            
            self.orders_collection.upsert(
                ids=[order_id],
                documents=[order_text],
                metadatas=[{
                    "order_id": order_id,
                    "customer_name": customer_name,
                    "customer_email": order.get('customer_email', ''),
                    "status": status,
                    "total_amount": total_amount,
                    "created_at": order['created_at'],
                    "updated_at": order.get('updated_at', order['created_at'])
                }]
            )
        except Exception as e:
            print("ChromaDB order save error: {}".format(e))
    
    def _record_amendment(self, order_id: str, action: str, details: Dict):
        """Record order amendment in ChromaDB"""
        if not self.chromadb_available:
            return
        
        try:
            amendment_id = f"{order_id}_amendment_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            amendment_text = f"Order {order_id} - {action}: {json.dumps(details)}"
            
            self.order_history_collection.add(
                ids=[amendment_id],
                documents=[amendment_text],
                metadatas=[{
                    "order_id": order_id,
                    "action": action,
                    "timestamp": datetime.now().isoformat(),
                    "details": json.dumps(details)
                }]
            )
        except Exception as e:
            print(f"ChromaDB amendment record error: {e}")
    
    def _update_stock_chromadb(self, product_id: str, stock_data: Dict):
        """Update stock in ChromaDB"""
        if not self.chromadb_available:
            return
        
        try:
            stock_text = f"Product {stock_data.get('product_name', product_id)} - " \
                        f"Total inventory: {stock_data.get('total_inventory', 0)}"
            
            self.stock_collection.upsert(
                ids=[str(product_id)],
                documents=[stock_text],
                metadatas=[{
                    "product_id": str(product_id),
                    "product_name": stock_data.get('product_name', ''),
                    "total_inventory": stock_data.get('total_inventory', 0),
                    "last_updated": datetime.now().isoformat()
                }]
            )
        except Exception as e:
            print(f"ChromaDB stock update error: {e}")
    
    def check_stock(self, product_id: str, size: str, color: str, quantity: int = 1) -> Tuple[bool, int]:
        """
        Check if stock is available for a specific variant.
        Returns: (is_available, current_quantity)
        """
        product_id = str(product_id)
        
        if product_id not in self.stock:
            return False, 0
        
        stock_data = self.stock[product_id]
        variants = stock_data.get('variants', [])
        
        for variant in variants:
            if variant.get('size', '').lower() == size.lower() and \
               variant.get('color', '').lower() == color.lower():
                current_qty = variant.get('quantity', 0)
                return current_qty >= quantity, current_qty
        
        return False, 0
    
    def update_stock(self, product_id: str, size: str, color: str, quantity_change: int) -> bool:
        """
        Update stock quantity for a specific variant.
        quantity_change: positive to add stock, negative to reduce
        Returns: True if successful
        """
        product_id = str(product_id)
        
        if product_id not in self.stock:
            print(f"Product {product_id} not found in stock")
            return False
        
        stock_data = self.stock[product_id]
        variants = stock_data.get('variants', [])
        
        for variant in variants:
            if variant.get('size', '').lower() == size.lower() and \
               variant.get('color', '').lower() == color.lower():
                
                new_qty = variant.get('quantity', 0) + quantity_change
                
                if new_qty < 0:
                    print(f"Insufficient stock for {product_id} {size}/{color}")
                    return False
                
                variant['quantity'] = new_qty
                variant['status'] = 'out_of_stock' if new_qty == 0 else \
                                   'low_stock' if new_qty <= 3 else 'in_stock'
                
                # Update total inventory
                stock_data['total_inventory'] = sum(v.get('quantity', 0) for v in variants)
                stock_data['last_updated'] = datetime.now().isoformat()
                
                # Save to JSON
                self._save_stock()
                
                # Save to ChromaDB
                self._update_stock_chromadb(product_id, stock_data)
                
                return True
        
        print(f"Variant {size}/{color} not found for product {product_id}")
        return False
    
    def create_order_simple(
        self,
        user_id: str,
        product: Dict,
        size: str = "Free Size",
        color: str = "Default",
        quantity: int = 1
    ) -> Dict:
        """
        Simplified order creation for agent use.
        Returns order dict directly or raises exception.
        Also UPDATES STOCK after successful order.
        """
        import random
        from datetime import datetime, timedelta
        
        # Generate order ID
        order_id = f"ORD-{random.randint(10000, 99999)}"
        
        # Get product details
        product_name = product.get('product_name', 'Unknown Product')
        product_id = str(product.get('product_id', ''))
        price = product.get('price_min', 0)
        currency = product.get('price_currency', 'MYR')
        
        # Create order dict
        order = {
            "order_id": order_id,
            "user_id": user_id,
            "product_id": product_id,
            "product_name": product_name,
            "size": size,
            "color": color,
            "quantity": quantity,
            "unit_price": price,
            "total_price": price * quantity,
            "currency": currency,
            "status": "confirmed",
            "created_at": datetime.now().isoformat(),
            "estimated_delivery": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        }
        
        # Save to orders - self.orders is a dict with "orders" list
        if "orders" not in self.orders:
            self.orders["orders"] = []
        self.orders["orders"].append(order)
        self._save_orders()
        
        # =====================================================
        # UPDATE STOCK - Reduce inventory after order
        # =====================================================
        self._reduce_stock(product_name, size, color, quantity)
        
        print(f"✅ Order created: {order_id}")
        print(f"📦 Stock updated: -{quantity} for {product_name} ({color}/{size})")
        return order
    
    def _reduce_stock(self, product_name: str, size: str, color: str, quantity: int):
        """Reduce stock for a product variant after order"""
        product_key = product_name.lower()
        
        if product_key not in self.stock:
            print(f"⚠️ Product '{product_name}' not found in stock data")
            return
        
        product_stock = self.stock[product_key]
        
        # Find and update the matching variant
        if 'variants' in product_stock:
            for variant in product_stock['variants']:
                variant_size = variant.get('size', '').lower()
                variant_color = variant.get('color', '').lower()
                
                # Match by size and color (case insensitive)
                if (variant_size == size.lower() or size.lower() in variant_size or variant_size in size.lower()) and \
                   (variant_color == color.lower() or color.lower() in variant_color or variant_color in color.lower()):
                    
                    old_qty = variant.get('quantity', 0)
                    new_qty = max(0, old_qty - quantity)
                    variant['quantity'] = new_qty
                    
                    # Update status if out of stock
                    if new_qty == 0:
                        variant['status'] = 'out_of_stock'
                    elif new_qty <= 3:
                        variant['status'] = 'low_stock'
                    
                    print(f"   Stock reduced: {old_qty} → {new_qty} for {color}/{size}")
                    break
            
            # Update total inventory
            total = sum(v.get('quantity', 0) for v in product_stock['variants'])
            product_stock['total_inventory'] = total
        
        # Save updated stock
        self._save_stock()
        
    def _restore_stock(self, product_name: str, size: str, color: str, quantity: int):
        """
        Restore stock for a product variant.
        Used when modifying or cancelling orders.
        """
        product_key = product_name.lower()
        
        if product_key not in self.stock:
            print(f"Warning: '{product_name}' not in stock data")
            return
        
        product_stock = self.stock[product_key]
        
        if 'variants' not in product_stock:
            return
            
        found = False
        for variant in product_stock['variants']:
            v_size = variant.get('size', '').upper()
            v_color = variant.get('color', '').lower()
            
            if v_size == size.upper() and v_color == color.lower():
                old_qty = variant.get('quantity', 0)
                new_qty = old_qty + quantity
                variant['quantity'] = new_qty
                
                # Update status
                if new_qty > 3:
                    variant['status'] = 'in_stock'
                elif new_qty > 0:
                    variant['status'] = 'low_stock'
                else:
                    variant['status'] = 'out_of_stock'
                
                print(f"  Stock restored: {old_qty} -> {new_qty} for {color}/{size}")
                found = True
                break
        
        if not found:
            print(f"Warning: Variant {size}/{color} not found")
        
        # Update total
        total = sum(v.get('quantity', 0) for v in product_stock['variants'])
        product_stock['total_inventory'] = total
        
        self._save_stock()
    
    def create_order(
        self,
        customer_name: str,
        customer_email: str,
        items: List[Dict],  # [{"product_id": "", "product_name": "", "size": "", "color": "", "quantity": 1, "price": 0}]
        shipping_address: str = "",
        notes: str = ""
    ) -> Tuple[Optional[Dict], str]:
        """
        Create a new order.
        Returns: (order_dict, message)
        """
        # Validate and check stock for all items
        validated_items = []
        total_amount = 0
        
        for item in items:
            product_id = str(item.get('product_id', ''))
            size = item.get('size', 'Free Size')
            color = item.get('color', '')
            quantity = item.get('quantity', 1)
            price = item.get('price', 0)
            
            # Check stock
            is_available, current_qty = self.check_stock(product_id, size, color, quantity)
            
            if not is_available:
                return None, f"Sorry, {item.get('product_name', 'item')} in {size}/{color} is not available. Only {current_qty} in stock."
            
            validated_items.append({
                "product_id": product_id,
                "product_name": item.get('product_name', ''),
                "size": size,
                "color": color,
                "quantity": quantity,
                "price": price,
                "subtotal": price * quantity
            })
            
            total_amount += price * quantity
        
        # Create order
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        order = {
            "order_id": order_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "items": validated_items,
            "total_amount": total_amount,
            "currency": "MYR",
            "shipping_address": shipping_address,
            "notes": notes,
            "status": OrderStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "history": [{
                "action": "created",
                "timestamp": datetime.now().isoformat(),
                "details": "Order created"
            }]
        }
        
        # Reduce stock for each item
        for item in validated_items:
            self.update_stock(
                item['product_id'],
                item['size'],
                item['color'],
                -item['quantity']  # Negative to reduce
            )
        
        # Save order
        self.orders["orders"].append(order)
        self._save_orders()
        self._save_order_to_chromadb(order)
        self._record_amendment(order_id, "created", {"items": validated_items, "total": total_amount})
        
        return order, f"Order {order_id} created successfully! Total: MYR {total_amount:.2f}"
    
    def modify_order(
        self,
        order_id: str,
        updates: Dict  # Can include: items, shipping_address, notes
    ) -> Tuple[bool, str]:
        """
        Modify an existing order.
        Returns: (success, message)
        """
        # Find order
        order = None
        order_index = None
        for i, o in enumerate(self.orders["orders"]):
            if o["order_id"] == order_id:
                order = o
                order_index = i
                break
        
        if not order:
            return False, f"Order {order_id} not found."
        
        # Check if order can be modified
        if order["status"] in [OrderStatus.SHIPPED.value, OrderStatus.DELIVERED.value, 
                               OrderStatus.CANCELLED.value, OrderStatus.REFUNDED.value]:
            return False, f"Order {order_id} cannot be modified. Status: {order['status']}"
        
        changes = []
        
        # Handle item changes
        if "items" in updates:
            old_items = order["items"]
            new_items = updates["items"]
            
            # Restore stock from old items
            for item in old_items:
                self.update_stock(
                    item['product_id'],
                    item['size'],
                    item['color'],
                    item['quantity']  # Positive to restore
                )
            
            # Validate and reduce stock for new items
            validated_items = []
            total_amount = 0
            
            for item in new_items:
                product_id = str(item.get('product_id', ''))
                size = item.get('size', 'Free Size')
                color = item.get('color', '')
                quantity = item.get('quantity', 1)
                price = item.get('price', 0)
                
                is_available, current_qty = self.check_stock(product_id, size, color, quantity)
                
                if not is_available:
                    # Restore original items stock
                    for old_item in old_items:
                        self.update_stock(
                            old_item['product_id'],
                            old_item['size'],
                            old_item['color'],
                            -old_item['quantity']
                        )
                    return False, f"{item.get('product_name', 'Item')} in {size}/{color} not available."
                
                validated_items.append({
                    "product_id": product_id,
                    "product_name": item.get('product_name', ''),
                    "size": size,
                    "color": color,
                    "quantity": quantity,
                    "price": price,
                    "subtotal": price * quantity
                })
                
                total_amount += price * quantity
            
            # Reduce stock for new items
            for item in validated_items:
                self.update_stock(
                    item['product_id'],
                    item['size'],
                    item['color'],
                    -item['quantity']
                )
            
            order["items"] = validated_items
            order["total_amount"] = total_amount
            changes.append(f"Items updated. New total: MYR {total_amount:.2f}")
        
        # Handle other updates
        if "shipping_address" in updates:
            order["shipping_address"] = updates["shipping_address"]
            changes.append("Shipping address updated")
        
        if "notes" in updates:
            order["notes"] = updates["notes"]
            changes.append("Notes updated")
        
        if "status" in updates:
            order["status"] = updates["status"]
            changes.append(f"Status changed to {updates['status']}")
        
        # Update timestamps and history
        order["updated_at"] = datetime.now().isoformat()
        order["history"].append({
            "action": "modified",
            "timestamp": datetime.now().isoformat(),
            "details": "; ".join(changes)
        })
        
        # Save changes
        self.orders["orders"][order_index] = order
        self._save_orders()
        self._save_order_to_chromadb(order)
        self._record_amendment(order_id, "modified", {"changes": changes})
        
        return True, f"Order {order_id} updated: {'; '.join(changes)}"
    
    def modify_order_simple(
        self,
        order_id: str,
        new_size: str = None,
        new_color: str = None,
        new_quantity: int = None,
        old_size: str = None,
        old_color: str = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Simplified order modification for agent use.
        Handles stock restoration and reduction automatically.
        
        Returns: (success, message, updated_order)
        """
        # Find order
        order = None
        order_index = None
        for i, o in enumerate(self.orders["orders"]):
            if o["order_id"] == order_id:
                order = o
                order_index = i
                break
        
        if not order:
            return False, f"Order {order_id} not found.", None
        
        # Check status
        blocked_statuses = [OrderStatus.SHIPPED.value, OrderStatus.DELIVERED.value, 
                          OrderStatus.CANCELLED.value, OrderStatus.REFUNDED.value]
        if order.get("status") in blocked_statuses:
            return False, f"Cannot modify order. Status: {order.get('status')}", None
        
        # Get current values
        old_size = old_size or order.get('size', '')
        old_color = old_color or order.get('color', '')
        quantity = new_quantity if new_quantity else order.get('quantity', 1)
        
        # New values
        final_size = new_size if new_size else order.get('size', '')
        final_color = new_color if new_color else order.get('color', '')
        product_name = order.get('product_name', '').lower()
        
        changes = []
        
        # Handle stock if variant changed
        size_changed = new_size and new_size.upper() != old_size.upper()
        color_changed = new_color and new_color.lower() != old_color.lower()
        
        if size_changed or color_changed:
            # Restore old variant stock
            self._restore_stock(product_name, old_size, old_color, quantity)
            print(f"Stock restored: +{quantity} for {product_name} ({old_color}/{old_size})")
            
            # Reduce new variant stock
            self._reduce_stock(product_name, final_size, final_color, quantity)
            print(f"Stock reduced: -{quantity} for {product_name} ({final_color}/{final_size})")
        
        # Apply changes
        if new_size and new_size.upper() != order.get('size', '').upper():
            old_val = order.get('size', '')
            order['size'] = new_size.upper()
            changes.append(f"Size: {old_val} -> {new_size.upper()}")
        
        if new_color and new_color.lower() != order.get('color', '').lower():
            old_val = order.get('color', '')
            order['color'] = new_color.capitalize()
            changes.append(f"Color: {old_val} -> {new_color.capitalize()}")
        
        if new_quantity and new_quantity != order.get('quantity'):
            old_val = order.get('quantity', 1)
            order['quantity'] = new_quantity
            order['total_price'] = order.get('unit_price', 0) * new_quantity
            changes.append(f"Quantity: {old_val} -> {new_quantity}")
        
        if not changes:
            return False, "No changes to apply.", order
        
        # Update metadata
        order["updated_at"] = datetime.now().isoformat()
        if "history" not in order:
            order["history"] = []
        order["history"].append({
            "action": "modified",
            "timestamp": datetime.now().isoformat(),
            "details": "; ".join(changes)
        })
        
        # Save
        self.orders["orders"][order_index] = order
        self._save_orders()
        
        return True, "; ".join(changes), order
    
    def cancel_order(self, order_id: str, reason: str = "") -> Tuple[bool, str]:
        """
        Cancel an order and restore stock.
        Returns: (success, message)
        """
        # Find order
        order = None
        order_index = None
        for i, o in enumerate(self.orders["orders"]):
            if o["order_id"] == order_id:
                order = o
                order_index = i
                break
        
        if not order:
            return False, f"Order {order_id} not found."
        
        # Check if order can be cancelled
        if order["status"] in [OrderStatus.SHIPPED.value, OrderStatus.DELIVERED.value, 
                               OrderStatus.CANCELLED.value]:
            return False, f"Order {order_id} cannot be cancelled. Status: {order['status']}"
        
        # Restore stock
        for item in order["items"]:
            self.update_stock(
                item['product_id'],
                item['size'],
                item['color'],
                item['quantity']  # Positive to restore
            )
        
        # Update order status
        order["status"] = OrderStatus.CANCELLED.value
        order["updated_at"] = datetime.now().isoformat()
        order["history"].append({
            "action": "cancelled",
            "timestamp": datetime.now().isoformat(),
            "details": f"Order cancelled. Reason: {reason}" if reason else "Order cancelled"
        })
        
        # Save changes
        self.orders["orders"][order_index] = order
        self._save_orders()
        self._save_order_to_chromadb(order)
        self._record_amendment(order_id, "cancelled", {"reason": reason})
        
        return True, f"Order {order_id} has been cancelled. Stock has been restored."
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID"""
        for order in self.orders["orders"]:
            if order["order_id"] == order_id:
                return order
        return None
    
    def get_orders_by_customer(self, customer_email: str) -> List[Dict]:
        """Get all orders for a customer"""
        return [o for o in self.orders["orders"] if o.get("customer_email") == customer_email]
    
    def get_orders_by_user(self, user_id: str) -> List[Dict]:
        """Get all orders for a user by user_id"""
        return [o for o in self.orders["orders"] if o.get("user_id") == user_id]
    
    def can_cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Check if an order can be cancelled"""
        order = self.get_order(order_id)
        if not order:
            return False, "Order not found"
        
        status = order.get("status", "").lower()
        
        # Orders that can be cancelled
        cancellable_statuses = ["confirmed", "pending", "processing"]
        if status in cancellable_statuses:
            return True, "Order can be cancelled"
        
        # Orders that cannot be cancelled
        if status in ["shipped", "delivered", "cancelled"]:
            return False, f"Order is already {status}"
        
        return False, f"Order status '{status}' does not allow cancellation"
    
    def can_modify_order(self, order_id: str) -> Tuple[bool, str]:
        """Check if an order can be modified"""
        order = self.get_order(order_id)
        if not order:
            return False, "Order not found"
        
        status = order.get("status", "").lower()
        
        # Orders that can be modified
        modifiable_statuses = ["confirmed", "pending", "processing"]
        if status in modifiable_statuses:
            return True, "Order can be modified"
        
        # Orders that cannot be modified
        if status in ["shipped", "delivered", "cancelled", "refunded"]:
            return False, f"Order is already {status} and cannot be modified"
        
        return False, f"Order status '{status}' does not allow modification"
    
    def modify_order_simple(
        self,
        order_id: str,
        new_size: str = None,
        new_color: str = None,
        new_quantity: int = None,
        old_size: str = None,
        old_color: str = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Simplified order modification for agent use.
        Handles stock restoration and reduction automatically.
        
        Returns: (success, message, updated_order)
        """
        # Find order
        order = None
        order_index = None
        for i, o in enumerate(self.orders["orders"]):
            if o["order_id"] == order_id:
                order = o
                order_index = i
                break
        
        if not order:
            return False, f"Order {order_id} not found.", None
        
        # Check status
        can_modify, reason = self.can_modify_order(order_id)
        if not can_modify:
            return False, reason, None
        
        # Get current values
        old_size = old_size or order.get('size', '')
        old_color = old_color or order.get('color', '')
        quantity = new_quantity if new_quantity else order.get('quantity', 1)
        
        # New values
        final_size = new_size if new_size else order.get('size', '')
        final_color = new_color if new_color else order.get('color', '')
        product_name = order.get('product_name', '')
        
        changes = []
        
        # Handle stock if variant changed
        size_changed = new_size and new_size.upper() != old_size.upper()
        color_changed = new_color and new_color.lower() != old_color.lower()
        
        if size_changed or color_changed:
            # Restore old variant stock
            self._restore_stock(product_name, old_size, old_color, quantity)
            print(f"   Stock restored: +{quantity} for {product_name} ({old_color}/{old_size})")
            
            # Reduce new variant stock
            self._reduce_stock(product_name, final_size, final_color, quantity)
            print(f"   Stock reduced: -{quantity} for {product_name} ({final_color}/{final_size})")
        
        # Apply changes
        if new_size and new_size.upper() != order.get('size', '').upper():
            old_val = order.get('size', '')
            order['size'] = new_size.upper()
            changes.append(f"Size: {old_val} → {new_size.upper()}")
        
        if new_color and new_color.lower() != order.get('color', '').lower():
            old_val = order.get('color', '')
            order['color'] = new_color.capitalize()
            changes.append(f"Color: {old_val} → {new_color.capitalize()}")
        
        if new_quantity and new_quantity != order.get('quantity'):
            old_val = order.get('quantity', 1)
            order['quantity'] = new_quantity
            order['total_price'] = order.get('unit_price', 0) * new_quantity
            changes.append(f"Quantity: {old_val} → {new_quantity}")
        
        if not changes:
            return False, "No changes to apply.", order
        
        # Update metadata
        order["updated_at"] = datetime.now().isoformat()
        if "history" not in order:
            order["history"] = []
        order["history"].append({
            "action": "modified",
            "timestamp": datetime.now().isoformat(),
            "details": "; ".join(changes)
        })
        
        # Save
        self.orders["orders"][order_index] = order
        self._save_orders()
        
        return True, "; ".join(changes), order

    
    def cancel_order(self, order_id: str) -> Tuple[bool, str, Optional[Dict]]:
        """Cancel an order and restore stock"""
        order = self.get_order(order_id)
        if not order:
            return False, "Order not found", None
        
        can_cancel, reason = self.can_cancel_order(order_id)
        if not can_cancel:
            return False, reason, None
        
        # Update order status
        for i, o in enumerate(self.orders["orders"]):
            if o["order_id"] == order_id:
                self.orders["orders"][i]["status"] = "cancelled"
                self.orders["orders"][i]["cancelled_at"] = datetime.now().isoformat()
                break
        
        # Restore stock
        product_name = order.get("product_name", "")
        size = order.get("size", "")
        color = order.get("color", "")
        quantity = order.get("quantity", 1)
        
        self._restore_stock(product_name, size, color, quantity)
        
        self._save_orders()
        
        return True, "Order cancelled successfully", order
    
    def _restore_stock(self, product_name: str, size: str, color: str, quantity: int):
        """Restore stock after order cancellation"""
        product_key = product_name.lower()
        
        if product_key not in self.stock:
            print(f"⚠️ Product '{product_name}' not found in stock for restoration")
            return
        
        product_stock = self.stock[product_key]
        
        if 'variants' in product_stock:
            for variant in product_stock['variants']:
                variant_size = variant.get('size', '').lower()
                variant_color = variant.get('color', '').lower()
                
                if (variant_size == size.lower() or size.lower() in variant_size) and \
                   (variant_color == color.lower() or color.lower() in variant_color):
                    
                    old_qty = variant.get('quantity', 0)
                    new_qty = old_qty + quantity
                    variant['quantity'] = new_qty
                    variant['status'] = 'in_stock'
                    
                    print(f"   Stock restored: {old_qty} → {new_qty} for {color}/{size}")
                    break
            
            # Update total inventory
            total = sum(v.get('quantity', 0) for v in product_stock['variants'])
            product_stock['total_inventory'] = total
        
        self._save_stock()
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status"""
        order = self.get_order(order_id)
        return order["status"] if order else None
    
    def get_stock_for_product(self, product_id: str) -> Optional[Dict]:
        """Get stock information for a product"""
        return self.stock.get(str(product_id))


# Convenience functions
def create_order_manager() -> OrderManager:
    """Create and return an OrderManager instance"""
    return OrderManager()
