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
        """Load stock from JSON"""
        if os.path.exists(self.stock_json_path):
            with open(self.stock_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert to dict keyed by product_id
                if isinstance(data, list):
                    return {str(item['product_id']): item for item in data}
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