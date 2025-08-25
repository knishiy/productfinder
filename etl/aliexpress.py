"""
AliExpress ETL Module for Winning Product Finder
Handles supplier data collection, product sourcing, and shipping calculations
"""

import requests
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import hmac
import base64
from urllib.parse import urlencode
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AliExpressProduct:
    """Data class for AliExpress product information"""
    product_id: str
    title: str
    unit_price: float
    currency: str
    min_order_quantity: int
    max_order_quantity: int
    available_quantity: int
    image_urls: List[str]
    product_url: str
    seller_id: str
    seller_name: str
    seller_rating: float
    seller_feedback_score: int
    ship_from: str
    shipping_options: List[Dict[str, Any]]  # cost, days, method
    product_rating: float
    review_count: int
    category: str
    attributes: Dict[str, str]  # color, size, material, etc.
    is_verified: bool
    is_premium: bool

@dataclass
class ShippingOption:
    """Data class for shipping information"""
    method: str
    cost: float
    currency: str
    delivery_days: int
    tracking_available: bool
    insurance_available: bool

class AliExpressETL:
    """AliExpress ETL class for supplier data collection"""
    
    def __init__(self, app_key: str = None, app_secret: str = None, 
                 access_token: str = None, use_apify: bool = False, apify_token: str = None):
        """
        Initialize AliExpress ETL
        
        Args:
            app_key: AliExpress Open Platform app key
            app_secret: AliExpress Open Platform app secret
            access_token: AliExpress access token
            use_apify: Whether to use Apify as fallback
            apify_token: Apify API token
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.access_token = access_token
        self.use_apify = use_apify
        self.apify_token = apify_token
        
        # API endpoints
        self.base_url = "https://api.aliexpress.com/v2"
        self.apify_url = "https://api.apify.com/v2"
        
        self.session = requests.Session()
        self.rate_limit_delay = 0.5  # Be respectful to APIs
        
    def _generate_signature(self, params: Dict[str, Any], app_secret: str) -> str:
        """Generate signature for AliExpress API requests"""
        # Sort parameters alphabetically
        sorted_params = sorted(params.items())
        
        # Create string to sign
        string_to_sign = app_secret
        for key, value in sorted_params:
            string_to_sign += f"{key}{value}"
        string_to_sign += app_secret
        
        # Generate MD5 hash
        signature = hashlib.md5(string_to_sign.encode('utf-8')).hexdigest().upper()
        return signature
    
    def _make_aliexpress_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make a request to AliExpress Open Platform API"""
        if not self.app_key or not self.app_secret:
            logger.warning("AliExpress credentials not configured")
            return None
        
        try:
            # Add common parameters
            params.update({
                "app_key": self.app_key,
                "timestamp": str(int(time.time())),
                "format": "json",
                "v": "2.0",
                "sign_method": "md5"
            })
            
            # Generate signature
            signature = self._generate_signature(params, self.app_secret)
            params["sign"] = signature
            
            # Make request
            url = f"{self.base_url}/{method}"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "error_response" in data:
                logger.error(f"AliExpress API error: {data['error_response']}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"AliExpress API request failed for {method}: {e}")
            return None
    
    def _make_apify_request(self, query: str, page: int = 1) -> Optional[Dict]:
        """Make a request to Apify AliExpress scraper"""
        if not self.use_apify or not self.apify_token:
            return None
        
        try:
            url = f"{self.apify_url}/acts/pintostudio~aliexpress-product-search/run-sync-get-dataset-items"
            params = {
                "token": self.apify_token
            }
            
            data = {
                "query": query,
                "page": page
            }
            
            response = self.session.post(url, params=params, json=data)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Apify API request failed: {e}")
            return None
    
    def search_products(self, query: str, category_id: Optional[str] = None, 
                       min_price: Optional[float] = None, max_price: Optional[float] = None,
                       limit: int = 50) -> List[AliExpressProduct]:
        """Search for products on AliExpress"""
        products = []
        
        # Try official API first
        if self.app_key and self.app_secret:
            products = self._search_aliexpress_official(query, category_id, min_price, max_price, limit)
        
        # Fallback to Apify if needed
        if not products and self.use_apify:
            products = self._search_aliexpress_apify(query, limit)
        
        return products
    
    def _search_aliexpress_official(self, query: str, category_id: Optional[str] = None,
                                   min_price: Optional[float] = None, max_price: Optional[float] = None,
                                   limit: int = 50) -> List[AliExpressProduct]:
        """Search using official AliExpress API"""
        try:
            params = {
                "method": "aliexpress.ds.product.search",
                "keywords": query,
                "page_size": min(limit, 50),  # API limit
                "page_no": 1
            }
            
            if category_id:
                params["category_id"] = category_id
            
            if min_price:
                params["min_price"] = str(min_price)
            
            if max_price:
                params["max_price"] = str(max_price)
            
            data = self._make_aliexpress_request("aliexpress.ds.product.search", params)
            if not data:
                return []
            
            products = []
            product_list = data.get("result", {}).get("products", [])
            
            for item in product_list:
                try:
                    product = self._parse_aliexpress_product(item)
                    if product:
                        products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to parse AliExpress product: {e}")
                    continue
            
            return products
            
        except Exception as e:
            logger.error(f"Official AliExpress search failed: {e}")
            return []
    
    def _search_aliexpress_apify(self, query: str, limit: int = 50) -> List[AliExpressProduct]:
        """Search using Apify AliExpress scraper"""
        try:
            products = []
            page = 1
            
            while len(products) < limit:
                data = self._make_apify_request(query, page)
                if not data:
                    break
                
                for item in data:
                    try:
                        product = self._parse_apify_product(item)
                        if product:
                            products.append(product)
                            
                        if len(products) >= limit:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse Apify product: {e}")
                        continue
                
                page += 1
                time.sleep(self.rate_limit_delay)
            
            return products[:limit]
            
        except Exception as e:
            logger.error(f"Apify AliExpress search failed: {e}")
            return []
    
    def _parse_aliexpress_product(self, item: Dict[str, Any]) -> Optional[AliExpressProduct]:
        """Parse product data from official AliExpress API"""
        try:
            # Extract shipping options
            shipping_options = []
            if "shipping_info" in item:
                shipping_info = item["shipping_info"]
                shipping_options.append({
                    "method": shipping_info.get("shipping_method", "Standard"),
                    "cost": float(shipping_info.get("shipping_cost", 0)),
                    "currency": shipping_info.get("currency", "USD"),
                    "delivery_days": int(shipping_info.get("delivery_time", 15)),
                    "tracking_available": shipping_info.get("tracking_available", False),
                    "insurance_available": shipping_info.get("insurance_available", False)
                })
            
            product = AliExpressProduct(
                product_id=str(item.get("product_id", "")),
                title=item.get("product_title", ""),
                unit_price=float(item.get("min_price", 0)),
                currency=item.get("currency", "USD"),
                min_order_quantity=int(item.get("min_order_quantity", 1)),
                max_order_quantity=int(item.get("max_order_quantity", 999999)),
                available_quantity=int(item.get("available_quantity", 0)),
                image_urls=item.get("product_images", []),
                product_url=item.get("product_url", ""),
                seller_id=str(item.get("seller_id", "")),
                seller_name=item.get("seller_name", ""),
                seller_rating=float(item.get("seller_rating", 0)),
                seller_feedback_score=int(item.get("seller_feedback_score", 0)),
                ship_from=item.get("ship_from", ""),
                shipping_options=shipping_options,
                product_rating=float(item.get("product_rating", 0)),
                review_count=int(item.get("review_count", 0)),
                category=item.get("category_name", ""),
                attributes=item.get("product_attributes", {}),
                is_verified=item.get("is_verified", False),
                is_premium=item.get("is_premium", False)
            )
            
            return product
            
        except Exception as e:
            logger.warning(f"Failed to parse AliExpress product data: {e}")
            return None
    
    def _parse_apify_product(self, item: Dict[str, Any]) -> Optional[AliExpressProduct]:
        """Parse product data from Apify scraper"""
        try:
            # Extract shipping options
            shipping_options = []
            if "shipping" in item:
                shipping = item["shipping"]
                shipping_options.append({
                    "method": shipping.get("method", "Standard"),
                    "cost": float(shipping.get("cost", 0)),
                    "currency": shipping.get("currency", "USD"),
                    "delivery_days": int(shipping.get("days", 15)),
                    "tracking_available": shipping.get("tracking", False),
                    "insurance_available": False
                })
            
            product = AliExpressProduct(
                product_id=str(item.get("id", "")),
                title=item.get("title", ""),
                unit_price=float(item.get("price", 0)),
                currency=item.get("currency", "USD"),
                min_order_quantity=int(item.get("min_order", 1)),
                max_order_quantity=999999,
                available_quantity=int(item.get("stock", 0)),
                image_urls=[item.get("image", "")] if item.get("image") else [],
                product_url=item.get("url", ""),
                seller_id=str(item.get("seller_id", "")),
                seller_name=item.get("seller", ""),
                seller_rating=float(item.get("seller_rating", 0)),
                seller_feedback_score=int(item.get("seller_feedback", 0)),
                ship_from=item.get("ship_from", ""),
                shipping_options=shipping_options,
                product_rating=float(item.get("rating", 0)),
                review_count=int(item.get("reviews", 0)),
                category=item.get("category", ""),
                attributes={},
                is_verified=False,
                is_premium=False
            )
            
            return product
            
        except Exception as e:
            logger.warning(f"Failed to parse Apify product data: {e}")
            return None
    
    def get_product_details(self, product_id: str) -> Optional[AliExpressProduct]:
        """Get detailed product information"""
        try:
            params = {
                "method": "aliexpress.ds.product.get",
                "product_id": product_id
            }
            
            data = self._make_aliexpress_request("aliexpress.ds.product.get", params)
            if not data:
                return None
            
            product_data = data.get("result", {}).get("product", {})
            return self._parse_aliexpress_product(product_data)
            
        except Exception as e:
            logger.error(f"Failed to get product details for {product_id}: {e}")
            return None
    
    def get_shipping_quote(self, product_id: str, quantity: int, 
                          country_code: str = "US") -> List[ShippingOption]:
        """Get shipping quote for a product"""
        try:
            params = {
                "method": "aliexpress.ds.shipping.get",
                "product_id": product_id,
                "quantity": quantity,
                "country_code": country_code
            }
            
            data = self._make_aliexpress_request("aliexpress.ds.shipping.get", params)
            if not data:
                return []
            
            shipping_options = []
            shipping_list = data.get("result", {}).get("shipping_options", [])
            
            for option in shipping_list:
                try:
                    shipping_option = ShippingOption(
                        method=option.get("shipping_method", ""),
                        cost=float(option.get("shipping_cost", 0)),
                        currency=option.get("currency", "USD"),
                        delivery_days=int(option.get("delivery_time", 15)),
                        tracking_available=option.get("tracking_available", False),
                        insurance_available=option.get("insurance_available", False)
                    )
                    shipping_options.append(shipping_option)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse shipping option: {e}")
                    continue
            
            return shipping_options
            
        except Exception as e:
            logger.error(f"Failed to get shipping quote for {product_id}: {e}")
            return []
    
    def calculate_landed_cost(self, product: AliExpressProduct, quantity: int = 1,
                             country_code: str = "US", include_buffer: bool = True) -> Dict[str, Any]:
        """Calculate total landed cost including shipping and fees"""
        try:
            # Get shipping options
            shipping_options = self.get_shipping_quote(product.product_id, quantity, country_code)
            
            if not shipping_options:
                # Use default shipping if no quote available
                default_shipping = {
                    "cost": 2.0,  # Default shipping cost
                    "delivery_days": 15,
                    "method": "Standard"
                }
            else:
                # Use cheapest shipping option
                cheapest_shipping = min(shipping_options, key=lambda x: x.cost)
                default_shipping = {
                    "cost": cheapest_shipping.cost,
                    "delivery_days": cheapest_shipping.delivery_days,
                    "method": cheapest_shipping.method
                }
            
            # Calculate costs
            unit_cost = product.unit_price
            total_product_cost = unit_cost * quantity
            shipping_cost = default_shipping["cost"]
            
            # Add buffer if requested
            buffer_multiplier = 1.05 if include_buffer else 1.0
            
            # Calculate total landed cost
            landed_cost = (total_product_cost + shipping_cost) * buffer_multiplier
            
            # Calculate risk factor for higher-priced items
            risk_factor = self._calculate_risk_factor(landed_cost)
            
            return {
                "unit_cost": unit_cost,
                "total_product_cost": total_product_cost,
                "shipping_cost": shipping_cost,
                "buffer_amount": (total_product_cost + shipping_cost) * 0.05 if include_buffer else 0,
                "landed_cost": landed_cost,
                "risk_factor": risk_factor,
                "delivery_days": default_shipping["delivery_days"],
                "shipping_method": default_shipping["method"],
                "is_acceptable": landed_cost <= 10.0 or risk_factor <= 0.7
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate landed cost for {product.product_id}: {e}")
            return {}
    
    def _calculate_risk_factor(self, landed_cost: float) -> float:
        """Calculate risk factor for higher-priced items (0-1, higher = more risky)"""
        if landed_cost <= 10.0:
            return 0.0  # No risk for items within cap
        
        # Calculate risk based on cost increase
        cost_ratio = landed_cost / 10.0
        
        if cost_ratio <= 1.5:
            # 1.0x to 1.5x: Low risk
            risk_factor = (cost_ratio - 1.0) * 0.2
        elif cost_ratio <= 2.0:
            # 1.5x to 2.0x: Medium risk
            risk_factor = 0.1 + (cost_ratio - 1.5) * 0.3
        else:
            # 2.0x+: High risk
            risk_factor = 0.25 + (cost_ratio - 2.0) * 0.15
        
        return min(1.0, risk_factor)
    
    def analyze_supplier_performance(self, products: List[AliExpressProduct]) -> Dict[str, Any]:
        """Analyze overall supplier performance metrics"""
        if not products:
            return {}
        
        # Calculate metrics
        total_products = len(products)
        avg_price = sum(p.unit_price for p in products) / total_products
        avg_rating = sum(p.seller_rating for p in products if p.seller_rating > 0) / max(1, sum(1 for p in products if p.seller_rating > 0))
        verified_count = sum(1 for p in products if p.is_verified)
        premium_count = sum(1 for p in products if p.is_premium)
        
        # Shipping analysis
        all_shipping_options = []
        for product in products:
            all_shipping_options.extend(product.shipping_options)
        
        if all_shipping_options:
            avg_shipping_cost = sum(opt["cost"] for opt in all_shipping_options) / len(all_shipping_options)
            avg_delivery_days = sum(opt["delivery_days"] for opt in all_shipping_options) / len(all_shipping_options)
        else:
            avg_shipping_cost = 0
            avg_delivery_days = 15
        
        return {
            "total_products": total_products,
            "avg_unit_price": avg_price,
            "avg_seller_rating": avg_rating,
            "verified_suppliers_percentage": verified_count / total_products if total_products > 0 else 0,
            "premium_suppliers_percentage": premium_count / total_products if total_products > 0 else 0,
            "avg_shipping_cost": avg_shipping_cost,
            "avg_delivery_days": avg_delivery_days,
            "price_range": {
                "min": min(p.unit_price for p in products),
                "max": max(p.unit_price for p in products)
            }
        }
    
    def export_supplier_data(self, products: List[AliExpressProduct], 
                            output_dir: str = "data/suppliers") -> None:
        """Export supplier data to CSV files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Convert to DataFrame
            data = []
            for product in products:
                data.append({
                    "product_id": product.product_id,
                    "title": product.title,
                    "unit_price": product.unit_price,
                    "currency": product.currency,
                    "seller_name": product.seller_name,
                    "seller_rating": product.seller_rating,
                    "ship_from": product.ship_from,
                    "product_rating": product.product_rating,
                    "review_count": product.review_count,
                    "category": product.category,
                    "is_verified": product.is_verified,
                    "is_premium": product.is_premium
                })
            
            df = pd.DataFrame(data)
            filename = f"{output_dir}/aliexpress_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(products)} supplier products to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export supplier data: {e}")

def main():
    """Test function for AliExpress ETL"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test with Apify fallback
    apify_token = os.getenv("APIFY_TOKEN")
    
    if not apify_token:
        print("Please set APIFY_TOKEN environment variable for testing")
        return
    
    aliexpress_etl = AliExpressETL(use_apify=True, apify_token=apify_token)
    
    # Test search
    print("Searching for 'wireless earbuds'...")
    products = aliexpress_etl.search_products("wireless earbuds", limit=10)
    
    if products:
        print(f"Found {len(products)} products")
        
        # Show first product details
        first_product = products[0]
        print(f"\nFirst product: {first_product.title}")
        print(f"Price: ${first_product.unit_price}")
        print(f"Seller: {first_product.seller_name} (Rating: {first_product.seller_rating})")
        print(f"Ship from: {first_product.ship_from}")
        
        # Calculate landed cost
        cost_analysis = aliexpress_etl.calculate_landed_cost(first_product, quantity=1)
        if cost_analysis:
            print(f"Landed cost: ${cost_analysis['landed_cost']:.2f}")
            print(f"Risk factor: {cost_analysis['risk_factor']:.3f}")
            print(f"Acceptable: {cost_analysis['is_acceptable']}")
        
        # Analyze supplier performance
        analysis = aliexpress_etl.analyze_supplier_performance(products)
        print(f"\nSupplier analysis: {analysis}")
        
        # Export data
        aliexpress_etl.export_supplier_data(products)
        
    else:
        print("No products found")

if __name__ == "__main__":
    main()

# Simplified AliExpress search via Apify (temporary solution)
def search_aliexpress_apify(query, page=1, token=None):
    """Search AliExpress products using Apify actor"""
    token = token or os.environ.get("APIFY_TOKEN")
    if not token:
        raise ValueError("APIFY_TOKEN must be set in .env file")
    
    url = f"https://api.apify.com/v2/acts/pintostudio~aliexpress-product-search/run-sync-get-dataset-items?token={token}"
    payload = {"query": query, "page": page}
    
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        items = r.json()
        
        # Normalize fields to match pipeline expectations
        out = []
        for it in items[:20]:  # Limit to 20 items
            out.append({
                "supplier_id": it.get("productId"),
                "title": it.get("title"),
                "unit_price": it.get("price", 0.0),
                "ship_cost": it.get("shippingCost", 0.0),
                "ship_days_est": it.get("deliveryTime", 15),
                "seller_rating": it.get("storePositiveFeedbackRate", 4.7),
                "ship_from_country": it.get("shipFrom") or "CN",
                "image_url": (it.get("images") or [None])[0],
                "url": it.get("url")
            })
        
        logger.info(f"Found {len(out)} AliExpress products via Apify for query: {query}")
        return out
        
    except Exception as e:
        logger.error(f"Apify AliExpress search failed for query '{query}': {e}")
        return []
