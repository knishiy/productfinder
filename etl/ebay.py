"""
eBay ETL Module for Winning Product Finder
Handles authentication, product search, and best-selling products collection
"""

import requests
import base64
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from rapidfuzz import fuzz

# eBay API URLs
EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
MARKETING_URL = "https://api.ebay.com/buy/marketing/v1_beta/merchandised_product"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EbayProduct:
    """Data class for eBay product information"""
    item_id: str
    title: str
    price: float
    currency: str
    image_url: str
    item_web_url: str
    seller_username: str
    seller_feedback_score: int
    seller_positive_feedback_percent: float
    buying_options: List[str]
    condition: str
    category_id: str
    category_name: str
    location: str
    shipping_cost: float
    shipping_type: str
    top_rated_seller: bool
    best_offer_enabled: bool
    listing_type: str
    end_time: str
    is_best_seller: bool = False
    merchandised_rank: Optional[int] = None

class EbayETL:
    """eBay ETL class for data collection"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry = None
        self.base_url = "https://api.ebay.com"
        self.session = requests.Session()
        
    def _get_auth_header(self) -> str:
        """Get Basic auth header for client credentials"""
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_credentials}"
    
    def authenticate(self) -> bool:
        """Authenticate with eBay API using client credentials"""
        try:
            url = f"{self.base_url}/identity/v1/oauth2/token"
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": self._get_auth_header()
            }
            data = {
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope"
            }
            
            response = self.session.post(url, headers=headers, data=data)
            
            # Log the response for debugging
            logger.info(f"eBay auth response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"eBay auth error response: {response.text}")
            
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.token_expiry = datetime.now() + timedelta(seconds=token_data["expires_in"] - 300)  # 5 min buffer
            
            # Update session headers
            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            })
            
            logger.info("eBay authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"eBay authentication failed: {e}")
            return False
    
    def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid access token"""
        if not self.access_token or (self.token_expiry and datetime.now() >= self.token_expiry):
            return self.authenticate()
        return True
    
    def search_products(self, query: str, category_id: str, limit: int = 200) -> List[EbayProduct]:
        """Search for products using eBay Browse API"""
        if not self._ensure_authenticated():
            return []
        
        try:
            url = f"{self.base_url}/buy/browse/v1/item_summary/search"
            params = {
                "q": query,
                "category_ids": category_id,
                "limit": limit,
                "filter": "buyingOptions:{AUCTION|FIXED_PRICE|BEST_OFFER},conditions:{NEW|USED_EXCELLENT|USED_VERY_GOOD|USED_GOOD}",
                "sort": "newlyListed"  # Can be: newlyListed, price, distance, newlyListed
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            products = []
            
            for item in data.get("itemSummaries", []):
                try:
                    product = EbayProduct(
                        item_id=item.get("itemId", ""),
                        title=item.get("title", ""),
                        price=float(item.get("price", {}).get("value", 0)),
                        currency=item.get("price", {}).get("currency", "USD"),
                        image_url=item.get("image", {}).get("imageUrl", ""),
                        item_web_url=item.get("itemWebUrl", ""),
                        seller_username=item.get("seller", {}).get("username", ""),
                        seller_feedback_score=int(item.get("seller", {}).get("feedbackScore", 0)),
                        seller_positive_feedback_percent=float(item.get("seller", {}).get("positiveFeedbackPercent", 0)),
                        buying_options=item.get("buyingOptions", []),
                        condition=item.get("condition", ""),
                        category_id=item.get("categoryId", ""),
                        category_name=item.get("categoryName", ""),
                        location=item.get("itemLocation", {}).get("country", ""),
                        shipping_cost=float(item.get("shippingOptions", [{}])[0].get("shippingCost", {}).get("value", 0) if item.get("shippingOptions") else 0),
                        shipping_type=item.get("shippingOptions", [{}])[0].get("shippingType", "") if item.get("shippingOptions") else "",
                        top_rated_seller=item.get("seller", {}).get("topRatedSeller", False),
                        best_offer_enabled=item.get("bestOfferEnabled", False),
                        listing_type=item.get("listingType", ""),
                        end_time=item.get("itemEndDate", "")
                    )
                    products.append(product)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse product {item.get('itemId', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(products)} products for query: {query}")
            return products
            
        except Exception as e:
            logger.error(f"eBay search failed for query '{query}': {e}")
            return []
    
    def get_merchandised_products(self, category_id: str, limit: int = 100) -> List[EbayProduct]:
        """Get merchandised best-selling products for a category"""
        if not self._ensure_authenticated():
            return []
        
        try:
            url = f"{self.base_url}/buy/marketing/v1_beta/merchandised_product"
            params = {
                "metric_name": "BEST_SELLING",
                "category_id": category_id,
                "limit": limit
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            products = []
            
            for item in data.get("merchandisedProducts", []):
                try:
                    # Get detailed product info for merchandised items
                    detailed_product = self._get_product_details(item.get("productId", ""))
                    if detailed_product:
                        detailed_product.is_best_seller = True
                        detailed_product.merchandised_rank = item.get("rank", 0)
                        products.append(detailed_product)
                        
                except Exception as e:
                    logger.warning(f"Failed to get details for merchandised product {item.get('productId', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(products)} merchandised products for category: {category_id}")
            return products
            
        except Exception as e:
            logger.error(f"eBay merchandised products failed for category {category_id}: {e}")
            return []
    
    def _get_product_details(self, product_id: str) -> Optional[EbayProduct]:
        """Get detailed product information by product ID"""
        if not self._ensure_authenticated():
            return None
        
        try:
            url = f"{self.base_url}/buy/browse/v1/item/{product_id}"
            response = self.session.get(url)
            response.raise_for_status()
            
            item = response.json()
            
            product = EbayProduct(
                item_id=item.get("itemId", ""),
                title=item.get("title", ""),
                price=float(item.get("price", {}).get("value", 0)),
                currency=item.get("price", {}).get("currency", "USD"),
                image_url=item.get("image", {}).get("imageUrl", ""),
                item_web_url=item.get("itemWebUrl", ""),
                seller_username=item.get("seller", {}).get("username", ""),
                seller_feedback_score=int(item.get("seller", {}).get("feedbackScore", 0)),
                seller_positive_feedback_percent=float(item.get("seller", {}).get("positiveFeedbackPercent", 0)),
                buying_options=item.get("buyingOptions", []),
                condition=item.get("condition", ""),
                category_id=item.get("categoryId", ""),
                category_name=item.get("categoryName", ""),
                location=item.get("itemLocation", {}).get("country", ""),
                shipping_cost=float(item.get("shippingOptions", [{}])[0].get("shippingCost", {}).get("value", 0) if item.get("shippingOptions") else 0),
                shipping_type=item.get("shippingOptions", [{}])[0].get("shippingType", "") if item.get("shippingOptions") else "",
                top_rated_seller=item.get("seller", {}).get("topRatedSeller", False),
                best_offer_enabled=item.get("bestOfferEnabled", False),
                listing_type=item.get("listingType", ""),
                end_time=item.get("itemEndDate", "")
            )
            return product
            
        except Exception as e:
            logger.warning(f"Failed to get product details for {product_id}: {e}")
            return None
    
    def collect_category_data(self, category_id: str, keywords: List[str]) -> Dict[str, List[EbayProduct]]:
        """Collect comprehensive data for a category including search results and merchandised products"""
        all_products = {}
        
        # Get merchandised best-sellers first
        merchandised_products = self.get_merchandised_products(category_id)
        all_products["merchandised"] = merchandised_products
        
        # Search for each keyword
        for keyword in keywords:
            products = self.search_products(keyword, category_id)
            all_products[keyword] = products
            
            # Rate limiting - be respectful to eBay API
            time.sleep(0.1)
        
        return all_products
    
    def analyze_competition(self, products: List[EbayProduct]) -> Dict[str, float]:
        """Analyze competition density and market saturation"""
        if not products:
            return {"competition_density": 0.0, "saturation_score": 0.0}
        
        # Calculate competition density based on similar titles
        competition_scores = []
        for i, product1 in enumerate(products):
            similar_count = 0
            for j, product2 in enumerate(products):
                if i != j:
                    # Use token set ratio for better title similarity
                    similarity = fuzz.token_set_ratio(product1.title.lower(), product2.title.lower())
                    if similarity > 70:  # High similarity threshold
                        similar_count += 1
            
            competition_scores.append(similar_count)
        
        avg_competition = sum(competition_scores) / len(competition_scores) if competition_scores else 0
        competition_density = min(1.0, avg_competition / 10)  # Normalize to 0-1
        
        # Calculate saturation based on price clustering
        prices = [p.price for p in products if p.price > 0]
        if len(prices) > 1:
            price_std = (max(prices) - min(prices)) / (sum(prices) / len(prices))
            saturation_score = min(1.0, price_std * 2)  # Normalize to 0-1
        else:
            saturation_score = 0.0
        
        return {
            "competition_density": competition_density,
            "saturation_score": saturation_score
        }

def main():
    """Test function for eBay ETL"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client_id = os.getenv("EBAY_CLIENT_ID")
    client_secret = os.getenv("EBAY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Please set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET environment variables")
        return
    
    ebay_etl = EbayETL(client_id, client_secret)
    
    # Test authentication
    if ebay_etl.authenticate():
        print("Authentication successful!")
        
        # Test search
        products = ebay_etl.search_products("wireless earbuds", "15032", limit=10)
        print(f"Found {len(products)} products")
        
        # Test merchandised products
        merchandised = ebay_etl.get_merchandised_products("15032", limit=5)
        print(f"Found {len(merchandised)} merchandised products")
        
        # Test competition analysis
        if products:
            analysis = ebay_etl.analyze_competition(products)
            print(f"Competition analysis: {analysis}")
    else:
        print("Authentication failed!")

if __name__ == "__main__":
    main()

# Simplified eBay functions for quick integration
def get_token():
    """Get eBay access token using environment variables"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    cid = os.environ.get("EBAY_CLIENT_ID")
    cs = os.environ.get("EBAY_CLIENT_SECRET")
    
    if not cid or not cs:
        raise ValueError("EBAY_CLIENT_ID and EBAY_CLIENT_SECRET must be set in .env file")
    
    auth = requests.auth.HTTPBasicAuth(cid, cs)
    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }
    r = requests.post(EBAY_OAUTH_URL, auth=auth, data=data,
                      headers={"Content-Type":"application/x-www-form-urlencoded"})
    r.raise_for_status()
    return r.json()["access_token"]

def search_items(token, q=None, category_id=None, limit=100):
    """Search eBay items using the Browse API"""
    params = {"limit": limit}
    if q: 
        params["q"] = q
    if category_id: 
        params["category_ids"] = category_id
    
    h = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    r = requests.get(BROWSE_URL, params=params, headers=h, timeout=30)
    r.raise_for_status()
    return r.json().get("itemSummaries", [])

def best_sellers(token, category_id, limit=100):
    """Get best-selling products for a category using the Marketing API"""
    params = {"metric_name": "BEST_SELLING", "category_id": category_id, "limit": limit}
    h = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    r = requests.get(MARKETING_URL, params=params, headers=h, timeout=30)
    r.raise_for_status()
    return r.json().get("merchandisedProducts", [])
