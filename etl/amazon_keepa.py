"""
Amazon Keepa ETL Module for Winning Product Finder
Handles product research, price history, and sales rank analysis
"""

import requests
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KeepaProduct:
    """Data class for Keepa product information"""
    asin: str
    title: str
    brand: str
    category: str
    current_price: float
    current_sales_rank: int
    price_history: List[Tuple[int, float]]  # (timestamp, price)
    sales_rank_history: List[Tuple[int, int]]  # (timestamp, rank)
    price_stability_score: float
    rank_momentum_score: float
    price_trend_14d: float
    rank_trend_14d: float
    is_prime: bool
    rating: float
    review_count: int
    image_url: str
    product_url: str

class KeepaETL:
    """Keepa ETL class for Amazon product data collection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.keepa.com"
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # Keepa allows 100 requests per minute
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make a request to Keepa API with rate limiting"""
        try:
            url = f"{self.base_url}/{endpoint}"
            params["key"] = self.api_key
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Keepa API request failed for {endpoint}: {e}")
            return None
    
    def get_product(self, asin: str, history: bool = True) -> Optional[KeepaProduct]:
        """Get detailed product information including price and sales rank history"""
        params = {
            "domain": 1,  # US domain
            "asin": asin,
            "history": 1 if history else 0
        }
        
        data = self._make_request("product", params)
        if not data:
            return None
        
        try:
            product_data = data.get("products", [{}])[0]
            
            # Parse price history
            price_history = []
            if "csv" in product_data and history:
                price_csv = product_data["csv"][0]  # Amazon price history
                for i, price in enumerate(price_csv):
                    if price != -1:  # Keepa uses -1 for no data
                        timestamp = product_data["timestamp"] + (i * 3600000)  # Convert to milliseconds
                        price_history.append((timestamp, price / 100.0))  # Convert from cents
            
            # Parse sales rank history
            sales_rank_history = []
            if "csv" in product_data and history:
                rank_csv = product_data["csv"][1]  # Sales rank history
                for i, rank in enumerate(rank_csv):
                    if rank != -1:
                        timestamp = product_data["timestamp"] + (i * 3600000)
                        sales_rank_history.append((timestamp, rank))
            
            # Calculate metrics
            price_stability_score = self._calculate_price_stability(price_history)
            rank_momentum_score = self._calculate_rank_momentum(sales_rank_history)
            price_trend_14d = self._calculate_price_trend(price_history, days=14)
            rank_trend_14d = self._calculate_rank_trend(sales_rank_history, days=14)
            
            product = KeepaProduct(
                asin=asin,
                title=product_data.get("title", ""),
                brand=product_data.get("brand", ""),
                category=product_data.get("categoryTree", [""])[-1] if product_data.get("categoryTree") else "",
                current_price=product_data.get("stats", {}).get("current", {}).get("AMAZON", 0) / 100.0,
                current_sales_rank=product_data.get("stats", {}).get("current", {}).get("SALES", 0),
                price_history=price_history,
                sales_rank_history=sales_rank_history,
                price_stability_score=price_stability_score,
                rank_momentum_score=rank_momentum_score,
                price_trend_14d=price_trend_14d,
                rank_trend_14d=rank_trend_14d,
                is_prime=product_data.get("isPrime", False),
                rating=product_data.get("rating", 0.0),
                review_count=product_data.get("reviewCount", 0),
                image_url=product_data.get("imagesCSV", "").split(",")[0] if product_data.get("imagesCSV") else "",
                product_url=f"https://www.amazon.com/dp/{asin}"
            )
            
            return product
            
        except Exception as e:
            logger.error(f"Failed to parse Keepa product data for {asin}: {e}")
            return None
    
    def search_products(self, query: str, category_id: Optional[str] = None, limit: int = 50) -> List[str]:
        """Search for products and return ASINs"""
        params = {
            "domain": 1,
            "term": query,
            "limit": limit
        }
        
        if category_id:
            params["category"] = category_id
        
        data = self._make_request("search", params)
        if not data:
            return []
        
        try:
            asins = []
            for product in data.get("products", []):
                asin = product.get("asin")
                if asin:
                    asins.append(asin)
            
            return asins[:limit]
            
        except Exception as e:
            logger.error(f"Failed to parse Keepa search results: {e}")
            return []
    
    def get_deals(self, category_id: Optional[str] = None, limit: int = 100) -> List[str]:
        """Get current deals and return ASINs"""
        params = {
            "domain": 1,
            "limit": limit
        }
        
        if category_id:
            params["category"] = category_id
        
        data = self._make_request("deals", params)
        if not data:
            return []
        
        try:
            asins = []
            for deal in data.get("deals", []):
                asin = deal.get("asin")
                if asin:
                    asins.append(asin)
            
            return asins[:limit]
            
        except Exception as e:
            logger.error(f"Failed to parse Keepa deals: {e}")
            return []
    
    def get_categories(self) -> Dict[str, str]:
        """Get category mapping"""
        data = self._make_request("category", {"domain": 1})
        if not data:
            return {}
        
        try:
            categories = {}
            for category in data.get("categories", []):
                cat_id = str(category.get("catId"))
                cat_name = category.get("name", "")
                if cat_id and cat_name:
                    categories[cat_id] = cat_name
            
            return categories
            
        except Exception as e:
            logger.error(f"Failed to parse Keepa categories: {e}")
            return {}
    
    def _calculate_price_stability(self, price_history: List[Tuple[int, float]]) -> float:
        """Calculate price stability score (0-1, higher = more stable)"""
        if len(price_history) < 2:
            return 0.5
        
        prices = [price for _, price in price_history]
        
        # Calculate coefficient of variation (std/mean)
        mean_price = np.mean(prices)
        if mean_price == 0:
            return 0.5
        
        std_price = np.std(prices)
        cv = std_price / mean_price
        
        # Convert to stability score (0-1)
        stability_score = max(0, 1 - (cv * 2))
        return min(1.0, stability_score)
    
    def _calculate_rank_momentum(self, rank_history: List[Tuple[int, float]]) -> float:
        """Calculate sales rank momentum score (0-1, higher = improving rank)"""
        if len(rank_history) < 2:
            return 0.5
        
        # Recent ranks (last 7 days) vs previous 7 days
        recent_cutoff = datetime.now().timestamp() * 1000 - (7 * 24 * 3600 * 1000)
        
        recent_ranks = [rank for timestamp, rank in rank_history if timestamp > recent_cutoff]
        previous_ranks = [rank for timestamp, rank in rank_history if timestamp <= recent_cutoff]
        
        if not recent_ranks or not previous_ranks:
            return 0.5
        
        recent_avg = np.mean(recent_ranks)
        previous_avg = np.mean(previous_ranks)
        
        # Lower rank = better sales, so we invert the comparison
        if previous_avg == 0:
            return 0.5
        
        improvement = (previous_avg - recent_avg) / previous_avg
        momentum_score = 0.5 + (improvement * 0.5)  # Center at 0.5
        
        return max(0, min(1.0, momentum_score))
    
    def _calculate_price_trend(self, price_history: List[Tuple[int, float]], days: int = 14) -> float:
        """Calculate price trend over specified days (-1 to 1, negative = decreasing)"""
        if len(price_history) < 2:
            return 0.0
        
        cutoff = datetime.now().timestamp() * 1000 - (days * 24 * 3600 * 1000)
        
        recent_prices = [price for timestamp, price in price_history if timestamp > cutoff]
        previous_prices = [price for timestamp, price in price_history if timestamp <= cutoff]
        
        if not recent_prices or not previous_prices:
            return 0.0
        
        recent_avg = np.mean(recent_prices)
        previous_avg = np.mean(previous_prices)
        
        if previous_avg == 0:
            return 0.0
        
        trend = (recent_avg - previous_avg) / previous_avg
        return max(-1.0, min(1.0, trend))
    
    def _calculate_rank_trend(self, rank_history: List[Tuple[int, float]], days: int = 14) -> float:
        """Calculate sales rank trend over specified days (-1 to 1, negative = improving rank)"""
        if len(rank_history) < 2:
            return 0.0
        
        cutoff = datetime.now().timestamp() * 1000 - (days * 24 * 3600 * 1000)
        
        recent_ranks = [rank for timestamp, rank in rank_history if timestamp > cutoff]
        previous_ranks = [rank for timestamp, rank in rank_history if timestamp <= cutoff]
        
        if not recent_ranks or not previous_ranks:
            return 0.0
        
        recent_avg = np.mean(recent_ranks)
        previous_avg = np.mean(previous_ranks)
        
        if previous_avg == 0:
            return 0.0
        
        # Invert because lower rank = better sales
        trend = (previous_avg - recent_avg) / previous_avg
        return max(-1.0, min(1.0, trend))
    
    def analyze_product_performance(self, products: List[KeepaProduct]) -> Dict[str, float]:
        """Analyze overall performance metrics for a set of products"""
        if not products:
            return {
                "avg_price_stability": 0.5,
                "avg_rank_momentum": 0.5,
                "avg_price_trend": 0.0,
                "avg_rank_trend": 0.0,
                "prime_percentage": 0.0,
                "avg_rating": 0.0
            }
        
        price_stabilities = [p.price_stability_score for p in products if p.price_stability_score is not None]
        rank_momenta = [p.rank_momentum_score for p in products if p.rank_momentum_score is not None]
        price_trends = [p.price_trend_14d for p in products if p.price_trend_14d is not None]
        rank_trends = [p.rank_trend_14d for p in products if p.rank_trend_14d is not None]
        ratings = [p.rating for p in products if p.rating > 0]
        prime_count = sum(1 for p in products if p.is_prime)
        
        return {
            "avg_price_stability": np.mean(price_stabilities) if price_stabilities else 0.5,
            "avg_rank_momentum": np.mean(rank_momenta) if rank_momenta else 0.5,
            "avg_price_trend": np.mean(price_trends) if price_trends else 0.0,
            "avg_rank_trend": np.mean(rank_trends) if rank_trends else 0.0,
            "prime_percentage": prime_count / len(products) if products else 0.0,
            "avg_rating": np.mean(ratings) if ratings else 0.0
        }
    
    def batch_get_products(self, asins: List[str]) -> List[KeepaProduct]:
        """Get multiple products efficiently with rate limiting"""
        products = []
        
        for asin in asins:
            product = self.get_product(asin)
            if product:
                products.append(product)
            
            # Rate limiting between requests
            time.sleep(self.rate_limit_delay)
        
        return products

def main():
    """Test function for Keepa ETL"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("KEEPA_API_KEY")
    
    if not api_key:
        print("Please set KEEPA_API_KEY environment variable")
        return
    
    keepa_etl = KeepaETL(api_key)
    
    # Test product lookup
    asin = "B08N5WRWNW"  # Example ASIN
    product = keepa_etl.get_product(asin)
    
    if product:
        print(f"Product: {product.title}")
        print(f"Brand: {product.brand}")
        print(f"Current Price: ${product.current_price}")
        print(f"Sales Rank: {product.current_sales_rank}")
        print(f"Price Stability: {product.price_stability_score:.3f}")
        print(f"Rank Momentum: {product.rank_momentum_score:.3f}")
        print(f"Price Trend 14d: {product.price_trend_14d:.3f}")
        print(f"Rank Trend 14d: {product.rank_trend_14d:.3f}")
        print(f"Prime: {product.is_prime}")
        print(f"Rating: {product.rating}/5 ({product.review_count} reviews)")
    else:
        print("Failed to retrieve product")
    
    # Test search
    print("\nSearching for 'wireless earbuds'...")
    asins = keepa_etl.search_products("wireless earbuds", limit=5)
    print(f"Found {len(asins)} products")
    
    # Test deals
    print("\nGetting current deals...")
    deal_asins = keepa_etl.get_deals(limit=5)
    print(f"Found {len(deal_asins)} deals")

if __name__ == "__main__":
    main()
