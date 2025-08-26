"""
TikTok Shop ETL Module for Winning Product Finder
Tracks product sales, engagement, and virality signals
"""

import requests
import time
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TikTokShopProduct:
    """Data class for TikTok Shop product information"""
    product_id: str
    title: str
    price: float
    currency: str
    image_url: str
    shop_url: str
    shop_name: str
    shop_rating: float
    total_sold: int
    sales_timeframe_days: int
    engagement_rate: float
    video_count: int
    avg_views: int
    avg_likes: int
    avg_shares: int
    avg_comments: int
    is_trending: bool
    category: str
    tags: List[str]

class TikTokShopETL:
    """TikTok Shop ETL class for tracking product performance"""
    
    def __init__(self, access_token: str = None, use_apify: bool = False, apify_token: str = None, apify_actor: str = None):
        """
        Initialize TikTok Shop ETL
        
        Args:
            access_token: TikTok Shop API access token (if available)
            use_apify: Whether to use Apify as fallback
            apify_token: Apify API token
            apify_actor: Apify actor to use (e.g., "clockworks/tiktok-scraper")
        """
        self.access_token = access_token
        self.use_apify = use_apify
        self.apify_token = apify_token
        self.apify_actor = apify_actor or "clockworks/tiktok-scraper"
        
        # API endpoints
        self.base_url = "https://open.tiktokapis.com/v2"
        self.apify_url = "https://api.apify.com/v2"
        
        self.session = requests.Session()
        self.rate_limit_delay = 1.0  # Be respectful to APIs
    
    def search_trending_products(self, query: str, limit: int = 50) -> List[TikTokShopProduct]:
        """
        Search for trending products on TikTok Shop
        
        Args:
            query: Search query
            limit: Maximum number of products to return
        
        Returns:
            List of TikTok Shop products
        """
        if self.access_token:
            return self._search_official_api(query, limit)
        elif self.use_apify and self.apify_token:
            return self._search_apify_fallback(query, limit)
        else:
            logger.warning("No TikTok Shop access available")
            return []
    
    def _search_official_api(self, query: str, limit: int) -> List[TikTokShopProduct]:
        """Search using official TikTok Shop API"""
        try:
            # This would use the official TikTok Shop API
            # For now, return empty list as we don't have official access
            logger.info("Official TikTok Shop API not yet implemented")
            return []
            
        except Exception as e:
            logger.error(f"Official TikTok Shop search failed: {e}")
            return []
    
    def _search_apify_fallback(self, query: str, limit: int) -> List[TikTokShopProduct]:
        """Search using Apify TikTok Shop actor as fallback"""
        try:
            url = f"{self.apify_url}/acts/{self.apify_actor}/run-sync-get-dataset-items?token={self.apify_token}"
            payload = {
                "query": query,
                "limit": limit,
                "country": "US"
            }
            
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            items = response.json()
            products = []
            
            for item in items[:limit]:
                try:
                    product = TikTokShopProduct(
                        product_id=item.get("productId", ""),
                        title=item.get("title", ""),
                        price=float(item.get("price", 0.0)),
                        currency=item.get("currency", "USD"),
                        image_url=item.get("imageUrl", ""),
                        shop_url=item.get("shopUrl", ""),
                        shop_name=item.get("shopName", ""),
                        shop_rating=float(item.get("shopRating", 4.5)),
                        total_sold=int(item.get("totalSold", 0)),
                        sales_timeframe_days=int(item.get("salesTimeframeDays", 30)),
                        engagement_rate=float(item.get("engagementRate", 0.0)),
                        video_count=int(item.get("videoCount", 0)),
                        avg_views=int(item.get("avgViews", 0)),
                        avg_likes=int(item.get("avgLikes", 0)),
                        avg_shares=int(item.get("avgShares", 0)),
                        avg_comments=int(item.get("avgComments", 0)),
                        is_trending=bool(item.get("isTrending", False)),
                        category=item.get("category", ""),
                        tags=item.get("tags", [])
                    )
                    products.append(product)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse TikTok Shop product: {e}")
                    continue
            
            logger.info(f"Found {len(products)} TikTok Shop products for query: {query}")
            return products
            
        except Exception as e:
            logger.error(f"Apify TikTok Shop search failed: {e}")
            return []
    
    def get_product_performance(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed performance metrics for a specific product
        
        Args:
            product_id: TikTok Shop product ID
        
        Returns:
            Performance metrics dictionary
        """
        try:
            # This would fetch detailed metrics from TikTok Shop API
            # For now, return mock data structure
            return {
                "product_id": product_id,
                "sales_velocity": 0.0,  # Sales per day
                "engagement_trend": 0.0,  # Engagement growth rate
                "competition_level": 0.5,  # Normalized competition score
                "virality_score": 0.0,    # Viral potential (0-1)
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get product performance for {product_id}: {e}")
            return None
    
    def analyze_virality_signals(self, products: List[TikTokShopProduct]) -> Dict[str, Any]:
        """
        Analyze virality signals across products
        
        Args:
            products: List of TikTok Shop products
        
        Returns:
            Virality analysis results
        """
        if not products:
            return {"virality_score": 0.0, "trending_products": 0, "avg_engagement": 0.0}
        
        try:
            # Calculate virality metrics
            trending_count = sum(1 for p in products if p.is_trending)
            avg_engagement = sum(p.engagement_rate for p in products) / len(products)
            
            # Calculate virality score based on engagement and sales velocity
            virality_scores = []
            for product in products:
                # Combine engagement rate and sales velocity
                engagement_score = min(1.0, product.engagement_rate / 10.0)  # Normalize to 0-1
                sales_score = min(1.0, product.total_sold / 1000.0)  # Normalize to 0-1
                virality_score = (engagement_score * 0.7) + (sales_score * 0.3)
                virality_scores.append(virality_score)
            
            avg_virality = sum(virality_scores) / len(virality_scores)
            
            return {
                "virality_score": round(avg_virality, 3),
                "trending_products": trending_count,
                "avg_engagement": round(avg_engagement, 3),
                "total_products": len(products)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze virality signals: {e}")
            return {"virality_score": 0.0, "trending_products": 0, "avg_engagement": 0.0}

# Simplified TikTok Shop search function for pipeline integration
def search_tiktok_shop_apify(query: str, token: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search TikTok Shop products using Apify (fallback method)
    
    Args:
        query: Search query
        token: Apify API token
        limit: Maximum results
    
    Returns:
        List of product dictionaries
    """
    if not token:
        logger.warning("No Apify token provided for TikTok Shop search")
        return []
    
    try:
        # Use the correct actor from environment variable
        import os
        actor = os.environ.get("APIFY_TIKTOK_ACTOR", "clockworks~tiktok-scraper")
        url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items?token={token}"
        payload = {"query": query, "limit": limit, "country": "US"}
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        items = response.json()
        
        # Normalize to pipeline format
        normalized_products = []
        for item in items[:limit]:
            normalized_product = {
                "tiktok_product_id": item.get("productId", ""),
                "title": item.get("title", ""),
                "price": float(item.get("price", 0.0)),
                "shop_name": item.get("shopName", ""),
                "shop_rating": float(item.get("shopRating", 4.5)),
                "total_sold": int(item.get("totalSold", 0)),
                "sales_timeframe_days": int(item.get("salesTimeframeDays", 30)),
                "engagement_rate": float(item.get("engagementRate", 0.0)),
                "video_count": int(item.get("videoCount", 0)),
                "avg_views": int(item.get("avgViews", 0)),
                "is_trending": bool(item.get("isTrending", False)),
                "shop_url": item.get("shopUrl", ""),
                "image_url": item.get("imageUrl", "")
            }
            normalized_products.append(normalized_product)
        
        logger.info(f"Found {len(normalized_products)} TikTok Shop products for query: {query}")
        return normalized_products
        
    except Exception as e:
        logger.error(f"TikTok Shop search failed: {e}")
        return []
