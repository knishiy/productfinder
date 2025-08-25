"""
Product Matching Module for Winning Product Finder
Handles matching market products with supplier products using title similarity and image analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from rapidfuzz import fuzz
import imagehash
from PIL import Image
import requests
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Data class for product matching results"""
    market_product_id: str
    supplier_product_id: str
    title_similarity: float
    image_similarity: float
    overall_score: float
    match_confidence: str  # "high", "medium", "low"
    is_match: bool
    match_reason: str

@dataclass
class ProductMatch:
    """Data class for a complete product match"""
    market_product: Any  # eBay/Amazon product
    supplier_product: Any  # AliExpress product
    match_score: float
    title_similarity: float
    image_similarity: float
    landed_cost: float
    risk_factor: float
    margin_percentage: float
    is_viable: bool

class ProductMatcher:
    """Product matching engine using title similarity and image analysis"""
    
    def __init__(self, min_title_similarity: float = 0.65, 
                 min_image_similarity: float = 0.65, 
                 min_overall_score: float = 0.65, use_image_phash: bool = True):
        """
        Initialize Product Matcher
        
        Args:
            min_title_similarity: Minimum title similarity threshold
            min_image_similarity: Minimum image similarity threshold
            min_overall_score: Minimum overall match score threshold
            use_image_phash: Whether to use image perceptual hashing
        """
        self.min_title_similarity = min_title_similarity
        self.min_image_similarity = min_image_similarity
        self.min_overall_score = min_overall_score
        self.use_image_phash = use_image_phash
        
        # Image hash cache to avoid re-downloading
        self.image_hash_cache = {}
        
    def find_matches(self, market_products: List[Any], supplier_products: List[Any],
                     max_suppliers_per_item: int = 5) -> Dict[str, List[ProductMatch]]:
        """
        Find matching suppliers for market products
        
        Args:
            market_products: List of market products (eBay/Amazon)
            supplier_products: List of supplier products (AliExpress)
            max_suppliers_per_item: Maximum number of suppliers per market item
        
        Returns:
            Dictionary mapping market product IDs to list of ProductMatch objects
        """
        matches = {}
        
        logger.info(f"Finding matches for {len(market_products)} market products with {len(supplier_products)} suppliers")
        
        # Process each market product
        for market_product in market_products:
            try:
                product_matches = self._find_matches_for_product(
                    market_product, supplier_products, max_suppliers_per_item
                )
                
                if product_matches:
                    matches[market_product.item_id] = product_matches
                    
            except Exception as e:
                logger.error(f"Failed to find matches for market product {getattr(market_product, 'item_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Found matches for {len(matches)} market products")
        return matches
    
    def title_sim(self, a: str, b: str) -> float:
        """Calculate title similarity using RapidFuzz"""
        if not a or not b:
            return 0.0
        return 0.01 * fuzz.token_set_ratio(a, b) + 0.99 * fuzz.WRatio(a, b)
    
    def image_sim(self, url_a: str, url_b: str) -> float:
        """Calculate image similarity using perceptual hashing"""
        if not self.use_image_phash or not url_a or not url_b:
            return 0.0
        
        try:
            # Get image hashes
            ha = self._get_image_hash(url_a)
            hb = self._get_image_hash(url_b)
            
            if ha is None or hb is None:
                return 0.0
            
            # Calculate similarity (0..1)
            return 1.0 - (ha - hb) / 64.0
            
        except Exception as e:
            logger.warning(f"Image similarity calculation failed: {e}")
            return 0.0
    
    def match_score(self, title_a: str, title_b: str, img_a: str, img_b: str) -> float:
        """Calculate overall match score"""
        t = self.title_sim(title_a or "", title_b or "") / 100.0
        i = self.image_sim(img_a, img_b)
        return 0.6 * t + 0.4 * i
    
    def _find_matches_for_product(self, market_product: Any, supplier_products: List[Any],
                                 max_suppliers_per_item: int) -> List[ProductMatch]:
        """Find matches for a single market product"""
        matches = []
        
        # Generate tokens from market product title
        market_tokens = self._extract_tokens(getattr(market_product, 'title', ''))
        
        for supplier_product in supplier_products:
            try:
                # Calculate similarity scores
                title_similarity = self._calculate_title_similarity(
                    getattr(market_product, 'title', ''),
                    getattr(supplier_product, 'title', '')
                )
                
                image_similarity = self._calculate_image_similarity(
                    getattr(market_product, 'image_url', ''),
                    getattr(supplier_product, 'image_urls', [])
                )
                
                # Calculate overall match score
                overall_score = self._calculate_overall_score(title_similarity, image_similarity)
                
                # Check if this is a valid match
                if overall_score >= self.min_overall_score:
                    # Calculate additional metrics
                    landed_cost = self._calculate_landed_cost(supplier_product)
                    risk_factor = self._calculate_risk_factor(landed_cost)
                    margin_percentage = self._calculate_margin_percentage(
                        getattr(market_product, 'price', 0),
                        landed_cost
                    )
                    
                    # Determine if match is viable
                    is_viable = self._is_viable_match(
                        overall_score, landed_cost, risk_factor, margin_percentage
                    )
                    
                    product_match = ProductMatch(
                        market_product=market_product,
                        supplier_product=supplier_product,
                        match_score=overall_score,
                        title_similarity=title_similarity,
                        image_similarity=image_similarity,
                        landed_cost=landed_cost,
                        risk_factor=risk_factor,
                        margin_percentage=margin_percentage,
                        is_viable=is_viable
                    )
                    
                    matches.append(product_match)
                    
            except Exception as e:
                logger.warning(f"Failed to process supplier product {getattr(supplier_product, 'product_id', 'unknown')}: {e}")
                continue
        
        # Sort by match score and limit results
        matches.sort(key=lambda x: x.match_score, reverse=True)
        return matches[:max_suppliers_per_item]
    
    def _extract_tokens(self, title: str) -> List[str]:
        """Extract meaningful tokens from product title"""
        if not title:
            return []
        
        # Convert to lowercase and split
        tokens = title.lower().split()
        
        # Remove common stop words and short tokens
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within',
            'new', 'best', 'top', 'quality', 'premium', 'original', 'genuine',
            'authentic', 'official', 'brand', 'name', 'model', 'version', 'edition'
        }
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Remove punctuation and clean token
            clean_token = ''.join(c for c in token if c.isalnum())
            if (len(clean_token) >= 3 and 
                clean_token not in stop_words and
                not clean_token.isdigit()):
                filtered_tokens.append(clean_token)
        
        return filtered_tokens
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity using multiple fuzzy matching algorithms"""
        if not title1 or not title2:
            return 0.0
        
        # Use multiple similarity algorithms and take the best
        ratios = [
            fuzz.ratio(title1.lower(), title2.lower()) / 100.0,
            fuzz.partial_ratio(title1.lower(), title2.lower()) / 100.0,
            fuzz.token_sort_ratio(title1.lower(), title2.lower()) / 100.0,
            fuzz.token_set_ratio(title1.lower(), title2.lower()) / 100.0
        ]
        
        # Return the best ratio
        return max(ratios)
    
    def _calculate_image_similarity(self, market_image_url: str, supplier_image_urls: List[str]) -> float:
        """Calculate image similarity using perceptual hashing"""
        if not market_image_url or not supplier_image_urls:
            return 0.0
        
        try:
            # Get market product image hash
            market_hash = self._get_image_hash(market_image_url)
            if market_hash is None:
                return 0.0
            
            # Compare with supplier images
            best_similarity = 0.0
            
            for supplier_url in supplier_image_urls[:3]:  # Limit to first 3 images
                try:
                    supplier_hash = self._get_image_hash(supplier_url)
                    if supplier_hash is not None:
                        # Calculate hash similarity
                        similarity = self._calculate_hash_similarity(market_hash, supplier_hash)
                        best_similarity = max(best_similarity, similarity)
                        
                except Exception as e:
                    logger.debug(f"Failed to process supplier image {supplier_url}: {e}")
                    continue
            
            return best_similarity
            
        except Exception as e:
            logger.warning(f"Failed to calculate image similarity: {e}")
            return 0.0
    
    def _get_image_hash(self, image_url: str) -> Optional[imagehash.ImageHash]:
        """Get perceptual hash for an image URL"""
        if not image_url:
            return None
        
        # Check cache first
        if image_url in self.image_hash_cache:
            return self.image_hash_cache[image_url]
        
        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Open image and calculate hash
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Calculate perceptual hash
            img_hash = imagehash.average_hash(image)
            
            # Cache the result
            self.image_hash_cache[image_url] = img_hash
            
            return img_hash
            
        except Exception as e:
            logger.debug(f"Failed to get image hash for {image_url}: {e}")
            return None
    
    def _calculate_hash_similarity(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
        """Calculate similarity between two image hashes"""
        try:
            # Calculate Hamming distance
            distance = hash1 - hash2
            
            # Convert to similarity score (0-1)
            # Maximum Hamming distance for 64-bit hash is 64
            similarity = 1.0 - (distance / 64.0)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Failed to calculate hash similarity: {e}")
            return 0.0
    
    def _calculate_overall_score(self, title_similarity: float, image_similarity: float) -> float:
        """Calculate overall match score"""
        # Weighted combination: 60% title, 40% image
        overall_score = (0.6 * title_similarity) + (0.4 * image_similarity)
        return overall_score
    
    def _calculate_landed_cost(self, supplier_product: Any) -> float:
        """Calculate estimated landed cost for supplier product"""
        try:
            unit_price = getattr(supplier_product, 'unit_price', 0)
            shipping_cost = 2.0  # Default shipping cost
            
            # Add 5% buffer
            landed_cost = (unit_price + shipping_cost) * 1.05
            
            return landed_cost
            
        except Exception as e:
            logger.warning(f"Failed to calculate landed cost: {e}")
            return 0.0
    
    def _calculate_risk_factor(self, landed_cost: float) -> float:
        """Calculate risk factor for higher-priced items"""
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
    
    def _calculate_margin_percentage(self, sell_price: float, landed_cost: float) -> float:
        """Calculate margin percentage"""
        if sell_price <= 0 or landed_cost <= 0:
            return 0.0
        
        margin = sell_price - landed_cost
        margin_percentage = (margin / sell_price) * 100
        
        return max(0.0, margin_percentage)
    
    def _is_viable_match(self, overall_score: float, landed_cost: float, 
                         risk_factor: float, margin_percentage: float) -> bool:
        """Determine if a match is viable based on all criteria"""
        # Check minimum score
        if overall_score < self.min_overall_score:
            return False
        
        # Check cost constraints
        if landed_cost > 25.0:  # Absolute maximum
            return False
        
        # Check margin requirements
        if margin_percentage < 25.0:  # Minimum 25% margin
            return False
        
        # Check risk tolerance
        if risk_factor > 0.8:  # High risk threshold
            return False
        
        return True
    
    def batch_find_matches(self, market_products: List[Any], supplier_products: List[Any],
                          max_workers: int = 4) -> Dict[str, List[ProductMatch]]:
        """Find matches using parallel processing"""
        matches = {}
        
        # Split work into chunks
        chunk_size = max(1, len(market_products) // max_workers)
        chunks = [market_products[i:i + chunk_size] for i in range(0, len(market_products), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, supplier_products): chunk 
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_matches = future.result()
                    matches.update(chunk_matches)
                except Exception as e:
                    logger.error(f"Failed to process chunk: {e}")
        
        return matches
    
    def _process_chunk(self, market_products_chunk: List[Any], 
                       supplier_products: List[Any]) -> Dict[str, List[ProductMatch]]:
        """Process a chunk of market products"""
        chunk_matches = {}
        
        for market_product in market_products_chunk:
            try:
                product_matches = self._find_matches_for_product(
                    market_product, supplier_products, max_suppliers_per_item=5
                )
                
                if product_matches:
                    chunk_matches[market_product.item_id] = product_matches
                    
            except Exception as e:
                logger.error(f"Failed to process market product in chunk: {e}")
                continue
        
        return chunk_matches
    
    def analyze_match_quality(self, matches: Dict[str, List[ProductMatch]]) -> Dict[str, Any]:
        """Analyze the quality of matches found"""
        if not matches:
            return {}
        
        all_matches = []
        for product_matches in matches.values():
            all_matches.extend(product_matches)
        
        if not all_matches:
            return {}
        
        # Calculate statistics
        match_scores = [m.match_score for m in all_matches]
        title_similarities = [m.title_similarity for m in all_matches]
        image_similarities = [m.image_similarity for m in all_matches]
        landed_costs = [m.landed_cost for m in all_matches]
        risk_factors = [m.risk_factor for m in all_matches]
        margin_percentages = [m.margin_percentage for m in all_matches]
        viable_count = sum(1 for m in all_matches if m.is_viable)
        
        return {
            "total_matches": len(all_matches),
            "viable_matches": viable_count,
            "viability_rate": viable_count / len(all_matches) if all_matches else 0,
            "avg_match_score": np.mean(match_scores) if match_scores else 0,
            "avg_title_similarity": np.mean(title_similarities) if title_similarities else 0,
            "avg_image_similarity": np.mean(image_similarities) if image_similarities else 0,
            "avg_landed_cost": np.mean(landed_costs) if landed_costs else 0,
            "avg_risk_factor": np.mean(risk_factors) if risk_factors else 0,
            "avg_margin_percentage": np.mean(margin_percentages) if margin_percentages else 0,
            "score_distribution": {
                "excellent": sum(1 for s in match_scores if s >= 0.9),
                "good": sum(1 for s in match_scores if 0.8 <= s < 0.9),
                "fair": sum(1 for s in match_scores if 0.7 <= s < 0.8),
                "poor": sum(1 for s in match_scores if s < 0.7)
            }
        }
    
    def export_match_results(self, matches: Dict[str, List[ProductMatch]], 
                           output_dir: str = "data/matches") -> None:
        """Export match results to CSV files"""
        import os
        import pandas as pd
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Convert to DataFrame
            data = []
            for market_id, product_matches in matches.items():
                for match in product_matches:
                    data.append({
                        "market_product_id": market_id,
                        "supplier_product_id": match.supplier_product.product_id,
                        "market_title": match.market_product.title,
                        "supplier_title": match.supplier_product.title,
                        "match_score": match.match_score,
                        "title_similarity": match.title_similarity,
                        "image_similarity": match.image_similarity,
                        "landed_cost": match.landed_cost,
                        "risk_factor": match.risk_factor,
                        "margin_percentage": match.margin_percentage,
                        "is_viable": match.is_viable,
                        "market_price": getattr(match.market_product, 'price', 0),
                        "supplier_price": match.supplier_product.unit_price
                    })
            
            df = pd.DataFrame(data)
            filename = f"{output_dir}/product_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(data)} match results to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export match results: {e}")

def main():
    """Test function for Product Matcher"""
    # Mock data for testing
    class MockMarketProduct:
        def __init__(self, item_id, title, price, image_url):
            self.item_id = item_id
            self.title = title
            self.price = price
            self.image_url = image_url
    
    class MockSupplierProduct:
        def __init__(self, product_id, title, unit_price, image_urls):
            self.product_id = product_id
            self.title = title
            self.unit_price = unit_price
            self.image_urls = image_urls
    
    # Create test data
    market_products = [
        MockMarketProduct("1", "Wireless Bluetooth Earbuds", 25.99, "https://example.com/earbuds1.jpg"),
        MockMarketProduct("2", "Phone Case for iPhone", 15.99, "https://example.com/case1.jpg")
    ]
    
    supplier_products = [
        MockSupplierProduct("s1", "Wireless Bluetooth Earbuds", 8.50, ["https://example.com/earbuds2.jpg"]),
        MockSupplierProduct("s2", "Phone Case for iPhone", 5.20, ["https://example.com/case2.jpg"]),
        MockSupplierProduct("s3", "Bluetooth Speaker", 12.00, ["https://example.com/speaker1.jpg"])
    ]
    
    # Test matching
    matcher = ProductMatcher()
    
    print("Finding product matches...")
    matches = matcher.find_matches(market_products, supplier_products)
    
    if matches:
        print(f"Found matches for {len(matches)} market products")
        
        # Analyze match quality
        analysis = matcher.analyze_match_quality(matches)
        print(f"Match analysis: {analysis}")
        
        # Show some matches
        for market_id, product_matches in matches.items():
            print(f"\nMarket product {market_id}:")
            for match in product_matches[:2]:  # Show top 2 matches
                print(f"  - Supplier: {match.supplier_product.title}")
                print(f"    Score: {match.match_score:.3f}")
                print(f"    Landed cost: ${match.landed_cost:.2f}")
                print(f"    Viable: {match.is_viable}")
        
        # Export results
        matcher.export_match_results(matches)
        
    else:
        print("No matches found")

if __name__ == "__main__":
    main()
