"""
Main Pipeline Module for Winning Product Finder
Orchestrates the entire product discovery and scoring process
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import json

# Import our modules
from config_loader import load_config, dget, get_nested_config
from etl.ebay import get_token, search_items, best_sellers
from etl.trends import TrendsETL
from etl.aliexpress import search_aliexpress_apify
from etl.tiktok_shop import search_tiktok_shop_apify
from matching import ProductMatcher
from costing import CostCalculator
from scoring import ProductScorer
from risk_calculator import risk_calculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Data class for pipeline execution results"""
    execution_time: float
    total_market_products: int
    total_supplier_products: int
    total_matches: int
    winning_products: int
    success: bool
    errors: List[str]
    output_files: List[str]

class WinningProductPipeline:
    """Main pipeline for finding winning products"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._load_env_vars()
        
        # Initialize ETL components
        self.ebay_etl = None
        self.keepa_etl = None
        self.trends_etl = None
        self.aliexpress_etl = None
        self.tiktok_shop_etl = None
        
        # Initialize processing components
        self.matcher = None
        self.cost_calculator = None
        self.scorer = None
        
        # Data storage
        self.market_products = {}
        self.supplier_products = {}
        self.product_matches = {}
        self.scoring_results = {}
        self.trends_data = {}
        self.tiktok_shop_data = {}
        
        # Pipeline status
        self.status = "idle"
        self.current_step = ""
        self.error_message = ""
        
        # Output directories
        self.output_dirs = {
            "raw": "data/raw",
            "processed": "data/processed",
            "matches": "data/matches",
            "scoring": "data/scoring",
            "reports": "data/reports"
        }
        
        self._create_output_directories()
        
        # Build optional clients based on config
        self._initialize_core_components()
    
    def _initialize_core_components(self):
        """Initialize core components that are needed throughout the pipeline"""
        try:
            # Initialize Trends client based on config
            trends_cfg = dget(self.config, "sources", {}).get("trends") or {}
            if dget(trends_cfg, "enabled", False):
                self.trends_etl = TrendsETL()
                logger.info("Trends ETL initialized in core components")
            else:
                logger.info("Trends ETL disabled in config")
            
            # Initialize ProductMatcher with config
            matching_cfg = dget(self.config, "matching", {})
            self.matcher = ProductMatcher(
                use_image_phash=dget(matching_cfg, "use_image_phash", True)
            )
            logger.info("ProductMatcher initialized in core components")
            
            # Initialize ProductScorer with config
            self.scorer = ProductScorer(self.config)
            logger.info("ProductScorer initialized in core components")
            
            # Initialize CostCalculator with config
            fees_cfg = dget(self.config, "fees", {})
            self.cost_calculator = CostCalculator(
                fee_rate=dget(fees_cfg, "marketplace_rate", 0.13),
                buffer=dget(fees_cfg, "buffer", 0.05)
            )
            logger.info("CostCalculator initialized in core components")
            
        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration using bulletproof config loader"""
        try:
            config = load_config(self.config_path)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # This should never happen with bulletproof loader, but just in case
            raise
    
    def _load_env_vars(self):
        """Load environment variables from .env file"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Environment variables loaded from .env file")
        except ImportError:
            logger.warning("python-dotenv not installed, using system environment variables")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")
    

    
    def _create_output_directories(self):
        """Create necessary output directories"""
        for directory in self.output_dirs.values():
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_etl_components(self):
        """Initialize all ETL components using null-safe config access"""
        try:
            # Use null-safe getters for config access
            sources = dget(self.config, "sources", {})
            
            # Initialize eBay ETL
            ebay_config = dget(sources, "ebay", {})
            if dget(ebay_config, "enabled", False):
                try:
                    token = get_token()  # This will use environment variables
                    self.ebay_etl = {"token": token, "enabled": True}
                    logger.info("eBay ETL initialized successfully")
                except Exception as e:
                    logger.warning(f"eBay ETL initialization failed: {e}")
                    self.ebay_etl = {"enabled": False}
            else:
                logger.info("eBay ETL disabled in configuration")
                self.ebay_etl = {"enabled": False}
            
            # Initialize AliExpress ETL
            aliexpress_config = dget(sources, "aliexpress", {})
            if dget(aliexpress_config, "enabled", False):
                try:
                    self.aliexpress_etl = {"enabled": True, "provider": "apify"}
                    logger.info("AliExpress ETL initialized (Apify provider)")
                except Exception as e:
                    logger.warning(f"AliExpress ETL initialization failed: {e}")
                    self.aliexpress_etl = {"enabled": False}
            else:
                logger.info("AliExpress ETL disabled in configuration")
                self.aliexpress_etl = {"enabled": False}
            
            # Initialize TikTok Shop ETL
            tiktok_config = dget(sources, "tiktok", {})
            if dget(tiktok_config, "enabled", False):
                try:
                    self.tiktok_shop_etl = {"enabled": True, "provider": "apify"}
                    logger.info("TikTok Shop ETL initialized (Apify provider)")
                except Exception as e:
                    logger.warning(f"TikTok Shop ETL initialization failed: {e}")
                    self.tiktok_shop_etl = {"enabled": False}
            else:
                logger.info("TikTok Shop ETL disabled in configuration")
                self.tiktok_shop_etl = {"enabled": False}
            
            logger.info("ETL components initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize ETL components: {e}")
            raise
    
    def _validate_components(self):
        """Validate that all required components are properly initialized"""
        required_components = {
            'matcher': self.matcher,
            'cost_calculator': self.cost_calculator,
            'scorer': self.scorer
        }
        
        missing_components = []
        for name, component in required_components.items():
            if component is None:
                missing_components.append(name)
        
        if missing_components:
            error_msg = f"Required components not initialized: {', '.join(missing_components)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("All required components validated successfully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status for web interface"""
        return {
            "status": self.status,
            "current_step": self.current_step,
            "error_message": self.error_message,
            "market_products_count": len(self.market_products),
            "supplier_products_count": len(self.supplier_products),
            "matches_count": len(self.product_matches),
            "trends_data_count": len(self.trends_data)
        }
    
    def test_initialization(self) -> bool:
        """Test if all components can be initialized properly"""
        try:
            logger.info("Testing component initialization...")
            self._initialize_etl_components()
            self._validate_components()
            logger.info("✅ All components initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    def run_pipeline(self) -> PipelineResult:
        """
        Run the complete product finding pipeline
        
        Returns:
            PipelineResult object with execution summary
        """
        start_time = time.time()
        errors = []
        
        try:
            logger.info("Starting Winning Product Pipeline")
            self.status = "running"
            
            # Initialize components
            self._initialize_etl_components()
            
            # Validate that all required components are initialized
            self._validate_components()
            
            # Step 1: Collect market data (eBay)
            logger.info("Step 1: Collecting market data from eBay")
            self._collect_market_data()
            
            # Step 2: Collect trends data
            logger.info("Step 2: Collecting Google Trends data")
            self._collect_trends_data()
            
            # Step 3: Collect TikTok Shop data
            logger.info("Step 3: Collecting TikTok Shop data")
            self._collect_tiktok_shop_data()
            
            # Step 3: Collect supplier data
            logger.info("Step 3: Collecting supplier data from AliExpress")
            self._collect_supplier_data()
            
            # Guards before matching - ensure we have data
            market_items = list(self.market_products.values())
            supplier_pool = list(self.supplier_products.values())
            
            if not market_items:
                self.status = "no_market_data"
                logger.warning("No market items collected; stopping before matching.")
                return PipelineResult(
                    execution_time=time.time() - start_time,
                    total_market_products=0,
                    total_supplier_products=len(supplier_pool),
                    total_matches=0,
                    winning_products=0,
                    success=False,
                    errors=["No market data collected"],
                    output_files=[]
                )
            
            if not supplier_pool:
                self.status = "no_supplier_data"
                logger.warning("No supplier items collected; stopping before matching.")
                return PipelineResult(
                    execution_time=time.time() - start_time,
                    total_market_products=len(market_items),
                    total_supplier_products=0,
                    total_matches=0,
                    winning_products=0,
                    success=False,
                    errors=["No supplier data collected"],
                    output_files=[]
                )
            
            if not self.matcher:
                raise RuntimeError("ProductMatcher not initialized.")
            
            # Step 4: Match products
            logger.info("Step 4: Matching market products with suppliers")
            self._match_products()
            
            # Step 5: Score and rank products
            logger.info("Step 5: Scoring and ranking products")
            self._score_products()
            
            # Step 6: Generate reports
            logger.info("Step 6: Generating reports")
            output_files = self._generate_reports()
            
            execution_time = time.time() - start_time
            
            # Determine final status and success
            if self.status in ["done_no_matches", "done_no_features"]:
                final_status = "completed_no_data"
                success = True  # Not an error, just no data
            else:
                final_status = "completed"
                success = True
            
            self.status = final_status
            
            result = PipelineResult(
                execution_time=execution_time,
                total_market_products=len(self.market_products),
                total_supplier_products=len(self.supplier_products),
                total_matches=len(self.product_matches),
                winning_products=len(self.scoring_results.get("winning_products", [])) if self.scoring_results else 0,
                success=success,
                errors=errors,
                output_files=output_files
            )
            
            logger.info(f"Pipeline completed with status '{final_status}' in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            import traceback
            execution_time = time.time() - start_time
            
            # Capture full traceback for debugging
            full_traceback = traceback.format_exc()
            error_details = f"Error: {str(e)}\nTraceback:\n{full_traceback}"
            
            errors.append(error_details)
            logger.error(f"Pipeline failed after {execution_time:.2f} seconds")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Full traceback:\n{full_traceback}")
            
            self.status = "error"
            self.error_message = error_details
            
            return PipelineResult(
                execution_time=execution_time,
                total_market_products=len(self.market_products),
                total_supplier_products=len(self.supplier_products),
                total_matches=len(self.product_matches),
                winning_products=0,
                success=False,
                errors=errors,
                output_files=[]
            )
    
    def _collect_market_data(self):
        """Collect market data from eBay using null-safe config access"""
        # Check if eBay ETL is enabled
        if not dget(self.ebay_etl, "enabled", False):
            logger.warning("eBay ETL not enabled, skipping market data collection")
            return
        
        try:
            self.current_step = "collecting_ebay_data"
            
            # Use null-safe config access
            sources_config = dget(self.config, "sources", {})
            ebay_config = dget(sources_config, "ebay", {})
            
            categories = dget(ebay_config, "categories", [])
            keywords = dget(ebay_config, "keywords", [])
            limit = dget(ebay_config, "limit", 120)
            
            logger.info(f"Collecting eBay data for {len(categories)} categories and {len(keywords)} keywords")
            
            # Collect data for each category
            for category_id in categories:
                try:
                    # Get best sellers for this category
                    best_seller_items = best_sellers(self.ebay_etl["token"], category_id, limit=limit//2)
                    
                    # Normalize eBay results to ensure we always have lists
                    if best_seller_items is None:
                        best_seller_items = []
                    elif isinstance(best_seller_items, dict):
                        best_seller_items = best_seller_items.get("items", [])
                    
                    # Search for each keyword in this category
                    for keyword in keywords:
                        search_items_result = search_items(
                            self.ebay_etl["token"], 
                            q=keyword, 
                            category_id=category_id, 
                            limit=limit//len(keywords)
                        )
                        
                        # Normalize search results
                        if search_items_result is None:
                            search_items_result = []
                        elif isinstance(search_items_result, dict):
                            search_items_result = search_items_result.get("items", [])
                        
                        # Store products
                        for item in best_seller_items + search_items_result:
                            if item.get("itemId"):
                                # Create a simple product object
                                product = type('Product', (), {
                                    'item_id': item.get("itemId"),
                                    'title': item.get("title", ""),
                                    'price': float(item.get("price", {}).get("value", 0)),
                                    'image_url': item.get("image", {}).get("imageUrl", ""),
                                    'category_id': category_id,
                                    'is_best_seller': item in best_seller_items
                                })()
                                
                                self.market_products[product.item_id] = product
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.warning(f"Failed to collect data for category {category_id}: {e}")
                    continue
            
            logger.info(f"Collected {len(self.market_products)} market products from eBay")
            
        except Exception as e:
            logger.error(f"Failed to collect market data: {e}")
            self.status = "error"
            self.error_message = str(e)
            raise
    
    def _collect_amazon_data(self):
        """Collect Amazon data from Keepa"""
        if not self.keepa_etl:
            logger.warning("Keepa ETL not configured, skipping Amazon data collection")
            return
        
        try:
            # For now, we'll collect data for a sample of market products
            # In practice, you'd want to extract ASINs from eBay product URLs or titles
            sample_products = list(self.market_products.values())[:10]
            
            for product in sample_products:
                try:
                    # Extract potential ASINs from title (simplified)
                    asin = self._extract_potential_asin(product.title)
                    if asin:
                        keepa_product = self.keepa_etl.get_product(asin)
                        if keepa_product:
                            # Store Keepa data with the market product
                            product.keepa_data = keepa_product
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Failed to get Keepa data for product {product.item_id}: {e}")
                    continue
            
            logger.info("Amazon data collection completed")
            
        except Exception as e:
            logger.error(f"Failed to collect Amazon data: {e}")
            # Don't raise - this is optional data
    
    def _extract_potential_asin(self, title: str) -> Optional[str]:
        """Extract potential ASIN from product title (simplified)"""
        # This is a simplified approach - in practice you'd use more sophisticated methods
        # ASINs are typically 10 characters long and contain alphanumeric characters
        import re
        
        # Look for patterns that might be ASINs
        asin_pattern = r'\b[A-Z0-9]{10}\b'
        matches = re.findall(asin_pattern, title.upper())
        
        return matches[0] if matches else None
    
    def _collect_trends_data(self):
        """Collect Google Trends data using null-safe config access"""
        try:
            self.current_step = "collecting_trends_data"
            
            # Use null-safe config access
            sources_config = dget(self.config, "sources", {})
            trends_config = dget(sources_config, "trends", {})
            
            if not dget(trends_config, "enabled", False):
                logger.info("Trends collection disabled, skipping")
                return
            
            keywords = dget(trends_config, "keywords", [])
            timeframe = dget(trends_config, "timeframe", "today 1-m")
            geo = dget(trends_config, "geo", "US")
            
            logger.info(f"Collecting trends data for {len(keywords)} keywords")
            
            # Check if trends ETL is available
            if not self.trends_etl:
                logger.warning("Trends ETL not initialized, skipping trends collection")
                return
            
            # Get growth data using the simplified method
            trends_data = self.trends_etl.get_growth_14d(keywords, timeframe, geo)
            
            # Log the type and preview of trends data
            logger.info(f"Trends data type: {type(trends_data)}, length: {len(trends_data) if trends_data else 0}")
            
            # Store trends data
            self.trends_data = trends_data or {}
            
            logger.info(f"Collected trends data for {len(self.trends_data)} keywords")
            
        except Exception as e:
            logger.error(f"Failed to collect trends data: {e}")
            self.trends_data = {}
            # Don't fail the pipeline for trends data
    
    def _collect_tiktok_shop_data(self):
        """Collect TikTok Shop data using null-safe config access"""
        try:
            self.current_step = "collecting_tiktok_shop_data"
            
            # Use null-safe config access
            sources_config = dget(self.config, "sources", {})
            tiktok_config = dget(sources_config, "tiktok", {})
            
            if not dget(tiktok_config, "enabled", False):
                logger.info("TikTok Shop collection disabled, skipping")
                return
            
            keywords = dget(tiktok_config, "keywords", [])
            max_results = dget(tiktok_config, "max_results", 20)
            
            logger.info(f"Collecting TikTok Shop data for {len(keywords)} keywords")
            
            # Check if TikTok Shop ETL is available
            if not dget(self.tiktok_shop_etl, "enabled", False):
                logger.warning("TikTok Shop ETL not initialized, skipping collection")
                return
            
            # Collect TikTok Shop data for each keyword
            for keyword in keywords:
                try:
                    # Search TikTok Shop using Apify
                    tiktok_products = search_tiktok_shop_apify(
                        keyword, 
                        token=os.environ.get("APIFY_TOKEN"),
                        limit=max_results
                    )
                    
                    # Store TikTok Shop data
                    if tiktok_products:
                        self.tiktok_shop_data[keyword] = tiktok_products
                        logger.info(f"Found {len(tiktok_products)} TikTok Shop products for '{keyword}'")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to collect TikTok Shop data for '{keyword}': {e}")
                    continue
            
            logger.info(f"Collected TikTok Shop data for {len(self.tiktok_shop_data)} keywords")
            
        except Exception as e:
            logger.error(f"Failed to collect TikTok Shop data: {e}")
            self.tiktok_shop_data = {}
            # Don't fail the pipeline for trends data
    
    def _collect_supplier_data(self):
        """Collect supplier data from AliExpress using null-safe config access"""
        # Check if AliExpress ETL is enabled
        if not dget(self.aliexpress_etl, "enabled", False):
            logger.warning("AliExpress ETL not enabled, skipping supplier data collection")
            return
        
        try:
            self.current_step = "collecting_supplier_data"
            
            # Use null-safe config access
            sources_config = dget(self.config, "sources", {})
            aliexpress_config = dget(sources_config, "aliexpress", {})
            max_results = dget(aliexpress_config, "max_results", 20)
            
            logger.info(f"Collecting supplier data for {len(self.market_products)} market products")
            
            # Collect supplier data for each market product
            for market_product in self.market_products.values():
                try:
                    # Use the title for supplier search
                    search_query = getattr(market_product, 'title', '')
                    
                    if not search_query:
                        continue
                    
                    # Search for suppliers using Apify
                    supplier_items = search_aliexpress_apify(search_query, page=1, token=os.environ.get("APIFY_TOKEN"))
                    
                    # Normalize supplier items to ensure we always have a list
                    if supplier_items is None:  # network hiccup fallback
                        supplier_items = []
                    elif isinstance(supplier_items, dict):
                        supplier_items = supplier_items.get("items", [])
                    
                    # Log the type and preview of supplier items
                    logger.info(f"Supplier items type: {type(supplier_items)}, length: {len(supplier_items) if supplier_items else 0}")
                    
                    # Store supplier products
                    for supplier_item in supplier_items[:max_results]:
                        # Create a simple supplier product object
                        supplier_product = type('SupplierProduct', (), {
                            'product_id': supplier_item.get("supplier_id"),
                            'title': supplier_item.get("title", ""),
                            'unit_price': supplier_item.get("unit_price", 0.0),
                            'image_url': supplier_item.get("image_url", ""),
                            'seller_rating': supplier_item.get("seller_rating", 4.0),
                            'market_product_id': market_product.item_id
                        })()
                        
                        self.supplier_products[supplier_product.product_id] = supplier_product
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to collect supplier data for {market_product.item_id}: {e}")
                    continue
            
            logger.info(f"Collected {len(self.supplier_products)} supplier products")
            
        except Exception as e:
            logger.error(f"Failed to collect supplier data: {e}")
            self.status = "error"
            self.error_message = str(e)
            raise
    
    def _match_products(self):
        """Match market products with supplier products with explicit guards"""
        try:
            # Validate that matcher is properly initialized
            if not self.matcher:
                error_msg = "ProductMatcher not initialized. Cannot proceed with matching."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Convert to lists for matching
            market_products_list = list(self.market_products.values())
            supplier_products_list = list(self.supplier_products.values())
            
            logger.info(f"Matching {len(market_products_list)} market products with {len(supplier_products_list)} suppliers")
            
            # Find matches
            matches = self.matcher.find_matches(
                market_products_list,
                supplier_products_list,
                max_suppliers_per_item=5
            )
            
            # Normalize matches to ensure we always have a dict
            matches = matches or {}
            total_matches = sum(len(v or []) for v in matches.values())
            logger.info(f"Found {total_matches} supplier matches across {len(matches)} market products")
            
            # Handle case when there are no matches
            if total_matches == 0:
                # TEMP: if matches empty, add one fake item to exercise scoring
                if self.config.get("run_mode") == "mvp":
                    logger.info("MVP mode: adding fake match for testing scoring")
                    matches = {
                        "ebay:demo123": [{
                            "market": {"title": "pet nail grinder", "price_now": 19.99, "image_url": None},
                            "supplier": {"unit_price": 6.8, "ship_cost": 1.2, "lead_time_days": 9, "seller_rating": 4.7, "image_url": None},
                            "features": {"margin_pct": 0.59, "sales_velocity": 8, "trend_growth_14d": 0.31, "price_stability": 0.7, "competition_density": 24, "lead_time_days": 9}
                        }]
                    }
                    total_matches = 1
                    logger.info("Added fake match for MVP testing")
                else:
                    self.status = "done_no_matches"
                    self.product_matches = {}
                    logger.warning("No matches found; finishing without scoring.")
                    return
            
            # Log the type and preview of matches
            logger.info(f"Matches type: {type(matches)}, count: {total_matches}")
            
            # Process matches to add costing and risk analysis
            for market_id, product_matches in matches.items():
                for match in product_matches:
                    try:
                        # Validate that cost_calculator is properly initialized
                        if not self.cost_calculator:
                            error_msg = "CostCalculator not initialized. Cannot proceed with costing analysis."
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        
                        # Calculate landed cost
                        cost_breakdown = self.cost_calculator.calculate_landed_cost(
                            match.supplier_product
                        )
                        
                        # Assess risk
                        risk_assessment = self.cost_calculator.assess_risk(
                            match.supplier_product, cost_breakdown
                        )
                        
                        # Analyze profitability
                        sell_price = getattr(match.market_product, 'price', 0)
                        profit_analysis = self.cost_calculator.analyze_profitability(
                            sell_price, cost_breakdown
                        )
                        
                        # Store analysis results
                        match.cost_breakdown = cost_breakdown
                        match.risk_assessment = risk_assessment
                        match.profit_analysis = profit_analysis
                        
                        # Calculate margin percentage
                        match.margin_percentage = profit_analysis.net_margin_percentage
                        
                        # Calculate risk factor
                        match.risk_factor = risk_assessment.overall_risk_score
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze match {market_id}: {e}")
                        continue
            
            self.product_matches = matches
            
            logger.info(f"Found {len(matches)} product matches")
            
        except Exception as e:
            logger.error(f"Failed to match products: {e}")
            raise
    
    def _score_products(self):
        """Score and rank products"""
        try:
            # Validate that scorer is properly initialized
            if not self.scorer:
                error_msg = "ProductScorer not initialized. Cannot proceed with scoring."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Build features list from matches
            features_list = self._build_features_from_matches()
            
            if not features_list:
                self.status = "done_no_features"
                self.scoring_results = []
                logger.warning("No features to score; finishing.")
                return
            
            # Score products
            scored_results = self.scorer.score_all(features_list)
            
            # Add risk calculation and success likelihood to each scored product
            enhanced_results = []
            for product in (scored_results or []):
                try:
                    # Calculate risk factors
                    risk_data = risk_calculator.calculate_overall_risk(product)
                    
                    # Calculate success likelihood
                    success_data = risk_calculator.calculate_success_likelihood(
                        score=product.get("score_overall", 0.0),
                        risk_percentage=risk_data["overall_risk"],
                        margin_pct=product.get("margin_pct", 0.0),
                        trend_growth=product.get("trend_growth_14d", 0.0)
                    )
                    
                    # Add TikTok Shop data if available
                    tiktok_data = self._get_tiktok_shop_data_for_product(product)
                    
                    # Enhance product with risk and success data
                    enhanced_product = {
                        **product,
                        "risk_percentage": risk_data["overall_risk"],
                        "risk_level": risk_data["risk_level"],
                        "risk_breakdown": risk_data["risk_breakdown"],
                        "success_likelihood": success_data["success_likelihood"],
                        "success_level": success_data["success_level"],
                        "success_factors": success_data["factors"],
                        "tiktok_shop_data": tiktok_data,
                        "analysis_date": datetime.now().isoformat()
                    }
                    
                    enhanced_results.append(enhanced_product)
                    
                except Exception as e:
                    logger.warning(f"Failed to enhance product {product.get('match_id', 'unknown')}: {e}")
                    enhanced_results.append(product)
            
            # Create report structure that the report generator expects
            min_score = dget(self.config, "scan", {}).get("min_score", 0.65)
            winners = [x for x in enhanced_results if x.get("score_overall", 0) >= min_score]
            flagged = [x for x in enhanced_results if any([
                x.get("ip_brand_flag"), 
                x.get("lead_time_days", 99) > 15
            ])]
            
            report_data = {
                "winning_products": winners,
                "flagged_products": flagged,
                "all_scored": enhanced_results,
                "summary": {
                    "count_all": len(enhanced_results),
                    "count_winners": len(winners),
                    "avg_score": (sum([p.get("score_overall", 0) for p in enhanced_results]) / max(len(enhanced_results), 1)),
                    "avg_risk": (sum([p.get("risk_percentage", 0) for p in enhanced_results]) / max(len(enhanced_results), 1)),
                    "avg_success_likelihood": (sum([p.get("success_likelihood", 0) for p in enhanced_results]) / max(len(enhanced_results), 1))
                },
            }
            
            self.scoring_results = report_data
            self.results = winners  # what /api/results serves
            
            logger.info(f"Scored {len(scored_results)} products, {len(winners)} winners")
            
        except Exception as e:
            logger.error(f"Failed to score products: {e}")
            raise
    
    def _build_features_from_matches(self) -> List[Dict[str, Any]]:
        """Build features list from product matches for scoring"""
        try:
            features_list = []
            
            for market_id, product_matches in self.product_matches.items():
                for match in product_matches:
                    try:
                        # Extract features from the match
                        features = {
                            "match_id": f"{market_id}_{getattr(match, 'supplier_product', {}).get('product_id', 'unknown')}",
                            "margin_pct": getattr(match, 'margin_percentage', 0.0),
                            "sales_velocity": getattr(match, 'sales_velocity', 0.0),
                            "trend_growth_14d": self.trends_data.get(getattr(match, 'keyword', ''), 0.0),
                            "price_stability": getattr(match, 'price_stability', 0.5),
                            "competition_density": getattr(match, 'competition_density', 25.0),
                            "lead_time_days": getattr(match, 'lead_time_days', 15.0),
                            "seller_rating": getattr(match, 'supplier_product', {}).get('seller_rating', 4.5),
                            "ip_brand_flag": getattr(match, 'ip_brand_flag', False),
                            "saturation_cluster": getattr(match, 'saturation_cluster', 0),
                            "landed_cost": getattr(match, 'landed_cost', 0.0),
                            "title": getattr(match, 'market_product', {}).get('title', ''),
                            "image_url": getattr(match, 'market_product', {}).get('image_url', ''),
                            "market_url": getattr(match, 'market_product', {}).get('item_web_url', ''),
                            "supplier_url": getattr(match, 'supplier_product', {}).get('url', '')
                        }
                        
                        features_list.append(features)
                        
                    except Exception as e:
                        logger.warning(f"Failed to build features for match {market_id}: {e}")
                        continue
            
            logger.info(f"Built features for {len(features_list)} matches")
            return features_list
            
        except Exception as e:
            logger.error(f"Failed to build features from matches: {e}")
            return []
    
    def _get_tiktok_shop_data_for_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Get TikTok Shop data for a specific product"""
        try:
            title = product.get("title", "").lower()
            if not title or not self.tiktok_shop_data:
                return {}
            
            # Find matching TikTok Shop products by title similarity
            best_match = None
            best_score = 0.0
            
            for keyword, tiktok_products in self.tiktok_shop_data.items():
                for tiktok_product in tiktok_products:
                    # Simple title similarity check
                    if keyword.lower() in title or any(word in title for word in keyword.lower().split()):
                        # Calculate a simple similarity score
                        similarity = sum(1 for word in keyword.lower().split() if word in title) / len(keyword.lower().split())
                        if similarity > best_score:
                            best_score = similarity
                            best_match = tiktok_product
            
            if best_match and best_score > 0.3:  # Minimum similarity threshold
                return {
                    "total_sold": best_match.get("total_sold", 0),
                    "sales_timeframe_days": best_match.get("sales_timeframe_days", 30),
                    "engagement_rate": best_match.get("engagement_rate", 0.0),
                    "is_trending": best_match.get("is_trending", False),
                    "shop_name": best_match.get("shop_name", ""),
                    "shop_rating": best_match.get("shop_rating", 4.5),
                    "similarity_score": round(best_score, 2)
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to get TikTok Shop data for product: {e}")
            return {}
    
    def _export_scoring_results(self) -> Optional[str]:
        """Export scoring results to file"""
        try:
            if not self.scoring_results or not isinstance(self.scoring_results, dict):
                logger.warning("No scoring results to export or invalid format")
                return None
            
            import json
            from datetime import datetime
            
            # Create scoring results file
            filename = f"{self.output_dirs['scoring']}/scoring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.scoring_results, f, indent=2, default=str)
            
            logger.info(f"Exported scoring results to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export scoring results: {e}")
            return None
    
    def _prepare_market_data_for_scoring(self) -> Dict[str, Any]:
        """Prepare market data for scoring"""
        try:
            # Calculate competition density
            competition_analysis = {}
            
            for market_product in self.market_products.values():
                category = getattr(market_product, 'category_name', 'unknown')
                
                if category not in competition_analysis:
                    competition_analysis[category] = []
                
                competition_analysis[category].append(market_product)
            
            # Calculate competition density for each category
            market_data = {}
            for category, products in competition_analysis.items():
                if len(products) > 1:
                    # Simple competition density calculation
                    competition_density = min(1.0, len(products) / 50)  # Normalize to 0-1
                    market_data[category] = {
                        'competition_density': competition_density,
                        'product_count': len(products)
                    }
            
            return market_data
            
        except Exception as e:
            logger.warning(f"Failed to prepare market data: {e}")
            return {}
    
    def _generate_reports(self) -> List[str]:
        """Generate comprehensive reports"""
        output_files = []
        
        try:
            # Export raw data
            if self.config.get("output", {}).get("save_raw_data", True):
                raw_data_file = self._export_raw_data()
                if raw_data_file:
                    output_files.append(raw_data_file)
            
            # Export processed data
            if self.config.get("output", {}).get("save_processed_data", True):
                processed_data_file = self._export_processed_data()
                if processed_data_file:
                    output_files.append(processed_data_file)
            
            # Export scoring results
            if self.scoring_results and isinstance(self.scoring_results, dict):
                try:
                    scoring_file = self._export_scoring_results()
                    if scoring_file:
                        output_files.append(scoring_file)
                except Exception as e:
                    logger.error(f"Failed to export scoring results: {e}")
            
            # Generate summary report
            summary_file = self._generate_summary_report()
            if summary_file:
                output_files.append(summary_file)
            
            # Generate winning products report
            if self.scoring_results and isinstance(self.scoring_results, dict):
                winning_count = len(self.scoring_results.get("winning_products", []))
                if winning_count > 0:
                    winners_file = self._generate_winners_report()
                    if winners_file:
                        output_files.append(winners_file)
            
            logger.info(f"Generated {len(output_files)} output files")
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
        
        return output_files
    
    def _export_raw_data(self) -> Optional[str]:
        """Export raw data to CSV files"""
        try:
            # Export market products
            market_data = []
            for product in self.market_products.values():
                market_data.append({
                    "item_id": product.item_id,
                    "title": product.title,
                    "price": product.price,
                    "category": getattr(product, 'category_name', ''),
                    "seller_rating": product.seller_rating,
                    "is_best_seller": getattr(product, 'is_best_seller', False)
                })
            
            if market_data:
                df = pd.DataFrame(market_data)
                filename = f"{self.output_dirs['raw']}/market_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Exported {len(market_data)} market products to {filename}")
                return filename
            
        except Exception as e:
            logger.error(f"Failed to export raw data: {e}")
        
        return None
    
    def _export_processed_data(self) -> Optional[str]:
        """Export processed data to CSV files"""
        try:
            # Export supplier products
            supplier_data = []
            for product in self.supplier_products.values():
                supplier_data.append({
                    "product_id": product.product_id,
                    "title": product.title,
                    "unit_price": product.unit_price,
                    "seller_rating": product.seller_rating,
                    "market_product_id": getattr(product, 'market_product_id', '')
                })
            
            if supplier_data:
                df = pd.DataFrame(supplier_data)
                filename = f"{self.output_dirs['processed']}/supplier_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Exported {len(supplier_data)} supplier products to {filename}")
                return filename
            
        except Exception as e:
            logger.error(f"Failed to export processed data: {e}")
        
        return None
    
    def _generate_summary_report(self) -> Optional[str]:
        """Generate summary report"""
        try:
            report_data = {
                "pipeline_execution": {
                    "timestamp": datetime.now().isoformat(),
                    "config_file": self.config_path,
                    "total_market_products": len(self.market_products),
                    "total_supplier_products": len(self.supplier_products),
                    "total_matches": len(self.product_matches),
                    "winning_products": len(self.scoring_results.get("winning_products", [])) if self.scoring_results else 0
                },
                "scoring_summary": self.scoring_results.__dict__ if self.scoring_results else {},
                "categories_analyzed": list(set(getattr(p, 'category_name', '') for p in self.market_products.values())),
                "trends_analyzed": list(self.trends_data.keys()) if hasattr(self, 'trends_data') else []
            }
            
            filename = f"{self.output_dirs['reports']}/pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Generated summary report: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return None
    
    def _generate_winners_report(self) -> Optional[str]:
        """Generate detailed winners report"""
        try:
            if not self.scoring_results or not isinstance(self.scoring_results, dict):
                return None
            
            # Get winning products from the new structure
            winners = self.scoring_results.get("winning_products", [])
            
            if not winners:
                return None
            
            # Create detailed report
            winners_data = []
            for winner in winners:
                # Find the corresponding match
                match = None
                for market_id, matches in self.product_matches.items():
                    for m in matches:
                        if m.supplier_product.product_id == winner.product_id:
                            match = m
                            break
                    if match:
                        break
                
                if match:
                                     winners_data.append({
                     "rank": winner.get("rank", 0),
                     "product_id": winner.get("match_id", "unknown"),
                     "title": winner.get("title", "Unknown"),
                     "final_score": winner.get("score_overall", 0),
                     "margin_percentage": winner.get("margin_pct", 0),
                     "landed_cost": winner.get("landed_cost", 0),
                     "risk_level": winner.get("risk_level", "unknown"),
                     "market_price": winner.get("market_price", 0),
                     "supplier_price": winner.get("supplier_price", 0),
                     "seller_rating": winner.get("seller_rating", 0),
                     "category": winner.get("category", "unknown")
                 })
            
            if winners_data:
                df = pd.DataFrame(winners_data)
                filename = f"{self.output_dirs['reports']}/winning_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Generated winners report: {filename}")
                return filename
            
        except Exception as e:
            logger.error(f"Failed to generate winners report: {e}")
        
        return None

def main():
    """Main function to run the pipeline"""
    try:
        # Initialize pipeline
        pipeline = WinningProductPipeline("config.yaml")
        
        # Run pipeline
        result = pipeline.run_pipeline()
        
        # Print results
        print("\n" + "="*50)
        print("WINNING PRODUCT PIPELINE RESULTS")
        print("="*50)
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time:.2f} seconds")
        print(f"Market Products: {result.total_market_products}")
        print(f"Supplier Products: {result.total_supplier_products}")
        print(f"Product Matches: {result.total_matches}")
        print(f"Winning Products: {result.winning_products}")
        
        if result.errors:
            print(f"\nErrors: {len(result.errors)}")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.output_files:
            print(f"\nOutput Files: {len(result.output_files)}")
            for file in result.output_files:
                print(f"  - {file}")
        
        if result.success and result.winning_products > 0:
            print(f"\n🎉 Found {result.winning_products} winning products!")
        else:
            print("\n❌ No winning products found or pipeline failed")
        
    except Exception as e:
        print(f"Pipeline failed to start: {e}")
        logger.error(f"Pipeline failed to start: {e}")

if __name__ == "__main__":
    main()
