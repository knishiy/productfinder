"""
Flask Web Application for Winning Product Finder
Provides web interface and API endpoints for the product discovery pipeline
"""

import os
import json
import logging
import threading
import glob
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pipeline import WinningProductPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global variables for pipeline status and results
pipeline_status = {
    "running": False,
    "current_step": "",
    "progress": 0,
    "error": None,
    "results": None
}

cached_results = None
pipeline_thread = None

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/pipeline_status')
def get_pipeline_status():
    """Get current pipeline status"""
    return jsonify(pipeline_status)

@app.route('/api/start_pipeline', methods=['POST'])
def start_pipeline():
    """Start the product discovery pipeline"""
    global pipeline_status, pipeline_thread
    
    if pipeline_status["running"]:
        return jsonify({"error": "Pipeline is already running"}), 400
    
    try:
        # Get configuration from request
        config_data = request.get_json(silent=True) or {}
        
        # Reset pipeline status
        pipeline_status.update({
            "running": True,
            "current_step": "Starting pipeline...",
            "progress": 0,
            "error": None,
            "results": None
        })
        
        # Start pipeline in background thread
        pipeline_thread = threading.Thread(
            target=run_pipeline_background,
            args=(config_data,),
            daemon=True
        )
        pipeline_thread.start()
        
        return jsonify({"message": "Pipeline started successfully"})
        
    except Exception as e:
        pipeline_status["error"] = str(e)
        pipeline_status["running"] = False
        logger.error(f"Failed to start pipeline: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop_pipeline', methods=['POST'])
def stop_pipeline():
    """Stop the running pipeline"""
    global pipeline_status
    
    if not pipeline_status["running"]:
        return jsonify({"error": "No pipeline running"}), 400
    
    try:
        pipeline_status["running"] = False
        pipeline_status["current_step"] = "Pipeline stop requested..."
        return jsonify({"message": "Pipeline stop requested"})
        
    except Exception as e:
        logger.error(f"Failed to stop pipeline: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/results_view')
def results_view():
    """Get results for display - shows exact same data as terminal pipeline"""
    global cached_results
    
    # Try to get scoring results first
    if cached_results and cached_results.get("top_products"):
        return jsonify(cached_results["top_products"])
    
    # If no scoring results, try to show market data from cached results
    if cached_results and cached_results.get("market_products"):
        return jsonify(cached_results["market_products"])
    
    # If no cached results, try to show actual market products from latest pipeline run
    # This will show the EXACT same data as the terminal pipeline
    try:
        # Find the latest pipeline run data
        pipeline_data_dir = "data/pipeline_runs"
        if os.path.exists(pipeline_data_dir):
            run_files = sorted([f for f in os.listdir(pipeline_data_dir) if f.endswith('.json')], reverse=True)
            if run_files:
                latest_run_file = os.path.join(pipeline_data_dir, run_files[0])
                with open(latest_run_file, 'r', encoding='utf-8') as f:
                    run_data = json.load(f)
                    
                # Extract market products exactly as they appear in terminal
                market_products = run_data.get("market_products", {})
                if market_products:
                    # Convert to list format for display, preserving all original attributes
                    products_list = []
                    for i, (item_id, product_data) in enumerate(market_products.items()):
                        # Create product object with ALL the same attributes as terminal
                        product_obj = {
                            "rank": i + 1,
                            "item_id": item_id,
                            "title": product_data.get("title", "Unknown Product"),
                            "price": product_data.get("price", 0.0),
                            "currency": product_data.get("currency", "USD"),
                            "image_url": product_data.get("image_url", ""),
                            "item_web_url": product_data.get("item_web_url", ""),
                            "seller_username": product_data.get("seller_username", ""),
                            "seller_feedback_score": product_data.get("seller_feedback_score", 0),
                            "seller_positive_feedback_percent": product_data.get("seller_positive_feedback_percent", 0.0),
                            "buying_options": product_data.get("buying_options", []),
                            "condition": product_data.get("condition", ""),
                            "category_id": product_data.get("category_id", ""),
                            "category_name": product_data.get("category_name", ""),
                            "location": product_data.get("location", ""),
                            "shipping_cost": product_data.get("shipping_cost", 0.0),
                            "shipping_type": product_data.get("shipping_type", ""),
                            "top_rated_seller": product_data.get("top_rated_seller", False),
                            "best_offer_enabled": product_data.get("best_offer_enabled", False),
                            "listing_type": product_data.get("listing_type", ""),
                            "end_time": product_data.get("end_time", ""),
                            "is_best_seller": product_data.get("is_best_seller", False),
                            "merchandised_rank": product_data.get("merchandised_rank"),
                            # Add display fields for UI
                            "final_score": "N/A",
                            "margin_score": "N/A",
                            "demand_score": "N/A",
                            "trend_score": "N/A",
                            "is_winner": False,
                            "landed_cost": "N/A",
                            "lead_time_days": "N/A",
                            "competition_density": "N/A",
                            "price_stability": "N/A"
                        }
                        products_list.append(product_obj)
                    
                    logger.info(f"Returning {len(products_list)} market products from pipeline run data")
                    return jsonify(products_list)
        
        # Fallback: try to show summary from pipeline reports
        report_files = sorted(glob.glob("data/reports/pipeline_summary_*.json"), reverse=True)
        if report_files:
            with open(report_files[0], 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                pipeline_execution = report_data.get("pipeline_execution", {})
                total_market_products = pipeline_execution.get("total_market_products", 0)
                
                if total_market_products > 0:
                    # Create a summary view since we don't have individual products
                    summary_data = [{
                        "rank": 1,
                        "title": f"Pipeline Results Summary - {total_market_products} Products Collected",
                        "final_score": "N/A",
                        "margin_score": "N/A", 
                        "demand_score": "N/A",
                        "trend_score": "N/A",
                        "is_winner": False,
                        "landed_cost": "N/A",
                        "sell_price": 0.0,
                        "lead_time_days": "N/A",
                        "seller_rating": "N/A",
                        "competition_density": "N/A",
                        "price_stability": "N/A",
                        "summary_info": {
                            "total_market_products": total_market_products,
                            "total_supplier_products": pipeline_execution.get("total_supplier_products", 0),
                            "total_matches": pipeline_execution.get("total_matches", 0),
                            "winning_products": pipeline_execution.get("winning_products", 0),
                            "trends_analyzed": report_data.get("trends_analyzed", []),
                            "message": "Click 'Start Pipeline' to collect fresh data and see individual products"
                        }
                    }]
                    return jsonify(summary_data)
    
    except Exception as e:
        logger.error(f"Error in results_view: {e}")
        return jsonify([{"error": f"Failed to load results: {str(e)}"}])
    
    # No results available
    return jsonify([])

@app.route('/api/categories')
def get_categories():
    """Get available product categories"""
    try:
        with open('config.yaml', 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        categories = config.get('categories', {})
        return jsonify(categories)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update_config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        data = request.json
        
        # Load current config
        with open('config.yaml', 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Update categories
        if 'categories' in data:
            config['categories'] = data['categories']
        
        # Save updated config
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return jsonify({"message": "Configuration updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_pipeline_background(config_data):
    """Run pipeline in background thread with configuration"""
    global pipeline_status, cached_results
    
    try:
        # Initialize pipeline
        pipeline_status["current_step"] = "Initializing pipeline..."
        pipeline_status["progress"] = 10
        
        # Check if pipeline was stopped
        if not pipeline_status["running"]:
            logger.info("Pipeline stopped by user request")
            return
        
        # Create dynamic config based on user input
        dynamic_config = create_dynamic_config(config_data)
        
        # Initialize pipeline with config.yaml (same as terminal)
        pipeline = WinningProductPipeline("config.yaml")
        
        # Apply dynamic configuration
        apply_dynamic_config(pipeline, dynamic_config)
        
        # Use the SAME pipeline execution as terminal - call run_pipeline() directly
        pipeline_status["current_step"] = "Running pipeline with real data collection..."
        pipeline_status["progress"] = 20
        
        if not pipeline_status["running"]:
            return
            
        # This is the key fix - use the exact same method as terminal
        result = pipeline.run_pipeline()
        
        # Update progress
        pipeline_status["progress"] = 90
        
        if not pipeline_status["running"]:
            return
            
        # Generate reports
        pipeline_status["current_step"] = "Generating reports..."
        pipeline_status["progress"] = 95
        output_files = pipeline._generate_reports()
        
        # Complete with proper status
        if result.success:
            if len(pipeline.supplier_products) == 0:
                pipeline_status["current_step"] = "Pipeline completed with market data only (no suppliers available)"
            else:
                pipeline_status["current_step"] = "Pipeline completed successfully!"
        else:
            # PipelineResult doesn't have status attribute, use errors instead
            error_msg = "; ".join(result.errors) if result.errors else "Unknown error"
            pipeline_status["current_step"] = f"Pipeline completed with issues: {error_msg}"
        
        pipeline_status["progress"] = 100
        
        # Add results data to pipeline status for frontend display
        pipeline_status["results"] = {
            "total_market_products": len(pipeline.market_products),
            "total_supplier_products": len(pipeline.supplier_products),
            "total_matches": len(pipeline.product_matches),
            "winning_products": len(pipeline.scoring_results.get("winning_products", [])) if pipeline.scoring_results else 0,
            "output_files": output_files,
            "data_summary": {
                "ebay_enabled": hasattr(pipeline, 'ebay_etl') and pipeline.ebay_etl and pipeline.ebay_etl.get("enabled", False),
                "trends_enabled": hasattr(pipeline, 'trends') and pipeline.trends is not None,
                "aliexpress_enabled": hasattr(pipeline, 'aliexpress_etl') and pipeline.aliexpress_etl and pipeline.aliexpress_etl.get("enabled", False),
                "tiktok_enabled": hasattr(pipeline, 'tiktok_shop_etl') and pipeline.tiktok_shop_etl and pipeline.tiktok_shop_etl.get("enabled", False),
                "amazon_enabled": hasattr(pipeline, 'amazon_etl') and pipeline.amazon_etl and pipeline.amazon_etl.get("enabled", False)
            }
        }
        
        # Store results - ONLY real data, no demo products
        cached_results = {
            "timestamp": datetime.now().isoformat(),
            "total_market_products": len(pipeline.market_products),
            "total_supplier_products": len(pipeline.supplier_products),
            "total_matches": len(pipeline.product_matches),
            "winning_products": len(pipeline.scoring_results.get("winning_products", [])) if pipeline.scoring_results else 0,
            "output_files": output_files,
            "scoring_summary": pipeline.scoring_results if pipeline.scoring_results else {},
            "top_products": [],
            "market_products": []
        }
        
        # Add ONLY real market products (no demo data)
        for market_product in pipeline.market_products.values():
            # Only include products with real data
            if hasattr(market_product, 'title') and getattr(market_product, 'title', '').strip():
                cached_results["market_products"].append({
                    "title": getattr(market_product, 'title', 'Unknown Product'),
                    "price": getattr(market_product, 'price', 0.0),
                    "image_url": getattr(market_product, 'image_url', ''),
                    "item_web_url": getattr(market_product, 'item_web_url', ''),
                    "seller_username": getattr(market_product, 'seller_username', ''),
                    "seller_feedback_score": getattr(market_product, 'seller_feedback_score', 0)
                })
        
        # Add ONLY real scored products (no demo data)
        if pipeline.scoring_results and isinstance(pipeline.scoring_results, dict):
            all_scored = pipeline.scoring_results.get("all_scored", [])
            for i, product in enumerate(all_scored[:10]):
                # Only include products with real scoring data
                if isinstance(product, dict) and product.get("title") and product.get("title") != "Unknown Product":
                    cached_results["top_products"].append({
                        "rank": i + 1,
                        "title": product.get("title", "Unknown Product"),
                        "final_score": product.get("score_overall", 0.0),
                        "margin_score": product.get("margin_pct", 0.0),
                        "demand_score": product.get("sales_velocity", 0.0),
                        "trend_score": product.get("trend_growth_14d", 0.0),
                        "is_winner": product.get("score_overall", 0.0) >= 0.65
                    })
        
        # Save results to file
        save_results(cached_results)
        
        logger.info("Pipeline completed successfully with real data")
        
    except Exception as e:
        pipeline_status["error"] = str(e)
        pipeline_status["current_step"] = f"Pipeline failed: {str(e)}"
        logger.error(f"Pipeline failed: {e}")
        
    finally:
        pipeline_status["running"] = False

def create_dynamic_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create dynamic configuration based on user input"""
    dynamic_config = {}
    
    # Handle category - only add if provided and not empty
    category = config_data.get("category")
    if category and isinstance(category, str) and category.strip():
        dynamic_config["category"] = category.strip()
    
    # Handle keywords - only add if provided and not empty
    keywords = config_data.get("keywords")
    if keywords and isinstance(keywords, str) and keywords.strip():
        keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        if keyword_list:  # Only add if we have valid keywords
            dynamic_config["keywords"] = keyword_list
    
    # Handle product link - only add if provided and not empty
    product_link = config_data.get("productLink")
    if product_link and isinstance(product_link, str) and product_link.strip():
        dynamic_config["product_link"] = product_link.strip()
    
    # Handle max results - only add if provided and not empty
    max_results = config_data.get("maxResults")
    if max_results and isinstance(max_results, str) and max_results.strip():
        dynamic_config["max_results"] = max_results.strip()
    
    # Handle sources - always include with defaults
    sources = config_data.get("sources", {})
    dynamic_config["sources"] = {
        "ebay": sources.get("ebay", True),
        "trends": sources.get("trends", True),
        "aliexpress": sources.get("aliexpress", True),
        "tiktok": sources.get("tiktok", False),
        "amazon": sources.get("amazon", False)
    }
    
    # Handle TikTok video limit only if TikTok is selected
    if sources.get("tiktok", False) and config_data.get("tiktokVideoLimit"):
        dynamic_config["tiktok_video_limit"] = config_data["tiktokVideoLimit"]
    
    return dynamic_config

def apply_dynamic_config(pipeline, dynamic_config: Dict[str, Any]):
    """Apply dynamic configuration to pipeline"""
    try:
        # Update pipeline config with dynamic values - only when provided
        if "category" in dynamic_config and dynamic_config["category"]:
            # Validate category - if it's not a valid eBay category ID, use defaults
            category = str(dynamic_config["category"]).strip()
            if category.isdigit() and len(category) >= 3:
                # Valid numeric category ID
                pipeline.config["sources"]["ebay"]["categories"] = [category]
                logger.info(f"Applied custom category: {category}")
            else:
                # Invalid category, use defaults from config
                logger.warning(f"Invalid category '{category}' provided, using default categories")
                pipeline.config["sources"]["ebay"]["categories"] = ["1281", "15032"]  # Pet Supplies (1281), Cell Phones (15032)
        else:
            logger.info("No custom category specified, using default categories from config")
        
        if "keywords" in dynamic_config and dynamic_config["keywords"]:
            # Use provided keywords if they're not empty
            pipeline.config["sources"]["ebay"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["trends"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["aliexpress"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["tiktok"]["keywords"] = dynamic_config["keywords"]
            logger.info(f"Applied custom keywords: {dynamic_config['keywords']}")
        else:
            # Use default keywords if none provided - EXPLICITLY set them
            default_keywords = pipeline.config["sources"]["ebay"].get("keywords", [])
            if default_keywords:
                pipeline.config["sources"]["ebay"]["keywords"] = default_keywords
                pipeline.config["sources"]["trends"]["keywords"] = default_keywords
                pipeline.config["sources"]["aliexpress"]["keywords"] = default_keywords
                pipeline.config["sources"]["tiktok"]["keywords"] = default_keywords
                logger.info(f"Applied default keywords from config: {default_keywords}")
            else:
                logger.warning("No default keywords found in config, pipeline may not collect data")
        
        if "max_results" in dynamic_config and dynamic_config["max_results"]:
            max_results = str(dynamic_config["max_results"]).strip()
            pipeline.config["sources"]["aliexpress"]["max_results"] = max_results
            pipeline.config["sources"]["tiktok"]["max_results"] = max_results
            logger.info(f"Applied custom max results: {max_results}")
        else:
            logger.info("No custom max results specified, using defaults from config")
        
        # Handle TikTok video limit only if TikTok is enabled
        if "tiktok_video_limit" in dynamic_config and dynamic_config["sources"]["tiktok"]:
            pipeline.config["sources"]["tiktok"]["video_limit"] = dynamic_config["tiktok_video_limit"]
            logger.info(f"Applied TikTok video limit: {dynamic_config['tiktok_video_limit']}")
        
        # Update source enablement
        for source, enabled in dynamic_config["sources"].items():
            if source in pipeline.config["sources"]:
                pipeline.config["sources"][source]["enabled"] = enabled
        
        logger.info(f"Applied dynamic configuration: {dynamic_config}")
        
    except Exception as e:
        logger.warning(f"Failed to apply dynamic configuration: {e}")

def save_results(results: Dict[str, Any]):
    """Save results to file"""
    try:
        os.makedirs('data/web', exist_ok=True)
        filename = f"data/web/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def load_latest_results() -> Optional[Dict[str, Any]]:
    """Load latest results from file"""
    try:
        results_dir = 'data/web'
        if not os.path.exists(results_dir):
            return None
        
        # Find latest results file
        files = [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.json')]
        if not files:
            return None
        
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
        
        with open(os.path.join(results_dir, latest_file), 'r') as f:
            results = json.load(f)
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return None

@app.route('/api/export_csv')
def export_csv():
    """Export results to CSV"""
    global cached_results
    
    if cached_results is None or not cached_results.get("top_products"):
        return jsonify({"error": "No results to export"}), 400
    
    try:
        # Create DataFrame
        df = pd.DataFrame(cached_results["top_products"])
        
        # Export to CSV
        os.makedirs('data/exports', exist_ok=True)
        filename = f"data/exports/winning_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        return jsonify({
            "message": "CSV exported successfully",
            "filename": filename
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_running": pipeline_status["running"]
    })

@app.route('/api/debug/results')
def debug_results():
    """Debug endpoint to see what results are available"""
    global cached_results
    
    # Check cached results
    cached_info = {
        "cached_results_exists": cached_results is not None,
        "total_market_products": cached_results.get("total_market_products", 0) if cached_results else 0,
        "total_supplier_products": cached_results.get("total_supplier_products", 0) if cached_results else 0,
        "market_products_count": len(cached_results.get("market_products", [])) if cached_results else 0,
        "top_products_count": len(cached_results.get("top_products", [])) if cached_results else 0
    }
    
    # Check pipeline reports
    report_files = sorted(glob.glob("data/reports/pipeline_summary_*.json"), reverse=True)
    latest_report = None
    if report_files:
        try:
            with open(report_files[0], 'r', encoding='utf-8') as f:
                latest_report = json.load(f)
        except Exception as e:
            latest_report = {"error": str(e)}
    
    return jsonify({
        "cached_results": cached_info,
        "latest_report": latest_report,
        "pipeline_status": pipeline_status
    })

# ML Analysis endpoints
@app.route('/ml_analysis')
def ml_analysis_page():
    """ML Analysis page"""
    return render_template('ml_analysis.html')

@app.route('/api/ml/model_info')
def get_ml_model_info():
    """Get ML model information"""
    try:
        from ml_scorer import ml_scorer
        return jsonify(ml_scorer.get_model_info())
    except Exception as e:
        logger.error(f"Failed to get ML model info: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
