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
    """Get results for display"""
    global cached_results
    
    # Try to get scoring results first
    if cached_results and cached_results.get("top_products"):
        return jsonify(cached_results["top_products"])
    
    # If no scoring results, try to show market data from pipeline reports
    report_files = sorted(glob.glob("data/reports/pipeline_summary_*.json"), reverse=True)
    if report_files:
        with open(report_files[0], 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            market_products = report_data.get("market_products", [])
            
            if market_products:
                # Convert market products to view format (without scoring data)
                view_data = []
                for i, product in enumerate(market_products):
                    view_item = {
                        "rank": i + 1,
                        "title": product.get("title", "Unknown Product"),
                        "final_score": "N/A",
                        "margin_score": "N/A",
                        "demand_score": "N/A",
                        "trend_score": "N/A",
                        "is_winner": False,
                        "landed_cost": "N/A",
                        "sell_price": product.get("price", 0.0),
                        "lead_time_days": "N/A",
                        "seller_rating": "N/A",
                        "competition_density": "N/A",
                        "price_stability": "N/A"
                    }
                    view_data.append(view_item)
                
                return jsonify(view_data)
    
    # Fallback to cached results
    if cached_results and cached_results.get("top_products"):
        return jsonify(cached_results["top_products"])
    
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
            pipeline_status["current_step"] = f"Pipeline completed with issues: {result.status}"
        
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
    
    # Handle category
    if config_data.get("category"):
        dynamic_config["category"] = config_data["category"]
    
    # Handle keywords
    if config_data.get("keywords"):
        keywords = [kw.strip() for kw in config_data["keywords"].split(",") if kw.strip()]
        dynamic_config["keywords"] = keywords
    
    # Handle product link
    if config_data.get("productLink"):
        dynamic_config["product_link"] = config_data["productLink"]
    
    # Handle max results
    if config_data.get("maxResults"):
        dynamic_config["max_results"] = config_data["maxResults"]
    
    # Handle sources
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
        # Update pipeline config with dynamic values
        if "category" in dynamic_config and dynamic_config["category"].strip():
            # Validate category - if it's not a valid eBay category ID, use defaults
            category = dynamic_config["category"].strip()
            if category.isdigit() and len(category) >= 3:
                # Valid numeric category ID
                pipeline.config["sources"]["ebay"]["categories"] = [category]
            else:
                # Invalid category, use defaults from config
                logger.warning(f"Invalid category '{category}' provided, using default categories")
                pipeline.config["sources"]["ebay"]["categories"] = ["177772", "63514"]  # Pet Supplies, Cell Phone Accessories
        
        if "keywords" in dynamic_config and dynamic_config["keywords"]:
            # Use provided keywords if they're not empty
            pipeline.config["sources"]["ebay"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["trends"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["aliexpress"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["tiktok"]["keywords"] = dynamic_config["keywords"]
        else:
            # Use default keywords if none provided
            logger.info("No keywords provided, using default keywords from config")
        
        if "max_results" in dynamic_config:
            pipeline.config["sources"]["aliexpress"]["max_results"] = dynamic_config["max_results"]
            pipeline.config["sources"]["tiktok"]["max_results"] = dynamic_config["max_results"]
        
        # Handle TikTok video limit only if TikTok is enabled
        if "tiktok_video_limit" in dynamic_config and dynamic_config["sources"]["tiktok"]:
            pipeline.config["sources"]["tiktok"]["video_limit"] = dynamic_config["tiktok_video_limit"]
        
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
