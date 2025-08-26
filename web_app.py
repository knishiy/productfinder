"""
Web Application for Winning Product Finder
Provides a modern web interface for product discovery and analysis
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import os
import json
import glob
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import threading
import time

# Import our pipeline
from pipeline import WinningProductPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'winning_product_finder_secret_key'

# Enable CORS for GitHub Pages
CORS(app, resources={r"/api/*": {"origins": ["https://*.github.io", "http://localhost:5000"]}})

# Global variables for pipeline status
pipeline_status = {
    "running": False,
    "progress": 0,
    "current_step": "",
    "results": None,
    "error": None
}

# Data storage
cached_results = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with results"""
    global cached_results
    
    if cached_results is None:
        # Try to load latest results
        cached_results = load_latest_results()
    
    return render_template('dashboard.html', results=cached_results)

@app.route('/api/start_pipeline', methods=['POST'])
def start_pipeline():
    """Start the product finding pipeline"""
    global pipeline_status
    
    if pipeline_status["running"]:
        return jsonify({"error": "Pipeline already running"}), 400
    
    try:
        # Get configuration from request
        config_data = request.json or {}
        
        # Store configuration for the background thread
        pipeline_status["config"] = config_data
        pipeline_status["running"] = True
        pipeline_status["progress"] = 0
        pipeline_status["current_step"] = "Initializing..."
        pipeline_status["error"] = None
        
        thread = threading.Thread(target=run_pipeline_background, args=(config_data,))
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Pipeline started successfully"})
        
    except Exception as e:
        pipeline_status["running"] = False
        pipeline_status["error"] = str(e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop_pipeline', methods=['POST'])
def stop_pipeline():
    """Stop the running pipeline"""
    global pipeline_status
    
    if not pipeline_status["running"]:
        return jsonify({"error": "No pipeline running"}), 400
    
    try:
        # Set stop flag
        pipeline_status["running"] = False
        pipeline_status["current_step"] = "Stopping..."
        pipeline_status["progress"] = 0
        
        return jsonify({"message": "Pipeline stop requested"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline_status')
def get_pipeline_status():
    """Get current pipeline status"""
    return jsonify(pipeline_status)

@app.route('/api/results')
def get_results():
    """Get pipeline results"""
    global cached_results
    
    if cached_results is None:
        cached_results = load_latest_results()
    
    return jsonify(cached_results)

@app.route('/api/results_view')
def results_view():
    """Get all scored products for display (not just winners)"""
    try:
        # Try to load from scoring results first
        scoring_files = sorted(glob.glob("data/scoring/scoring_results_*.json"), reverse=True)
        if scoring_files:
            with open(scoring_files[0], 'r', encoding='utf-8') as f:
                scoring_data = json.load(f)
                all_scored = scoring_data.get("all_scored", [])
                winning_products = scoring_data.get("winning_products", [])
                
                # Convert to view format
                view_data = []
                for i, product in enumerate(all_scored):
                    view_item = {
                        "rank": i + 1,
                        "title": product.get("title", "Unknown Product"),
                        "final_score": product.get("score_overall", 0.0),
                        "margin_score": product.get("margin_pct", 0.0),
                        "demand_score": product.get("sales_velocity", 0.0),
                        "trend_score": product.get("trend_growth_14d", 0.0),
                        "is_winner": product.get("score_overall", 0.0) >= 0.65,
                        "landed_cost": product.get("landed_cost", 0.0),
                        "sell_price": product.get("sell_price", 0.0),
                        "lead_time_days": product.get("lead_time_days", 15),
                        "seller_rating": product.get("seller_rating", 4.5),
                        "competition_density": product.get("competition_density", 25.0),
                        "price_stability": product.get("price_stability", 0.5)
                    }
                    view_data.append(view_item)
                
                return jsonify({
                    "all_scored": view_data,
                    "winners": [item for item in view_data if item["is_winner"]],
                    "total_count": len(view_data),
                    "winners_count": len([item for item in view_data if item["is_winner"]])
                })
        
        # Fallback to cached results
        if cached_results and cached_results.get("top_products"):
            return jsonify({
                "all_scored": cached_results["top_products"],
                "winners": [item for item in cached_results["top_products"] if item.get("is_winner", False)],
                "total_count": len(cached_results["top_products"]),
                "winners_count": len([item for item in cached_results["top_products"] if item.get("is_winner", False)])
            })
        
        return jsonify({
            "all_scored": [],
            "winners": [],
            "total_count": 0,
            "winners_count": 0
        })
        
    except Exception as e:
        logger.error(f"Failed to load results view: {e}")
        return jsonify({"error": str(e)}), 500

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
        
        pipeline = WinningProductPipeline("config.yaml")
        
        # Apply dynamic configuration
        apply_dynamic_config(pipeline, dynamic_config)
        
        # Step 1: Market data collection (conditional)
        if config_data.get("sources", {}).get("ebay", True):
            pipeline_status["current_step"] = "Collecting market data from eBay..."
            pipeline_status["progress"] = 20
            if not pipeline_status["running"]:
                return
            pipeline._collect_market_data()
        
        # Step 2: Amazon data collection (conditional)
        if config_data.get("sources", {}).get("amazon", False):
            pipeline_status["current_step"] = "Collecting Amazon data from Keepa..."
            pipeline_status["progress"] = 30
            if not pipeline_status["running"]:
                return
            pipeline._collect_amazon_data()
        
        # Step 3: Trends data collection (conditional)
        if config_data.get("sources", {}).get("trends", True):
            pipeline_status["current_step"] = "Collecting Google Trends data..."
            pipeline_status["progress"] = 40
            if not pipeline_status["running"]:
                return
            pipeline._collect_trends_data()
        
        # Step 4: TikTok Shop data collection (conditional)
        if config_data.get("sources", {}).get("tiktok", False):
            pipeline_status["current_step"] = "Collecting TikTok Shop data..."
            pipeline_status["progress"] = 50
            if not pipeline_status["running"]:
                return
            pipeline._collect_tiktok_shop_data()
        
        # Step 5: Supplier data collection (conditional)
        if config_data.get("sources", {}).get("aliexpress", True):
            pipeline_status["current_step"] = "Collecting supplier data from AliExpress..."
            pipeline_status["progress"] = 60
            if not pipeline_status["running"]:
                return
            pipeline._collect_supplier_data()
        
        # Step 6: Product matching
        pipeline_status["current_step"] = "Matching products..."
        pipeline_status["progress"] = 75
        if not pipeline_status["running"]:
            return
        pipeline._match_products()
        
        # Step 7: Scoring
        pipeline_status["current_step"] = "Scoring and ranking products..."
        pipeline_status["progress"] = 85
        if not pipeline_status["running"]:
            return
        pipeline._score_products()
        
        # Step 7.5: ML Scoring (if model available)
        pipeline_status["current_step"] = "Applying ML predictions..."
        pipeline_status["progress"] = 87
        if not pipeline_status["running"]:
            return
        
        try:
            from ml_scorer import ml_scorer
            from data_aggregator import data_aggregator
            
            # Add ML predictions to scored products
            if pipeline.scoring_results and isinstance(pipeline.scoring_results, dict):
                all_scored = pipeline.scoring_results.get("all_scored", [])
                
                for product in all_scored:
                    # Get ML prediction
                    ml_prob, ml_confidence = ml_scorer.predict_success(product)
                    product['ml_prediction'] = ml_prob
                    product['ml_confidence'] = ml_confidence.get('confidence', 0.0)
                
                # Add products to data aggregator for ML training
                pipeline_run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                data_aggregator.add_products(all_scored, pipeline_run_id)
                
                logger.info(f"Added {len(all_scored)} products to ML training data")
                
        except Exception as e:
            logger.warning(f"ML scoring failed: {e}")
            # Continue without ML scoring
        
        # Step 8: Reports
        pipeline_status["current_step"] = "Generating reports..."
        pipeline_status["progress"] = 95
        if not pipeline_status["running"]:
            return
        output_files = pipeline._generate_reports()
        
        # Complete
        pipeline_status["current_step"] = "Pipeline completed successfully!"
        pipeline_status["progress"] = 100
        
        # Store results
        cached_results = {
            "timestamp": datetime.now().isoformat(),
            "total_market_products": len(pipeline.market_products),
            "total_supplier_products": len(pipeline.supplier_products),
            "total_matches": len(pipeline.product_matches),
            "winning_products": len(pipeline.scoring_results.get("winning_products", [])) if pipeline.scoring_results else 0,
            "output_files": output_files,
            "scoring_summary": pipeline.scoring_results if pipeline.scoring_results else {},
            "top_products": []
        }
        
        # Add top products
        if pipeline.scoring_results and isinstance(pipeline.scoring_results, dict):
            winning_products = pipeline.scoring_results.get("winning_products", [])
            for i, product in enumerate(winning_products[:10]):
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
        
        logger.info("Pipeline completed successfully")
        
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
    
    # Handle TikTok video limit
    if config_data.get("tiktokVideoLimit"):
        dynamic_config["tiktok_video_limit"] = config_data["tiktokVideoLimit"]
    
    return dynamic_config

def apply_dynamic_config(pipeline, dynamic_config: Dict[str, Any]):
    """Apply dynamic configuration to pipeline"""
    try:
        # Update pipeline config with dynamic values
        if "category" in dynamic_config:
            pipeline.config["sources"]["ebay"]["categories"] = [dynamic_config["category"]]
        
        if "keywords" in dynamic_config:
            pipeline.config["sources"]["ebay"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["trends"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["aliexpress"]["keywords"] = dynamic_config["keywords"]
            pipeline.config["sources"]["tiktok"]["keywords"] = dynamic_config["keywords"]
        
        if "max_results" in dynamic_config:
            pipeline.config["sources"]["aliexpress"]["max_results"] = dynamic_config["max_results"]
            pipeline.config["sources"]["tiktok"]["max_results"] = dynamic_config["max_results"]
        
        # Handle TikTok video limit
        if "tiktok_video_limit" in dynamic_config:
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

@app.route('/api/ml/data_stats')
def get_ml_data_stats():
    """Get ML data statistics"""
    try:
        from data_aggregator import data_aggregator
        return jsonify(data_aggregator.get_product_stats())
    except Exception as e:
        logger.error(f"Failed to get ML data stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/train', methods=['POST'])
def train_ml_model():
    """Trigger ML model training"""
    try:
        from ml_trainer import ml_trainer
        from data_aggregator import data_aggregator
        
        # Get training data
        training_data = data_aggregator.get_training_data()
        
        if training_data.empty:
            return jsonify({"error": "No training data available"}), 400
        
        # Train model
        training_results = ml_trainer.train_model(training_data)
        
        if training_results:
            return jsonify({
                "message": "Model training completed successfully",
                "results": training_results
            })
        else:
            return jsonify({"error": "Model training failed"}), 500
            
    except Exception as e:
        logger.error(f"Failed to train ML model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/export')
def export_ml_data():
    """Export ML data for analysis"""
    try:
        format_type = request.args.get('format', 'csv')
        from data_aggregator import data_aggregator
        
        filepath = data_aggregator.export_for_analysis(format_type)
        
        if filepath:
            return jsonify({
                "message": f"Data exported successfully to {filepath}",
                "filepath": filepath
            })
        else:
            return jsonify({"error": "Export failed"}), 500
            
    except Exception as e:
        logger.error(f"Failed to export ML data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/web', exist_ok=True)
    os.makedirs('data/exports', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
