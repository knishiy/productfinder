"""
Data Aggregator for Winning Product Finder
Collects and maintains product data for ML training and analysis
"""

import pandas as pd
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAggregator:
    """Aggregates product data for ML training and analysis"""
    
    def __init__(self, data_dir: str = "data", csv_filename: str = "product_history.csv"):
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.csv_path = os.path.join(data_dir, csv_filename)
        self.backup_dir = os.path.join(data_dir, "backups")
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialize or load existing CSV
        self.product_history = self._load_or_create_csv()
        
        # Track product count for ML training trigger
        self.products_since_last_training = 0
        self.last_training_count = 0
        
        # Load training trigger count
        self.training_trigger_count = 100  # Train every 100 products
        
    def _load_or_create_csv(self) -> pd.DataFrame:
        """Load existing CSV or create new one with proper schema"""
        try:
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                logger.info(f"Loaded existing product history: {len(df)} products")
                return df
            else:
                # Create new CSV with schema
                schema = {
                    'product_id': [],
                    'title': [],
                    'category': [],
                    'analysis_date': [],
                    'score_overall': [],
                    'margin_pct': [],
                    'landed_cost': [],
                    'seller_rating': [],
                    'lead_time_days': [],
                    'competition_density': [],
                    'trend_growth_14d': [],
                    'price_stability': [],
                    'engagement_rate': [],
                    'total_sold': [],
                    'avg_views': [],
                    'success_label': [],
                    'ml_prediction': [],
                    'rule_based_score': [],
                    'score_difference': [],
                    'ml_confidence': [],
                    'pipeline_run_id': [],
                    'data_sources': [],
                    'tiktok_video_limit': []
                }
                
                df = pd.DataFrame(schema)
                df.to_csv(self.csv_path, index=False)
                logger.info("Created new product history CSV with schema")
                return df
                
        except Exception as e:
            logger.error(f"Failed to load/create CSV: {e}")
            return pd.DataFrame()
    
    def add_products(self, products: List[Dict[str, Any]], pipeline_run_id: str = None) -> int:
        """
        Add new products to the history
        
        Args:
            products: List of product dictionaries from pipeline
            pipeline_run_id: Unique identifier for this pipeline run
            
        Returns:
            Number of products added
        """
        try:
            if not products:
                return 0
            
            new_products = []
            current_time = datetime.now().isoformat()
            
            for product in products:
                # Extract data for CSV
                product_data = {
                    'product_id': product.get('match_id', f"prod_{len(self.product_history) + len(new_products)}"),
                    'title': product.get('title', 'Unknown'),
                    'category': product.get('category', 'General'),
                    'analysis_date': current_time,
                    'score_overall': product.get('score_overall', 0.0),
                    'margin_pct': product.get('margin_pct', 0.0),
                    'landed_cost': product.get('landed_cost', 0.0),
                    'seller_rating': product.get('seller_rating', 4.5),
                    'lead_time_days': product.get('lead_time_days', 15.0),
                    'competition_density': product.get('competition_density', 25.0),
                    'trend_growth_14d': product.get('trend_growth_14d', 0.0),
                    'price_stability': product.get('price_stability', 0.5),
                    'engagement_rate': product.get('engagement_rate', 0.0),
                    'total_sold': product.get('total_sold', 0),
                    'avg_views': product.get('avg_views', 0),
                    'success_label': 1 if product.get('score_overall', 0) >= 0.75 else 0,
                    'ml_prediction': product.get('ml_prediction', 0.0),
                    'rule_based_score': product.get('score_overall', 0.0),
                    'score_difference': abs(product.get('ml_prediction', 0.0) - product.get('score_overall', 0.0)),
                    'ml_confidence': product.get('ml_confidence', 0.0),
                    'pipeline_run_id': pipeline_run_id or f"run_{current_time}",
                    'data_sources': self._extract_data_sources(product),
                    'tiktok_video_limit': product.get('tiktok_video_limit', 5)
                }
                
                new_products.append(product_data)
            
            # Add to DataFrame
            new_df = pd.DataFrame(new_products)
            self.product_history = pd.concat([self.product_history, new_df], ignore_index=True)
            
            # Update product count
            self.products_since_last_training += len(new_products)
            
            # Save to CSV
            self._save_csv()
            
            # Check if ML training should be triggered
            if self.products_since_last_training >= self.training_trigger_count:
                self._trigger_ml_training()
            
            logger.info(f"Added {len(new_products)} products to history (total: {len(self.product_history)})")
            return len(new_products)
            
        except Exception as e:
            logger.error(f"Failed to add products: {e}")
            return 0
    
    def _extract_data_sources(self, product: Dict[str, Any]) -> str:
        """Extract data sources used for this product"""
        sources = []
        
        if product.get('ebay_data'):
            sources.append('ebay')
        if product.get('trends_data'):
            sources.append('trends')
        if product.get('aliexpress_data'):
            sources.append('aliexpress')
        if product.get('tiktok_shop_data'):
            sources.append('tiktok')
        if product.get('amazon_data'):
            sources.append('amazon')
        
        return ','.join(sources) if sources else 'none'
    
    def _save_csv(self):
        """Save current data to CSV"""
        try:
            # Create backup before saving
            if os.path.exists(self.csv_path):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(self.backup_dir, f"product_history_{timestamp}.csv")
                os.rename(self.csv_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Save new data
            self.product_history.to_csv(self.csv_path, index=False)
            logger.info(f"Saved product history to {self.csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
    
    def _trigger_ml_training(self):
        """Trigger ML model training"""
        try:
            logger.info(f"Triggering ML training after {self.products_since_last_training} new products")
            
            # Import ML trainer
            from ml_trainer import ml_trainer
            
            # Prepare training data
            training_data = ml_trainer.prepare_training_data(self.product_history.to_dict('records'))
            
            if not training_data.empty:
                # Train model
                training_results = ml_trainer.train_model(training_data)
                
                if training_results:
                    logger.info("ML model training completed successfully")
                    
                    # Update training count
                    self.last_training_count = len(self.product_history)
                    self.products_since_last_training = 0
                    
                    # Save training metadata
                    self._save_training_metadata(training_results)
                else:
                    logger.warning("ML model training failed")
            else:
                logger.warning("No training data available for ML training")
                
        except Exception as e:
            logger.error(f"Failed to trigger ML training: {e}")
    
    def _save_training_metadata(self, training_results: Dict[str, Any]):
        """Save metadata about ML training"""
        try:
            metadata = {
                'training_date': datetime.now().isoformat(),
                'total_products': len(self.product_history),
                'training_samples': training_results.get('metrics', {}).get('training_samples', 0),
                'test_samples': training_results.get('metrics', {}).get('test_samples', 0),
                'auc_score': training_results.get('metrics', {}).get('auc', 0.0),
                'logloss': training_results.get('metrics', {}).get('logloss', 0.0),
                'feature_importance': training_results.get('feature_importance', {})
            }
            
            metadata_file = os.path.join(self.data_dir, 'ml_training_metadata.json')
            
            # Load existing metadata
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = []
            
            # Add new metadata
            existing_metadata.append(metadata)
            
            # Keep only last 10 training records
            if len(existing_metadata) > 10:
                existing_metadata = existing_metadata[-10:]
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
            
            logger.info("ML training metadata saved")
            
        except Exception as e:
            logger.error(f"Failed to save training metadata: {e}")
    
    def get_training_data(self, limit: int = None) -> pd.DataFrame:
        """
        Get training data for ML
        
        Args:
            limit: Maximum number of products to return (None for all)
            
        Returns:
            DataFrame ready for ML training
        """
        try:
            if limit:
                return self.product_history.tail(limit)
            return self.product_history.copy()
            
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return pd.DataFrame()
    
    def get_product_stats(self) -> Dict[str, Any]:
        """Get statistics about the product history"""
        try:
            if self.product_history.empty:
                return {'total_products': 0}
            
            stats = {
                'total_products': len(self.product_history),
                'products_since_last_training': self.products_since_last_training,
                'last_training_count': self.last_training_count,
                'next_training_trigger': self.training_trigger_count - self.products_since_last_training,
                'date_range': {
                    'earliest': self.product_history['analysis_date'].min(),
                    'latest': self.product_history['analysis_date'].max()
                },
                'success_rate': (self.product_history['success_label'] == 1).mean(),
                'avg_score': self.product_history['score_overall'].mean(),
                'avg_margin': self.product_history['margin_pct'].mean(),
                'categories': self.product_history['category'].value_counts().to_dict(),
                'data_sources': self.product_history['data_sources'].value_counts().to_dict()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get product stats: {e}")
            return {'error': str(e)}
    
    def export_for_analysis(self, format: str = 'csv') -> str:
        """
        Export data for external analysis
        
        Args:
            format: Export format ('csv', 'json', 'excel')
            
        Returns:
            Path to exported file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'csv':
                filename = f"product_analysis_{timestamp}.csv"
                filepath = os.path.join(self.data_dir, filename)
                self.product_history.to_csv(filepath, index=False)
                
            elif format == 'json':
                filename = f"product_analysis_{timestamp}.json"
                filepath = os.path.join(self.data_dir, filename)
                self.product_history.to_json(filepath, orient='records', indent=2)
                
            elif format == 'excel':
                filename = f"product_analysis_{timestamp}.xlsx"
                filepath = os.path.join(self.data_dir, filename)
                self.product_history.to_csv(filepath, index=False)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return ""
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Remove old product data to keep CSV manageable
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        try:
            if self.product_history.empty:
                return
            
            # Convert analysis_date to datetime
            self.product_history['analysis_date'] = pd.to_datetime(self.product_history['analysis_date'])
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Filter data
            old_count = len(self.product_history)
            self.product_history = self.product_history[
                self.product_history['analysis_date'] >= cutoff_date
            ]
            new_count = len(self.product_history)
            
            if old_count != new_count:
                logger.info(f"Cleaned up {old_count - new_count} old products (keeping last {days_to_keep} days)")
                self._save_csv()
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

# Global instance
data_aggregator = DataAggregator()
