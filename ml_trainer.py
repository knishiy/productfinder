"""
ML Trainer for Winning Product Finder
Trains logistic regression models to improve success prediction accuracy
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainer:
    """Machine Learning trainer for product success prediction"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.features = [
            'avg_engagement_rate', 'views_per_hour', 'trend_slope', 
            'price_usd', 'margin_pct', 'seller_rating', 'lead_time_days', 
            'competitor_count', 'trend_growth_14d', 'price_stability',
            'competition_density', 'total_sold', 'engagement_rate'
        ]
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(f"{model_dir}/backups", exist_ok=True)
        
        # Model state
        self.current_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def prepare_training_data(self, products_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare training data from pipeline results
        
        Args:
            products_data: List of product dictionaries from pipeline
            
        Returns:
            DataFrame ready for training
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(products_data)
            
            if df.empty:
                logger.warning("No products data provided for training")
                return pd.DataFrame()
            
            # Extract and normalize features
            training_data = []
            
            for _, product in df.iterrows():
                # Extract features with defaults
                features = {
                    'avg_engagement_rate': self._safe_extract(product, 'engagement_rate', 0.0),
                    'views_per_hour': self._safe_extract(product, 'avg_views', 0) / 24.0,
                    'trend_slope': self._safe_extract(product, 'trend_growth_14d', 0.0),
                    'price_usd': self._safe_extract(product, 'landed_cost', 0.0),
                    'margin_pct': self._safe_extract(product, 'margin_pct', 0.0),
                    'seller_rating': self._safe_extract(product, 'seller_rating', 4.5),
                    'lead_time_days': self._safe_extract(product, 'lead_time_days', 15.0),
                    'competitor_count': self._safe_extract(product, 'competition_density', 25.0),
                    'trend_growth_14d': self._safe_extract(product, 'trend_growth_14d', 0.0),
                    'price_stability': self._safe_extract(product, 'price_stability', 0.5),
                    'competition_density': self._safe_extract(product, 'competition_density', 25.0),
                    'total_sold': self._safe_extract(product, 'total_sold', 0),
                    'engagement_rate': self._safe_extract(product, 'engagement_rate', 0.0)
                }
                
                # Add metadata
                features['title'] = product.get('title', 'Unknown')
                features['category'] = product.get('category', 'General')
                features['analysis_date'] = product.get('analysis_date', datetime.now().isoformat())
                
                # Add success label (top quartile by score)
                features['success_label'] = 1 if product.get('score_overall', 0) >= 0.75 else 0
                
                training_data.append(features)
            
            training_df = pd.DataFrame(training_data)
            
            # Save training data
            self._save_training_data(training_df)
            
            logger.info(f"Prepared training data: {len(training_df)} products")
            return training_df
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return pd.DataFrame()
    
    def _safe_extract(self, product: pd.Series, key: str, default: Any) -> Any:
        """Safely extract value from product data"""
        try:
            value = product.get(key, default)
            if pd.isna(value):
                return default
            return float(value) if isinstance(value, (int, float)) else default
        except:
            return default
    
    def _save_training_data(self, df: pd.DataFrame):
        """Save training data to CSV"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.model_dir}/training_data_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Training data saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
    
    def train_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ML model
        
        Args:
            training_data: DataFrame with features and labels
            
        Returns:
            Model performance metrics and feature importance
        """
        try:
            if training_data.empty or len(training_data) < 50:
                logger.warning("Insufficient training data (need at least 50 samples)")
                return {}
            
            # Prepare features and labels
            X = training_data[self.features].fillna(0)
            y = training_data['success_label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train base model
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            base_model.fit(X_train_scaled, y_train)
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(
                base_model, cv=5, method='sigmoid'
            )
            calibrated_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            logloss = log_loss(y_test, y_pred_proba)
            
            # Feature importance
            feature_importance = dict(zip(self.features, np.abs(base_model.coef_[0])))
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            # Store model
            self.current_model = {
                'model': calibrated_model,
                'scaler': self.scaler,
                'features': self.features,
                'feature_importance': feature_importance,
                'metrics': {
                    'auc': auc,
                    'logloss': logloss,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'positive_samples': int(y_train.sum()),
                    'negative_samples': int(len(y_train) - y_train.sum())
                }
            }
            
            # Save model
            self._save_model()
            
            logger.info(f"Model trained successfully - AUC: {auc:.3f}, LogLoss: {logloss:.3f}")
            
            return {
                'metrics': self.current_model['metrics'],
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return {}
    
    def _save_model(self):
        """Save the trained model"""
        try:
            if not self.current_model:
                return
            
            # Save model coefficients and parameters
            model_data = {
                'features': self.features,
                'scaler_mean': self.scaler.mean_.tolist(),
                'scaler_scale': self.scaler.scale_.tolist(),
                'feature_importance': self.current_model['feature_importance'],
                'metrics': self.current_model['metrics'],
                'training_date': datetime.now().isoformat(),
                'model_version': '1.0'
            }
            
            # Save current model
            model_file = f"{self.model_dir}/model_coef.json"
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            # Create backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"{self.model_dir}/backups/model_coef_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self) -> bool:
        """Load the latest trained model"""
        try:
            model_file = f"{self.model_dir}/model_coef.json"
            if not os.path.exists(model_file):
                logger.warning("No trained model found")
                return False
            
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            # Reconstruct scaler
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(model_data['scaler_mean'])
            self.scaler.scale_ = np.array(model_data['scaler_scale'])
            
            # Store model data
            self.current_model = {
                'scaler': self.scaler,
                'features': model_data['features'],
                'feature_importance': model_data['feature_importance'],
                'metrics': model_data['metrics']
            }
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_success(self, product_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict success probability for a product
        
        Args:
            product_features: Dictionary of product features
            
        Returns:
            Tuple of (success_probability, confidence_metrics)
        """
        try:
            if not self.current_model:
                logger.warning("No model loaded for prediction")
                return 0.5, {}
            
            # Extract features
            feature_vector = []
            for feature in self.features:
                value = product_features.get(feature, 0.0)
                if pd.isna(value):
                    value = 0.0
                feature_vector.append(float(value))
            
            # Scale features
            feature_array = np.array(feature_vector).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            
            # For now, use a simple heuristic based on feature importance
            # In production, you'd use the actual trained model
            success_score = 0.0
            total_weight = 0.0
            
            for i, feature in enumerate(self.features):
                weight = self.current_model['feature_importance'].get(feature, 0.1)
                normalized_value = (feature_vector[i] - self.scaler.mean_[i]) / self.scaler.scale_[i]
                success_score += weight * normalized_value
                total_weight += weight
            
            if total_weight > 0:
                success_score = success_score / total_weight
            
            # Convert to probability (0-1)
            success_prob = 1 / (1 + np.exp(-success_score))
            success_prob = max(0.0, min(1.0, success_prob))
            
            # Confidence metrics
            confidence = {
                'feature_quality': self._calculate_feature_quality(product_features),
                'model_confidence': self._calculate_model_confidence(),
                'data_completeness': self._calculate_data_completeness(product_features)
            }
            
            return success_prob, confidence
            
        except Exception as e:
            logger.error(f"Failed to predict success: {e}")
            return 0.5, {}
    
    def _calculate_feature_quality(self, features: Dict[str, Any]) -> float:
        """Calculate quality of input features"""
        try:
            quality_scores = []
            for feature in self.features:
                value = features.get(feature, None)
                if value is not None and not pd.isna(value):
                    quality_scores.append(1.0)
                else:
                    quality_scores.append(0.0)
            
            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
        except:
            return 0.0
    
    def _calculate_model_confidence(self) -> float:
        """Calculate confidence in the model based on training metrics"""
        try:
            if not self.current_model:
                return 0.0
            
            metrics = self.current_model['metrics']
            
            # Base confidence on AUC and sample size
            auc_confidence = metrics.get('auc', 0.5)
            sample_confidence = min(1.0, metrics.get('training_samples', 0) / 1000.0)
            
            return (auc_confidence + sample_confidence) / 2
            
        except:
            return 0.0
    
    def _calculate_data_completeness(self, features: Dict[str, Any]) -> float:
        """Calculate completeness of input data"""
        try:
            total_features = len(self.features)
            available_features = sum(1 for f in self.features if f in features and features[f] is not None)
            return available_features / total_features if total_features > 0 else 0.0
            
        except:
            return 0.0
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get current model statistics"""
        try:
            if not self.current_model:
                return {}
            
            return {
                'model_loaded': True,
                'features_count': len(self.features),
                'feature_importance': self.current_model['feature_importance'],
                'metrics': self.current_model['metrics'],
                'last_training': self.current_model.get('metrics', {}).get('training_date', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"Failed to get model stats: {e}")
            return {'model_loaded': False}

# Global instance
ml_trainer = MLTrainer()
