"""
ML Scorer for Winning Product Finder
Uses trained ML models to predict product success probabilities
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLScorer:
    """Machine Learning-based product success scorer"""
    
    def __init__(self, model_path: str = "models/model_coef.json"):
        self.model_path = model_path
        self.model_data = None
        self.features = []
        self.scaler_mean = None
        self.scaler_scale = None
        self.feature_importance = {}
        self.model_loaded = False
        
        # Load model if available
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained ML model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            with open(self.model_path, 'r') as f:
                self.model_data = json.load(f)
            
            # Extract model components
            self.features = self.model_data.get('features', [])
            self.scaler_mean = np.array(self.model_data.get('scaler_mean', []))
            self.scaler_scale = np.array(self.model_data.get('scaler_scale', []))
            self.feature_importance = self.model_data.get('feature_importance', {})
            
            self.model_loaded = True
            logger.info(f"ML model loaded successfully with {len(self.features)} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return False
    
    def extract_features(self, product: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract ML features from a product
        
        Args:
            product: Product dictionary from pipeline
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # Map product data to ML features
            feature_mapping = {
                'avg_engagement_rate': product.get('engagement_rate', 0.0),
                'views_per_hour': product.get('avg_views', 0) / 24.0 if product.get('avg_views') else 0.0,
                'trend_slope': product.get('trend_growth_14d', 0.0),
                'price_usd': product.get('landed_cost', 0.0),
                'margin_pct': product.get('margin_pct', 0.0),
                'seller_rating': product.get('seller_rating', 4.5),
                'lead_time_days': product.get('lead_time_days', 15.0),
                'competitor_count': product.get('competition_density', 25.0),
                'trend_growth_14d': product.get('trend_growth_14d', 0.0),
                'price_stability': product.get('price_stability', 0.5),
                'competition_density': product.get('competition_density', 25.0),
                'total_sold': product.get('total_sold', 0),
                'engagement_rate': product.get('engagement_rate', 0.0)
            }
            
            # Extract only the features the model expects
            for feature in self.features:
                features[feature] = feature_mapping.get(feature, 0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return {}
    
    def predict_success(self, product: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict success probability using ML model
        
        Args:
            product: Product dictionary from pipeline
            
        Returns:
            Tuple of (success_probability, confidence_metrics)
        """
        try:
            if not self.model_loaded:
                logger.warning("ML model not loaded, using fallback prediction")
                return self._fallback_prediction(product)
            
            # Extract features
            features = self.extract_features(product)
            if not features:
                return 0.5, {'error': 'Failed to extract features'}
            
            # Convert to feature vector
            feature_vector = np.array([features[f] for f in self.features])
            
            # Scale features
            scaled_features = (feature_vector - self.scaler_mean) / self.scaler_scale
            
            # Calculate success score using feature importance
            success_score = 0.0
            total_weight = 0.0
            
            for i, feature in enumerate(self.features):
                weight = self.feature_importance.get(feature, 0.1)
                success_score += weight * scaled_features[i]
                total_weight += weight
            
            if total_weight > 0:
                success_score = success_score / total_weight
            
            # Convert to probability using sigmoid
            success_prob = 1 / (1 + np.exp(-success_score))
            success_prob = max(0.0, min(1.0, success_prob))
            
            # Calculate confidence metrics
            confidence = self._calculate_confidence(features, success_prob)
            
            return success_prob, confidence
            
        except Exception as e:
            logger.error(f"Failed to predict success: {e}")
            return 0.5, {'error': str(e)}
    
    def _fallback_prediction(self, product: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Fallback prediction when ML model is not available"""
        try:
            # Simple heuristic based on key metrics
            margin_score = min(1.0, product.get('margin_pct', 0.0) * 2)  # 50% margin = 1.0
            trend_score = max(0.0, min(1.0, product.get('trend_growth_14d', 0.0) + 0.5))
            seller_score = max(0.0, min(1.0, (product.get('seller_rating', 4.5) - 3.0) / 2.0))
            
            # Weighted average
            success_prob = (margin_score * 0.4 + trend_score * 0.3 + seller_score * 0.3)
            
            confidence = {
                'method': 'fallback_heuristic',
                'confidence': 0.3,  # Low confidence for fallback
                'features_used': ['margin_pct', 'trend_growth_14d', 'seller_rating']
            }
            
            return success_prob, confidence
            
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return 0.5, {'error': 'Fallback prediction failed'}
    
    def _calculate_confidence(self, features: Dict[str, Any], prediction: float) -> Dict[str, Any]:
        """Calculate confidence in the ML prediction"""
        try:
            # Feature quality score
            feature_quality = 0.0
            for feature in self.features:
                value = features.get(feature, None)
                if value is not None and not pd.isna(value):
                    feature_quality += 1.0
            feature_quality = feature_quality / len(self.features)
            
            # Model confidence based on training metrics
            model_metrics = self.model_data.get('metrics', {})
            auc_score = model_metrics.get('auc', 0.5)
            training_samples = model_metrics.get('training_samples', 0)
            
            # Sample size confidence (more samples = higher confidence)
            sample_confidence = min(1.0, training_samples / 1000.0)
            
            # Overall confidence
            overall_confidence = (feature_quality * 0.3 + auc_score * 0.4 + sample_confidence * 0.3)
            
            return {
                'method': 'ml_model',
                'confidence': overall_confidence,
                'feature_quality': feature_quality,
                'model_auc': auc_score,
                'training_samples': training_samples,
                'prediction_confidence': 'high' if overall_confidence > 0.7 else 'medium' if overall_confidence > 0.5 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return {'method': 'ml_model', 'confidence': 0.5, 'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded ML model"""
        try:
            if not self.model_loaded:
                return {'model_loaded': False}
            
            return {
                'model_loaded': True,
                'features_count': len(self.features),
                'feature_importance': self.feature_importance,
                'model_metrics': self.model_data.get('metrics', {}),
                'training_date': self.model_data.get('training_date', 'Unknown'),
                'model_version': self.model_data.get('model_version', '1.0')
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'model_loaded': False, 'error': str(e)}
    
    def compare_with_rule_based(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare ML prediction with rule-based scoring
        
        Args:
            product: Product dictionary from pipeline
            
        Returns:
            Comparison results
        """
        try:
            # Get ML prediction
            ml_prob, ml_confidence = self.predict_success(product)
            
            # Get rule-based score (assuming it's in the product data)
            rule_score = product.get('score_overall', 0.0)
            
            # Calculate difference
            score_difference = abs(ml_prob - rule_score)
            
            # Determine which method to trust more
            ml_confidence_score = ml_confidence.get('confidence', 0.5)
            rule_confidence = 0.7  # Assume rule-based has medium confidence
            
            if ml_confidence_score > rule_confidence:
                recommended_score = ml_prob
                recommended_method = 'ML Model'
                reasoning = f"ML model has higher confidence ({ml_confidence_score:.2f} vs {rule_confidence:.2f})"
            else:
                recommended_score = rule_score
                recommended_method = 'Rule-based'
                reasoning = f"Rule-based scoring has higher confidence ({rule_confidence:.2f} vs {ml_confidence_score:.2f})"
            
            return {
                'ml_prediction': ml_prob,
                'rule_based_score': rule_score,
                'score_difference': score_difference,
                'ml_confidence': ml_confidence_score,
                'rule_confidence': rule_confidence,
                'recommended_score': recommended_score,
                'recommended_method': recommended_method,
                'reasoning': reasoning,
                'agreement_level': 'high' if score_difference < 0.1 else 'medium' if score_difference < 0.2 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Failed to compare scoring methods: {e}")
            return {'error': str(e)}

# Global instance
ml_scorer = MLScorer()
