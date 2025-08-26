"""
Risk Factor Calculator for Winning Product Finder
Calculates risk percentage and success likelihood for products
"""

import math
from typing import Dict, Any, Optional
from datetime import datetime

class RiskCalculator:
    """Calculates risk factors and success likelihood for products"""
    
    def __init__(self):
        self.risk_weights = {
            'price_risk': 0.35,      # Higher price = higher risk
            'competition_risk': 0.25, # More competition = higher risk
            'seller_risk': 0.20,      # Lower seller rating = higher risk
            'lead_time_risk': 0.15,   # Longer lead time = higher risk
            'trend_risk': 0.05        # Declining trends = higher risk
        }
    
    def calculate_price_risk(self, landed_cost: float, price_cap: float = 10.0) -> float:
        """
        Calculate price-based risk (1% = low risk, 100% = high risk)
        
        Args:
            landed_cost: Total landed cost of the product
            price_cap: Base price cap (default $10)
        
        Returns:
            Risk percentage (1-100)
        """
        if landed_cost <= price_cap:
            # Below cap: low risk (1-25%)
            return max(1.0, (landed_cost / price_cap) * 25.0)
        else:
            # Above cap: exponential risk increase (25-100%)
            excess_ratio = (landed_cost - price_cap) / price_cap
            risk = 25.0 + (75.0 * (1 - math.exp(-excess_ratio * 2)))
            return min(100.0, risk)
    
    def calculate_competition_risk(self, competition_density: float) -> float:
        """
        Calculate competition-based risk
        
        Args:
            competition_density: Normalized competition score (0-1)
        
        Returns:
            Risk percentage (1-100)
        """
        # Higher competition = higher risk
        return max(1.0, competition_density * 100.0)
    
    def calculate_seller_risk(self, seller_rating: float, min_rating: float = 4.5) -> float:
        """
        Calculate seller-based risk
        
        Args:
            seller_rating: Seller rating (0-5)
            min_rating: Minimum acceptable rating
        
        Returns:
            Risk percentage (1-100)
        """
        if seller_rating >= min_rating:
            # Good rating: low risk (1-20%)
            return max(1.0, (1 - (seller_rating - min_rating) / (5.0 - min_rating)) * 20.0)
        else:
            # Poor rating: high risk (20-100%)
            return min(100.0, 20.0 + (80.0 * (1 - seller_rating / min_rating)))
    
    def calculate_lead_time_risk(self, lead_time_days: float, max_days: float = 15.0) -> float:
        """
        Calculate lead time risk
        
        Args:
            lead_time_days: Estimated delivery time in days
            max_days: Maximum acceptable lead time
        
        Returns:
            Risk percentage (1-100)
        """
        if lead_time_days <= max_days:
            # Within acceptable range: low risk (1-30%)
            return max(1.0, (lead_time_days / max_days) * 30.0)
        else:
            # Beyond acceptable range: high risk (30-100%)
            excess_ratio = (lead_time_days - max_days) / max_days
            risk = 30.0 + (70.0 * (1 - math.exp(-excess_ratio)))
            return min(100.0, risk)
    
    def calculate_trend_risk(self, trend_growth_14d: float) -> float:
        """
        Calculate trend-based risk
        
        Args:
            trend_growth_14d: 14-day trend growth (-1 to +1)
        
        Returns:
            Risk percentage (1-100)
        """
        if trend_growth_14d >= 0:
            # Positive trend: low risk (1-20%)
            return max(1.0, 20.0 - (trend_growth_14d * 19.0))
        else:
            # Negative trend: high risk (20-100%)
            return min(100.0, 20.0 + (80.0 * abs(trend_growth_14d)))
    
    def calculate_overall_risk(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall risk factor and breakdown
        
        Args:
            product_data: Product data dictionary with risk metrics
        
        Returns:
            Dictionary with risk breakdown and overall risk
        """
        # Extract metrics with defaults
        landed_cost = product_data.get('landed_cost', 0.0)
        competition_density = product_data.get('competition_density', 0.5)
        seller_rating = product_data.get('seller_rating', 4.5)
        lead_time_days = product_data.get('lead_time_days', 15.0)
        trend_growth_14d = product_data.get('trend_growth_14d', 0.0)
        
        # Calculate individual risk components
        price_risk = self.calculate_price_risk(landed_cost)
        comp_risk = self.calculate_competition_risk(competition_density)
        seller_risk = self.calculate_seller_risk(seller_rating)
        lead_risk = self.calculate_lead_time_risk(lead_time_days)
        trend_risk = self.calculate_trend_risk(trend_growth_14d)
        
        # Calculate weighted overall risk
        overall_risk = (
            price_risk * self.risk_weights['price_risk'] +
            comp_risk * self.risk_weights['competition_risk'] +
            seller_risk * self.risk_weights['seller_risk'] +
            lead_risk * self.risk_weights['lead_time_risk'] +
            trend_risk * self.risk_weights['trend_risk']
        )
        
        return {
            'overall_risk': round(overall_risk, 1),
            'risk_breakdown': {
                'price_risk': round(price_risk, 1),
                'competition_risk': round(comp_risk, 1),
                'seller_risk': round(seller_risk, 1),
                'lead_time_risk': round(lead_risk, 1),
                'trend_risk': round(trend_risk, 1)
            },
            'risk_level': self._get_risk_level(overall_risk)
        }
    
    def calculate_success_likelihood(self, score: float, risk_percentage: float, 
                                   margin_pct: float, trend_growth: float) -> Dict[str, Any]:
        """
        Calculate success likelihood based on score, risk, margin, and trends
        
        Args:
            score: Overall product score (0-1)
            risk_percentage: Overall risk percentage (1-100)
            margin_pct: Profit margin percentage (0-1)
            trend_growth: Trend growth rate (-1 to +1)
        
        Returns:
            Dictionary with success likelihood and factors
        """
        # Base success score from product score (0-100%)
        base_success = score * 100.0
        
        # Risk penalty (higher risk reduces success likelihood)
        risk_penalty = (risk_percentage / 100.0) * 40.0  # Max 40% penalty
        
        # Margin bonus (higher margins increase success likelihood)
        margin_bonus = min(20.0, margin_pct * 100.0)  # Max 20% bonus
        
        # Trend bonus/penalty
        trend_bonus = 0.0
        if trend_growth > 0:
            trend_bonus = min(15.0, trend_growth * 100.0)  # Max 15% bonus
        elif trend_growth < 0:
            trend_bonus = max(-15.0, trend_growth * 100.0)  # Max 15% penalty
        
        # Calculate final success likelihood
        success_likelihood = base_success - risk_penalty + margin_bonus + trend_bonus
        success_likelihood = max(0.0, min(100.0, success_likelihood))
        
        return {
            'success_likelihood': round(success_likelihood, 1),
            'factors': {
                'base_score': round(base_success, 1),
                'risk_penalty': round(risk_penalty, 1),
                'margin_bonus': round(margin_bonus, 1),
                'trend_bonus': round(trend_bonus, 1)
            },
            'success_level': self._get_success_level(success_likelihood)
        }
    
    def _get_risk_level(self, risk_percentage: float) -> str:
        """Get human-readable risk level"""
        if risk_percentage <= 25:
            return "Low Risk"
        elif risk_percentage <= 50:
            return "Medium Risk"
        elif risk_percentage <= 75:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _get_success_level(self, success_likelihood: float) -> str:
        """Get human-readable success level"""
        if success_likelihood >= 80:
            return "Excellent"
        elif success_likelihood >= 60:
            return "Good"
        elif success_likelihood >= 40:
            return "Fair"
        elif success_likelihood >= 20:
            return "Poor"
        else:
            return "Very Poor"

# Global instance for easy access
risk_calculator = RiskCalculator()
