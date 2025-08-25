#!/usr/bin/env python3
"""
Simple cost calculation for Winning Product Pipeline
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CostCalculator:
    """Simple cost calculator for product analysis"""
    
    def __init__(self, fee_rate: float = 0.13, buffer: float = 0.05):
        self.fee_rate = fee_rate
        self.buffer = buffer
        logger.info(f"CostCalculator initialized with fee_rate={fee_rate}, buffer={buffer}")

    def landed_cost(self, unit_price: float, ship_cost: float) -> float:
        """Calculate total landed cost including buffer"""
        unit_price = float(unit_price or 0.0)
        ship_cost = float(ship_cost or 0.0)
        return round(unit_price + ship_cost + self.buffer * (unit_price + ship_cost), 2)

    def margin(self, sell_price: float, landed: float) -> tuple[float, float]:
        """Calculate profit and margin percentage"""
        sell_price = float(sell_price or 0.0)
        landed = float(landed or 0.0)
        fees = self.fee_rate * sell_price
        profit = sell_price - fees - landed
        pct = (profit / sell_price) if sell_price > 0 else 0.0
        return (profit, pct)

    def calculate_landed_cost(self, supplier_product: Any) -> Dict[str, Any]:
        """Calculate landed cost for a supplier product"""
        try:
            unit_price = getattr(supplier_product, 'unit_price', 0.0)
            ship_cost = getattr(supplier_product, 'ship_cost', 0.0)
            
            total_landed = self.landed_cost(unit_price, ship_cost)
            
            return {
                'unit_price': unit_price,
                'shipping_cost': ship_cost,
                'buffer_amount': self.buffer * (unit_price + ship_cost),
                'total_landed_cost': total_landed
            }
        except Exception as e:
            logger.warning(f"Failed to calculate landed cost: {e}")
            return {
                'unit_price': 0.0,
                'shipping_cost': 0.0,
                'buffer_amount': 0.0,
                'total_landed_cost': 0.0
            }

    def assess_risk(self, supplier_product: Any, cost_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk based on cost and supplier factors"""
        try:
            landed_cost = cost_breakdown.get('total_landed_cost', 0.0)
            seller_rating = getattr(supplier_product, 'seller_rating', 4.0)
            
            # Simple risk scoring
            cost_risk = min(1.0, landed_cost / 10.0)  # Higher cost = higher risk
            seller_risk = max(0.0, (5.0 - seller_rating) / 5.0)  # Lower rating = higher risk
            
            overall_risk = (cost_risk * 0.6) + (seller_risk * 0.4)
            
            if overall_risk < 0.3:
                risk_level = "low"
            elif overall_risk < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            return {
                'cost_risk_score': cost_risk,
                'seller_risk_score': seller_risk,
                'overall_risk_score': overall_risk,
                'risk_level': risk_level
            }
        except Exception as e:
            logger.warning(f"Failed to assess risk: {e}")
            return {
                'cost_risk_score': 0.5,
                'seller_risk_score': 0.5,
                'overall_risk_score': 0.5,
                'risk_level': "unknown"
            }

    def analyze_profitability(self, sell_price: float, cost_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze profitability of a product"""
        try:
            landed_cost = cost_breakdown.get('total_landed_cost', 0.0)
            profit, margin_pct = self.margin(sell_price, landed_cost)
            
            return {
                'sell_price': sell_price,
                'landed_cost': landed_cost,
                'fees': self.fee_rate * sell_price,
                'net_profit': profit,
                'net_margin_percentage': margin_pct
            }
        except Exception as e:
            logger.warning(f"Failed to analyze profitability: {e}")
            return {
                'sell_price': 0.0,
                'landed_cost': 0.0,
                'fees': 0.0,
                'net_profit': 0.0,
                'net_margin_percentage': 0.0
            }
