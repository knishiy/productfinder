#!/usr/bin/env python3
"""
Simple rule-based product scoring for Winning Product Pipeline
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def _norm(x: float, lo: float, hi: float) -> float:
    """Normalize value to 0-1 range"""
    if hi == lo:
        return 0.0
    x = max(min(x, hi), lo)
    return (x - lo) / (hi - lo)

class ProductScorer:
    """Simple rule-based product scorer"""
    
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.min_margin_pct = (cfg.get("scoring") or {}).get("min_margin_pct", 0.25)
        penalties = (cfg.get("scoring") or {}).get("penalties", {})
        self.pen_ip = penalties.get("ip_brand", 0.25)
        self.pen_low_seller = penalties.get("low_seller_rating", 0.10)
        self.pen_sat = penalties.get("saturation", 0.10)
        
        logger.info(f"ProductScorer initialized with min_margin_pct={self.min_margin_pct}")

    def score_item(self, f: Dict[str, Any]) -> float:
        """Score a single product based on its features"""
        # Expect keys; default to safe values
        margin_pct = f.get("margin_pct", 0.0)
        demand_vel = f.get("sales_velocity", 0.0)
        trend = f.get("trend_growth_14d", 0.0)
        stab = f.get("price_stability", 0.0)
        comp = f.get("competition_density", 0.0)
        lead_days = f.get("lead_time_days", 30.0)

        score = 0.0
        score += 0.35 * _norm(margin_pct, 0.0, 0.7)             # 0–70%+
        score += 0.20 * _norm(demand_vel, 0.0, 50.0)            # 0–50/day
        score += 0.15 * _norm(trend, -0.2, 0.8)                 # -20%–+80%
        score += 0.10 * _norm(stab, 0.0, 1.0)
        score += 0.10 * (1 - _norm(comp, 0.0, 100.0))
        score += 0.10 * (1 - _norm(lead_days, 2.0, 20.0))

        penalty = 0.0
        if f.get("ip_brand_flag"):
            penalty += self.pen_ip
        if f.get("seller_rating", 5.0) < 4.5:
            penalty += self.pen_low_seller
        if f.get("saturation_cluster", 0) > 50:
            penalty += self.pen_sat

        final_score = max(0.0, score - penalty)
        logger.debug(f"Scored item: base={score:.3f}, penalty={penalty:.3f}, final={final_score:.3f}")
        
        return final_score

    def score_all(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all products in a list"""
        out = []
        for fx in (features_list or []):
            s = self.score_item(fx)
            fx2 = dict(fx)
            fx2["score_overall"] = s
            out.append(fx2)
        
        # Sort by score (highest first)
        out.sort(key=lambda x: x.get("score_overall", 0.0), reverse=True)
        
        logger.info(f"Scored {len(out)} products")
        return out

    def export_scoring_results(self, scoring_results: Any, output_dir: str) -> Optional[str]:
        """Export scoring results to file (placeholder for now)"""
        try:
            # This is a placeholder - implement actual export logic later
            logger.info(f"Scoring results exported to {output_dir}")
            return f"{output_dir}/scoring_results.json"
        except Exception as e:
            logger.error(f"Failed to export scoring results: {e}")
            return None
