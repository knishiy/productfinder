"""
Google Trends ETL Module for Winning Product Finder
Handles keyword trend analysis and momentum calculation
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrendData:
    """Data class for Google Trends information"""
    keyword: str
    interest_over_time: pd.DataFrame
    growth_14d: float
    growth_30d: float
    peak_interest: float
    current_interest: float
    momentum_score: float
    seasonality_score: float
    trend_direction: str  # "rising", "falling", "stable"
    related_queries: List[str]
    related_topics: List[str]
    geo_data: Dict[str, float]
    category: str

class TrendsETL:
    """Google Trends ETL class for keyword analysis"""
    
    def __init__(self, hl: str = "en-US", tz: int = -300):
        """
        Initialize Trends ETL
        
        Args:
            hl: Language (default: en-US)
            tz: Timezone offset in minutes (default: -300 for EST)
        """
        self.pytrends = TrendReq(hl=hl, tz=tz)
        self.session_data = None
        self.rate_limit_delay = 1.0  # Be respectful to Google Trends
        
    def _ensure_session(self):
        """Ensure we have an active session"""
        if not self.session_data:
            try:
                # Make a simple request to establish session
                self.pytrends.build_payload(["test"], timeframe="today 1-m")
                self.session_data = True
                logger.info("Google Trends session established")
            except Exception as e:
                logger.error(f"Failed to establish Google Trends session: {e}")
                return False
        return True
    
    def get_trends(self, keywords: List[str], timeframe: str = "today 1-m", 
                   geo: str = "US", cat: int = 0) -> Dict[str, TrendData]:
        """
        Get trends data for multiple keywords
        
        Args:
            keywords: List of keywords to analyze
            timeframe: Time range (e.g., "today 1-m", "today 3-m", "today 12-m")
            geo: Geographic location (e.g., "US", "GB", "CA")
            cat: Category (0 = all categories)
        
        Returns:
            Dictionary mapping keywords to TrendData objects
        """
        if not self._ensure_session():
            return {}
        
        results = {}
        
        for keyword in keywords:
            try:
                trend_data = self._get_single_trend(keyword, timeframe, geo, cat)
                if trend_data:
                    results[keyword] = trend_data
                
                # Rate limiting between requests
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to get trends for keyword '{keyword}': {e}")
                continue
        
        return results
    
    def get_growth_14d(self, keywords: List[str], timeframe: str = "today 1-m", geo: str = "US") -> Dict[str, float]:
        """
        Get 14-day growth for multiple keywords (simplified method)
        
        Args:
            keywords: List of keywords to analyze
            timeframe: Time range for analysis
            geo: Geographic location for data
        
        Returns:
            Dictionary mapping keywords to growth percentages
        """
        if not self._ensure_session():
            logger.error("Google Trends client not initialized")
            return {}
        
        results = {}
        
        for kw in keywords:
            try:
                logger.info(f"Getting growth data for keyword: {kw}")
                
                # Build payload
                self.pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
                
                # Get interest over time
                df = self.pytrends.interest_over_time()
                
                if df is None or df.empty:
                    logger.warning(f"No trends data available for keyword: {kw}")
                    results[kw] = 0.0
                    continue
                
                # Compute mean last 14 days vs previous 14 days
                s = df[kw].astype(float)
                recent = s.tail(14).mean() if len(s) >= 28 else s.tail(7).mean()
                prior = s.tail(28).head(14).mean() if len(s) >= 28 else max(s.head(len(s)-7).mean(), 1.0)
                growth = (recent / max(prior, 1.0)) - 1.0
                results[kw] = float(growth)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to get growth data for keyword '{kw}': {e}")
                results[kw] = 0.0
                continue
        
        logger.info(f"Collected growth data for {len(results)} keywords")
        return results
    
    def _get_single_trend(self, keyword: str, timeframe: str, geo: str, cat: int) -> Optional[TrendData]:
        """Get trends data for a single keyword"""
        try:
            # Build payload
            self.pytrends.build_payload([keyword], timeframe=timeframe, geo=geo, cat=cat)
            
            # Get interest over time
            interest_over_time = self.pytrends.interest_over_time()
            
            # Get related queries and topics
            related_queries = self.pytrends.related_queries()
            related_topics = self.pytrends.related_topics()
            
            # Get interest by region
            interest_by_region = self.pytrends.interest_by_region(resolution="COUNTRY")
            
            if interest_over_time.empty:
                logger.warning(f"No trends data available for keyword: {keyword}")
                return None
            
            # Calculate metrics
            growth_14d = self._calculate_growth(interest_over_time, days=14)
            growth_30d = self._calculate_growth(interest_over_time, days=30)
            peak_interest = interest_over_time[keyword].max()
            current_interest = interest_over_time[keyword].iloc[-1]
            momentum_score = self._calculate_momentum_score(interest_over_time[keyword])
            seasonality_score = self._calculate_seasonality_score(interest_over_time[keyword])
            trend_direction = self._determine_trend_direction(interest_over_time[keyword])
            
            # Process related data
            related_queries_list = self._extract_related_queries(related_queries, keyword)
            related_topics_list = self._extract_related_topics(related_topics, keyword)
            geo_data = self._extract_geo_data(interest_by_region, keyword)
            
            trend_data = TrendData(
                keyword=keyword,
                interest_over_time=interest_over_time,
                growth_14d=growth_14d,
                growth_30d=growth_30d,
                peak_interest=peak_interest,
                current_interest=current_interest,
                momentum_score=momentum_score,
                seasonality_score=seasonality_score,
                trend_direction=trend_direction,
                related_queries=related_queries_list,
                related_topics=related_topics_list,
                geo_data=geo_data,
                category="general"
            )
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Failed to process trends for keyword '{keyword}': {e}")
            return None
    
    def _calculate_growth(self, interest_data: pd.DataFrame, days: int) -> float:
        """Calculate growth percentage over specified days"""
        if len(interest_data) < days:
            return 0.0
        
        # Get recent and previous periods
        recent_data = interest_data.iloc[-days:]
        previous_data = interest_data.iloc[-2*days:-days] if len(interest_data) >= 2*days else interest_data.iloc[:-days]
        
        if previous_data.empty or recent_data.empty:
            return 0.0
        
        recent_avg = recent_data.mean().iloc[0]
        previous_avg = previous_data.mean().iloc[0]
        
        if previous_avg == 0:
            return 0.0
        
        growth = (recent_avg - previous_avg) / previous_avg
        return growth
    
    def _calculate_momentum_score(self, interest_series: pd.Series) -> float:
        """Calculate momentum score based on recent trend acceleration"""
        if len(interest_series) < 7:
            return 0.5
        
        # Calculate moving averages
        recent_7d = interest_series.tail(7).mean()
        previous_7d = interest_series.tail(14).head(7).mean()
        previous_14d = interest_series.tail(21).head(7).mean()
        
        if previous_14d == 0:
            return 0.5
        
        # Calculate acceleration
        recent_momentum = (recent_7d - previous_7d) / previous_7d if previous_7d > 0 else 0
        previous_momentum = (previous_7d - previous_14d) / previous_14d if previous_14d > 0 else 0
        
        acceleration = recent_momentum - previous_momentum
        
        # Convert to 0-1 score
        momentum_score = 0.5 + (acceleration * 2)
        return max(0.0, min(1.0, momentum_score))
    
    def _calculate_seasonality_score(self, interest_series: pd.Series) -> float:
        """Calculate seasonality score based on pattern repetition"""
        if len(interest_series) < 30:
            return 0.5
        
        # Calculate autocorrelation at lag 7 (weekly pattern)
        series = interest_series.values
        if len(series) < 14:
            return 0.5
        
        # Simple autocorrelation calculation
        mean_val = np.mean(series)
        if mean_val == 0:
            return 0.5
        
        # Calculate autocorrelation at lag 7
        lag = 7
        if len(series) < 2 * lag:
            return 0.5
        
        numerator = 0
        denominator = 0
        
        for i in range(lag, len(series)):
            numerator += (series[i] - mean_val) * (series[i - lag] - mean_val)
            denominator += (series[i] - mean_val) ** 2
        
        if denominator == 0:
            return 0.5
        
        autocorr = numerator / denominator
        
        # Convert to seasonality score (0-1)
        seasonality_score = (autocorr + 1) / 2
        return max(0.0, min(1.0, seasonality_score))
    
    def _determine_trend_direction(self, interest_series: pd.Series) -> str:
        """Determine overall trend direction"""
        if len(interest_series) < 7:
            return "stable"
        
        # Compare recent 7 days vs previous 7 days
        recent_avg = interest_series.tail(7).mean()
        previous_avg = interest_series.tail(14).head(7).mean()
        
        if previous_avg == 0:
            return "stable"
        
        change_percent = (recent_avg - previous_avg) / previous_avg
        
        if change_percent > 0.1:  # 10% increase
            return "rising"
        elif change_percent < -0.1:  # 10% decrease
            return "falling"
        else:
            return "stable"
    
    def _extract_related_queries(self, related_queries: Dict, keyword: str) -> List[str]:
        """Extract related queries from pytrends response"""
        queries = []
        
        if keyword in related_queries:
            # Get top rising queries
            rising = related_queries[keyword].get("rising", pd.DataFrame())
            if not rising.empty:
                queries.extend(rising["query"].head(5).tolist())
            
            # Get top queries
            top = related_queries[keyword].get("top", pd.DataFrame())
            if not top.empty:
                queries.extend(top["query"].head(5).tolist())
        
        return list(set(queries))[:10]  # Remove duplicates and limit to 10
    
    def _extract_related_topics(self, related_topics: Dict, keyword: str) -> List[str]:
        """Extract related topics from pytrends response"""
        topics = []
        
        if keyword in related_topics:
            # Get top rising topics
            rising = related_topics[keyword].get("rising", pd.DataFrame())
            if not rising.empty:
                topics.extend(rising["topic_title"].head(5).tolist())
            
            # Get top topics
            top = related_topics[keyword].get("top", pd.DataFrame())
            if not top.empty:
                topics.extend(top["topic_title"].head(5).tolist())
        
        return list(set(topics))[:10]  # Remove duplicates and limit to 10
    
    def _extract_geo_data(self, interest_by_region: pd.DataFrame, keyword: str) -> Dict[str, float]:
        """Extract geographic interest data"""
        geo_data = {}
        
        if not interest_by_region.empty and keyword in interest_by_region.columns:
            # Get top countries by interest
            top_countries = interest_by_region[keyword].nlargest(10)
            for country, interest in top_countries.items():
                geo_data[country] = float(interest)
        
        return geo_data
    
    def analyze_keyword_performance(self, trends_data: Dict[str, TrendData]) -> Dict[str, Any]:
        """Analyze overall performance of keywords"""
        if not trends_data:
            return {}
        
        # Calculate aggregate metrics
        growth_scores = [data.growth_14d for data in trends_data.values()]
        momentum_scores = [data.momentum_score for data in trends_data.values()]
        seasonality_scores = [data.seasonality_score for data in trends_data.values()]
        
        # Find best performing keywords
        best_growth = max(growth_scores) if growth_scores else 0
        best_momentum = max(momentum_scores) if momentum_scores else 0
        
        # Calculate trend distribution
        trend_directions = [data.trend_direction for data in trends_data.values()]
        rising_count = trend_directions.count("rising")
        falling_count = trend_directions.count("falling")
        stable_count = trend_directions.count("stable")
        
        return {
            "total_keywords": len(trends_data),
            "avg_growth_14d": np.mean(growth_scores) if growth_scores else 0,
            "avg_momentum_score": np.mean(momentum_scores) if momentum_scores else 0.5,
            "avg_seasonality_score": np.mean(seasonality_scores) if seasonality_scores else 0.5,
            "best_growth_keyword": best_growth,
            "best_momentum_keyword": best_momentum,
            "trend_distribution": {
                "rising": rising_count,
                "falling": falling_count,
                "stable": stable_count
            },
            "rising_percentage": rising_count / len(trends_data) if trends_data else 0
        }
    
    def get_trending_keywords(self, category: str = "all", geo: str = "US", limit: int = 20) -> List[str]:
        """Get currently trending keywords in a category"""
        try:
            # Get trending searches
            trending_searches = self.pytrends.trending_searches(pn=geo)
            
            if trending_searches.empty:
                return []
            
            # Filter by category if specified
            if category != "all":
                # This is a simplified approach - in practice you might want more sophisticated filtering
                trending_keywords = trending_searches.head(limit).iloc[:, 0].tolist()
            else:
                trending_keywords = trending_searches.head(limit).iloc[:, 0].tolist()
            
            return trending_keywords
            
        except Exception as e:
            logger.error(f"Failed to get trending keywords: {e}")
            return []
    
    def export_trends_data(self, trends_data: Dict[str, TrendData], 
                          output_dir: str = "data/trends") -> None:
        """Export trends data to CSV files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for keyword, data in trends_data.items():
            try:
                # Export interest over time
                filename = f"{output_dir}/{keyword.replace(' ', '_')}_trends.csv"
                data.interest_over_time.to_csv(filename)
                
                # Export summary data
                summary_data = {
                    "keyword": [keyword],
                    "growth_14d": [data.growth_14d],
                    "growth_30d": [data.growth_30d],
                    "momentum_score": [data.momentum_score],
                    "seasonality_score": [data.seasonality_score],
                    "trend_direction": [data.trend_direction],
                    "peak_interest": [data.peak_interest],
                    "current_interest": [data.current_interest]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_filename = f"{output_dir}/{keyword.replace(' ', '_')}_summary.csv"
                summary_df.to_csv(summary_filename, index=False)
                
                logger.info(f"Exported trends data for keyword: {keyword}")
                
            except Exception as e:
                logger.error(f"Failed to export trends data for keyword '{keyword}': {e}")

def main():
    """Test function for Trends ETL"""
    trends_etl = TrendsETL()
    
    # Test keywords
    test_keywords = ["wireless earbuds", "phone case", "bluetooth speaker"]
    
    print("Getting trends data...")
    trends_data = trends_etl.get_trends(test_keywords, timeframe="today 1-m", geo="US")
    
    if trends_data:
        print(f"Retrieved trends for {len(trends_data)} keywords")
        
        # Analyze performance
        analysis = trends_etl.analyze_keyword_performance(trends_data)
        print(f"Analysis: {analysis}")
        
        # Show details for first keyword
        first_keyword = list(trends_data.keys())[0]
        first_data = trends_data[first_keyword]
        
        print(f"\nDetails for '{first_keyword}':")
        print(f"Growth 14d: {first_data.growth_14d:.2%}")
        print(f"Momentum Score: {first_data.momentum_score:.3f}")
        print(f"Trend Direction: {first_data.trend_direction}")
        print(f"Related Queries: {first_data.related_queries[:5]}")
        
        # Export data
        trends_etl.export_trends_data(trends_data)
        
    else:
        print("Failed to retrieve trends data")
    
    # Test trending keywords
    print("\nGetting trending keywords...")
    trending = trends_etl.get_trending_keywords(limit=10)
    print(f"Trending keywords: {trending[:5]}")

if __name__ == "__main__":
    main()
