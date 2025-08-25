#!/usr/bin/env python3
"""
Bulletproof configuration loader for Winning Product Pipeline
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)

# Default configuration that will always be available
DEFAULT_CFG = {
    "run_mode": "mvp",
    "scan": {
        "ship_to": "US",
        "landed_cap_usd": 10.0,
        "lead_time_max_days": 15,
        "min_score": 0.65
    },
    "sources": {
        "ebay": {
            "enabled": False,
            "categories": ["177772", "63514"],
            "keywords": ["pet nail grinder", "phone stand", "cable organizer"],
            "limit": 120
        },
        "amazon": {
            "enabled": False,
            "provider": "keepa"
        },
        "trends": {
            "enabled": True,
            "keywords": ["pet nail grinder", "phone stand", "cable organizer",
                        "lint remover", "mini flashlight", "soap dispenser"],
            "timeframe": "today 1-m",
            "geo": "US"
        },
        "aliexpress": {
            "enabled": True,
            "provider": "apify",
            "max_results": 20
        }
    },
    "matching": {
        "title_threshold": 0.65,
        "use_image_phash": True
    },
    "scoring": {
        "min_margin_pct": 0.25,
        "penalties": {
            "ip_brand": 0.25,
            "low_seller_rating": 0.10,
            "saturation": 0.10
        }
    }
}

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, preserving nested structure"""
    result = deepcopy(base)
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration with bulletproof error handling
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary (never None, always has defaults)
    """
    try:
        # Always start with defaults
        config = deepcopy(DEFAULT_CFG)
        
        # Try to load user config
        if os.path.exists(config_path):
            logger.info(f"Loading configuration from {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                
            if user_config is not None:
                # Deep merge user config with defaults
                config = deep_merge(config, user_config)
                logger.info("Configuration loaded and merged successfully")
            else:
                logger.warning("User config file is empty or invalid, using defaults")
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        logger.info("Using default configuration due to YAML error")
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_path}: {e}")
        logger.info("Using default configuration due to loading error")
    
    # Validate critical sections exist
    required_sections = ["sources", "matching", "scoring"]
    for section in required_sections:
        if section not in config:
            logger.warning(f"Missing required section '{section}', adding defaults")
            config[section] = deepcopy(DEFAULT_CFG[section])
    
    logger.info(f"Configuration ready with {len(config)} sections")
    return config

def dget(obj: Any, key: str, default: Any = None) -> Any:
    """
    Null-safe dictionary getter
    
    Args:
        obj: Object to get from (dict or any other type)
        key: Key to look up
        default: Default value if key not found or obj not a dict
        
    Returns:
        Value at key or default
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def get_nested_config(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested configuration values
    
    Args:
        config: Configuration dictionary
        *keys: Sequence of keys to traverse
        default: Default value if path not found
        
    Returns:
        Value at nested path or default
    """
    current = config
    
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = dget(current, key)
        if current is None:
            return default
    
    return current if current is not None else default
