#!/usr/bin/env python3
"""
Simple test script to debug pipeline initialization issues
"""

import logging
import sys
from pipeline import WinningProductPipeline

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_pipeline():
    """Test pipeline initialization"""
    print("🔍 Testing Winning Product Pipeline Initialization...")
    print("=" * 60)
    
    try:
        # Try to create pipeline
        print("1. Creating pipeline instance...")
        pipeline = WinningProductPipeline("config.yaml")
        print("✅ Pipeline instance created successfully")
        
        # Test initialization
        print("\n2. Testing component initialization...")
        if pipeline.test_initialization():
            print("✅ All components initialized successfully!")
            
            # Show component status
            print("\n3. Component Status:")
            print(f"   - ProductMatcher: {'✅' if pipeline.matcher else '❌'}")
            print(f"   - CostCalculator: {'✅' if pipeline.cost_calculator else '❌'}")
            print(f"   - ProductScorer: {'✅' if pipeline.scorer else '❌'}")
            print(f"   - eBay ETL: {'✅' if pipeline.ebay_etl else '⚠️ (Not configured)'}")
            print(f"   - Keepa ETL: {'✅' if pipeline.keepa_etl else '⚠️ (Not configured)'}")
            print(f"   - Trends ETL: {'✅' if pipeline.trends_etl else '❌'}")
            print(f"   - AliExpress ETL: {'✅' if pipeline.aliexpress_etl else '⚠️ (Not configured)'}")
            
            return True
        else:
            print("❌ Component initialization failed!")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_imports():
    """Test if individual modules can be imported"""
    print("\n🔍 Testing Individual Module Imports...")
    print("=" * 60)
    
    modules_to_test = [
        ("etl.ebay", "EbayETL"),
        ("etl.amazon_keepa", "KeepaETL"),
        ("etl.trends", "TrendsETL"),
        ("etl.aliexpress", "AliExpressETL"),
        ("matching", "ProductMatcher"),
        ("costing", "CostCalculator"),
        ("scoring", "ProductScorer")
    ]
    
    all_imports_ok = True
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} imported successfully")
        except Exception as e:
            print(f"❌ Failed to import {module_name}.{class_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

if __name__ == "__main__":
    print("🚀 Winning Product Pipeline - Initialization Test")
    print("=" * 60)
    
    # Test individual imports first
    imports_ok = test_individual_imports()
    
    if imports_ok:
        print("\n✅ All modules imported successfully, testing pipeline...")
        # Test pipeline initialization
        pipeline_ok = test_pipeline()
        
        if pipeline_ok:
            print("\n🎉 All tests passed! Pipeline is ready to use.")
            sys.exit(0)
        else:
            print("\n❌ Pipeline initialization failed!")
            sys.exit(1)
    else:
        print("\n❌ Some modules failed to import!")
        sys.exit(1)
