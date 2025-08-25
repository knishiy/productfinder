#!/usr/bin/env python3
"""
Simple test script to verify basic pipeline functionality
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_env_vars():
    """Test if environment variables are loaded"""
    print("🔍 Testing Environment Variables...")
    
    required_vars = ["EBAY_CLIENT_ID", "EBAY_CLIENT_SECRET", "APIFY_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if value and value != f"xxxxxxxxxxxxxxx{var.split('_')[-1].lower()}":
            print(f"✅ {var}: {value[:8]}...")
        else:
            print(f"❌ {var}: Not set or using placeholder")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Please set these environment variables in your .env file:")
        for var in missing_vars:
            print(f"   {var}=your_actual_value")
        return False
    
    return True

def test_imports():
    """Test if all modules can be imported"""
    print("\n🔍 Testing Module Imports...")
    
    try:
        from pipeline import WinningProductPipeline
        print("✅ Pipeline module imported successfully")
        
        from etl.ebay import get_token, search_items, best_sellers
        print("✅ eBay ETL functions imported successfully")
        
        from etl.trends import TrendsETL
        print("✅ Trends ETL imported successfully")
        
        from etl.aliexpress import search_aliexpress_apify
        print("✅ AliExpress ETL functions imported successfully")
        
        from matching import ProductMatcher
        print("✅ ProductMatcher imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test if configuration can be loaded"""
    print("\n🔍 Testing Configuration...")
    
    try:
        import yaml
        
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully")
        
        # Check required sections
        required_sections = ["sources", "matching", "scoring"]
        for section in required_sections:
            if section in config:
                print(f"✅ {section} section found")
            else:
                print(f"❌ {section} section missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_pipeline_creation():
    """Test if pipeline can be created"""
    print("\n🔍 Testing Pipeline Creation...")
    
    try:
        from pipeline import WinningProductPipeline
        
        pipeline = WinningProductPipeline("config.yaml")
        print("✅ Pipeline instance created successfully")
        
        # Test initialization
        if pipeline.test_initialization():
            print("✅ Pipeline initialization test passed")
            return True
        else:
            print("❌ Pipeline initialization test failed")
            return False
        
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Winning Product Pipeline - Simple Test")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_env_vars),
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Pipeline Creation", test_pipeline_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Set your actual API keys in .env file")
        print("2. Run: py -3.12 web_app.py")
        print("3. Open http://localhost:5000 in your browser")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
