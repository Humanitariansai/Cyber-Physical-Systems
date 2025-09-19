#!/usr/bin/env python3
"""
Pre-deployment test to verify system readiness
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all critical imports work"""
    print("Testing imports...")
    
    try:
        import mlflow
        print("✅ MLflow imported successfully")
    except ImportError as e:
        print(f"❌ MLflow import failed: {e}")
        return False
    
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import optuna
        print("✅ Optuna imported successfully")
    except ImportError as e:
        print(f"❌ Optuna import failed: {e}")
        return False
    
    # Test model imports
    sys.path.append(str(Path(__file__).parent / "ml-models"))
    
    try:
        from basic_forecaster import BasicTimeSeriesForecaster
        print("✅ BasicTimeSeriesForecaster imported successfully")
    except ImportError as e:
        print(f"⚠️ BasicTimeSeriesForecaster import failed: {e}")
    
    try:
        from xgboost_forecaster import XGBoostForecaster
        print("✅ XGBoostForecaster imported successfully")
    except ImportError as e:
        print(f"⚠️ XGBoostForecaster import failed: {e}")
    
    return True

def test_directories():
    """Test that required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = [
        "ml-models",
        "data-collection", 
        "docker",
        "mlruns"
    ]
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")
            return False
    
    return True

def test_files():
    """Test that critical files exist"""
    print("\nTesting critical files...")
    
    required_files = [
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "docker/prediction_api.py",
        "ml-models/basic_forecaster.py"
    ]
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    return True

def main():
    print("🔍 PRE-DEPLOYMENT READINESS CHECK")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_directories()
    all_tests_passed &= test_files()
    all_tests_passed &= test_imports()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ SYSTEM READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("1. Build Docker image: docker-compose build")
        print("2. Start services: docker-compose up -d")
        print("3. Test endpoints:")
        print("   - MLflow: http://localhost:5000")
        print("   - API: http://localhost:8080/health")
    else:
        print("❌ DEPLOYMENT NOT READY - Fix issues above first")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)