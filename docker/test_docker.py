"""
Docker Implementation Test Suite
==============================

Basic tests to verify Docker containers are working correctly.
"""

import requests
import json
import time
import sys

def test_mlflow_server():
    """Test MLflow server connectivity"""
    try:
        response = requests.get("http://localhost:5000", timeout=10)
        if response.status_code == 200:
            print("✅ MLflow Server: PASSED")
            return True
        else:
            print(f"❌ MLflow Server: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ MLflow Server: FAILED (Error: {e})")
        return False

def test_prediction_api_health():
    """Test prediction API health endpoint"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction API Health: PASSED")
            print(f"   Model Loaded: {data.get('model_loaded', False)}")
            return True
        else:
            print(f"❌ Prediction API Health: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Prediction API Health: FAILED (Error: {e})")
        return False

def test_prediction_api_predict():
    """Test prediction API prediction endpoint"""
    try:
        # Test data
        test_payload = {
            "temperature_history": [22.5, 23.1, 22.8, 23.3, 22.9, 23.0]
        }
        
        response = requests.post(
            "http://localhost:8080/predict",
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            predicted_temp = data.get('predicted_temperature')
            
            # Basic validation
            if isinstance(predicted_temp, (int, float)) and 15 <= predicted_temp <= 35:
                print("✅ Prediction API Predict: PASSED")
                print(f"   Predicted Temperature: {predicted_temp}°C")
                print(f"   Model Used: {data.get('model_used', 'unknown')}")
                return True
            else:
                print(f"❌ Prediction API Predict: FAILED (Invalid prediction: {predicted_temp})")
                return False
        else:
            print(f"❌ Prediction API Predict: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Prediction API Predict: FAILED (Error: {e})")
        return False

def test_prediction_api_root():
    """Test prediction API root endpoint"""
    try:
        response = requests.get("http://localhost:8080/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'service' in data and 'Cyber-Physical' in data['service']:
                print("✅ Prediction API Root: PASSED")
                return True
            else:
                print("❌ Prediction API Root: FAILED (Invalid response)")
                return False
        else:
            print(f"❌ Prediction API Root: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Prediction API Root: FAILED (Error: {e})")
        return False

def main():
    """Run all tests"""
    print("🧪 Docker Implementation Test Suite")
    print("=" * 50)
    
    print("\n📋 Testing Services...")
    
    # Wait for services to be ready
    print("⏳ Waiting for services to start (30 seconds)...")
    time.sleep(30)
    
    tests = [
        test_mlflow_server,
        test_prediction_api_health,
        test_prediction_api_root,
        test_prediction_api_predict
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(2)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Docker implementation is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the logs and ensure all services are running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
