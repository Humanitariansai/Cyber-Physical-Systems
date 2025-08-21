"""
Quick MLflow Integration Test
============================
"""

import numpy as np
import pandas as pd
from basic_forecaster import BasicTimeSeriesForecaster
from mlflow_tracking import setup_mlflow_tracking

def create_test_data():
    """Create simple test data."""
    np.random.seed(42)
    timestamps = pd.date_range(start='2023-01-01', periods=100, freq='H')
    temperature = 20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100)
    return pd.DataFrame({'timestamp': timestamps, 'temperature': temperature})

def main():
    print("🧪 Quick MLflow Integration Test")
    print("=" * 40)
    
    # Create test data
    data = create_test_data()
    train_data = data[:80]
    test_data = data[80:]
    
    print(f"✅ Data created: {len(train_data)} train, {len(test_data)} test")
    
    # Test Basic Forecaster with MLflow
    print("🔬 Testing Basic Forecaster with MLflow...")
    forecaster = BasicTimeSeriesForecaster(n_lags=3, enable_mlflow=True)
    forecaster.fit(train_data, target_col='temperature', run_name='quick_test')
    
    metrics = forecaster.evaluate(test_data, log_to_mlflow=True)
    print(f"   RMSE: {metrics['rmse']:.3f}")
    
    forecaster.finish_mlflow_run()
    
    print("✅ Test completed successfully!")
    print("💡 Check MLflow UI with: mlflow ui")

if __name__ == "__main__":
    main()
