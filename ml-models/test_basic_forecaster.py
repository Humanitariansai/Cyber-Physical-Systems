"""
Tests for Basic Time-Series Forecasting
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Simple tests for the basic forecasting model.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basic_forecaster import BasicTimeSeriesForecaster, create_sample_data


def test_forecaster_initialization():
    """Test basic forecaster initialization."""
    forecaster = BasicTimeSeriesForecaster(n_lags=3)
    
    assert forecaster.n_lags == 3
    assert not forecaster.is_fitted
    assert len(forecaster.feature_names) == 0
    print("✓ Forecaster initialization test passed")


def test_sample_data_generation():
    """Test sample data generation."""
    data = create_sample_data(n_points=50)
    
    assert len(data) == 50
    assert 'timestamp' in data.columns
    assert 'temperature' in data.columns
    assert not data['temperature'].isna().any()
    print("✓ Sample data generation test passed")


def test_lag_feature_creation():
    """Test lag feature creation."""
    forecaster = BasicTimeSeriesForecaster(n_lags=3)
    data = create_sample_data(n_points=20)
    
    df_with_lags = forecaster.create_lag_features(data, 'temperature')
    
    # Should have lag columns
    expected_cols = ['temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3']
    for col in expected_cols:
        assert col in df_with_lags.columns
    
    # Should have fewer rows due to NaN removal
    assert len(df_with_lags) == len(data) - forecaster.n_lags
    print("✓ Lag feature creation test passed")


def test_model_fitting_and_prediction():
    """Test model fitting and prediction."""
    forecaster = BasicTimeSeriesForecaster(n_lags=5)
    data = create_sample_data(n_points=50)
    
    # Fit the model
    forecaster.fit(data, 'temperature')
    assert forecaster.is_fitted
    assert len(forecaster.feature_names) == 5
    
    # Make predictions
    predictions = forecaster.predict(data, n_steps=3)
    assert len(predictions) == 3
    assert all(isinstance(p, (int, float)) for p in predictions)
    
    print("✓ Model fitting and prediction test passed")


def test_model_evaluation():
    """Test model evaluation."""
    forecaster = BasicTimeSeriesForecaster(n_lags=4)
    data = create_sample_data(n_points=100)
    
    # Split data
    train_data = data[:80]
    test_data = data[80:]
    
    # Fit and evaluate
    forecaster.fit(train_data, 'temperature')
    metrics = forecaster.evaluate(test_data)
    
    # Check metrics exist and are reasonable
    required_metrics = ['mse', 'mae', 'rmse', 'r2', 'n_samples']
    for metric in required_metrics:
        assert metric in metrics
    
    assert metrics['mse'] > 0
    assert metrics['mae'] > 0
    assert metrics['rmse'] > 0
    assert metrics['n_samples'] > 0
    
    print("✓ Model evaluation test passed")


def test_error_handling():
    """Test error handling."""
    forecaster = BasicTimeSeriesForecaster(n_lags=3)
    data = create_sample_data(n_points=10)
    
    # Test prediction without fitting
    try:
        forecaster.predict(data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be fitted" in str(e)
    
    # Test insufficient data
    small_data = data[:2]  # Only 2 points, need at least 3 for n_lags=3
    try:
        forecaster.fit(small_data, 'temperature')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Not enough data" in str(e)
    
    print("✓ Error handling test passed")


def run_all_tests():
    """Run all tests."""
    print("Running Basic Forecaster Tests...")
    print("=" * 40)
    
    test_forecaster_initialization()
    test_sample_data_generation()
    test_lag_feature_creation()
    test_model_fitting_and_prediction()
    test_model_evaluation()
    test_error_handling()
    
    print("=" * 40)
    print("✓ All tests passed successfully!")


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print("\nRunning End-to-End Workflow Test...")
    print("-" * 40)
    
    # Generate data
    data = create_sample_data(n_points=100)
    print(f"Generated {len(data)} data points")
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create and train forecaster
    forecaster = BasicTimeSeriesForecaster(n_lags=6)
    forecaster.fit(train_data, 'temperature')
    print(f"Trained model with {len(train_data)} samples")
    
    # Evaluate
    metrics = forecaster.evaluate(test_data)
    print(f"RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
    
    # Make future predictions
    predictions = forecaster.predict(train_data, n_steps=5)
    print(f"Future predictions: {[f'{p:.2f}' for p in predictions]}")
    
    print("✓ End-to-end workflow completed successfully!")


if __name__ == "__main__":
    try:
        run_all_tests()
        test_end_to_end_workflow()
    except Exception as e:
        print(f"\n✗ Tests failed: {e}")
        import traceback
        traceback.print_exc()
