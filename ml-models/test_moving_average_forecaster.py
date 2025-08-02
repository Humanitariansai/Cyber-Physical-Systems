"""
Tests for Simple Moving Average Forecasting Model
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Simple tests for the moving average forecasting model.
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    
import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_arima_forecaster import SimpleMovingAverageForecaster, create_sample_data


class TestMovingAverageForecaster:
    """Test the simple moving average forecaster."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = create_sample_data(n_points=50)
    
    def test_sma_initialization(self):
        """Test SMA forecaster initialization."""
        forecaster = SimpleMovingAverageForecaster(window=5, method='sma')
        assert forecaster.window == 5
        assert forecaster.method == 'sma'
        assert not forecaster.is_fitted
    
    def test_ema_initialization(self):
        """Test EMA forecaster initialization."""
        forecaster = SimpleMovingAverageForecaster(method='ema', alpha=0.3)
        assert forecaster.alpha == 0.3
        assert forecaster.method == 'ema'
        assert not forecaster.is_fitted
    
    def test_wma_initialization(self):
        """Test WMA forecaster initialization."""
        forecaster = SimpleMovingAverageForecaster(window=7, method='wma')
        assert forecaster.window == 7
        assert forecaster.method == 'wma'
        assert not forecaster.is_fitted
    
    def test_invalid_method(self):
        """Test invalid method raises error."""
        try:
            SimpleMovingAverageForecaster(method='invalid')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Method must be" in str(e)
    
    def test_invalid_alpha(self):
        """Test invalid alpha raises error."""
        try:
            SimpleMovingAverageForecaster(method='ema', alpha=1.5)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Alpha must be between" in str(e)
    
    def test_sma_fitting(self):
        """Test SMA model fitting."""
        forecaster = SimpleMovingAverageForecaster(window=5, method='sma')
        forecaster.fit(self.test_data, 'temperature')
        
        assert forecaster.is_fitted
        assert forecaster.target_col == 'temperature'
        assert len(forecaster.fitted_values) == len(self.test_data)
    
    def test_ema_fitting(self):
        """Test EMA model fitting."""
        forecaster = SimpleMovingAverageForecaster(method='ema', alpha=0.3)
        forecaster.fit(self.test_data, 'temperature')
        
        assert forecaster.is_fitted
        assert forecaster.target_col == 'temperature'
    
    def test_wma_fitting(self):
        """Test WMA model fitting."""
        forecaster = SimpleMovingAverageForecaster(window=3, method='wma')
        forecaster.fit(self.test_data, 'temperature')
        
        assert forecaster.is_fitted
        assert forecaster.target_col == 'temperature'
    
    def test_sma_predictions(self):
        """Test SMA predictions."""
        forecaster = SimpleMovingAverageForecaster(window=5, method='sma')
        forecaster.fit(self.test_data, 'temperature')
        
        predictions = forecaster.predict(n_steps=3)
        
        assert len(predictions) == 3
        assert all(isinstance(p, (int, float)) for p in predictions)
        
        # Predictions should be reasonable
        data_mean = self.test_data['temperature'].mean()
        data_std = self.test_data['temperature'].std()
        
        for pred in predictions:
            assert abs(pred - data_mean) < 5 * data_std
    
    def test_ema_predictions(self):
        """Test EMA predictions."""
        forecaster = SimpleMovingAverageForecaster(method='ema', alpha=0.3)
        forecaster.fit(self.test_data, 'temperature')
        
        predictions = forecaster.predict(n_steps=2)
        
        assert len(predictions) == 2
        assert all(isinstance(p, (int, float)) for p in predictions)
    
    def test_wma_predictions(self):
        """Test WMA predictions."""
        forecaster = SimpleMovingAverageForecaster(window=4, method='wma')
        forecaster.fit(self.test_data, 'temperature')
        
        predictions = forecaster.predict(n_steps=3)
        
        assert len(predictions) == 3
        assert all(isinstance(p, (int, float)) for p in predictions)
    
    def test_evaluation(self):
        """Test model evaluation."""
        forecaster = SimpleMovingAverageForecaster(window=5, method='sma')
        forecaster.fit(self.test_data, 'temperature')
        
        metrics = forecaster.evaluate(self.test_data)
        
        # Check metrics exist
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'n_samples' in metrics
        assert 'method' in metrics
        
        # Metrics should be positive (except R² which can be negative)
        assert metrics['mse'] > 0
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert metrics['n_samples'] > 0
        assert metrics['method'] == 'sma'
    
    def test_unfitted_model_error(self):
        """Test error when using unfitted model."""
        forecaster = SimpleMovingAverageForecaster(window=5, method='sma')
        
        try:
            forecaster.predict(n_steps=1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model must be fitted" in str(e)
        
        try:
            forecaster.evaluate(self.test_data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model must be fitted" in str(e)
    
    def test_insufficient_data(self):
        """Test error with insufficient data."""
        small_data = self.test_data.head(2)
        forecaster = SimpleMovingAverageForecaster(window=5, method='sma')
        
        try:
            forecaster.fit(small_data, 'temperature')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Need at least" in str(e)
    
    def test_missing_target_column(self):
        """Test error with missing target column."""
        forecaster = SimpleMovingAverageForecaster(window=5, method='sma')
        
        try:
            forecaster.fit(self.test_data, 'missing_column')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found in data" in str(e)


def test_simple_workflow():
    """Test a simple end-to-end workflow."""
    # Create simple data
    data = create_sample_data(n_points=30)
    
    # Test each method
    methods = ['sma', 'ema', 'wma']
    
    for method in methods:
        if method == 'ema':
            forecaster = SimpleMovingAverageForecaster(method=method, alpha=0.3)
        else:
            forecaster = SimpleMovingAverageForecaster(method=method, window=3)
        
        forecaster.fit(data, 'temperature')
        predictions = forecaster.predict(n_steps=2)
        metrics = forecaster.evaluate(data)
        
        assert len(predictions) == 2
        assert 'rmse' in metrics
        print(f"✓ {method.upper()} workflow test passed")


def test_comparison_workflow():
    """Test comparing different methods."""
    data = create_sample_data(n_points=40)
    
    methods = [
        ('sma', {'window': 5}),
        ('ema', {'alpha': 0.3}),
        ('wma', {'window': 5})
    ]
    
    results = {}
    
    for method, params in methods:
        if method == 'ema':
            forecaster = SimpleMovingAverageForecaster(method=method, alpha=params['alpha'])
        else:
            forecaster = SimpleMovingAverageForecaster(method=method, window=params['window'])
        
        forecaster.fit(data, 'temperature')
        metrics = forecaster.evaluate(data)
        results[method] = metrics['rmse']
    
    # All should produce reasonable results
    for method, rmse in results.items():
        assert rmse > 0
        assert rmse < 10  # Should be reasonable for temperature data
        print(f"✓ {method.upper()} RMSE: {rmse:.3f}")


if __name__ == "__main__":
    # Run basic tests
    test_simple_workflow()
    test_comparison_workflow()
    
    # Run class tests manually
    test_class = TestMovingAverageForecaster()
    test_class.setup_method()
    
    try:
        test_class.test_sma_initialization()
        test_class.test_ema_initialization()
        test_class.test_wma_initialization()
        test_class.test_invalid_method()
        test_class.test_invalid_alpha()
        test_class.test_sma_fitting()
        test_class.test_ema_fitting()
        test_class.test_wma_fitting()
        test_class.test_sma_predictions()
        test_class.test_ema_predictions()
        test_class.test_wma_predictions()
        test_class.test_evaluation()
        test_class.test_unfitted_model_error()
        test_class.test_insufficient_data()
        test_class.test_missing_target_column()
        print("✓ All class tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✓ All basic tests completed!")
    
    # Run pytest if available
    if PYTEST_AVAILABLE:
        try:
            pytest.main([__file__, "-v"])
        except Exception as e:
            print(f"pytest execution failed: {e}")
    else:
        print("pytest not available, but manual tests completed")
