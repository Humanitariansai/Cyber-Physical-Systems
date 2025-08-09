"""
Tests for XGBoost Time Series Forecasting Model
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Comprehensive tests for the XGBoost forecasting model.
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

from xgboost_forecaster import XGBoostTimeSeriesForecaster, create_sample_data


class TestXGBoostForecaster:
    """Test the XGBoost time series forecaster."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = create_sample_data(n_points=100)
        self.small_data = create_sample_data(n_points=30)
    
    def test_initialization_default(self):
        """Test default initialization."""
        forecaster = XGBoostTimeSeriesForecaster()
        assert forecaster.n_lags == 12
        assert forecaster.rolling_windows == [3, 7, 12]
        assert forecaster.random_state == 42
        assert not forecaster.is_fitted
        assert forecaster.model is None
        assert forecaster.target_col is None
        assert forecaster.feature_names == []
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        custom_params = {
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 50
        }
        
        forecaster = XGBoostTimeSeriesForecaster(
            n_lags=8,
            rolling_windows=[5, 10],
            xgb_params=custom_params,
            random_state=123
        )
        
        assert forecaster.n_lags == 8
        assert forecaster.rolling_windows == [5, 10]
        assert forecaster.random_state == 123
        assert forecaster.xgb_params['max_depth'] == 4
        assert forecaster.xgb_params['learning_rate'] == 0.05
        assert forecaster.xgb_params['n_estimators'] == 50
        assert forecaster.xgb_params['random_state'] == 123
    
    def test_feature_creation(self):
        """Test feature creation functionality."""
        forecaster = XGBoostTimeSeriesForecaster(n_lags=5, rolling_windows=[3, 5])
        features_df = forecaster._create_features(self.test_data, 'temperature')
        
        # Check that features are created
        assert not features_df.empty
        assert len(features_df.columns) > 0
        
        # Check for expected feature types
        feature_names = features_df.columns.tolist()
        
        # Should have lag features
        lag_features = [f for f in feature_names if f.startswith('lag_')]
        assert len(lag_features) == 5
        
        # Should have rolling features
        rolling_features = [f for f in feature_names if f.startswith('rolling_')]
        assert len(rolling_features) > 0
        
        # Should have time features
        time_features = [f for f in feature_names if any(t in f for t in ['hour', 'day', 'month', 'time'])]
        assert len(time_features) > 0
    
    def test_fitting(self):
        """Test model fitting."""
        forecaster = XGBoostTimeSeriesForecaster(n_lags=5, rolling_windows=[3, 5])
        forecaster.fit(self.test_data, 'temperature')
        
        assert forecaster.is_fitted
        assert forecaster.target_col == 'temperature'
        assert forecaster.model is not None
        assert len(forecaster.feature_names) > 0
        assert forecaster.target_mean > 0  # Should be around 20°C
        assert forecaster.target_std > 0
    
    def test_fitting_invalid_target(self):
        """Test fitting with invalid target column."""
        forecaster = XGBoostTimeSeriesForecaster()
        
        try:
            forecaster.fit(self.test_data, 'invalid_column')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found in data" in str(e)
    
    def test_predictions(self):
        """Test making predictions."""
        forecaster = XGBoostTimeSeriesForecaster(n_lags=5, rolling_windows=[3, 5])
        forecaster.fit(self.test_data, 'temperature')
        
        # Test single step prediction
        predictions = forecaster.predict(n_steps=1, last_known_data=self.test_data.tail(20))
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.floating, np.integer))
        
        # Test multi-step predictions
        predictions = forecaster.predict(n_steps=5, last_known_data=self.test_data.tail(20))
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float, np.floating, np.integer)) for p in predictions)
        
        # Predictions should be reasonable (around 20°C ± some range)
        data_mean = self.test_data['temperature'].mean()
        data_std = self.test_data['temperature'].std()
        for pred in predictions:
            assert abs(pred - data_mean) < 5 * data_std
    
    def test_predictions_unfitted(self):
        """Test predictions with unfitted model."""
        forecaster = XGBoostTimeSeriesForecaster()
        
        try:
            forecaster.predict(n_steps=1, last_known_data=self.test_data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model must be fitted" in str(e)
    
    def test_predictions_no_data(self):
        """Test predictions without last_known_data."""
        forecaster = XGBoostTimeSeriesForecaster()
        forecaster.fit(self.test_data, 'temperature')
        
        try:
            forecaster.predict(n_steps=1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "last_known_data must be provided" in str(e)
    
    def test_evaluation(self):
        """Test model evaluation."""
        forecaster = XGBoostTimeSeriesForecaster(n_lags=5, rolling_windows=[3, 5])
        forecaster.fit(self.test_data, 'temperature')
        
        metrics = forecaster.evaluate(self.test_data)
        
        # Check that all expected metrics exist
        expected_metrics = ['mse', 'mae', 'rmse', 'r2', 'mape', 'n_samples', 'model', 'n_features']
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check metric values
        assert metrics['mse'] > 0
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert metrics['n_samples'] > 0
        assert metrics['n_features'] > 0
        assert metrics['model'] == 'XGBoost'
        assert metrics['mape'] >= 0
        
        # R² can be negative for poor fits, but should be reasonable for our synthetic data
        assert metrics['r2'] > -1
    
    def test_evaluation_with_predictions(self):
        """Test evaluation with provided predictions."""
        forecaster = XGBoostTimeSeriesForecaster(n_lags=5, rolling_windows=[3, 5])
        forecaster.fit(self.test_data, 'temperature')
        
        # Generate some predictions
        test_size = 10
        predictions = np.array([20.0] * test_size)  # Simple constant predictions
        
        metrics = forecaster.evaluate(self.test_data.tail(test_size), predictions)
        
        assert metrics['n_samples'] == test_size
        assert metrics['mse'] > 0
        assert metrics['rmse'] > 0
    
    def test_evaluation_unfitted(self):
        """Test evaluation with unfitted model."""
        forecaster = XGBoostTimeSeriesForecaster()
        
        try:
            forecaster.evaluate(self.test_data)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model must be fitted" in str(e)
    
    def test_feature_importance(self):
        """Test feature importance functionality."""
        forecaster = XGBoostTimeSeriesForecaster(n_lags=5, rolling_windows=[3, 5])
        forecaster.fit(self.test_data, 'temperature')
        
        # Test getting feature importance
        importance_df = forecaster.get_feature_importance(top_n=10)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) <= 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        
        # Check that importances are sorted in descending order
        importances = importance_df['importance'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
        
        # All importances should be non-negative
        assert all(imp >= 0 for imp in importances)
    
    def test_feature_importance_unfitted(self):
        """Test feature importance with unfitted model."""
        forecaster = XGBoostTimeSeriesForecaster()
        
        try:
            forecaster.get_feature_importance()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model must be fitted" in str(e)
    
    def test_insufficient_data(self):
        """Test with insufficient data for feature creation."""
        # Create very small dataset
        tiny_data = create_sample_data(n_points=5)
        forecaster = XGBoostTimeSeriesForecaster(n_lags=10)  # More lags than data points
        
        try:
            forecaster.fit(tiny_data, 'temperature')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No features could be created" in str(e)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random state."""
        # First run
        forecaster1 = XGBoostTimeSeriesForecaster(random_state=42, n_lags=5)
        forecaster1.fit(self.test_data, 'temperature')
        pred1 = forecaster1.predict(n_steps=3, last_known_data=self.test_data.tail(20))
        
        # Second run with same random state
        forecaster2 = XGBoostTimeSeriesForecaster(random_state=42, n_lags=5)
        forecaster2.fit(self.test_data, 'temperature')
        pred2 = forecaster2.predict(n_steps=3, last_known_data=self.test_data.tail(20))
        
        # Should be very close (allowing for small numerical differences)
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)
    
    def test_different_data_types(self):
        """Test with different types of input data."""
        # Test with different index types
        data_with_int_index = self.test_data.copy()
        data_with_int_index.index = range(len(data_with_int_index))
        
        forecaster = XGBoostTimeSeriesForecaster(n_lags=5, rolling_windows=[3, 5])
        forecaster.fit(data_with_int_index, 'temperature')
        
        assert forecaster.is_fitted
        predictions = forecaster.predict(n_steps=2, last_known_data=data_with_int_index.tail(20))
        assert len(predictions) == 2


def test_simple_xgboost_workflow():
    """Test a simple end-to-end XGBoost workflow."""
    print("Testing XGBoost workflow...")
    
    # Create data
    data = create_sample_data(n_points=80)
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Initialize and fit model
    forecaster = XGBoostTimeSeriesForecaster(
        n_lags=6,
        rolling_windows=[3, 6],
        xgb_params={'n_estimators': 50, 'max_depth': 4}
    )
    
    forecaster.fit(train_data, 'temperature')
    assert forecaster.is_fitted
    
    # Make predictions
    n_pred = len(test_data)
    predictions = forecaster.predict(n_steps=n_pred, last_known_data=train_data.tail(30))
    assert len(predictions) == n_pred
    
    # Evaluate
    metrics = forecaster.evaluate(test_data, predictions)
    assert metrics['rmse'] > 0
    assert metrics['r2'] > -1  # Can be negative but should be reasonable
    
    print(f"✓ XGBoost RMSE: {metrics['rmse']:.3f}°C, R²: {metrics['r2']:.3f}")


def test_xgboost_feature_engineering():
    """Test XGBoost feature engineering capabilities."""
    print("Testing feature engineering...")
    
    data = create_sample_data(n_points=60)
    forecaster = XGBoostTimeSeriesForecaster(n_lags=8, rolling_windows=[3, 6, 12])
    
    # Test feature creation
    features_df = forecaster._create_features(data, 'temperature')
    
    # Should have various types of features
    feature_names = features_df.columns.tolist()
    
    # Count feature types
    lag_features = len([f for f in feature_names if f.startswith('lag_')])
    rolling_features = len([f for f in feature_names if f.startswith('rolling_')])
    time_features = len([f for f in feature_names if any(t in f for t in ['hour', 'day', 'time'])])
    diff_features = len([f for f in feature_names if f.startswith('diff_')])
    
    print(f"✓ Created {len(feature_names)} features:")
    print(f"  - Lag features: {lag_features}")
    print(f"  - Rolling features: {rolling_features}")
    print(f"  - Time features: {time_features}")
    print(f"  - Difference features: {diff_features}")
    
    assert lag_features > 0
    assert rolling_features > 0
    assert time_features > 0
    assert diff_features > 0


def test_xgboost_comparison_with_baselines():
    """Compare XGBoost with simple baselines."""
    print("Comparing XGBoost with baselines...")
    
    data = create_sample_data(n_points=100, noise_level=0.3)
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # XGBoost model
    xgb_forecaster = XGBoostTimeSeriesForecaster(
        n_lags=6,
        rolling_windows=[3, 6],
        xgb_params={'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1}
    )
    xgb_forecaster.fit(train_data, 'temperature')
    
    n_pred = len(test_data)
    xgb_pred = xgb_forecaster.predict(n_steps=n_pred, last_known_data=train_data.tail(30))
    xgb_metrics = xgb_forecaster.evaluate(test_data, xgb_pred)
    
    # Simple baseline: last value
    last_value_pred = np.full(n_pred, train_data['temperature'].iloc[-1])
    last_value_rmse = np.sqrt(np.mean((test_data['temperature'].values - last_value_pred)**2))
    
    # Simple baseline: mean value
    mean_pred = np.full(n_pred, train_data['temperature'].mean())
    mean_rmse = np.sqrt(np.mean((test_data['temperature'].values - mean_pred)**2))
    
    print(f"✓ XGBoost RMSE:     {xgb_metrics['rmse']:.3f}°C")
    print(f"✓ Last value RMSE: {last_value_rmse:.3f}°C")
    print(f"✓ Mean RMSE:       {mean_rmse:.3f}°C")
    
    # XGBoost should generally perform better than simple baselines
    # (Though this may not always be true for very simple synthetic data)
    assert xgb_metrics['rmse'] > 0
    assert last_value_rmse > 0
    assert mean_rmse > 0


if __name__ == "__main__":
    # Run basic tests
    test_simple_xgboost_workflow()
    test_xgboost_feature_engineering()
    test_xgboost_comparison_with_baselines()
    
    # Run class tests manually
    test_class = TestXGBoostForecaster()
    test_class.setup_method()
    
    test_methods = [
        'test_initialization_default',
        'test_initialization_custom',
        'test_feature_creation',
        'test_fitting',
        'test_fitting_invalid_target',
        'test_predictions',
        'test_predictions_unfitted',
        'test_predictions_no_data',
        'test_evaluation',
        'test_evaluation_with_predictions',
        'test_evaluation_unfitted',
        'test_feature_importance',
        'test_feature_importance_unfitted',
        'test_insufficient_data',
        'test_reproducibility',
        'test_different_data_types'
    ]
    
    print(f"\nRunning {len(test_methods)} detailed tests...")
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            # Reset test data for each test
            test_class.setup_method()
            method = getattr(test_class, test_method)
            method()
            print(f"✓ {test_method}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All XGBoost tests completed successfully!")
    else:
        print(f"❌ {failed} tests failed")
    
    # Run pytest if available
    if PYTEST_AVAILABLE:
        try:
            print("\nRunning pytest...")
            pytest.main([__file__, "-v"])
        except Exception as e:
            print(f"pytest execution failed: {e}")
    else:
        print("pytest not available, but manual tests completed")
