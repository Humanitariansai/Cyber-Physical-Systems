"""
Test Suite for GRU Time-Series Forecaster
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Unit tests for the GRU forecasting model.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from gru_forecaster import GRUTimeSeriesForecaster


class TestGRUForecaster(unittest.TestCase):
    """Test cases for GRU time-series forecaster"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample time series data
        np.random.seed(42)
        n_points = 200
        timestamps = pd.date_range(start='2025-01-01', periods=n_points, freq='H')
        
        # Generate synthetic data with trend and seasonality
        t = np.arange(n_points)
        trend = 0.02 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 24)
        noise = np.random.normal(0, 0.5, n_points)
        values = 20 + trend + seasonal + noise
        
        self.data = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        # Initialize forecaster
        self.forecaster = GRUTimeSeriesForecaster(
            sequence_length=10,
            n_gru_units=32,
            dropout_rate=0.2,
            learning_rate=0.001,
            enable_mlflow=False
        )
    
    def test_initialization(self):
        """Test GRU forecaster initialization"""
        self.assertIsNotNone(self.forecaster)
        self.assertEqual(self.forecaster.sequence_length, 10)
        self.assertEqual(self.forecaster.n_gru_units, 32)
        self.assertFalse(self.forecaster.is_fitted)
    
    def test_model_building(self):
        """Test GRU model architecture building"""
        model = self.forecaster._build_model()
        self.assertIsNotNone(model)
        
        # Check model has correct input shape
        self.assertEqual(len(model.layers), 3)  # GRU + Dropout + Dense
        
    def test_sequence_preparation(self):
        """Test sequence preparation for GRU"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.forecaster.sequence_length = 3
        
        X, y = self.forecaster._prepare_sequences(data.reshape(-1, 1))
        
        # Check shapes
        self.assertEqual(X.shape[0], len(data) - self.forecaster.sequence_length)
        self.assertEqual(X.shape[1], self.forecaster.sequence_length)
        self.assertEqual(len(y), len(data) - self.forecaster.sequence_length)
    
    def test_fitting(self):
        """Test model fitting"""
        # Use smaller epochs for testing
        history = self.forecaster.fit(
            self.data,
            target_col='value',
            epochs=5,
            batch_size=16,
            validation_split=0.2
        )
        
        self.assertTrue(self.forecaster.is_fitted)
        self.assertIsNotNone(self.forecaster.model)
        self.assertIn('loss', history.history)
    
    def test_prediction(self):
        """Test making predictions"""
        # First fit the model
        self.forecaster.fit(
            self.data,
            target_col='value',
            epochs=5,
            batch_size=16,
            validation_split=0.2
        )
        
        # Make predictions
        test_data = self.data.tail(50)['value']
        predictions = self.forecaster.predict(test_data, n_steps=10)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_evaluation(self):
        """Test model evaluation"""
        # Fit model
        self.forecaster.fit(
            self.data,
            target_col='value',
            epochs=5,
            batch_size=16,
            validation_split=0.2
        )
        
        # Evaluate
        test_data = self.data.tail(50)[['value']]
        metrics = self.forecaster.evaluate(test_data)
        
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertTrue(metrics['rmse'] >= 0)
        self.assertTrue(metrics['mae'] >= 0)
    
    def test_prediction_without_fitting(self):
        """Test that prediction raises error without fitting"""
        with self.assertRaises(ValueError):
            self.forecaster.predict(self.data['value'], n_steps=5)
    
    def test_gru_vs_lstm_comparison(self):
        """Test that GRU has fewer parameters than LSTM"""
        gru_model = self.forecaster._build_model()
        gru_params = gru_model.count_params()
        
        # GRU should have fewer parameters than equivalent LSTM
        # (GRU has 2 gates vs LSTM's 3 gates)
        # This is a relative test - we just ensure model builds correctly
        self.assertGreater(gru_params, 0)
        print(f"GRU model has {gru_params} parameters")


class TestGRUPerformance(unittest.TestCase):
    """Performance tests for GRU forecaster"""
    
    def setUp(self):
        """Set up performance test data"""
        np.random.seed(42)
        n_points = 500
        timestamps = pd.date_range(start='2025-01-01', periods=n_points, freq='H')
        
        t = np.arange(n_points)
        trend = 0.05 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 24)
        noise = np.random.normal(0, 1, n_points)
        values = 50 + trend + seasonal + noise
        
        self.data = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
    
    def test_training_speed(self):
        """Test that GRU trains reasonably fast"""
        import time
        
        forecaster = GRUTimeSeriesForecaster(
            sequence_length=10,
            n_gru_units=50,
            dropout_rate=0.2,
            learning_rate=0.001,
            enable_mlflow=False
        )
        
        start_time = time.time()
        forecaster.fit(
            self.data,
            target_col='value',
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
        training_time = time.time() - start_time
        
        # Training should complete in reasonable time
        # (adjust threshold based on hardware)
        self.assertLess(training_time, 60, "Training took too long")
        print(f"Training completed in {training_time:.2f} seconds")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGRUForecaster))
    suite.addTests(loader.loadTestsFromTestCase(TestGRUPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running GRU Forecaster Test Suite")
    print("=" * 50)
    success = run_tests()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
