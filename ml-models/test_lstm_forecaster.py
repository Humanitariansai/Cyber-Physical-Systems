"""
Tests for LSTM Time-Series Forecasting
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Unit tests for the LSTM forecasting model.
"""

import numpy as np
import pandas as pd
import sys
import os
import tensorflow as tf

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_forecaster import LSTMTimeSeriesForecaster, create_sample_data


def test_forecaster_initialization():
    """Test LSTM forecaster initialization."""
    forecaster = LSTMTimeSeriesForecaster(sequence_length=10)
    
    assert forecaster.sequence_length == 10
    assert forecaster.n_lstm_units == 50  # default value
    assert not forecaster.is_fitted
    assert forecaster.model is None
    print("✓ Forecaster initialization test passed")


def test_sample_data_generation():
    """Test sample data generation."""
    data = create_sample_data(n_points=50)
    
    assert len(data) == 50
    assert 'timestamp' in data.columns
    assert 'value' in data.columns
    assert not data['value'].isna().any()
    print("✓ Sample data generation test passed")


def test_model_building():
    """Test LSTM model architecture building."""
    forecaster = LSTMTimeSeriesForecaster(
        sequence_length=10,
        n_lstm_units=32,
        n_dense_units=1
    )
    
    model = forecaster._build_model()
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 3  # LSTM + Dropout + Dense
    assert model.input_shape == (None, 10, 1)
    assert model.output_shape == (None, 1)
    print("✓ Model building test passed")


def test_sequence_preparation():
    """Test input sequence preparation."""
    forecaster = LSTMTimeSeriesForecaster(sequence_length=3)
    data = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    
    X, y = forecaster._prepare_sequences(data)
    
    assert X.shape == (2, 3, 1)  # 2 sequences of length 3
    assert y.shape == (2, 1)  # 2 target values
    assert np.array_equal(X[0], np.array([[1], [2], [3]]))
    assert np.array_equal(y[0], np.array([4]))
    print("✓ Sequence preparation test passed")


def test_model_fitting():
    """Test model fitting."""
    forecaster = LSTMTimeSeriesForecaster(sequence_length=5)
    data = create_sample_data(n_points=100)
    
    forecaster.fit(data, 'value', epochs=2, batch_size=32)
    
    assert forecaster.is_fitted
    assert forecaster.model is not None
    print("✓ Model fitting test passed")


def test_prediction():
    """Test prediction functionality."""
    forecaster = LSTMTimeSeriesForecaster(sequence_length=5)
    data = create_sample_data(n_points=100)
    
    # Fit the model
    forecaster.fit(data, 'value', epochs=2, batch_size=32)
    
    # Make predictions
    predictions = forecaster.predict(data['value'], n_steps=3)
    
    assert len(predictions) == 3
    assert all(isinstance(p, (int, float)) for p in predictions)
    print("✓ Prediction test passed")


def test_model_evaluation():
    """Test model evaluation."""
    forecaster = LSTMTimeSeriesForecaster(sequence_length=5)
    data = create_sample_data(n_points=100)
    train_data = data[:80]
    test_data = data[80:]
    
    # Fit model and make predictions
    forecaster.fit(train_data, 'value', epochs=2, batch_size=32)
    predictions = forecaster.predict(test_data['value'], n_steps=len(test_data))
    
    # Evaluate
    metrics = forecaster.evaluate(test_data['value'], predictions)
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    print("✓ Model evaluation test passed")


def run_all_tests():
    """Run all tests."""
    print("\nRunning LSTM Forecaster Tests...")
    print("=" * 40)
    
    test_forecaster_initialization()
    test_sample_data_generation()
    test_model_building()
    test_sequence_preparation()
    test_model_fitting()
    test_prediction()
    test_model_evaluation()
    
    print("=" * 40)
    print("✓ All tests passed successfully!")


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n✗ Tests failed: {e}")
        import traceback
        traceback.print_exc()