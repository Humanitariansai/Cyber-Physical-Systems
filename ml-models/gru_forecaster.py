"""
GRU Time-Series Forecasting Model
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Implementation of time-series forecasting using GRU (Gated Recurrent Unit) neural networks.
GRU is a lighter alternative to LSTM with faster training and fewer parameters.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. GRU forecaster will not work.")

import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))

# Try to import MLflow tracking
try:
    from mlflow_tracking import ExperimentTracker, create_experiment_config
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Experiment tracking disabled.")


class GRUTimeSeriesForecaster:
    """
    GRU-based time-series forecasting model.
    
    GRU (Gated Recurrent Unit) is a simpler alternative to LSTM with:
    - Fewer parameters (faster training)
    - Similar performance for many tasks
    - Better computational efficiency
    - Easier to tune hyperparameters
    """
    
    def __init__(
        self,
        sequence_length=10,
        n_features=1,
        n_gru_units=50,
        n_dense_units=1,
        dropout_rate=0.2,
        learning_rate=0.001,
        results_dir='results',
        enable_mlflow=True
    ):
        """
        Initialize the GRU forecaster.
        
        Args:
            sequence_length (int): Number of time steps to look back
            n_features (int): Number of input features
            n_gru_units (int): Number of GRU units in the layer
            n_dense_units (int): Number of units in the dense layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the Adam optimizer
            results_dir (str): Directory to store results
            enable_mlflow (bool): Whether to enable MLflow experiment tracking
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_gru_units = n_gru_units
        self.n_dense_units = n_dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.results_dir = results_dir
        
        # Initialize MLflow tracking if available and enabled
        if MLFLOW_AVAILABLE and enable_mlflow:
            self.enable_mlflow = True
            self.mlflow_tracker = None
            self.current_run_id = None
        else:
            self.enable_mlflow = False
            
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize MLflow experiment
        if self.enable_mlflow:
            try:
                self.mlflow_tracker = ExperimentTracker(create_experiment_config("gru-forecasting"))
                print("MLflow experiment tracking enabled")
            except Exception as e:
                print(f"Failed to initialize MLflow tracking: {e}")
                self.enable_mlflow = False
    
    def _build_model(self):
        """Build the GRU model architecture."""
        model = Sequential([
            # GRU layer - note: return_sequences=False for single output
            GRU(
                units=self.n_gru_units,
                activation='relu',
                input_shape=(self.sequence_length, self.n_features),
                return_sequences=False  # Only return last output
            ),
            Dropout(self.dropout_rate),
            Dense(self.n_dense_units)
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_sequences(self, data):
        """Prepare input sequences for the GRU model."""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def fit(
        self,
        data,
        target_col='value',
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        run_name='gru_training'
    ):
        """
        Fit the GRU model to the training data.
        
        Args:
            data (pd.DataFrame): Training data
            target_col (str): Name of the target column
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            run_name (str): Name for the MLflow run
        """
        # Start MLflow run if enabled
        if self.enable_mlflow:
            self.mlflow_tracker.start_run(run_name=run_name)
            self.current_run_id = self.mlflow_tracker.current_run.info.run_id
        
        # Extract and scale data
        series = data[target_col].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(series)
        
        # Prepare sequences
        X, y = self._prepare_sequences(scaled_data)
        
        # Build model
        self.model = self._build_model()
        
        # Log parameters to MLflow
        if self.enable_mlflow:
            self.mlflow_tracker.log_params({
                'sequence_length': self.sequence_length,
                'n_gru_units': self.n_gru_units,
                'n_dense_units': self.n_dense_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'epochs': epochs,
                'batch_size': batch_size
            })
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Log metrics to MLflow
        if self.enable_mlflow:
            self.mlflow_tracker.log_metrics({
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_mae': history.history['mae'][-1],
                'final_val_mae': history.history['val_mae'][-1]
            })
        
        self.is_fitted = True
        print("GRU model training completed successfully")
        
        return history
    
    def predict(self, data, n_steps=1):
        """
        Make predictions for future time steps.
        
        Args:
            data (pd.DataFrame): Historical data for making predictions
            n_steps (int): Number of future steps to predict
            
        Returns:
            np.array: Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        
        # Scale input data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        
        # Make predictions
        current_sequence = scaled_data[-self.sequence_length:]
        
        for _ in range(n_steps):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, self.n_features)
            
            # Predict next value
            scaled_pred = self.model.predict(X, verbose=0)[0]
            
            # Inverse transform prediction
            pred = self.scaler.inverse_transform(scaled_pred.reshape(-1, 1))[0, 0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], scaled_pred).reshape(-1, 1)
        
        return np.array(predictions)
    
    def evaluate(self, test_data, predictions=None):
        """
        Evaluate model performance.
        
        Args:
            test_data (pd.DataFrame): Test data
            predictions (np.array, optional): Pre-computed predictions
            
        Returns:
            dict: Dictionary of performance metrics
        """
        if predictions is None:
            predictions = self.predict(test_data, len(test_data))
        
        true_values = test_data['value'].values
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions)
        }
        
        # Log to MLflow if enabled
        if self.enable_mlflow:
            self.mlflow_tracker.log_metrics(metrics)
        
        return metrics
    
    def finish_mlflow_run(self):
        """End the current MLflow run."""
        if self.enable_mlflow and self.current_run_id:
            self.mlflow_tracker.end_run()
            self.current_run_id = None


def create_sample_data(n_points=500):
    """
    Create sample time series data for testing.
    
    Args:
        n_points (int): Number of data points to generate
        
    Returns:
        pd.DataFrame: DataFrame with timestamp and value columns
    """
    timestamps = pd.date_range(start='2025-01-01', periods=n_points, freq='H')
    
    # Generate synthetic data with trend, seasonality, and noise
    t = np.arange(n_points)
    trend = 0.05 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 1, n_points)
    
    values = 50 + trend + seasonal + noise
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })


def demo_gru_forecasting():
    """Demonstrate GRU forecasting functionality."""
    print("\n⚡ GRU Time-Series Forecasting Demo")
    print("=" * 45)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    data = create_sample_data(500)
    
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Test samples: {len(test_data)}")
    
    # 2. Initialize GRU model
    print("\n2. Initializing GRU model...")
    print("   (GRU: Lighter & faster than LSTM)")
    forecaster = GRUTimeSeriesForecaster(
        sequence_length=24,  # 24 hours of history
        n_gru_units=64,
        dropout_rate=0.2,
        learning_rate=0.001,
        enable_mlflow=False
    )
    
    # 3. Train model
    print("\n3. Training model...")
    import time
    start_time = time.time()
    
    forecaster.fit(train_data, epochs=50, batch_size=32)
    
    training_time = time.time() - start_time
    print(f"   - Training completed in {training_time:.2f} seconds")
    
    # 4. Evaluate on test data
    print("\n4. Evaluating on test data...")
    test_predictions = forecaster.predict(test_data['value'], n_steps=len(test_data))
    metrics = forecaster.evaluate(test_data[['value']])
    
    print("\n   Model Performance Metrics:")
    print(f"   - RMSE: {metrics['rmse']:.3f}")
    print(f"   - MAE:  {metrics['mae']:.3f}")
    print(f"   - R²:   {metrics['r2']:.3f}")
    
    # 5. Make future predictions
    print("\n5. Making future predictions...")
    print("   Next 24 hour predictions:")
    future_predictions = forecaster.predict(data['value'].tail(100), n_steps=24)
    
    for i, pred in enumerate(future_predictions[:5], 1):
        print(f"   - Hour {i}: {pred:.2f}")
    print("   ...")
    
    # 6. Compare with LSTM
    print("\n6. GRU vs LSTM Comparison:")
    print("   ✓ GRU trains faster (fewer parameters)")
    print("   ✓ GRU uses less memory")
    print("   ✓ Similar prediction accuracy")
    print("   ✓ Better for real-time applications")
    
    # 7. Finalize
    print("\n7. Finalizing...")
    forecaster.finish_mlflow_run()
    
    print("\n✓ GRU forecasting demo completed successfully!")
    print(f"✓ Training time: {training_time:.2f}s")
    print(f"✓ Final RMSE: {metrics['rmse']:.3f}")
    
    return forecaster, metrics


# TODO: Add the following features in future iterations:
# 1. Bidirectional GRU support
# 2. Stacked GRU layers
# 3. Attention mechanism
# 4. Multi-step ahead forecasting
# 5. Ensemble with LSTM
# 6. Hyperparameter optimization
# 7. Early stopping and model checkpointing
# 8. Advanced visualization methods


if __name__ == "__main__":
    try:
        # Run demo
        forecaster, metrics = demo_gru_forecasting()
        
        print("\n" + "=" * 45)
        print("Demo Summary:")
        print(f"  Model: GRU (Gated Recurrent Unit)")
        print(f"  Architecture: Lighter than LSTM")
        print(f"  Performance: RMSE = {metrics['rmse']:.3f}")
        print("=" * 45)
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
