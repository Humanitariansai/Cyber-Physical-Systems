"""
LSTM Time-Series Forecasting Model
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Implementation of time-series forecasting using LSTM (Long Short-Term Memory) neural networks.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import json

# Add data-collection path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))

# MLflow imports for experiment tracking
try:
    from mlflow_tracking import ExperimentTracker, create_experiment_config
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Experiment tracking disabled.")


class LSTMTimeSeriesForecaster:
    """
    LSTM-based time-series forecasting model.
    
    This implementation uses a Long Short-Term Memory (LSTM) neural network
    for time series prediction with support for multiple features and
    MLflow experiment tracking.
    """
    
    def __init__(self, 
                 sequence_length=10,
                 n_features=1,
                 n_lstm_units=50,
                 n_dense_units=1,
                 dropout_rate=0.2,
                 learning_rate=0.001,
                 results_dir='results',
                 enable_mlflow=True):
        """
        Initialize the LSTM forecaster.
        
        Args:
            sequence_length (int): Number of time steps to look back
            n_features (int): Number of input features
            n_lstm_units (int): Number of LSTM units in the layer
            n_dense_units (int): Number of units in the dense layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the Adam optimizer
            results_dir (str): Directory to store results
            enable_mlflow (bool): Whether to enable MLflow experiment tracking
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_lstm_units = n_lstm_units
        self.n_dense_units = n_dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.results_dir = results_dir
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.mlflow_tracker = None
        self.current_run_id = None
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize MLflow tracking if enabled
        if self.enable_mlflow:
            try:
                self.mlflow_tracker = ExperimentTracker("lstm-forecasting")
                print("MLflow experiment tracking enabled")
            except Exception as e:
                print(f"Failed to initialize MLflow tracking: {e}")
                self.enable_mlflow = False
    
    def _build_model(self):
        """Build the LSTM model architecture."""
        model = Sequential([
            LSTM(units=self.n_lstm_units, 
                 activation='relu',
                 input_shape=(self.sequence_length, self.n_features),
                 return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(units=self.n_dense_units)
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def _prepare_sequences(self, data):
        """Prepare input sequences for the LSTM model."""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def fit(self, data, target_col='value', epochs=100, batch_size=32, validation_split=0.2, run_name=None):
        """
        Fit the LSTM model to the training data.
        
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
            self.mlflow_tracker.start_run(run_name or "lstm_training")
            self.current_run_id = self.mlflow_tracker.current_run.info.run_id
        
        # Extract and scale the target variable
        series = data[target_col].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(series)
        
        # Prepare sequences
        X, y = self._prepare_sequences(scaled_data)
        
        # Build and train the model
        self.model = self._build_model()
        
        # Log parameters if MLflow is enabled
        if self.enable_mlflow:
            self.mlflow_tracker.log_params({
                'sequence_length': self.sequence_length,
                'n_lstm_units': self.n_lstm_units,
                'n_dense_units': self.n_dense_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'epochs': epochs,
                'batch_size': batch_size
            })
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Log metrics if MLflow is enabled
        if self.enable_mlflow:
            self.mlflow_tracker.log_metrics({
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_mae': history.history['mae'][-1],
                'final_val_mae': history.history['val_mae'][-1]
            })
        
        self.is_fitted = True
        print("Model training completed successfully")
    
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
        current_sequence = data[-self.sequence_length:].values.reshape(-1, 1)
        current_sequence = self.scaler.transform(current_sequence)
        
        for _ in range(n_steps):
            # Reshape sequence for prediction
            X = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Make prediction
            scaled_pred = self.model.predict(X, verbose=0)[0]
            
            # Inverse transform prediction
            pred = self.scaler.inverse_transform(scaled_pred.reshape(-1, 1))[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence, scaled_pred)
            current_sequence = current_sequence[1:]
        
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
            predictions = self.predict(test_data)
        
        true_values = test_data[-len(predictions):].values
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions)
        }
        
        # Log metrics if MLflow is enabled
        if self.enable_mlflow:
            self.mlflow_tracker.log_metrics(metrics)
        
        return metrics
    
    def plot_predictions(self, data, predictions, figsize=(12, 6), save_plot=True):
        """
        Plot actual vs predicted values.
        
        Args:
            data (pd.DataFrame): Historical data
            predictions (np.array): Model predictions
            figsize (tuple): Figure size
            save_plot (bool): Whether to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot actual values
        plt.plot(data.index[-len(predictions)-self.sequence_length:],
                data.values[-len(predictions)-self.sequence_length:],
                label='Actual', color='blue')
        
        # Plot predictions
        pred_index = data.index[-len(predictions):]
        plt.plot(pred_index, predictions, label='Predicted',
                color='red', linestyle='--')
        
        plt.title('LSTM Time Series Predictions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f'lstm_predictions_{timestamp}.png')
            plt.savefig(plot_path)
            print(f"Plot saved to: {plot_path}")
            
            if self.enable_mlflow:
                self.mlflow_tracker.log_artifact(plot_path)
        
        plt.close()
    
    def save_results(self, data, predictions, metrics):
        """
        Save forecasting results to files.
        
        Args:
            data (pd.DataFrame): Historical data
            predictions (np.array): Model predictions
            metrics (dict): Performance metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        pred_df = pd.DataFrame({
            'step': range(1, len(predictions) + 1),
            'prediction': predictions
        })
        pred_file = os.path.join(self.results_dir, f'lstm_predictions_{timestamp}.csv')
        pred_df.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")
        
        # Save metrics
        metrics_with_meta = {
            'timestamp': timestamp,
            'model_type': 'LSTMTimeSeriesForecaster',
            'sequence_length': self.sequence_length,
            'n_lstm_units': self.n_lstm_units,
            'metrics': metrics
        }
        
        metrics_file = os.path.join(self.results_dir, f'lstm_metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_with_meta, f, indent=4)
        print(f"Metrics saved to: {metrics_file}")
        
        # Log artifacts to MLflow if enabled
        if self.enable_mlflow:
            self.mlflow_tracker.log_artifact(pred_file)
            self.mlflow_tracker.log_artifact(metrics_file)
    
    def finish_mlflow_run(self):
        """End the current MLflow run."""
        if self.enable_mlflow and self.current_run_id:
            self.mlflow_tracker.end_run()
            self.current_run_id = None


def create_sample_data(n_points=100):
    """
    Create sample time series data for testing.
    
    Args:
        n_points (int): Number of data points to generate
        
    Returns:
        pd.DataFrame: DataFrame with timestamp and value columns
    """
    timestamps = pd.date_range(start='2025-01-01', periods=n_points, freq='H')
    
    # Generate synthetic time series with trend and seasonality
    t = np.arange(n_points)
    trend = 0.1 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 24)  # Daily seasonality
    noise = np.random.normal(0, 1, n_points)
    
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })


def demo_lstm_forecasting():
    """Demonstrate LSTM forecasting functionality."""
    print("\nLSTM Time-Series Forecasting Demo")
    print("=================================")
    
    # Create sample data
    print("1. Generating sample data...")
    data = create_sample_data(n_points=1000)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Initialize and train model
    print("\n2. Initializing LSTM model...")
    forecaster = LSTMTimeSeriesForecaster(
        sequence_length=24,  # Look back 24 hours
        n_lstm_units=50,
        dropout_rate=0.2
    )
    
    print("\n3. Training model...")
    forecaster.fit(train_data, 'value', epochs=50, batch_size=32)
    
    # Evaluate on test data
    print("\n4. Evaluating on test data...")
    test_predictions = forecaster.predict(test_data['value'], n_steps=len(test_data))
    metrics = forecaster.evaluate(test_data['value'], test_predictions)
    
    print("\nModel Performance Metrics:")
    print(f"   - RMSE: {metrics['rmse']:.3f}")
    print(f"   - MAE:  {metrics['mae']:.3f}")
    print(f"   - R²:   {metrics['r2']:.3f}")
    
    # Make future predictions
    print("\n5. Making future predictions...")
    future_predictions = forecaster.predict(train_data['value'], n_steps=24)
    
    print("\nNext 24 hour predictions:")
    for i, pred in enumerate(future_predictions[:5], 1):
        print(f"   - Hour {i}: {pred:.2f}")
    print("   ...")
    
    # Save results
    print("\n6. Saving results...")
    forecaster.save_results(train_data, future_predictions, metrics)
    
    # Plot results
    print("\n7. Plotting predictions...")
    forecaster.plot_predictions(train_data, future_predictions)
    
    # Finish MLflow run
    print("\n8. Finalizing MLflow experiment tracking...")
    forecaster.finish_mlflow_run()
    
    return forecaster, data, metrics


if __name__ == "__main__":
    try:
        forecaster, data, metrics = demo_lstm_forecasting()
        print("\n✓ LSTM forecasting demo completed successfully!")
        print(f"✓ Final RMSE: {metrics['rmse']:.3f}")
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()