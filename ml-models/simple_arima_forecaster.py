"""
Simple Moving Average Time-Series Forecasting Model
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Simple implementation of moving average-based time-series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import json

# Add data-collection path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))


class SimpleMovingAverageForecaster:
    """
    Simple Moving Average-based time-series forecasting.
    
    Uses different moving average strategies:
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Weighted Moving Average (WMA)
    """
    
    def __init__(self, window=5, method='sma', alpha=0.3, results_dir='results'):
        """
        Initialize the Moving Average forecaster.
        
        Args:
            window (int): Window size for moving average
            method (str): Method to use ('sma', 'ema', 'wma')
            alpha (float): Smoothing parameter for EMA (0 < alpha < 1)
            results_dir (str): Directory to store results
        """
        self.window = window
        self.method = method.lower()
        self.alpha = alpha
        self.is_fitted = False
        self.target_col = None
        self.results_dir = results_dir
        self.training_data = None
        self.last_values = None
        
        if self.method not in ['sma', 'ema', 'wma']:
            raise ValueError("Method must be 'sma', 'ema', or 'wma'")
        
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _simple_moving_average(self, data):
        """Calculate Simple Moving Average."""
        return data.rolling(window=self.window, min_periods=1).mean()
    
    def _exponential_moving_average(self, data):
        """Calculate Exponential Moving Average."""
        return data.ewm(alpha=self.alpha, adjust=False).mean()
    
    def _weighted_moving_average(self, data):
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, self.window + 1)
        weights = weights / weights.sum()
        
        def wma(x):
            if len(x) < self.window:
                return x.mean()
            return np.average(x[-self.window:], weights=weights)
        
        return data.rolling(window=self.window, min_periods=1).apply(wma, raw=True)
    
    def fit(self, data, target_col='value'):
        """
        Fit the moving average model.
        
        Args:
            data (pd.DataFrame): Training data with time series
            target_col (str): Name of the column to forecast
        """
        self.target_col = target_col
        self.training_data = data.copy()
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        series = data[target_col].dropna()
        
        if len(series) < self.window:
            raise ValueError(f"Need at least {self.window} data points for window size {self.window}")
        
        # Calculate moving averages
        if self.method == 'sma':
            self.fitted_values = self._simple_moving_average(series)
            print(f"Fitted Simple Moving Average with window={self.window}")
        elif self.method == 'ema':
            self.fitted_values = self._exponential_moving_average(series)
            print(f"Fitted Exponential Moving Average with alpha={self.alpha}")
        elif self.method == 'wma':
            self.fitted_values = self._weighted_moving_average(series)
            print(f"Fitted Weighted Moving Average with window={self.window}")
        
        # Store last values for prediction
        self.last_values = series.tail(self.window).values
        self.is_fitted = True
        
        print(f"Model fitted with {len(series)} observations")
    
    def predict(self, n_steps=1):
        """
        Make predictions for future time steps.
        
        Args:
            n_steps (int): Number of future steps to predict
            
        Returns:
            np.array: Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        current_values = self.last_values.copy()
        
        for _ in range(n_steps):
            if self.method == 'sma':
                # Simple Moving Average prediction
                pred = np.mean(current_values[-self.window:])
            elif self.method == 'ema':
                # Exponential Moving Average prediction (use last EMA value)
                pred = self.fitted_values.iloc[-1]
                # For subsequent predictions, use simple average
                if len(predictions) > 0:
                    pred = self.alpha * current_values[-1] + (1 - self.alpha) * pred
            elif self.method == 'wma':
                # Weighted Moving Average prediction
                weights = np.arange(1, self.window + 1)
                weights = weights / weights.sum()
                pred = np.average(current_values[-self.window:], weights=weights)
            
            predictions.append(pred)
            
            # Update current values for next prediction
            current_values = np.append(current_values[1:], pred)
        
        return np.array(predictions)
    
    def evaluate(self, test_data):
        """
        Evaluate model performance on test data.
        
        Args:
            test_data (pd.DataFrame): Test data
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        test_series = test_data[self.target_col].dropna()
        
        if len(test_series) == 0:
            raise ValueError("Test data is empty")
        
        # Use fitted values for evaluation
        min_len = min(len(self.fitted_values), len(test_series))
        
        if min_len == 0:
            raise ValueError("No overlapping data for evaluation")
        
        # Align the data
        y_true = test_series.iloc[:min_len].values
        y_pred = self.fitted_values.iloc[:min_len].values
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_samples': len(y_true),
            'method': self.method,
            'window': self.window if self.method != 'ema' else None,
            'alpha': self.alpha if self.method == 'ema' else None
        }
    
    def plot_predictions(self, n_steps=10, figsize=(12, 6), save_plot=True):
        """
        Plot historical data with predictions.
        
        Args:
            n_steps (int): Number of future steps to predict
            figsize (tuple): Figure size for the plot
            save_plot (bool): Whether to save the plot to file
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Make predictions
        predictions = self.predict(n_steps)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot historical data
        historical_data = self.training_data[self.target_col]
        plt.plot(historical_data.index, historical_data.values, 
                label='Historical Data', color='blue', linewidth=1)
        
        # Plot fitted values
        plt.plot(self.fitted_values.index, self.fitted_values.values, 
                label=f'Fitted {self.method.upper()}', color='green', alpha=0.7)
        
        # Plot predictions
        future_index = range(len(historical_data), len(historical_data) + n_steps)
        plt.plot(future_index, predictions, 
                label='Predictions', color='red', linestyle='--', marker='o')
        
        method_name = {
            'sma': f'Simple MA (window={self.window})',
            'ema': f'Exponential MA (α={self.alpha})',
            'wma': f'Weighted MA (window={self.window})'
        }
        
        plt.title(f'Time Series Forecasting - {method_name[self.method]}')
        plt.xlabel('Time')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(self.results_dir, f'ma_forecast_plot_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_file}")
        
        plt.show()
        return predictions
    
    def save_results(self, predictions, metrics):
        """
        Save forecasting results to files.
        
        Args:
            predictions (np.array): Array of predictions
            metrics (dict): Performance metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        pred_df = pd.DataFrame({
            'step': range(1, len(predictions) + 1),
            'prediction': predictions
        })
        
        pred_file = os.path.join(self.results_dir, f'ma_predictions_{timestamp}.csv')
        pred_df.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")
        
        # Save metrics
        metrics_file = os.path.join(self.results_dir, f'ma_metrics_{timestamp}.json')
        metrics_with_meta = {
            'timestamp': timestamp,
            'model_type': 'SimpleMovingAverageForecaster',
            'method': self.method,
            'window': self.window if self.method != 'ema' else None,
            'alpha': self.alpha if self.method == 'ema' else None,
            'target_column': self.target_col,
            'metrics': metrics
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
        
        return {
            'predictions_file': pred_file,
            'metrics_file': metrics_file,
            'timestamp': timestamp
        }


def create_sample_data(n_points=100):
    """
    Create sample time series data for testing moving average models.
    
    Args:
        n_points (int): Number of data points to generate
        
    Returns:
        pd.DataFrame: Sample time series data
    """
    np.random.seed(42)
    
    # Create timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='h')
    
    # Generate synthetic temperature data with trend and seasonality
    time_hours = np.arange(n_points)
    
    # Trend component
    trend = 0.01 * time_hours
    
    # Seasonal component (daily cycle)
    seasonal = 5 * np.sin(time_hours * 2 * np.pi / 24)
    
    # Base temperature + components + noise
    temperature = 20 + trend + seasonal + np.random.normal(0, 1, n_points)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature
    })


def demo_moving_average_forecasting():
    """Demonstrate Moving Average forecasting functionality."""
    print("=" * 60)
    print("SIMPLE MOVING AVERAGE TIME-SERIES FORECASTING DEMO")
    print("=" * 60)
    
    # Create sample data
    print("1. Generating sample temperature data...")
    data = create_sample_data(n_points=168)  # 1 week of hourly data
    print(f"   Generated {len(data)} data points")
    
    # Split into train and test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"2. Split data: {len(train_data)} train, {len(test_data)} test samples")
    
    # Test different moving average methods
    methods = [
        ('sma', {'window': 7}),
        ('ema', {'alpha': 0.3}),
        ('wma', {'window': 5})
    ]
    
    best_forecaster = None
    best_rmse = float('inf')
    best_method = None
    
    for method, params in methods:
        print(f"\n3. Training {method.upper()} forecaster...")
        try:
            if method == 'ema':
                forecaster = SimpleMovingAverageForecaster(
                    method=method, alpha=params['alpha'], results_dir='results'
                )
            else:
                forecaster = SimpleMovingAverageForecaster(
                    method=method, window=params['window'], results_dir='results'
                )
            
            forecaster.fit(train_data, target_col='temperature')
            
            # Evaluate on training data (in-sample)
            metrics = forecaster.evaluate(train_data)
            
            print(f"   {method.upper()} Performance Metrics:")
            print(f"   - RMSE: {metrics['rmse']:.3f}")
            print(f"   - MAE:  {metrics['mae']:.3f}")
            print(f"   - R²:   {metrics['r2']:.3f}")
            
            # Track best model
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_forecaster = forecaster
                best_method = method
                best_metrics = metrics
                
        except Exception as e:
            print(f"   ❌ {method.upper()} failed: {e}")
    
    if best_forecaster is None:
        print("\n❌ All forecasting methods failed!")
        return None, None, None
    
    print(f"\n🏆 Best model: {best_method.upper()} (RMSE: {best_rmse:.3f})")
    
    # Make future predictions with best model
    print("4. Making future predictions...")
    predictions = best_forecaster.predict(n_steps=5)
    
    print("   Next 5 hour predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"   - Hour {i}: {pred:.2f}°C")
    
    # Save results
    print("5. Saving results...")
    saved_files = best_forecaster.save_results(predictions, best_metrics)
    
    # Plot results
    print("6. Plotting and saving visualization...")
    best_forecaster.plot_predictions(n_steps=10, save_plot=True)
    
    print("\n📁 Files saved:")
    for key, filepath in saved_files.items():
        if key != 'timestamp':
            print(f"   - {key}: {filepath}")
    
    return best_forecaster, data, best_metrics


if __name__ == "__main__":
    try:
        forecaster, data, metrics = demo_moving_average_forecasting()
        if forecaster is not None:
            print("\n✓ Moving Average forecasting demo completed successfully!")
            print(f"✓ Final RMSE: {metrics['rmse']:.3f}")
        else:
            print("\n❌ Moving Average forecasting demo failed!")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
