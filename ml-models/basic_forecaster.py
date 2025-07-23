"""
Basic Time-Series Forecasting Model
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Simple implementation of time-series forecasting using scikit-learn linear regression.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import os

# Add data-collection path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))


class BasicTimeSeriesForecaster:
    """
    Basic time-series forecasting using linear regression with lag features.
    
    This is a simple implementation that creates lag features from the time series
    and uses linear regression to predict future values.
    """
    
    def __init__(self, n_lags=5):
        """
        Initialize the basic forecaster.
        
        Args:
            n_lags (int): Number of lag features to use for prediction
        """
        self.n_lags = n_lags
        self.model = LinearRegression()
        self.is_fitted = False
        self.feature_names = []
        self.target_col = None
        
    def create_lag_features(self, data, target_col):
        """
        Create lag features for time series data.
        
        Args:
            data (pd.DataFrame): Input time series data
            target_col (str): Name of the target column
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        df = data.copy()
        
        # Create lag features
        for i in range(1, self.n_lags + 1):
            df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
        
        # Store feature names
        self.feature_names = [f'{target_col}_lag_{i}' for i in range(1, self.n_lags + 1)]
        
        # Drop rows with NaN values (due to shifting)
        df = df.dropna()
        
        return df
    
    def fit(self, data, target_col='value'):
        """
        Fit the forecasting model.
        
        Args:
            data (pd.DataFrame): Training data with time series
            target_col (str): Name of the column to forecast
        """
        self.target_col = target_col
        
        # Create lag features
        df_with_lags = self.create_lag_features(data, target_col)
        
        if len(df_with_lags) == 0:
            raise ValueError("Not enough data to create lag features")
        
        # Prepare features and target
        X = df_with_lags[self.feature_names]
        y = df_with_lags[target_col]
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        print(f"Model fitted with {len(X)} samples and {len(self.feature_names)} lag features")
        
    def predict(self, data, n_steps=1):
        """
        Make predictions for future time steps.
        
        Args:
            data (pd.DataFrame): Recent data to base predictions on
            n_steps (int): Number of future steps to predict
            
        Returns:
            np.array: Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        
        # Get the last n_lags values
        last_values = data[self.target_col].tail(self.n_lags).values
        
        if len(last_values) < self.n_lags:
            raise ValueError(f"Need at least {self.n_lags} data points for prediction")
        
        predictions = []
        
        for _ in range(n_steps):
            # Create feature vector from last values (reverse order for lag features)
            X_pred = last_values[::-1].reshape(1, -1)
            
            # Make prediction
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            
            # Update last_values for next prediction (rolling window)
            last_values = np.append(last_values[1:], pred)
        
        return np.array(predictions)
    
    def evaluate(self, data):
        """
        Evaluate model performance on test data.
        
        Args:
            data (pd.DataFrame): Test data
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Create lag features for test data
        test_with_lags = self.create_lag_features(data, self.target_col)
        
        if len(test_with_lags) == 0:
            raise ValueError("Not enough test data to create lag features")
        
        X_test = test_with_lags[self.feature_names]
        y_test = test_with_lags[self.target_col]
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_samples': len(y_test)
        }
    
    def plot_predictions(self, data, n_steps=10, figsize=(12, 6)):
        """
        Plot historical data with predictions.
        
        Args:
            data (pd.DataFrame): Historical data
            n_steps (int): Number of future steps to predict
            figsize (tuple): Figure size for the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Make predictions
        predictions = self.predict(data, n_steps)
        
        # Create time index for predictions
        if 'timestamp' in data.columns:
            last_time = pd.to_datetime(data['timestamp'].iloc[-1])
            pred_times = pd.date_range(start=last_time, periods=n_steps+1, freq='H')[1:]
        else:
            pred_times = range(len(data), len(data) + n_steps)
        
        # Plot
        plt.figure(figsize=figsize)
        
        # Plot historical data
        if 'timestamp' in data.columns:
            plt.plot(pd.to_datetime(data['timestamp']), data[self.target_col], 
                    label='Historical Data', color='blue')
            plt.plot(pred_times, predictions, label='Predictions', 
                    color='red', linestyle='--', marker='o')
        else:
            plt.plot(data[self.target_col], label='Historical Data', color='blue')
            plt.plot(pred_times, predictions, label='Predictions', 
                    color='red', linestyle='--', marker='o')
        
        plt.title('Time Series Forecasting')
        plt.xlabel('Time')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_sample_data(n_points=100):
    """
    Create sample time series data for testing.
    
    Args:
        n_points (int): Number of data points to generate
        
    Returns:
        pd.DataFrame: Sample time series data
    """
    np.random.seed(42)
    
    # Create timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='H')
    
    # Generate synthetic temperature data with daily pattern
    time_hours = np.arange(n_points)
    daily_pattern = 20 + 5 * np.sin(time_hours * 2 * np.pi / 24)  # Daily cycle
    noise = np.random.normal(0, 1, n_points)  # Random noise
    
    temperature = daily_pattern + noise
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature
    })


def demo_basic_forecasting():
    """Demonstrate basic forecasting functionality."""
    print("=" * 60)
    print("BASIC TIME-SERIES FORECASTING DEMO")
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
    
    # Create and train forecaster
    print("3. Training forecaster...")
    forecaster = BasicTimeSeriesForecaster(n_lags=6)
    forecaster.fit(train_data, target_col='temperature')
    
    # Evaluate on test data
    print("4. Evaluating on test data...")
    metrics = forecaster.evaluate(test_data)
    
    print("   Model Performance Metrics:")
    print(f"   - RMSE: {metrics['rmse']:.3f}")
    print(f"   - MAE:  {metrics['mae']:.3f}")
    print(f"   - R²:   {metrics['r2']:.3f}")
    
    # Make future predictions
    print("5. Making future predictions...")
    future_predictions = forecaster.predict(train_data, n_steps=5)
    
    print("   Next 5 hour predictions:")
    for i, pred in enumerate(future_predictions, 1):
        print(f"   - Hour {i}: {pred:.2f}°C")
    
    # Optional: Plot results (if matplotlib is available)
    try:
        print("6. Plotting results...")
        forecaster.plot_predictions(train_data, n_steps=10)
    except Exception as e:
        print(f"   Plotting skipped: {e}")
    
    return forecaster, data, metrics


if __name__ == "__main__":
    try:
        forecaster, data, metrics = demo_basic_forecasting()
        print("\n✓ Basic forecasting demo completed successfully!")
        print(f"✓ Final RMSE: {metrics['rmse']:.3f}")
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
