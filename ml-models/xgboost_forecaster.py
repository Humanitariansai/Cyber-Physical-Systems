"""
XGBoost Time Series Forecasting Model
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Advanced time series forecasting using XGBoost with feature engineering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# MLflow imports for experiment tracking
try:
    from mlflow_tracking import ExperimentTracker, create_experiment_config
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Experiment tracking disabled.")


class XGBoostTimeSeriesForecaster:
    """
    XGBoost-based time series forecaster with advanced feature engineering.
    
    Features:
    - Lag features (1-12 lags)
    - Rolling statistics (mean, std, min, max)
    - Trend and seasonality features
    - Difference features
    - Target encoding for time-based features
    """
    
    def __init__(self, n_lags=12, rolling_windows=[3, 7, 12], 
                 xgb_params=None, random_state=42, enable_mlflow=True):
        """
        Initialize XGBoost forecaster.
        
        Parameters:
        -----------
        n_lags : int, default=12
            Number of lag features to create
        rolling_windows : list, default=[3, 7, 12]
            Window sizes for rolling statistics
        xgb_params : dict, optional
            XGBoost parameters. If None, uses default parameters.
        random_state : int, default=42
            Random state for reproducibility
        enable_mlflow : bool, default=True
            Whether to enable MLflow experiment tracking
        """
        self.n_lags = n_lags
        self.rolling_windows = rolling_windows
        self.random_state = random_state
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.mlflow_tracker = None
        self.current_run_id = None
        
        # Default XGBoost parameters optimized for time series
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': random_state,
                'n_jobs': -1
            }
        else:
            self.xgb_params = xgb_params.copy()
            self.xgb_params['random_state'] = random_state
        
        self.model = None
        self.is_fitted = False
        self.target_col = None
        self.feature_names = []
        self.target_mean = 0
        self.target_std = 1
        
        # Initialize MLflow tracking if enabled
        if self.enable_mlflow:
            try:
                self.mlflow_tracker = ExperimentTracker("xgboost-forecasting")
                print("MLflow experiment tracking enabled for XGBoost")
            except Exception as e:
                print(f"Failed to initialize MLflow tracking: {e}")
                self.enable_mlflow = False
        
    def _create_features(self, data, target_col, is_training=True):
        """
        Create comprehensive feature set for time series forecasting.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_col : str
            Target column name
        is_training : bool, default=True
            Whether this is training data (affects feature creation)
            
        Returns:
        --------
        pd.DataFrame : Feature matrix
        """
        df = data.copy()
        features_df = pd.DataFrame(index=df.index)
        
        # Target variable
        target = df[target_col].values
        
        # 1. Lag features
        for lag in range(1, self.n_lags + 1):
            features_df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # 2. Rolling statistics
        for window in self.rolling_windows:
            features_df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            features_df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            features_df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            features_df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        # 3. Difference features
        features_df['diff_1'] = df[target_col].diff(1)
        features_df['diff_2'] = df[target_col].diff(2)
        features_df['diff_seasonal'] = df[target_col].diff(12)  # Assuming monthly seasonality
        
        # 4. Time-based features (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            features_df['hour'] = df.index.hour
            features_df['day_of_week'] = df.index.dayofweek
            features_df['day_of_month'] = df.index.day
            features_df['month'] = df.index.month
            features_df['quarter'] = df.index.quarter
            features_df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        else:
            # Create synthetic time features based on position
            n = len(df)
            features_df['time_idx'] = np.arange(n)
            features_df['time_sin'] = np.sin(2 * np.pi * np.arange(n) / 24)  # Daily pattern
            features_df['time_cos'] = np.cos(2 * np.pi * np.arange(n) / 24)
            features_df['time_sin_weekly'] = np.sin(2 * np.pi * np.arange(n) / (24 * 7))  # Weekly pattern
            features_df['time_cos_weekly'] = np.cos(2 * np.pi * np.arange(n) / (24 * 7))
        
        # 5. Interaction features
        if len(self.rolling_windows) >= 2:
            w1, w2 = self.rolling_windows[0], self.rolling_windows[1]
            features_df[f'rolling_ratio_{w1}_{w2}'] = (
                features_df[f'rolling_mean_{w1}'] / (features_df[f'rolling_mean_{w2}'] + 1e-8)
            )
        
        # 6. Recent trend features
        for window in [3, 7]:
            if window <= len(df):
                recent_values = df[target_col].rolling(window=window)
                features_df[f'trend_{window}'] = recent_values.apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
                )
        
        # 7. Volatility features
        for window in self.rolling_windows:
            features_df[f'volatility_{window}'] = (
                features_df[f'rolling_std_{window}'] / (features_df[f'rolling_mean_{window}'] + 1e-8)
            )
        
        # Drop rows with NaN values (mainly from initial lags and rolling features)
        features_df = features_df.dropna()
        
        return features_df
    
    def fit(self, data, target_col, run_name=None):
        """
        Fit the XGBoost model on the training data with MLflow tracking.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        target_col : str
            Name of the target column
        run_name : str, optional
            Name for the MLflow run
        """
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        
        self.target_col = target_col
        
        # Start MLflow run if enabled
        if self.enable_mlflow and self.mlflow_tracker:
            if run_name is None:
                run_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_run_id = self.mlflow_tracker.start_run(
                run_name=run_name,
                model_type="xgboost",
                tags={"algorithm": "xgboost", "task": "time_series_forecasting"}
            )
            
            # Log model parameters
            params = {
                "n_lags": self.n_lags,
                "rolling_windows": str(self.rolling_windows),
                "target_column": target_col,
                "data_points": len(data),
                **self.xgb_params
            }
            self.mlflow_tracker.log_parameters(params)
            
            # Log dataset info
            self.mlflow_tracker.log_dataset_info(data, "training_data")
        
        # Store target statistics for normalization
        self.target_mean = data[target_col].mean()
        self.target_std = data[target_col].std()
        
        # Create features
        features_df = self._create_features(data, target_col, is_training=True)
        
        if features_df.empty:
            raise ValueError("No features could be created. Check your data and parameters.")
        
        # Get corresponding target values (aligned with features after dropna)
        target_values = data.loc[features_df.index, target_col].values
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        # Log feature information to MLflow
        if self.enable_mlflow and self.mlflow_tracker:
            feature_info = {
                "num_features": len(self.feature_names),
                "feature_names": self.feature_names[:10],  # Log first 10 feature names
                "total_features": len(self.feature_names)
            }
            self.mlflow_tracker.log_parameters(feature_info)
        
        # Create and fit XGBoost model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(features_df.values, target_values)
        
        # Log model artifact if MLflow is enabled
        if self.enable_mlflow and self.mlflow_tracker:
            try:
                # Create sample input for model signature
                sample_input = features_df.head(1)
                self.mlflow_tracker.log_model(
                    self.model,
                    "xgboost",
                    input_example=sample_input.values
                )
            except Exception as e:
                print(f"Failed to log XGBoost model to MLflow: {e}")
        
        
        self.is_fitted = True
        print(f"XGBoost model fitted with {len(self.feature_names)} features on {len(features_df)} samples.")
        
        # Print feature importance
        importance = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Most Important Features:")
        for i, (feature, imp) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feature:<25} {imp:.4f}")
    
    def predict(self, n_steps=1, last_known_data=None):
        """
        Make multi-step predictions.
        
        Parameters:
        -----------
        n_steps : int, default=1
            Number of steps to predict
        last_known_data : pd.DataFrame, optional
            Recent data to use for prediction. If None, uses training data.
            
        Returns:
        --------
        np.array : Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        if last_known_data is None:
            raise ValueError("last_known_data must be provided for predictions.")
        
        predictions = []
        current_data = last_known_data.copy()
        
        for step in range(n_steps):
            # Create features for the current state
            features_df = self._create_features(current_data, self.target_col, is_training=False)
            
            if features_df.empty:
                # If we can't create features, use the mean as prediction
                pred = self.target_mean
            else:
                # Get the last row of features (most recent)
                feature_vector = features_df.iloc[-1:].values
                
                # Handle missing features by filling with zeros
                if feature_vector.shape[1] < len(self.feature_names):
                    missing_cols = len(self.feature_names) - feature_vector.shape[1]
                    feature_vector = np.column_stack([
                        feature_vector, 
                        np.zeros((1, missing_cols))
                    ])
                elif feature_vector.shape[1] > len(self.feature_names):
                    feature_vector = feature_vector[:, :len(self.feature_names)]
                
                # Make prediction
                pred = self.model.predict(feature_vector)[0]
            
            predictions.append(pred)
            
            # Update current_data with the prediction for next iteration
            next_idx = current_data.index[-1] + pd.Timedelta(hours=1) if isinstance(current_data.index, pd.DatetimeIndex) else len(current_data)
            new_row = pd.DataFrame({self.target_col: [pred]}, 
                                 index=[next_idx] if isinstance(next_idx, (pd.Timestamp, int)) else [len(current_data)])
            current_data = pd.concat([current_data, new_row])
            
            # Keep only recent data to avoid memory issues
            if len(current_data) > 100:
                current_data = current_data.tail(100)
        
        return np.array(predictions)
    
    def evaluate(self, data, predictions=None, log_to_mlflow=True):
        """
        Evaluate the model performance with MLflow logging.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Test data
        predictions : np.array, optional
            Pre-computed predictions. If None, will generate predictions.
        log_to_mlflow : bool, default=True
            Whether to log metrics to MLflow
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")
        
        if predictions is None:
            # Generate in-sample predictions for evaluation
            features_df = self._create_features(data, self.target_col, is_training=False)
            if features_df.empty:
                raise ValueError("Cannot create features for evaluation.")
            
            # Ensure feature alignment
            feature_matrix = features_df.values
            if feature_matrix.shape[1] != len(self.feature_names):
                # Handle feature mismatch
                if feature_matrix.shape[1] < len(self.feature_names):
                    missing_cols = len(self.feature_names) - feature_matrix.shape[1]
                    feature_matrix = np.column_stack([
                        feature_matrix, 
                        np.zeros((feature_matrix.shape[0], missing_cols))
                    ])
                else:
                    feature_matrix = feature_matrix[:, :len(self.feature_names)]
            
            predictions = self.model.predict(feature_matrix)
            actual = data.loc[features_df.index, self.target_col].values
        else:
            # Use provided predictions (assumed to be for the full data)
            actual = data[self.target_col].values[-len(predictions):]
        
        # Calculate metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)
        
        # Additional metrics
        mape = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'n_samples': int(len(actual)),
            'n_features': int(len(self.feature_names))
        }
        
        # Log metrics and prediction results to MLflow
        if log_to_mlflow and self.enable_mlflow and self.mlflow_tracker:
            try:
                self.mlflow_tracker.log_metrics(metrics)
                self.mlflow_tracker.log_prediction_results(actual, predictions, "test_")
                
                # Log feature importance if available
                if hasattr(self.model, 'feature_importances_'):
                    feature_importance = self.get_feature_importance(top_n=10)
                    # Convert to metrics for MLflow
                    importance_metrics = {}
                    for i, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance'])):
                        importance_metrics[f"feature_importance_{i+1}_{feature[:20]}"] = importance
                    self.mlflow_tracker.log_metrics(importance_metrics)
                    
            except Exception as e:
                print(f"Failed to log XGBoost metrics to MLflow: {e}")
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from the trained model.
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        importance = self.model.feature_importances_
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_df.head(top_n)
    
    def finish_mlflow_run(self):
        """End the current MLflow run."""
        if self.enable_mlflow and self.mlflow_tracker and self.current_run_id:
            try:
                self.mlflow_tracker.end_run()
                print(f"XGBoost MLflow run {self.current_run_id} completed")
                self.current_run_id = None
            except Exception as e:
                print(f"Error ending XGBoost MLflow run: {e}")
    
    def plot_feature_importance(self, top_n=15, figsize=(10, 8), save_path=None):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int, default=15
            Number of top features to plot
        figsize : tuple, default=(10, 8)
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} XGBoost Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def plot_predictions(self, data, predictions, title="XGBoost Predictions", 
                        figsize=(12, 6), save_path=None):
        """
        Plot predictions vs actual values.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Actual data
        predictions : np.array
            Predicted values
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot actual values
        plt.plot(data.index, data[self.target_col], 
                label='Actual', color='blue', alpha=0.7)
        
        # Plot predictions
        pred_index = data.index[-len(predictions):]
        plt.plot(pred_index, predictions, 
                label='Predicted', color='red', alpha=0.8, linestyle='--')
        
        plt.xlabel('Time')
        plt.ylabel(self.target_col.capitalize())
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, data, predictions, metrics, base_name="xgboost_results"):
        """
        Save model results including metrics, predictions, and plots.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Original data
        predictions : np.array
            Model predictions
        metrics : dict
            Evaluation metrics
        base_name : str, default="xgboost_results"
            Base name for saved files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = f"{results_dir}/{base_name}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'predicted': predictions,
            'actual': data[self.target_col].values[-len(predictions):]
        })
        pred_file = f"{results_dir}/{base_name}_predictions_{timestamp}.csv"
        predictions_df.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")
        
        # Save prediction plot
        plot_file = f"{results_dir}/{base_name}_plot_{timestamp}.png"
        self.plot_predictions(data, predictions, save_path=plot_file)
        
        # Save feature importance plot
        importance_file = f"{results_dir}/{base_name}_importance_{timestamp}.png"
        self.plot_feature_importance(save_path=importance_file)
        
        print(f"All results saved to {results_dir}/ directory")


def create_sample_data(n_points=168, noise_level=0.5, trend=0.01, seasonal_period=24):
    """
    Create sample temperature data with trend and seasonality.
    
    Parameters:
    -----------
    n_points : int, default=168
        Number of data points (default: 1 week of hourly data)
    noise_level : float, default=0.5
        Standard deviation of noise
    trend : float, default=0.01
        Linear trend coefficient
    seasonal_period : int, default=24
        Seasonal period (24 for daily pattern)
        
    Returns:
    --------
    pd.DataFrame : Sample data
    """
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
    
    # Base temperature (around 20°C)
    base_temp = 20.0
    
    # Add trend
    trend_component = trend * np.arange(n_points)
    
    # Add daily seasonality (temperature varies throughout the day)
    seasonal_component = 5 * np.sin(2 * np.pi * np.arange(n_points) / seasonal_period)
    
    # Add weekly pattern (weekends slightly different)
    weekly_component = 2 * np.sin(2 * np.pi * np.arange(n_points) / (seasonal_period * 7))
    
    # Add random noise
    noise = np.random.normal(0, noise_level, n_points)
    
    # Combine all components
    temperature = base_temp + trend_component + seasonal_component + weekly_component + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    }, index=dates)
    
    return df


def demo_xgboost_forecaster():
    """
    Demonstrate XGBoost forecaster with sample data.
    """
    print("=" * 60)
    print("XGBoost Time Series Forecaster Demo")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    data = create_sample_data(n_points=168)  # 1 week of hourly data
    print(f"Created {len(data)} data points")
    print(f"Temperature range: {data['temperature'].min():.2f}°C to {data['temperature'].max():.2f}°C")
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"\nTrain set: {len(train_data)} points")
    print(f"Test set: {len(test_data)} points")
    
    # Initialize and fit model
    print("\n2. Training XGBoost model...")
    forecaster = XGBoostTimeSeriesForecaster(
        n_lags=12,
        rolling_windows=[3, 6, 12],
        xgb_params={
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    )
    
    forecaster.fit(train_data, 'temperature')
    
    # Make predictions
    print("\n3. Making predictions...")
    n_pred_steps = len(test_data)
    predictions = forecaster.predict(n_steps=n_pred_steps, 
                                   last_known_data=train_data.tail(50))
    
    print(f"Generated {len(predictions)} predictions")
    
    # Evaluate model
    print("\n4. Evaluating model...")
    metrics = forecaster.evaluate(test_data, predictions)
    
    print("\nModel Performance:")
    print(f"RMSE: {metrics['rmse']:.3f}°C")
    print(f"MAE:  {metrics['mae']:.3f}°C")
    print(f"R²:   {metrics['r2']:.3f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Features used: {metrics['n_features']}")
    
    # Show feature importance
    print("\n5. Top feature importance:")
    importance_df = forecaster.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")
    
    # Save results
    print("\n6. Saving results...")
    forecaster.save_results(test_data, predictions, metrics)
    
    print("\n" + "=" * 60)
    print("XGBoost demo completed successfully!")
    print("=" * 60)
    
    return forecaster, metrics


if __name__ == "__main__":
    # Run demonstration
    forecaster, metrics = demo_xgboost_forecaster()
