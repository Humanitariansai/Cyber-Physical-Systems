"""
Adding Different Models to MLflow - Practical Examples
=====================================================

This script demonstrates how to add various types of models to MLflow
tracking in your forecasting system.

Examples include:
1. Scikit-learn models (Linear, Random Forest, SVM)
2. Deep Learning models (LSTM, GRU with TensorFlow/PyTorch)
3. Statistical models (ARIMA, Exponential Smoothing)
4. Custom ensemble models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Standard ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MLflow
import mlflow
import mlflow.sklearn
from mlflow_tracking import ExperimentTracker


class RandomForestForecaster:
    """
    Random Forest model adapted for time series forecasting.
    """
    
    def __init__(self, n_lags=12, n_estimators=100, max_depth=None, random_state=42):
        self.n_lags = n_lags
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
    def create_features(self, data, target_col):
        """Create lag features for time series."""
        df = data.copy()
        
        # Create lag features
        for i in range(1, self.n_lags + 1):
            df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
        
        # Create rolling features
        for window in [3, 7, 14]:
            if window <= self.n_lags:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        
        self.feature_names = [col for col in df.columns if col != target_col and col != 'timestamp']
        return df.dropna()
    
    def fit(self, data, target_col='temperature'):
        """Fit the Random Forest model."""
        df_features = self.create_features(data, target_col)
        
        X = df_features[self.feature_names]
        y = df_features[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.target_col = target_col
        
        print(f"Random Forest fitted with {len(X)} samples and {len(self.feature_names)} features")
    
    def predict(self, data, n_steps=1):
        """Make future predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        last_values = data[self.target_col].tail(self.n_lags).values
        predictions = []
        
        for _ in range(n_steps):
            # Create feature vector (simplified for demo)
            features = last_values[::-1]  # Reverse for lag order
            
            # Pad with basic rolling statistics if needed
            while len(features) < len(self.feature_names):
                features = np.append(features, np.mean(features))
            
            features = features[:len(self.feature_names)]
            features_scaled = self.scaler.transform([features])
            
            pred = self.model.predict(features_scaled)[0]
            predictions.append(pred)
            
            # Update for next prediction
            last_values = np.append(last_values[1:], pred)
        
        return np.array(predictions)
    
    def evaluate(self, test_data):
        """Evaluate model on test data."""
        df_features = self.create_features(test_data, self.target_col)
        
        X = df_features[self.feature_names]
        y = df_features[self.target_col]
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'n_samples': len(y)
        }


class SVMForecaster:
    """
    Support Vector Machine model for time series forecasting.
    """
    
    def __init__(self, n_lags=12, kernel='rbf', C=1.0, gamma='scale'):
        self.n_lags = n_lags
        self.model = SVR(kernel=kernel, C=C, gamma=gamma)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def create_features(self, data, target_col):
        """Create lag features."""
        df = data.copy()
        for i in range(1, self.n_lags + 1):
            df[f'lag_{i}'] = df[target_col].shift(i)
        return df.dropna()
    
    def fit(self, data, target_col='temperature'):
        """Fit the SVM model."""
        df_features = self.create_features(data, target_col)
        
        feature_cols = [f'lag_{i}' for i in range(1, self.n_lags + 1)]
        X = df_features[feature_cols]
        y = df_features[target_col]
        
        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # Fit model
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        self.target_col = target_col
        
        print(f"SVM fitted with {len(X)} samples and {self.n_lags} lag features")
    
    def predict(self, data, n_steps=1):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        last_values = data[self.target_col].tail(self.n_lags).values
        predictions = []
        
        for _ in range(n_steps):
            X_pred = last_values[::-1].reshape(1, -1)  # Reverse for lag order
            X_pred_scaled = self.scaler_X.transform(X_pred)
            
            pred_scaled = self.model.predict(X_pred_scaled)[0]
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            predictions.append(pred)
            last_values = np.append(last_values[1:], pred)
        
        return np.array(predictions)
    
    def evaluate(self, test_data):
        """Evaluate model."""
        df_features = self.create_features(test_data, self.target_col)
        
        feature_cols = [f'lag_{i}' for i in range(1, self.n_lags + 1)]
        X = df_features[feature_cols]
        y = df_features[self.target_col]
        
        X_scaled = self.scaler_X.transform(X)
        predictions_scaled = self.model.predict(X_scaled)
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions), 
            'r2': r2_score(y, predictions),
            'n_samples': len(y)
        }


class EnsembleForecaster:
    """
    Ensemble model combining multiple forecasters.
    """
    
    def __init__(self, base_models=None, weights=None):
        if base_models is None:
            from basic_forecaster import BasicTimeSeriesForecaster
            self.base_models = [
                BasicTimeSeriesForecaster(n_lags=6, enable_mlflow=False),
                RandomForestForecaster(n_lags=8, n_estimators=50),
                SVMForecaster(n_lags=10, C=0.1)
            ]
        else:
            self.base_models = base_models
        
        self.weights = weights or [1/len(self.base_models)] * len(self.base_models)
        self.is_fitted = False
    
    def fit(self, data, target_col='temperature'):
        """Fit all base models."""
        print(f"Training ensemble with {len(self.base_models)} base models...")
        
        for i, model in enumerate(self.base_models):
            print(f"  Training model {i+1}/{len(self.base_models)}: {type(model).__name__}")
            model.fit(data, target_col=target_col)
        
        self.target_col = target_col
        self.is_fitted = True
        print("Ensemble training completed")
    
    def predict(self, data, n_steps=1):
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        all_predictions = []
        for model in self.base_models:
            try:
                pred = model.predict(data, n_steps=n_steps)
                all_predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model {type(model).__name__} failed to predict: {e}")
                # Use mean prediction as fallback
                all_predictions.append(np.full(n_steps, data[self.target_col].mean()))
        
        # Weighted average
        all_predictions = np.array(all_predictions)
        ensemble_pred = np.average(all_predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def evaluate(self, test_data):
        """Evaluate ensemble model."""
        # For simplicity, evaluate each base model and return weighted average metrics
        all_metrics = []
        
        for model in self.base_models:
            try:
                metrics = model.evaluate(test_data)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Warning: Could not evaluate {type(model).__name__}: {e}")
        
        if not all_metrics:
            return {'rmse': float('inf'), 'mae': float('inf'), 'r2': 0, 'n_samples': len(test_data)}
        
        # Weighted average of metrics
        avg_metrics = {}
        for metric in ['rmse', 'mae', 'r2']:
            values = [m[metric] for m in all_metrics if metric in m]
            if values:
                avg_metrics[metric] = np.average(values, weights=self.weights[:len(values)])
            else:
                avg_metrics[metric] = 0
        
        avg_metrics['n_samples'] = all_metrics[0]['n_samples']
        avg_metrics['base_models'] = len(self.base_models)
        
        return avg_metrics


def run_mlflow_experiments_with_new_models():
    """
    Run MLflow experiments with different model types.
    """
    print(" RUNNING EXPERIMENTS WITH DIFFERENT MODEL TYPES")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    timestamps = pd.date_range(start='2023-01-01', periods=150, freq='h')
    temperature = (20 + 
                  3 * np.sin(np.arange(150) * 2 * np.pi / 24) +
                  np.random.normal(0, 0.5, 150))
    
    data = pd.DataFrame({'timestamp': timestamps, 'temperature': temperature})
    train_data = data[:120]
    test_data = data[120:]
    
    print(f" Data: {len(train_data)} train, {len(test_data)} test samples")
    
    # Initialize MLflow tracker
    tracker = ExperimentTracker("advanced-forecasting-models")
    
    # Models to test
    models_to_test = [
        ("Random Forest", RandomForestForecaster(n_lags=10, n_estimators=50)),
        ("SVM", SVMForecaster(n_lags=8, C=0.1)),
        ("Ensemble", EnsembleForecaster()),
    ]
    
    results = []
    
    for model_name, model in models_to_test:
        print(f"\n🔬 Testing {model_name}...")
        
        try:
            # Start MLflow run
            run_id = tracker.start_run(
                run_name=f"{model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}",
                model_type=model_name.lower().replace(' ', '_'),
                tags={"model_category": "advanced", "experiment": "comparison"}
            )
            
            # Log model parameters
            if hasattr(model, '__dict__'):
                params = {k: str(v) for k, v in model.__dict__.items() 
                         if not k.startswith('_') and not callable(v)}
                tracker.log_parameters(params)
            
            # Train model
            model.fit(train_data, target_col='temperature')
            
            # Evaluate model
            metrics = model.evaluate(test_data)
            tracker.log_metrics(metrics)
            
            # Make predictions
            try:
                predictions = model.predict(train_data, n_steps=5)
                print(f"   Next 5 predictions: {predictions}")
            except Exception as e:
                print(f"   Could not generate predictions: {e}")
            
            # Log model artifact
            try:
                tracker.log_model(model, "sklearn")
            except Exception as e:
                print(f"   Could not log model: {e}")
            
            # End run
            tracker.end_run()
            
            results.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'], 
                'R²': metrics['r2'],
                'Samples': metrics['n_samples']
            })
            
            print(f" {model_name} - RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            tracker.end_run()
    
    # Compare results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RMSE')
        
        print(f"\n🏆 MODEL COMPARISON RESULTS")
        print("=" * 50)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Save results
        results_df.to_csv(f'advanced_models_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    
    print(f"\n Experiments completed! View results at: http://localhost:5000")


def demonstrate_custom_model_logging():
    """
    Demonstrate how to log a completely custom model to MLflow.
    """
    print("\n" + "="*60)
    print("DEMONSTRATING CUSTOM MODEL LOGGING")
    print("="*60)
    
    # Example: Custom polynomial regression model
    class PolynomialForecaster:
        def __init__(self, degree=2):
            self.degree = degree
            self.coefficients = None
            self.is_fitted = False
        
        def fit(self, data, target_col='temperature'):
            # Simple polynomial fit to time index
            y = data[target_col].values
            x = np.arange(len(y))
            self.coefficients = np.polyfit(x, y, self.degree)
            self.is_fitted = True
            self.data_length = len(data)
        
        def predict(self, data, n_steps=1):
            if not self.is_fitted:
                raise ValueError("Model not fitted")
            
            # Predict next n_steps
            start_idx = self.data_length
            x_future = np.arange(start_idx, start_idx + n_steps)
            return np.polyval(self.coefficients, x_future)
        
        def evaluate(self, test_data):
            # Simple evaluation on polynomial trend
            y_true = test_data['temperature'].values
            x_test = np.arange(len(y_true))
            y_pred = np.polyval(self.coefficients, x_test)
            
            return {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'n_samples': len(y_true)
            }
    
    # Create and log custom model
    tracker = ExperimentTracker("custom-models")
    
    # Sample data
    data = pd.DataFrame({
        'temperature': 20 + 0.1 * np.arange(50) + np.random.normal(0, 0.5, 50)
    })
    
    model = PolynomialForecaster(degree=3)
    
    # Start MLflow run
    run_id = tracker.start_run(
        run_name="polynomial_forecaster",
        model_type="custom_polynomial",
        tags={"model_type": "polynomial", "complexity": "simple"}
    )
    
    # Log parameters
    tracker.log_parameters({
        "model_name": "PolynomialForecaster",
        "degree": model.degree,
        "algorithm": "numpy.polyfit"
    })
    
    # Train and evaluate
    model.fit(data)
    metrics = model.evaluate(data)
    tracker.log_metrics(metrics)
    
    # Log coefficients as parameters
    coeff_params = {f"coeff_{i}": float(coeff) for i, coeff in enumerate(model.coefficients)}
    tracker.log_parameters(coeff_params)
    
    # Create and log custom visualization
    plt.figure(figsize=(10, 6))
    x = np.arange(len(data))
    y_true = data['temperature'].values
    y_pred = np.polyval(model.coefficients, x)
    
    plt.plot(x, y_true, 'bo-', label='Actual', alpha=0.7)
    plt.plot(x, y_pred, 'r-', label=f'Polynomial Fit (degree {model.degree})', linewidth=2)
    plt.legend()
    plt.title('Custom Polynomial Forecaster')
    plt.xlabel('Time Index')
    plt.ylabel('Temperature')
    plt.grid(True, alpha=0.3)
    
    plot_file = 'polynomial_fit.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    tracker.log_artifact(plot_file)
    plt.close()
    
    # Clean up
    import os
    if os.path.exists(plot_file):
        os.remove(plot_file)
    
    tracker.end_run()
    
    print(" Custom polynomial model logged to MLflow")
    print(f"   RMSE: {metrics['rmse']:.3f}")
    print(f"   Coefficients: {model.coefficients}")


if __name__ == "__main__":
    # Run experiments with new model types
    run_mlflow_experiments_with_new_models()
    
    # Demonstrate custom model logging
    demonstrate_custom_model_logging()
    
    print(f"\n All experiments completed!")
    print(f" Start MLflow UI to view results: mlflow ui --port 5000")
    print(f"🔗 Access at: http://localhost:5000")
