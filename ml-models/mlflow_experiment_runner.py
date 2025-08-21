"""
MLflow Experiment Runner for Forecasting Models
===============================================

This script demonstrates MLflow experiment tracking across all forecasting models:
1. Linear Regression (Basic Forecaster)
2. Moving Averages (Simple ARIMA Forecaster)  
3. XGBoost (Advanced ML Forecaster)

Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import forecasting models
from basic_forecaster import BasicTimeSeriesForecaster
from xgboost_forecaster import XGBoostTimeSeriesForecaster
from simple_arima_forecaster import SimpleMovingAverageForecaster
from mlflow_tracking import ExperimentTracker, setup_mlflow_tracking

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Experiment tracking disabled.")


def create_comprehensive_dataset(n_points=200, noise_level=0.1):
    """
    Create a comprehensive synthetic temperature dataset for model comparison.
    
    Args:
        n_points (int): Number of data points to generate
        noise_level (float): Level of noise to add to the data
    
    Returns:
        pd.DataFrame: Synthetic temperature dataset
    """
    np.random.seed(42)  # For reproducible results
    
    # Create time index
    timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='H')
    
    # Base temperature with daily and weekly patterns
    t = np.arange(n_points)
    
    # Daily pattern (24-hour cycle)
    daily_pattern = 5 * np.sin(2 * np.pi * t / 24)
    
    # Weekly pattern (7-day cycle)  
    weekly_pattern = 3 * np.sin(2 * np.pi * t / (24 * 7))
    
    # Seasonal trend (annual cycle simulation)
    seasonal_pattern = 10 * np.sin(2 * np.pi * t / (24 * 365))
    
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, 0.1, n_points))
    
    # Base temperature around 20°C
    base_temp = 20.0
    
    # Combine all patterns
    temperature = (base_temp + 
                  daily_pattern + 
                  weekly_pattern + 
                  seasonal_pattern + 
                  random_walk +
                  np.random.normal(0, noise_level, n_points))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'hour': timestamps.hour,
        'day_of_week': timestamps.dayofweek,
        'month': timestamps.month
    })
    
    return data


def run_linear_regression_experiment(data, experiment_tracker):
    """
    Run Linear Regression forecasting experiment with MLflow tracking.
    
    Args:
        data (pd.DataFrame): Training data
        experiment_tracker (ExperimentTracker): MLflow tracker instance
    
    Returns:
        dict: Experiment results
    """
    print("=" * 60)
    print("LINEAR REGRESSION FORECASTING EXPERIMENT")
    print("=" * 60)
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size].reset_index(drop=True)
    test_data = data[train_size:].reset_index(drop=True)
    
    print(f"Data split: {len(train_data)} train, {len(test_data)} test samples")
    
    # Create and train model
    forecaster = BasicTimeSeriesForecaster(n_lags=6, enable_mlflow=True)
    forecaster.fit(train_data, target_col='temperature', run_name='linear_regression_comprehensive')
    
    # Evaluate model
    metrics = forecaster.evaluate(test_data, log_to_mlflow=True)
    
    # Make predictions
    future_predictions = forecaster.predict(train_data, n_steps=5)
    
    print(f"✅ Linear Regression Results:")
    print(f"   - RMSE: {metrics['rmse']:.3f}")
    print(f"   - MAE:  {metrics['mae']:.3f}")
    print(f"   - R²:   {metrics['r2']:.3f}")
    
    # Finish MLflow run
    forecaster.finish_mlflow_run()
    
    return {
        'model_name': 'Linear Regression',
        'forecaster': forecaster,
        'metrics': metrics,
        'predictions': future_predictions
    }


def run_xgboost_experiment(data, experiment_tracker):
    """
    Run XGBoost forecasting experiment with MLflow tracking.
    
    Args:
        data (pd.DataFrame): Training data
        experiment_tracker (ExperimentTracker): MLflow tracker instance
    
    Returns:
        dict: Experiment results
    """
    print("=" * 60)
    print("XGBOOST FORECASTING EXPERIMENT")
    print("=" * 60)
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size].reset_index(drop=True)
    test_data = data[train_size:].reset_index(drop=True)
    
    print(f"Data split: {len(train_data)} train, {len(test_data)} test samples")
    
    # Create and train model with optimized parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }
    
    forecaster = XGBoostTimeSeriesForecaster(
        n_lags=12, 
        rolling_windows=[3, 6, 12, 24],
        xgb_params=xgb_params,
        enable_mlflow=True
    )
    
    forecaster.fit(train_data, target_col='temperature', run_name='xgboost_comprehensive')
    
    # Evaluate model
    metrics = forecaster.evaluate(test_data, log_to_mlflow=True)
    
    # Make predictions  
    future_predictions = forecaster.predict(n_steps=5, last_known_data=train_data)
    
    print(f"✅ XGBoost Results:")
    print(f"   - RMSE: {metrics['rmse']:.3f}")
    print(f"   - MAE:  {metrics['mae']:.3f}")
    print(f"   - R²:   {metrics['r2']:.3f}")
    print(f"   - Features: {metrics['n_features']}")
    
    # Log feature importance
    try:
        feature_importance = forecaster.get_feature_importance(top_n=10)
        print(f"   - Top features: {feature_importance['feature'].head(3).tolist()}")
    except Exception as e:
        print(f"   - Feature importance unavailable: {e}")
    
    # Finish MLflow run
    forecaster.finish_mlflow_run()
    
    return {
        'model_name': 'XGBoost',
        'forecaster': forecaster,
        'metrics': metrics,
        'predictions': future_predictions
    }


def run_moving_average_experiment(data, experiment_tracker):
    """
    Run Moving Average forecasting experiment.
    Note: Moving averages don't use MLflow as they're statistical methods.
    
    Args:
        data (pd.DataFrame): Training data
        experiment_tracker (ExperimentTracker): MLflow tracker instance (unused for MA)
    
    Returns:
        dict: Experiment results
    """
    print("=" * 60)
    print("MOVING AVERAGE FORECASTING EXPERIMENT")
    print("=" * 60)
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size].reset_index(drop=True)
    test_data = data[train_size:].reset_index(drop=True)
    
    print(f"Data split: {len(train_data)} train, {len(test_data)} test samples")
    
    # Create and train model
    forecaster = SimpleMovingAverageForecaster(window=12)
    forecaster.fit(train_data, target_col='temperature')
    
    # Evaluate model - try different methods
    results = {}
    for method in ['sma', 'ema', 'wma']:
        forecaster.method = method
        metrics = forecaster.evaluate(test_data)
        results[method] = metrics
        print(f"✅ {method.upper()} Results:")
        print(f"   - RMSE: {metrics['rmse']:.3f}")
        print(f"   - MAE:  {metrics['mae']:.3f}")
        print(f"   - R²:   {metrics['r2']:.3f}")
    
    # Use best performing method for final predictions
    best_method = min(results.keys(), key=lambda k: results[k]['rmse'])
    forecaster.method = best_method
    
    # Make predictions
    future_predictions = forecaster.predict(n_steps=5)
    
    print(f"🏆 Best method: {best_method.upper()} with RMSE: {results[best_method]['rmse']:.3f}")
    
    return {
        'model_name': f'Moving Average ({best_method.upper()})',
        'forecaster': forecaster,
        'metrics': results[best_method],
        'predictions': future_predictions,
        'all_results': results
    }


def compare_models(experiment_results):
    """
    Compare all model results and create summary.
    
    Args:
        experiment_results (list): List of experiment result dictionaries
    
    Returns:
        pd.DataFrame: Comparison summary
    """
    print("=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison_data = []
    for result in experiment_results:
        comparison_data.append({
            'Model': result['model_name'],
            'RMSE': result['metrics']['rmse'],
            'MAE': result['metrics']['mae'],
            'R²': result['metrics']['r2'],
            'N_Samples': result['metrics']['n_samples']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('RMSE').reset_index(drop=True)
    comparison_df['Rank'] = range(1, len(comparison_df) + 1)
    
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    # Save comparison results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'mlflow_results'
    os.makedirs(results_dir, exist_ok=True)
    
    comparison_file = os.path.join(results_dir, f'model_comparison_{timestamp}.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n📊 Comparison saved to: {comparison_file}")
    
    return comparison_df


def main():
    """Main experiment runner."""
    print("🚀 STARTING MLFLOW FORECASTING EXPERIMENTS")
    print("=" * 60)
    
    if not MLFLOW_AVAILABLE:
        print("❌ MLflow not available. Please install MLflow to continue.")
        return
    
    # Create comprehensive dataset
    print("📊 Creating comprehensive synthetic dataset...")
    data = create_comprehensive_dataset(n_points=200, noise_level=0.15)
    print(f"   Generated {len(data)} data points")
    print(f"   Temperature range: {data['temperature'].min():.1f}°C to {data['temperature'].max():.1f}°C")
    
    # Initialize MLflow tracking
    experiment_tracker = setup_mlflow_tracking("comprehensive-forecasting-comparison")
    
    # Run experiments
    experiment_results = []
    
    try:
        # 1. Linear Regression
        lr_result = run_linear_regression_experiment(data, experiment_tracker)
        experiment_results.append(lr_result)
        
        # 2. XGBoost
        xgb_result = run_xgboost_experiment(data, experiment_tracker)
        experiment_results.append(xgb_result)
        
        # 3. Moving Averages
        ma_result = run_moving_average_experiment(data, experiment_tracker)
        experiment_results.append(ma_result)
        
        # Compare all models
        comparison_df = compare_models(experiment_results)
        
        # Print final summary
        print("\n🏆 EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully ran {len(experiment_results)} experiments")
        print(f"🥇 Best model: {comparison_df.iloc[0]['Model']} (RMSE: {comparison_df.iloc[0]['RMSE']:.3f})")
        print(f"📈 MLflow UI: Run 'mlflow ui' in the terminal to view detailed results")
        print(f"🔗 MLflow tracking URI: {experiment_tracker.config.tracking_uri}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Experiments completed!")


if __name__ == "__main__":
    main()
