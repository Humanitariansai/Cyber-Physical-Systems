"""
Hyperparameter Optimization Demo
===============================

This script demonstrates how to use the hyperparameter optimization framework
with your existing forecasting models and MLflow tracking.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "ml-models"))

from hyperparameter_optimizer import (
    BasicForecasterOptimizer,
    XGBoostForecasterOptimizer,
    MultiModelOptimizer
)

def generate_realistic_temperature_data(n_points=500):
    """
    Generate realistic temperature time series data for optimization
    """
    np.random.seed(42)
    
    # Create time series
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
    
    # Generate complex temperature patterns
    time_hours = np.arange(n_points)
    
    # Daily cycle (24-hour period)
    daily_cycle = 8 * np.sin(2 * np.pi * time_hours / 24)
    
    # Weekly cycle (7-day period)  
    weekly_cycle = 3 * np.sin(2 * np.pi * time_hours / (24 * 7))
    
    # Seasonal trend (yearly cycle approximation)
    seasonal_trend = 10 * np.sin(2 * np.pi * time_hours / (24 * 365))
    
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, 0.5, n_points))
    
    # Noise
    noise = np.random.normal(0, 2, n_points)
    
    # Combine all components
    temperature = 20 + daily_cycle + weekly_cycle + seasonal_trend + random_walk + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'month': dates.month
    })
    
    return df

def demo_basic_forecaster_optimization():
    """
    Demonstrate BasicTimeSeriesForecaster hyperparameter optimization
    """
    print(" DEMO: BasicTimeSeriesForecaster Hyperparameter Optimization")
    print("=" * 70)
    
    # Generate data
    data = generate_realistic_temperature_data(400)
    print(f" Generated {len(data)} data points for optimization")
    
    # Create optimizer
    optimizer = BasicForecasterOptimizer(
        experiment_name="BasicForecaster_Demo_HPO",
        n_trials=20,  # Small number for demo
        cv_folds=3
    )
    
    # Run optimization
    study = optimizer.optimize(data, target_col='temperature')
    
    print(f"\n Optimization completed!")
    print(f"🏆 Best parameters: {study.best_params}")
    print(f" Best CV RMSE: {study.best_value:.4f}")
    
    return study

def demo_xgboost_optimization():
    """
    Demonstrate XGBoostForecaster hyperparameter optimization
    """
    print("\n DEMO: XGBoostForecaster Hyperparameter Optimization")
    print("=" * 70)
    
    # Generate data
    data = generate_realistic_temperature_data(400)
    print(f" Generated {len(data)} data points for optimization")
    
    # Create optimizer
    optimizer = XGBoostForecasterOptimizer(
        experiment_name="XGBoostForecaster_Demo_HPO",
        n_trials=15,  # Small number for demo
        cv_folds=3
    )
    
    # Run optimization
    study = optimizer.optimize(data, target_col='temperature')
    
    print(f"\n Optimization completed!")
    print(f"🏆 Best parameters: {study.best_params}")
    print(f" Best CV RMSE: {study.best_value:.4f}")
    
    return study

def demo_multi_model_optimization():
    """
    Demonstrate multi-model optimization and comparison
    """
    print("\n DEMO: Multi-Model Hyperparameter Optimization")
    print("=" * 70)
    
    # Generate data
    data = generate_realistic_temperature_data(400)
    print(f" Generated {len(data)} data points for optimization")
    
    # Create multi-model optimizer
    multi_optimizer = MultiModelOptimizer(
        tracking_uri="./mlruns",
        n_trials=15  # Small number for demo
    )
    
    # Run optimization for all models
    results = multi_optimizer.optimize_all_models(data, target_col='temperature')
    
    print(f"\n Multi-model optimization completed!")
    
    return results

def demo_grid_search_alternative():
    """
    Demonstrate a simple grid search as an alternative to Bayesian optimization
    """
    print("\n🔬 DEMO: Grid Search Alternative")
    print("=" * 70)
    
    import mlflow
    from itertools import product
    
    # Set up MLflow
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Grid_Search_Demo")
    
    # Generate data
    data = generate_realistic_temperature_data(300)
    train_data = data[:240]
    test_data = data[240:]
    
    # Define parameter grid
    param_grid = {
        'n_lags': [3, 6, 12, 18],
    }
    
    # Grid search
    results = []
    
    with mlflow.start_run(run_name="Grid_Search_BasicForecaster"):
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            
            try:
                # Import and create model
                from basic_forecaster import BasicTimeSeriesForecaster
                
                model = BasicTimeSeriesForecaster(
                    n_lags=param_dict['n_lags'],
                    enable_mlflow=False
                )
                
                # Train and evaluate
                with mlflow.start_run(nested=True, run_name=f"grid_lag_{param_dict['n_lags']}"):
                    model.fit(train_data, target_col='temperature')
                    predictions = model.predict(test_data)
                    
                    rmse = np.sqrt(np.mean((test_data['temperature'].values - predictions) ** 2))
                    
                    # Log to MLflow
                    mlflow.log_params(param_dict)
                    mlflow.log_metric("test_rmse", rmse)
                    mlflow.log_param("search_type", "grid_search")
                    
                    results.append({
                        'params': param_dict,
                        'rmse': rmse
                    })
                    
                    print(f" Grid point {param_dict}: RMSE = {rmse:.4f}")
                    
            except Exception as e:
                print(f"❌ Error with params {param_dict}: {e}")
    
    # Find best result
    best_result = min(results, key=lambda x: x['rmse'])
    print(f"\n🏆 Best grid search result: {best_result}")
    
    return results

if __name__ == "__main__":
    print(" Starting Hyperparameter Optimization Demonstration")
    print("=" * 80)
    print("This demo will show you how to:")
    print("1. Optimize BasicTimeSeriesForecaster hyperparameters")
    print("2. Optimize XGBoostForecaster hyperparameters") 
    print("3. Compare multiple models automatically")
    print("4. Run grid search as an alternative")
    print("=" * 80)
    
    try:
        # Demo 1: BasicTimeSeriesForecaster
        basic_study = demo_basic_forecaster_optimization()
        
        # Demo 2: XGBoostForecaster
        xgb_study = demo_xgboost_optimization()
        
        # Demo 3: Multi-model comparison
        multi_results = demo_multi_model_optimization()
        
        # Demo 4: Grid search alternative
        grid_results = demo_grid_search_alternative()
        
        print("\n" + "=" * 80)
        print(" ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(" Check your MLflow dashboard at http://127.0.0.1:5000")
        print(" Look for these experiments:")
        print("   - BasicForecaster_Demo_HPO")
        print("   - XGBoostForecaster_Demo_HPO")
        print("   - hyperparameter-optimization")
        print("   - Grid_Search_Demo")
        print("\n You can now use these optimized parameters in your models!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print(" Make sure your models are available and MLflow is running")
        import traceback
        traceback.print_exc()
