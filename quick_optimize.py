"""
Quick Hyperparameter Optimization Runner
========================================

Simple script to quickly optimize your models with sensible defaults.
Perfect for getting started with hyperparameter tuning.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "ml-models"))

def generate_sample_data(n_points=400):
    """Generate sample temperature data if no data file provided"""
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
    
    # Realistic temperature pattern
    time_hours = np.arange(n_points)
    daily_cycle = 8 * np.sin(2 * np.pi * time_hours / 24)
    weekly_cycle = 3 * np.sin(2 * np.pi * time_hours / (24 * 7))
    trend = 0.01 * time_hours
    noise = np.random.normal(0, 2, n_points)
    
    temperature = 20 + daily_cycle + weekly_cycle + trend + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature
    })

def quick_optimize(model_type='all', data_file=None, n_trials=30, target_col='temperature'):
    """
    Quick optimization with sensible defaults
    
    Args:
        model_type: 'basic', 'xgboost', or 'all'
        data_file: Path to CSV file (optional, will generate sample data if None)
        n_trials: Number of optimization trials
        target_col: Target column name
    """
    from hyperparameter_optimizer import (
        BasicForecasterOptimizer,
        XGBoostForecasterOptimizer,
        MultiModelOptimizer
    )
    
    print("🚀 Quick Hyperparameter Optimization")
    print("=" * 50)
    
    # Load or generate data
    if data_file and Path(data_file).exists():
        print(f"📊 Loading data from {data_file}")
        data = pd.read_csv(data_file)
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
    else:
        print("📊 Generating sample temperature data")
        data = generate_sample_data()
    
    print(f"📈 Data shape: {data.shape}")
    print(f"🎯 Target column: {target_col}")
    print(f"🔢 Optimization trials: {n_trials}")
    
    if model_type.lower() == 'basic':
        print("\n🔍 Optimizing BasicTimeSeriesForecaster...")
        optimizer = BasicForecasterOptimizer(
            experiment_name="Quick_BasicForecaster_HPO",
            n_trials=n_trials,
            cv_folds=3
        )
        study = optimizer.optimize(data, target_col=target_col)
        print(f"✅ Best parameters: {study.best_params}")
        print(f"📊 Best CV RMSE: {study.best_value:.4f}")
        
    elif model_type.lower() == 'xgboost':
        print("\n🚀 Optimizing XGBoostForecaster...")
        optimizer = XGBoostForecasterOptimizer(
            experiment_name="Quick_XGBoostForecaster_HPO",
            n_trials=n_trials,
            cv_folds=3
        )
        study = optimizer.optimize(data, target_col=target_col)
        print(f"✅ Best parameters: {study.best_params}")
        print(f"📊 Best CV RMSE: {study.best_value:.4f}")
        
    elif model_type.lower() == 'all':
        print("\n🎯 Optimizing All Models...")
        optimizer = MultiModelOptimizer(
            tracking_uri="./mlruns",
            n_trials=n_trials
        )
        results = optimizer.optimize_all_models(data, target_col=target_col)
        
        print("\n🏆 Final Results:")
        for model_name, result in results.items():
            print(f"   {model_name}: RMSE={result['best_score']:.4f}, Params={result['best_params']}")
    
    else:
        print(f"❌ Unknown model type: {model_type}")
        print("   Available options: 'basic', 'xgboost', 'all'")
        return
    
    print(f"\n🌐 View results at: http://127.0.0.1:5000")
    print("💡 Check the MLflow dashboard for detailed analysis!")

def main():
    parser = argparse.ArgumentParser(description='Quick Hyperparameter Optimization')
    parser.add_argument('--model', choices=['basic', 'xgboost', 'all'], default='all',
                        help='Model type to optimize (default: all)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV data file (optional)')
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of optimization trials (default: 30)')
    parser.add_argument('--target', type=str, default='temperature',
                        help='Target column name (default: temperature)')
    
    args = parser.parse_args()
    
    try:
        quick_optimize(
            model_type=args.model,
            data_file=args.data,
            n_trials=args.trials,
            target_col=args.target
        )
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("💡 Make sure MLflow is running and your models are available")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("🎯 Running quick optimization demo...")
        quick_optimize(model_type='all', n_trials=15)
    else:
        main()
