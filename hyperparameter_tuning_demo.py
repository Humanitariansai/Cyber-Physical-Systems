"""
Hyperparameter Tuning Demonstration with MLflow
==============================================

This script demonstrates hyperparameter optimization for your forecasting models
with full MLflow tracking and visualization.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import sys
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add ml-models to path
sys.path.append(str(Path(__file__).parent / "ml-models"))

def generate_demo_temperature_data(n_points=400):
    """Generate realistic temperature data for hyperparameter tuning demo"""
    np.random.seed(42)
    
    # Create time series
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
    
    # Generate complex temperature patterns
    time_hours = np.arange(n_points)
    
    # Daily cycle (24-hour period)
    daily_cycle = 8 * np.sin(2 * np.pi * time_hours / 24)
    
    # Weekly cycle (7-day period)  
    weekly_cycle = 3 * np.sin(2 * np.pi * time_hours / (24 * 7))
    
    # Seasonal trend
    seasonal_trend = 5 * np.sin(2 * np.pi * time_hours / (24 * 30))
    
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, 0.3, n_points))
    
    # Noise
    noise = np.random.normal(0, 1.5, n_points)
    
    # Combine all components
    temperature = 22 + daily_cycle + weekly_cycle + seasonal_trend + random_walk + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature
    })
    
    return df

def evaluate_model_with_cv(model, data, target_col='temperature', n_splits=3):
    """
    Evaluate model using time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=50)
    scores = {'rmse': [], 'mae': [], 'r2': []}
    
    for train_idx, test_idx in tscv.split(data):
        try:
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Fit model
            model.fit(train_data, target_col=target_col)
            
            # Predict
            predictions = model.predict(test_data)
            actual = test_data[target_col].values
            
            # Handle prediction format
            if np.isscalar(predictions):
                predictions = np.full(len(actual), predictions)
            elif len(predictions) != len(actual):
                predictions = np.full(len(actual), predictions[0] if len(predictions) > 0 else 0)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            r2 = r2_score(actual, predictions)
            
            if np.isfinite(rmse) and np.isfinite(mae) and np.isfinite(r2):
                scores['rmse'].append(rmse)
                scores['mae'].append(mae)
                scores['r2'].append(r2)
                
        except Exception as e:
            print(f"CV fold failed: {e}")
            continue
    
    if len(scores['rmse']) > 0:
        return {
            'mean_rmse': np.mean(scores['rmse']),
            'std_rmse': np.std(scores['rmse']),
            'mean_mae': np.mean(scores['mae']),
            'std_mae': np.std(scores['mae']),
            'mean_r2': np.mean(scores['r2']),
            'std_r2': np.std(scores['r2'])
        }
    else:
        return {
            'mean_rmse': 100.0,
            'std_rmse': 0.0,
            'mean_mae': 100.0,
            'std_mae': 0.0,
            'mean_r2': -10.0,
            'std_r2': 0.0
        }

def hyperparameter_tuning_demo():
    """
    Demonstrate hyperparameter tuning with MLflow tracking
    """
    print(" Hyperparameter Tuning Demonstration with MLflow")
    print("=" * 60)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Hyperparameter_Tuning_Demo")
    
    # Generate demo data
    data = generate_demo_temperature_data(350)
    print(f" Generated {len(data)} data points for tuning")
    print(f" Temperature range: {data['temperature'].min():.2f}°C to {data['temperature'].max():.2f}°C")
    
    try:
        # Import the basic forecaster
        from basic_forecaster import BasicTimeSeriesForecaster
        
        # Define hyperparameter grid to test
        lag_values = [3, 6, 9, 12, 15, 18]
        
        print(f"\n Testing {len(lag_values)} different lag configurations...")
        
        best_rmse = float('inf')
        best_params = {}
        all_results = []
        
        with mlflow.start_run(run_name="BasicForecaster_Hyperparameter_Search"):
            
            for i, n_lags in enumerate(lag_values):
                print(f"🧪 Trial {i+1}/{len(lag_values)}: Testing n_lags={n_lags}")
                
                with mlflow.start_run(run_name=f"trial_lags_{n_lags}", nested=True):
                    # Create model with current parameters
                    model = BasicTimeSeriesForecaster(n_lags=n_lags, enable_mlflow=False)
                    
                    # Evaluate with cross-validation
                    cv_results = evaluate_model_with_cv(model, data, target_col='temperature')
                    
                    # Log parameters to MLflow
                    mlflow.log_param("n_lags", n_lags)
                    mlflow.log_param("model_type", "BasicTimeSeriesForecaster")
                    mlflow.log_param("trial_number", i+1)
                    
                    # Log metrics to MLflow
                    mlflow.log_metric("cv_mean_rmse", cv_results['mean_rmse'])
                    mlflow.log_metric("cv_std_rmse", cv_results['std_rmse'])
                    mlflow.log_metric("cv_mean_mae", cv_results['mean_mae'])
                    mlflow.log_metric("cv_std_mae", cv_results['std_mae'])
                    mlflow.log_metric("cv_mean_r2", cv_results['mean_r2'])
                    mlflow.log_metric("cv_std_r2", cv_results['std_r2'])
                    
                    # Track best result
                    if cv_results['mean_rmse'] < best_rmse:
                        best_rmse = cv_results['mean_rmse']
                        best_params = {'n_lags': n_lags}
                    
                    # Store results
                    all_results.append({
                        'n_lags': n_lags,
                        'rmse': cv_results['mean_rmse'],
                        'mae': cv_results['mean_mae'],
                        'r2': cv_results['mean_r2']
                    })
                    
                    print(f"    RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
            
            # Log best overall results
            mlflow.log_params(best_params)
            mlflow.log_metric("best_rmse", best_rmse)
            mlflow.log_param("total_trials", len(lag_values))
            mlflow.log_param("optimization_method", "grid_search")
            
            print(f"\n" + "=" * 60)
            print(" HYPERPARAMETER TUNING RESULTS")
            print("=" * 60)
            print(f"{'n_lags':<8} {'RMSE':<12} {'MAE':<12} {'R²':<8}")
            print("-" * 45)
            
            # Sort results by RMSE
            all_results.sort(key=lambda x: x['rmse'])
            
            for result in all_results:
                print(f"{result['n_lags']:<8} {result['rmse']:<12.4f} {result['mae']:<12.4f} {result['r2']:<8.3f}")
            
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   Parameters: {best_params}")
            print(f"   Cross-validation RMSE: {best_rmse:.4f}")
            
            # Test best model on final validation
            print(f"\n🧪 Testing best model on validation set...")
            best_model = BasicTimeSeriesForecaster(n_lags=best_params['n_lags'], enable_mlflow=False)
            
            # Split data for final validation
            train_size = int(0.8 * len(data))
            train_data = data[:train_size]
            val_data = data[train_size:]
            
            # Train and evaluate best model
            best_model.fit(train_data, target_col='temperature')
            val_predictions = best_model.predict(val_data)
            val_actual = val_data['temperature'].values
            
            # Handle prediction format
            if np.isscalar(val_predictions):
                val_predictions = np.full(len(val_actual), val_predictions)
            elif len(val_predictions) != len(val_actual):
                val_predictions = np.full(len(val_actual), val_predictions[0] if len(val_predictions) > 0 else 0)
            
            # Calculate final metrics
            final_rmse = np.sqrt(mean_squared_error(val_actual, val_predictions))
            final_mae = mean_absolute_error(val_actual, val_predictions)
            final_r2 = r2_score(val_actual, val_predictions)
            
            # Log final validation results
            mlflow.log_metric("final_validation_rmse", final_rmse)
            mlflow.log_metric("final_validation_mae", final_mae)
            mlflow.log_metric("final_validation_r2", final_r2)
            
            # Save best model
            mlflow.sklearn.log_model(best_model.model, "best_model")
            
            print(f"   Final Validation RMSE: {final_rmse:.4f}")
            print(f"   Final Validation MAE: {final_mae:.4f}")
            print(f"   Final Validation R²: {final_r2:.4f}")
            
        print(f"\n View detailed results at: http://127.0.0.1:5000")
        print(" Look for the 'Hyperparameter_Tuning_Demo' experiment in MLflow!")
        print(" You can compare all trials and see parameter vs. performance relationships!")
        
        return best_params, best_rmse
        
    except ImportError:
        print("❌ Could not import BasicTimeSeriesForecaster")
        print(" Make sure you're in the correct directory with the ml-models folder")
        return None, None
    except Exception as e:
        print(f"❌ Error during hyperparameter tuning: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    best_params, best_score = hyperparameter_tuning_demo()
    
    if best_params:
        print(f"\n Hyperparameter tuning completed successfully!")
        print(f"🏆 Best parameters: {best_params}")
        print(f" Best performance: RMSE = {best_score:.4f}")
        print(f"\n Next steps:")
        print(f"   1. Check MLflow dashboard for detailed analysis")
        print(f"   2. Use best parameters in your production models")
        print(f"   3. Consider more advanced optimization algorithms (Bayesian, etc.)")
    else:
        print(f"\n❌ Hyperparameter tuning failed. Check the error messages above.")
