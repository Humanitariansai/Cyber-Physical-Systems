"""
MLflow Model Integration Guide
=============================

This guide explains how to add different models to MLflow tracking
in your cyber-physical systems forecasting project.

Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Import MLflow and forecasting models
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from mlflow_tracking import ExperimentTracker

# Import your existing models
from basic_forecaster import BasicTimeSeriesForecaster
from xgboost_forecaster import XGBoostTimeSeriesForecaster
from simple_arima_forecaster import SimpleMovingAverageForecaster


class MLflowModelManager:
    """
    Centralized manager for adding and tracking different models in MLflow.
    
    This class provides a unified interface for:
    1. Registering new model types
    2. Running experiments with different models
    3. Comparing model performance
    4. Managing model artifacts and metadata
    """
    
    def __init__(self, experiment_name="model-comparison"):
        self.experiment_name = experiment_name
        self.tracker = ExperimentTracker(experiment_name)
        self.registered_models = {}
        self.experiment_results = []
        
    def register_model(self, model_name, model_class, model_params=None, mlflow_type="sklearn"):
        """
        Register a new model type for experiment tracking.
        
        Args:
            model_name (str): Unique name for the model
            model_class: The model class to instantiate
            model_params (dict): Default parameters for the model
            mlflow_type (str): MLflow model type ('sklearn', 'xgboost', 'pytorch', etc.)
        """
        self.registered_models[model_name] = {
            'class': model_class,
            'params': model_params or {},
            'mlflow_type': mlflow_type
        }
        print(f"✅ Registered model: {model_name}")
    
    def run_experiment(self, model_name, train_data, test_data, target_col='temperature', 
                      custom_params=None, run_name=None):
        """
        Run a complete experiment for a specific model.
        
        Args:
            model_name (str): Name of the registered model
            train_data (pd.DataFrame): Training dataset
            test_data (pd.DataFrame): Test dataset  
            target_col (str): Target column name
            custom_params (dict): Custom parameters to override defaults
            run_name (str): Custom run name
        
        Returns:
            dict: Experiment results with metrics and artifacts
        """
        if model_name not in self.registered_models:
            raise ValueError(f"Model '{model_name}' not registered. Available: {list(self.registered_models.keys())}")
        
        model_config = self.registered_models[model_name]
        
        # Prepare parameters
        params = model_config['params'].copy()
        if custom_params:
            params.update(custom_params)
        
        # Generate run name if not provided
        if run_name is None:
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 Running experiment: {run_name}")
        print(f"   Model: {model_name}")
        print(f"   Parameters: {params}")
        
        try:
            # Start MLflow run
            run_id = self.tracker.start_run(
                run_name=run_name,
                model_type=model_name.lower().replace(' ', '_'),
                tags={
                    "model_family": model_name,
                    "experiment_type": "forecasting",
                    "data_type": "temperature"
                }
            )
            
            # Log experiment metadata
            self.tracker.log_parameters({
                "model_name": model_name,
                "target_column": target_col,
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                **params
            })
            
            # Log dataset information
            self.tracker.log_dataset_info(train_data, "training_data")
            self.tracker.log_dataset_info(test_data, "test_data")
            
            # Create and train model
            model = model_config['class'](**params)
            
            # Handle different model interfaces
            if hasattr(model, 'fit'):
                if 'enable_mlflow' in params:
                    # For models with built-in MLflow support
                    model.fit(train_data, target_col=target_col, run_name=run_name)
                else:
                    # For models without built-in MLflow support
                    model.fit(train_data, target_col=target_col)
            else:
                raise ValueError(f"Model {model_name} doesn't have a 'fit' method")
            
            # Evaluate model
            if hasattr(model, 'evaluate'):
                metrics = model.evaluate(test_data)
            else:
                # Custom evaluation for models without evaluate method
                metrics = self._evaluate_model(model, test_data, target_col)
            
            # Log metrics
            self.tracker.log_metrics(metrics)
            
            # Generate predictions for visualization
            if hasattr(model, 'predict'):
                try:
                    # Handle different predict interfaces
                    if model_name == "Moving Averages":
                        predictions = model.predict(n_steps=5)
                    elif model_name == "XGBoost":
                        predictions = model.predict(n_steps=5, last_known_data=train_data)
                    else:
                        predictions = model.predict(train_data, n_steps=5)
                    
                    # Log prediction results
                    pred_df = pd.DataFrame({
                        'step': range(1, len(predictions) + 1),
                        'prediction': predictions
                    })
                    pred_file = f"predictions_{run_name}.csv"
                    pred_df.to_csv(pred_file, index=False)
                    self.tracker.log_artifact(pred_file)
                    os.remove(pred_file)  # Clean up
                    
                except Exception as e:
                    print(f"   Warning: Could not generate predictions: {e}")
            
            # Log model artifact
            try:
                self.tracker.log_model(model, model_config['mlflow_type'])
            except Exception as e:
                print(f"   Warning: Could not log model artifact: {e}")
            
            # Create performance visualization
            self._create_performance_plot(metrics, run_name)
            
            # End MLflow run
            self.tracker.end_run()
            
            # Store results
            result = {
                'run_id': run_id,
                'model_name': model_name,
                'run_name': run_name,
                'metrics': metrics,
                'parameters': params,
                'model': model
            }
            
            self.experiment_results.append(result)
            
            print(f"✅ Experiment completed: {run_name}")
            print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
            print(f"   MAE: {metrics.get('mae', 'N/A')}")
            print(f"   R²: {metrics.get('r2', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            self.tracker.end_run()
            raise
    
    def _evaluate_model(self, model, test_data, target_col):
        """Custom evaluation for models without built-in evaluate method."""
        # This is a fallback evaluation method
        # You would implement specific evaluation logic here
        return {
            'rmse': 0.0,
            'mae': 0.0,
            'r2': 0.0,
            'n_samples': len(test_data)
        }
    
    def _create_performance_plot(self, metrics, run_name):
        """Create and log performance visualization."""
        try:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            metric_names = ['RMSE', 'MAE', 'R²']
            metric_values = [
                metrics.get('rmse', 0),
                metrics.get('mae', 0), 
                metrics.get('r2', 0)
            ]
            
            for i, (name, value) in enumerate(zip(metric_names, metric_values)):
                ax[i].bar([name], [value], color=['red', 'orange', 'green'][i])
                ax[i].set_title(f'{name}: {value:.3f}')
                ax[i].set_ylim(0, max(value * 1.2, 0.1))
            
            plt.suptitle(f'Performance Metrics: {run_name}')
            plt.tight_layout()
            
            plot_file = f"performance_{run_name}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.tracker.log_artifact(plot_file)
            plt.close()
            
            # Clean up
            if os.path.exists(plot_file):
                os.remove(plot_file)
                
        except Exception as e:
            print(f"Warning: Could not create performance plot: {e}")
    
    def compare_all_models(self):
        """
        Compare all experiment results and create comprehensive report.
        
        Returns:
            pd.DataFrame: Comparison results
        """
        if not self.experiment_results:
            print("No experiments to compare!")
            return pd.DataFrame()
        
        comparison_data = []
        for result in self.experiment_results:
            comparison_data.append({
                'Model': result['model_name'],
                'Run_Name': result['run_name'],
                'RMSE': result['metrics'].get('rmse', float('inf')),
                'MAE': result['metrics'].get('mae', float('inf')),
                'R²': result['metrics'].get('r2', 0),
                'Run_ID': result['run_id'][:8] + '...'  # Shortened for display
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE').reset_index(drop=True)
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        print("\n🏆 MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_file = f'model_comparison_{timestamp}.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\n📊 Comparison saved to: {comparison_file}")
        
        return comparison_df
    
    def get_best_model(self, metric='rmse'):
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric (str): Metric to use for comparison ('rmse', 'mae', 'r2')
        
        Returns:
            dict: Best model result
        """
        if not self.experiment_results:
            return None
        
        if metric == 'r2':
            # Higher is better for R²
            best_result = max(self.experiment_results, 
                            key=lambda x: x['metrics'].get(metric, 0))
        else:
            # Lower is better for RMSE and MAE
            best_result = min(self.experiment_results, 
                            key=lambda x: x['metrics'].get(metric, float('inf')))
        
        print(f"\n🥇 Best model by {metric.upper()}:")
        print(f"   Model: {best_result['model_name']}")
        print(f"   Run: {best_result['run_name']}")
        print(f"   {metric.upper()}: {best_result['metrics'].get(metric, 'N/A')}")
        
        return best_result


def setup_standard_models():
    """
    Set up the standard forecasting models for MLflow tracking.
    
    Returns:
        MLflowModelManager: Configured model manager
    """
    manager = MLflowModelManager("comprehensive-forecasting")
    
    # Register Linear Regression
    manager.register_model(
        model_name="Linear Regression",
        model_class=BasicTimeSeriesForecaster,
        model_params={
            'n_lags': 6,
            'enable_mlflow': False  # We'll handle MLflow externally
        },
        mlflow_type="sklearn"
    )
    
    # Register XGBoost
    manager.register_model(
        model_name="XGBoost",
        model_class=XGBoostTimeSeriesForecaster,
        model_params={
            'n_lags': 12,
            'rolling_windows': [3, 6, 12],
            'enable_mlflow': False  # We'll handle MLflow externally
        },
        mlflow_type="xgboost"
    )
    
    # Register Moving Averages
    manager.register_model(
        model_name="Moving Averages",
        model_class=SimpleMovingAverageForecaster,
        model_params={
            'window': 12,
            'method': 'sma'
        },
        mlflow_type="sklearn"  # Generic type for custom models
    )
    
    return manager


def add_new_model_example():
    """
    Example of how to add a new model type to MLflow tracking.
    """
    
    # Example: Adding a simple LSTM model (placeholder)
    class SimpleSequenceForecaster:
        """Example placeholder for a new model type."""
        
        def __init__(self, sequence_length=10, hidden_units=50):
            self.sequence_length = sequence_length
            self.hidden_units = hidden_units
            self.is_fitted = False
        
        def fit(self, data, target_col='temperature'):
            """Placeholder fit method."""
            print(f"Training sequence model with {len(data)} samples")
            self.is_fitted = True
        
        def predict(self, data, n_steps=1):
            """Placeholder predict method."""
            if not self.is_fitted:
                raise ValueError("Model must be fitted first")
            # Return dummy predictions
            return np.random.normal(20, 2, n_steps)
        
        def evaluate(self, test_data):
            """Placeholder evaluate method."""
            return {
                'rmse': np.random.uniform(0.5, 2.0),
                'mae': np.random.uniform(0.3, 1.5),
                'r2': np.random.uniform(0.8, 0.95),
                'n_samples': len(test_data)
            }
    
    # Set up model manager
    manager = setup_standard_models()
    
    # Add the new model
    manager.register_model(
        model_name="Sequence Model",
        model_class=SimpleSequenceForecaster,
        model_params={
            'sequence_length': 24,  # 24 hours
            'hidden_units': 64
        },
        mlflow_type="sklearn"  # Or "pytorch" if it's a PyTorch model
    )
    
    print("✅ Added new Sequence Model to MLflow tracking")
    return manager


def run_comprehensive_experiments():
    """
    Run experiments with all registered models and compare results.
    """
    print("🚀 STARTING COMPREHENSIVE MODEL EXPERIMENTS")
    print("=" * 60)
    
    # Create synthetic dataset
    np.random.seed(42)
    timestamps = pd.date_range(start='2023-01-01', periods=200, freq='h')
    temperature = (20 + 
                  5 * np.sin(np.arange(200) * 2 * np.pi / 24) +  # Daily pattern
                  2 * np.sin(np.arange(200) * 2 * np.pi / (24*7)) +  # Weekly pattern
                  np.random.normal(0, 0.5, 200))  # Noise
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature
    })
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"📊 Dataset: {len(train_data)} train, {len(test_data)} test samples")
    
    # Set up model manager with all standard models
    manager = setup_standard_models()
    
    # Run experiments for each model
    models_to_test = [
        ("Linear Regression", {"n_lags": 6}),
        ("Linear Regression", {"n_lags": 12}),  # Different configuration
        ("XGBoost", {"n_lags": 8, "rolling_windows": [3, 6]}),
        ("XGBoost", {"n_lags": 16, "rolling_windows": [3, 6, 12, 24]}),  # Different config
        ("Moving Averages", {"window": 8, "method": "sma"}),
        ("Moving Averages", {"window": 12, "method": "ema"}),
    ]
    
    for model_name, custom_params in models_to_test:
        try:
            manager.run_experiment(
                model_name=model_name,
                train_data=train_data,
                test_data=test_data,
                custom_params=custom_params
            )
        except Exception as e:
            print(f"❌ Failed to run {model_name}: {e}")
    
    # Compare all results
    comparison_df = manager.compare_all_models()
    
    # Get best model
    best_model = manager.get_best_model('rmse')
    
    print(f"\n🎯 Experiments completed! Check MLflow UI at: http://localhost:5000")
    
    return manager, comparison_df


if __name__ == "__main__":
    # Run the comprehensive experiments
    manager, results = run_comprehensive_experiments()
    
    # Example of adding a new model
    print("\n" + "="*60)
    print("EXAMPLE: Adding New Model Type")
    print("="*60)
    manager_with_new_model = add_new_model_example()
