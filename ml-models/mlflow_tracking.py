"""
MLflow Configuration for Cyber-Physical Systems Forecasting Models
==================================================================

This module provides centralized MLflow configuration and utilities for 
experiment tracking across all forecasting models.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os
import yaml
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


class MLflowConfig:
    """Central configuration for MLflow experiments."""
    
    def __init__(self, experiment_name: str = "forecasting-models"):
        self.experiment_name = experiment_name
        self.tracking_uri = self._setup_tracking_uri()
        self.artifact_location = self._setup_artifact_location()
        
    def _setup_tracking_uri(self) -> str:
        """Set up MLflow tracking URI."""
        # Use relative path approach for better Windows compatibility
        mlruns_path = Path(__file__).parent / "mlruns"
        mlruns_path.mkdir(exist_ok=True)
        return "./mlruns"
    
    def _setup_artifact_location(self) -> str:
        """Set up artifact storage location."""
        artifacts_path = Path(__file__).parent / "mlflow-artifacts"
        artifacts_path.mkdir(exist_ok=True)
        return "./mlflow-artifacts"


class ExperimentTracker:
    """
    MLflow experiment tracking utility for forecasting models.
    
    This class provides a unified interface for logging experiments,
    parameters, metrics, and artifacts across all forecasting models.
    """
    
    def __init__(self, experiment_name: str = "forecasting-models"):
        self.config = MLflowConfig(experiment_name)
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Initialize MLflow with proper configuration."""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(
                name=self.config.experiment_name,
                artifact_location=self.config.artifact_location
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(self.config.experiment_name)
        print(f"MLflow tracking URI: {self.config.tracking_uri}")
        print(f"Experiment: {self.config.experiment_name}")
        
    def start_run(self, run_name: str, model_type: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            model_type: Type of model (e.g., 'linear_regression', 'xgboost', 'moving_average')
            tags: Additional tags for the run
            
        Returns:
            Run ID
        """
        default_tags = {
            "model_type": model_type,
            "project": "cyber-physical-systems",
            "task": "temperature-forecasting",
            "timestamp": datetime.now().isoformat()
        }
        
        if tags:
            default_tags.update(tags)
            
        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        return run.info.run_id
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log model parameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log model metrics."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, model_type: str, signature=None, input_example=None):
        """
        Log model artifact.
        
        Args:
            model: The trained model
            model_type: Type of model for appropriate MLflow logging
            signature: Model signature
            input_example: Example input
        """
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                input_example=input_example
            )
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example
            )
        else:
            # For custom models (like moving averages), use generic pickle
            mlflow.log_artifact(model, "model")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log file or directory as artifact."""
        if artifact_name:
            mlflow.log_artifact(artifact_path, artifact_name)
        else:
            mlflow.log_artifact(artifact_path)
    
    def log_dataset_info(self, data: pd.DataFrame, dataset_name: str = "training_data"):
        """Log dataset information and statistics."""
        # Log dataset shape and basic stats
        self.log_parameters({
            f"{dataset_name}_shape": f"{data.shape[0]}x{data.shape[1]}",
            f"{dataset_name}_columns": list(data.columns),
            f"{dataset_name}_size": len(data)
        })
        
        # Log statistical summary
        stats = data.describe()
        stats_dict = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            stats_dict.update({
                f"{dataset_name}_{col}_mean": stats.loc['mean', col],
                f"{dataset_name}_{col}_std": stats.loc['std', col],
                f"{dataset_name}_{col}_min": stats.loc['min', col],
                f"{dataset_name}_{col}_max": stats.loc['max', col]
            })
        
        self.log_metrics(stats_dict)
    
    def log_prediction_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             prefix: str = ""):
        """Log prediction results and create comparison plots."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import matplotlib.pyplot as plt
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Log metrics
        metrics = {
            f"{prefix}mse": mse,
            f"{prefix}rmse": rmse,
            f"{prefix}mae": mae,
            f"{prefix}r2_score": r2
        }
        self.log_metrics(metrics)
        
        # Create and log prediction vs actual plot
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{prefix}Predictions vs Actual')
        
        # Create residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{prefix}Residual Plot')
        
        plt.tight_layout()
        
        # Save and log plot
        plot_path = f"prediction_analysis_{prefix.rstrip('_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.log_artifact(plot_path)
        plt.close()
        
        # Clean up
        if os.path.exists(plot_path):
            os.remove(plot_path)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def get_experiment_results(self) -> pd.DataFrame:
        """Get all runs from the current experiment."""
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return runs


def create_experiment_config(model_name: str, model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a standardized experiment configuration.
    
    Args:
        model_name: Name of the model
        model_params: Model-specific parameters
        
    Returns:
        Experiment configuration dictionary
    """
    config = {
        "experiment_info": {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "project": "cyber-physical-systems",
            "task": "temperature-forecasting"
        },
        "model_params": model_params,
        "data_params": {
            "train_test_split": 0.8,
            "random_state": 42,
            "data_source": "synthetic_temperature_data"
        }
    }
    return config


# Convenience function for quick setup
def setup_mlflow_tracking(experiment_name: str = "forecasting-models") -> ExperimentTracker:
    """
    Quick setup function for MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Configured ExperimentTracker instance
    """
    return ExperimentTracker(experiment_name)


if __name__ == "__main__":
    # Example usage
    tracker = setup_mlflow_tracking()
    print("MLflow tracking setup complete!")
    print(f"Tracking URI: {tracker.config.tracking_uri}")
    print(f"Experiment: {tracker.config.experiment_name}")
