"""
MLflow Experiment Tracking Integration for Cold Chain ML Models.
Provides a tracker class for logging parameters, metrics, and model artifacts.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time
import json
import os

try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class ExperimentConfig:
    """MLflow experiment configuration."""
    experiment_name: str = "cold-chain-forecasting"
    tracking_uri: str = "mlruns"
    artifact_location: str = "mlruns/artifacts"


class MLflowTracker:
    """Wrapper for MLflow experiment tracking."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.active_run = None
        self.run_metrics: Dict[str, float] = {}
        self.run_params: Dict[str, Any] = {}

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)

    @property
    def is_available(self) -> bool:
        return MLFLOW_AVAILABLE

    def start_run(self, run_name: str = None) -> str:
        """Start a new MLflow run."""
        self.run_metrics = {}
        self.run_params = {}

        if MLFLOW_AVAILABLE:
            self.active_run = mlflow.start_run(run_name=run_name)
            return self.active_run.info.run_id
        return f"local-{int(time.time())}"

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log training parameters."""
        self.run_params.update(params)
        if MLFLOW_AVAILABLE and self.active_run:
            mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log training metrics."""
        self.run_metrics.update(metrics)
        if MLFLOW_AVAILABLE and self.active_run:
            mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, model_name: str) -> None:
        """Log a trained model artifact."""
        if MLFLOW_AVAILABLE and self.active_run:
            try:
                mlflow.keras.log_model(model, model_name)
            except Exception:
                mlflow.log_dict({"model_type": model_name}, f"{model_name}/info.json")

    def log_artifact(self, filepath: str) -> None:
        """Log a file artifact."""
        if MLFLOW_AVAILABLE and self.active_run:
            mlflow.log_artifact(filepath)

    def end_run(self, status: str = "FINISHED") -> Dict[str, Any]:
        """End the current run and return summary."""
        summary = {
            "params": self.run_params,
            "metrics": self.run_metrics,
            "status": status,
        }

        if MLFLOW_AVAILABLE and self.active_run:
            summary["run_id"] = self.active_run.info.run_id
            mlflow.end_run(status=status)
            self.active_run = None

        return summary

    def get_best_run(self, metric: str = "mae_30min") -> Optional[Dict]:
        """Get the best run by a specific metric."""
        if not MLFLOW_AVAILABLE:
            return None

        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if not experiment:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )

        if len(runs) > 0:
            return runs.iloc[0].to_dict()
        return None
