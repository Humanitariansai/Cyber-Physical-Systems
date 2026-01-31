"""
Ensemble Forecaster - Combines LSTM, GRU, and XGBoost predictions
with weighted averaging for improved accuracy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class EnsembleConfig:
    """Configuration for ensemble forecaster."""
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        "lstm": 0.4,
        "gru": 0.35,
        "xgboost": 0.25
    })
    adaptive_weights: bool = True
    min_weight: float = 0.1
    prediction_horizons: Tuple[int, ...] = (30, 60)


class EnsembleForecaster:
    """Combines multiple model predictions with weighted averaging."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = dict(self.config.default_weights)
        self.error_history: Dict[str, List[float]] = {}

    def add_model(self, name: str, model: Any, weight: float = None) -> None:
        """Register a forecasting model."""
        self.models[name] = model
        if weight is not None:
            self.weights[name] = weight
        elif name not in self.weights:
            self.weights[name] = 1.0 / max(len(self.models), 1)
        self.error_history[name] = []

    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        self.models.pop(name, None)
        self.weights.pop(name, None)
        self.error_history.pop(name, None)

    def get_weights(self) -> Dict[str, float]:
        """Return current normalized weights."""
        total = sum(self.weights[n] for n in self.models)
        if total == 0:
            return {n: 1.0 / len(self.models) for n in self.models}
        return {n: self.weights[n] / total for n in self.models}

    def predict_ensemble(self, data: List[float], horizon: int = 30) -> Tuple[float, float]:
        """Generate weighted ensemble prediction."""
        if not self.models:
            raise ValueError("No models registered")

        predictions = {}
        confidences = {}

        for name, model in self.models.items():
            try:
                pred, conf = model.predict(horizon)
                predictions[name] = pred
                confidences[name] = conf
            except Exception:
                continue

        if not predictions:
            raise ValueError("All models failed to predict")

        normalized = self.get_weights()
        active_models = [n for n in predictions if n in normalized]
        weight_sum = sum(normalized[n] for n in active_models)

        ensemble_pred = sum(
            predictions[n] * normalized[n] / weight_sum
            for n in active_models
        )
        ensemble_conf = sum(
            confidences[n] * normalized[n] / weight_sum
            for n in active_models
        )

        return round(ensemble_pred, 2), round(min(ensemble_conf, 0.99), 2)

    def update_weights(self, actuals: Dict[int, float]) -> None:
        """Update weights based on recent prediction errors."""
        if not self.config.adaptive_weights:
            return

        for name, model in self.models.items():
            errors = []
            for horizon, actual in actuals.items():
                try:
                    pred, _ = model.predict(horizon)
                    errors.append(abs(pred - actual))
                except Exception:
                    errors.append(5.0)

            if errors:
                avg_error = sum(errors) / len(errors)
                self.error_history[name].append(avg_error)
                self.weights[name] = max(
                    self.config.min_weight,
                    1.0 / (1.0 + avg_error)
                )

    def predict_all_horizons(self, data: List[float]) -> Dict[int, Dict]:
        """Predict for all configured horizons."""
        results = {}
        for h in self.config.prediction_horizons:
            pred, conf = self.predict_ensemble(data, h)
            results[h] = {"prediction": pred, "confidence": conf}
        return results

    def get_model_contributions(self, horizon: int = 30) -> Dict[str, Dict]:
        """Get individual model predictions for comparison."""
        contributions = {}
        for name, model in self.models.items():
            try:
                pred, conf = model.predict(horizon)
                contributions[name] = {
                    "prediction": pred,
                    "confidence": conf,
                    "weight": self.get_weights().get(name, 0)
                }
            except Exception:
                contributions[name] = {"error": "prediction failed"}
        return contributions
