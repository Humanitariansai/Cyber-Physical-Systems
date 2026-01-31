"""
Basic Forecaster - Baseline models for cold chain temperature prediction.
Implements Moving Average and Exponential Smoothing as baseline comparisons.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BasicForecasterConfig:
    """Configuration for basic forecasters."""
    window_size: int = 10
    alpha: float = 0.3  # Exponential smoothing factor
    prediction_horizons: Tuple[int, ...] = (30, 60)


class MovingAverageForecaster:
    """Simple Moving Average forecaster as baseline."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: List[float] = []
        self.is_trained = False

    def train(self, data: List[float]) -> None:
        """Store historical data for prediction."""
        self.history = list(data)
        self.is_trained = True

    def predict(self, horizon: int = 30) -> Tuple[float, float]:
        """Predict future temperature using moving average."""
        if not self.is_trained or len(self.history) < self.window_size:
            raise ValueError("Insufficient data for prediction")

        window = self.history[-self.window_size:]
        prediction = sum(window) / len(window)
        std = np.std(window)
        confidence = max(0.5, 1.0 - (horizon / 120) - (std / 5.0))
        return round(prediction, 2), round(confidence, 2)

    def update(self, value: float) -> None:
        """Add new observation."""
        self.history.append(value)


class ExponentialSmoothingForecaster:
    """Exponential Smoothing forecaster."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.history: List[float] = []
        self.smoothed: Optional[float] = None
        self.is_trained = False

    def train(self, data: List[float]) -> None:
        """Train exponential smoothing model."""
        self.history = list(data)
        self.smoothed = data[0]
        for val in data[1:]:
            self.smoothed = self.alpha * val + (1 - self.alpha) * self.smoothed
        self.is_trained = True

    def predict(self, horizon: int = 30) -> Tuple[float, float]:
        """Predict using last smoothed value."""
        if not self.is_trained or self.smoothed is None:
            raise ValueError("Model not trained")

        trend = 0.0
        if len(self.history) >= 2:
            recent = self.history[-10:]
            trend = (recent[-1] - recent[0]) / len(recent)

        prediction = self.smoothed + trend * (horizon / 10)
        confidence = max(0.4, 0.9 - (horizon / 100))
        return round(prediction, 2), round(confidence, 2)

    def update(self, value: float) -> None:
        """Update with new observation."""
        self.history.append(value)
        if self.smoothed is not None:
            self.smoothed = self.alpha * value + (1 - self.alpha) * self.smoothed


class BasicForecaster:
    """Combined baseline forecaster using MA and ES ensemble."""

    def __init__(self, config: Optional[BasicForecasterConfig] = None):
        self.config = config or BasicForecasterConfig()
        self.ma = MovingAverageForecaster(self.config.window_size)
        self.es = ExponentialSmoothingForecaster(self.config.alpha)
        self.is_trained = False

    def train(self, data: List[float]) -> dict:
        """Train both baseline models."""
        self.ma.train(data)
        self.es.train(data)
        self.is_trained = True
        return {"status": "trained", "data_points": len(data)}

    def predict(self, horizon: int = 30) -> Tuple[float, float]:
        """Ensemble prediction from MA and ES."""
        ma_pred, ma_conf = self.ma.predict(horizon)
        es_pred, es_conf = self.es.predict(horizon)

        prediction = 0.4 * ma_pred + 0.6 * es_pred
        confidence = 0.4 * ma_conf + 0.6 * es_conf
        return round(prediction, 2), round(confidence, 2)

    def predict_all_horizons(self) -> dict:
        """Predict for all configured horizons."""
        results = {}
        for h in self.config.prediction_horizons:
            pred, conf = self.predict(h)
            results[h] = {"prediction": pred, "confidence": conf}
        return results

    def update(self, value: float) -> None:
        """Update models with new observation."""
        self.ma.update(value)
        self.es.update(value)
