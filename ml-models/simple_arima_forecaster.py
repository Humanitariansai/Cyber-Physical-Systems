"""
ARIMA Forecaster for cold chain temperature prediction.
Uses statsmodels ARIMA with fallback to simple trend extrapolation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@dataclass
class ARIMAConfig:
    """ARIMA model configuration."""
    order: tuple = (2, 1, 2)
    seasonal_order: tuple = (1, 0, 1, 60)
    use_seasonal: bool = False
    prediction_horizons: tuple = (30, 60)


class ARIMAForecaster:
    """ARIMA-based temperature forecaster."""

    def __init__(self, config: Optional[ARIMAConfig] = None):
        self.config = config or ARIMAConfig()
        self.model = None
        self.model_fit = None
        self.history: List[float] = []
        self.is_trained = False

    def train(self, data: List[float]) -> dict:
        """Train ARIMA model on historical data."""
        self.history = list(data)

        if STATSMODELS_AVAILABLE and len(data) >= 30:
            try:
                model = ARIMA(data, order=self.config.order)
                self.model_fit = model.fit()
                self.is_trained = True
                return {
                    "status": "trained",
                    "method": "ARIMA",
                    "order": self.config.order,
                    "aic": round(self.model_fit.aic, 2),
                    "data_points": len(data)
                }
            except Exception as e:
                return self._train_fallback(data, str(e))
        else:
            return self._train_fallback(data, "statsmodels not available or insufficient data")

    def _train_fallback(self, data: List[float], reason: str) -> dict:
        """Fallback to linear trend model."""
        self.history = list(data)
        self.is_trained = True
        self.model_fit = None
        return {
            "status": "trained",
            "method": "linear_trend_fallback",
            "reason": reason,
            "data_points": len(data)
        }

    def predict(self, horizon: int = 30) -> Tuple[float, float]:
        """Predict future temperature."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self.model_fit is not None and STATSMODELS_AVAILABLE:
            try:
                forecast = self.model_fit.forecast(steps=horizon)
                prediction = float(forecast.iloc[-1])
                confidence = max(0.5, 0.9 - (horizon / 200))
                return round(prediction, 2), round(confidence, 2)
            except Exception:
                pass

        # Fallback: linear trend extrapolation
        recent = self.history[-20:]
        if len(recent) < 2:
            return self.history[-1], 0.5

        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)
        prediction = coeffs[0] * (len(recent) + horizon) + coeffs[1]
        confidence = max(0.3, 0.75 - (horizon / 150))
        return round(float(prediction), 2), round(confidence, 2)

    def update(self, value: float) -> None:
        """Add new observation and optionally retrain."""
        self.history.append(value)

    def get_model_summary(self) -> str:
        """Get model summary string."""
        if self.model_fit is not None:
            return str(self.model_fit.summary())
        return "Using linear trend fallback model"
