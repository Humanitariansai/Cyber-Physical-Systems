"""
ML Models for Cold Chain Temperature Forecasting

This module provides machine learning models for predictive temperature
monitoring in pharmaceutical cold chain applications.

Models:
- LSTMForecaster: Long Short-Term Memory networks for 30-60 minute forecasts
- GRUForecaster: Gated Recurrent Unit networks (faster than LSTM)
- XGBoostForecaster: Gradient boosting with feature engineering

All models include:
- TensorFlow/XGBoost with fallback to statistical methods
- Min-max normalization
- Configurable prediction horizons
- Confidence estimation
"""

__version__ = "0.1.0"

from .lstm_forecaster import (
    LSTMForecaster,
    LSTMConfig,
    quick_forecast,
)

from .gru_forecaster import (
    GRUForecaster,
    GRUConfig,
    quick_gru_forecast,
)

from .xgboost_forecaster import (
    XGBoostForecaster,
    XGBoostConfig,
    quick_xgboost_forecast,
)

__all__ = [
    # LSTM
    "LSTMForecaster",
    "LSTMConfig",
    "quick_forecast",
    # GRU
    "GRUForecaster",
    "GRUConfig",
    "quick_gru_forecast",
    # XGBoost
    "XGBoostForecaster",
    "XGBoostConfig",
    "quick_xgboost_forecast",
]
