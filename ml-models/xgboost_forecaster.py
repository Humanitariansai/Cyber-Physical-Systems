"""
XGBoost Forecaster for Cold Chain Temperature Prediction
Gradient boosting approach for time series forecasting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# XGBoost import with fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available. Forecaster will use fallback mode.")


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost forecaster."""
    # Feature engineering
    lag_features: List[int] = None  # Lag values to use as features
    rolling_windows: List[int] = None  # Rolling statistics windows
    prediction_horizons: Tuple[int, ...] = (30, 60)

    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1

    # Training
    early_stopping_rounds: int = 10
    validation_split: float = 0.2

    def __post_init__(self):
        if self.lag_features is None:
            self.lag_features = [1, 2, 3, 5, 10, 15, 30, 60]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 30]


class XGBoostForecaster:
    """
    XGBoost-based temperature forecaster.

    Uses gradient boosting with engineered lag features for time series prediction.
    Generally faster to train than deep learning approaches.
    """

    def __init__(self, config: Optional[XGBoostConfig] = None):
        self.config = config or XGBoostConfig()
        self.models: Dict[int, any] = {}
        self.scalers: Dict[str, float] = {}
        self.is_trained = False
        self._feature_names: List[str] = []

        if not XGB_AVAILABLE:
            logger.warning("Running in fallback mode without XGBoost")

    def _create_features(self, temperatures: np.ndarray,
                         humidity: Optional[np.ndarray] = None) -> np.ndarray:
        """Create lag and rolling features from time series."""
        n = len(temperatures)
        features = []
        feature_names = []

        # Lag features
        for lag in self.config.lag_features:
            if lag < n:
                lagged = np.zeros(n)
                lagged[lag:] = temperatures[:-lag]
                lagged[:lag] = temperatures[0]
                features.append(lagged)
                feature_names.append(f"temp_lag_{lag}")

        # Rolling statistics
        for window in self.config.rolling_windows:
            if window < n:
                # Rolling mean
                rolling_mean = np.zeros(n)
                for i in range(n):
                    start = max(0, i - window + 1)
                    rolling_mean[i] = np.mean(temperatures[start:i + 1])
                features.append(rolling_mean)
                feature_names.append(f"temp_rolling_mean_{window}")

                # Rolling std
                rolling_std = np.zeros(n)
                for i in range(n):
                    start = max(0, i - window + 1)
                    rolling_std[i] = np.std(temperatures[start:i + 1])
                features.append(rolling_std)
                feature_names.append(f"temp_rolling_std_{window}")

                # Rolling min/max
                rolling_min = np.zeros(n)
                rolling_max = np.zeros(n)
                for i in range(n):
                    start = max(0, i - window + 1)
                    rolling_min[i] = np.min(temperatures[start:i + 1])
                    rolling_max[i] = np.max(temperatures[start:i + 1])
                features.append(rolling_min)
                features.append(rolling_max)
                feature_names.extend([f"temp_rolling_min_{window}",
                                     f"temp_rolling_max_{window}"])

        # Add humidity if available
        if humidity is not None:
            features.append(humidity)
            feature_names.append("humidity")

            # Humidity lag
            for lag in [1, 5, 10]:
                if lag < n:
                    lagged = np.zeros(n)
                    lagged[lag:] = humidity[:-lag]
                    lagged[:lag] = humidity[0]
                    features.append(lagged)
                    feature_names.append(f"humidity_lag_{lag}")

        self._feature_names = feature_names
        return np.column_stack(features) if features else temperatures.reshape(-1, 1)

    def _prepare_data(self, temperatures: np.ndarray,
                      humidity: Optional[np.ndarray],
                      horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        X = self._create_features(temperatures, humidity)

        # Target: temperature 'horizon' steps ahead
        y = np.zeros(len(temperatures))
        y[:-horizon] = temperatures[horizon:]
        y[-horizon:] = temperatures[-1]

        # Remove last 'horizon' samples (no valid target)
        valid_idx = len(temperatures) - horizon
        return X[:valid_idx], y[:valid_idx]

    def train(self, temperatures: np.ndarray,
              humidity: Optional[np.ndarray] = None,
              verbose: int = 1) -> Dict:
        """Train XGBoost models for all horizons."""
        if not XGB_AVAILABLE:
            self.is_trained = True
            return {"status": "fallback"}

        # Store scaling info
        self.scalers['temp_min'] = float(temperatures.min())
        self.scalers['temp_max'] = float(temperatures.max())

        histories = {}

        for horizon in self.config.prediction_horizons:
            logger.info(f"Training XGBoost for {horizon}-minute horizon...")

            X, y = self._prepare_data(temperatures, humidity, horizon)

            # Split for validation
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train,
                                feature_names=self._feature_names)
            dval = xgb.DMatrix(X_val, label=y_val,
                              feature_names=self._feature_names)

            # Parameters
            params = {
                'objective': 'reg:squarederror',
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'min_child_weight': self.config.min_child_weight,
                'eval_metric': 'mae',
                'seed': 42
            }

            # Train
            evals = [(dtrain, 'train'), (dval, 'val')]
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.config.n_estimators,
                evals=evals,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=verbose > 0
            )

            self.models[horizon] = model

            # Evaluate
            y_pred = model.predict(dval)
            mae = np.mean(np.abs(y_val - y_pred))
            histories[horizon] = {
                'val_mae': float(mae),
                'best_iteration': model.best_iteration,
                'feature_importance': dict(zip(
                    self._feature_names,
                    [float(x) for x in model.get_score(importance_type='gain').values()]
                ))
            }

            logger.info(f"  Validation MAE: {mae:.4f}")

        self.is_trained = True
        return histories

    def predict(self, temperatures: np.ndarray,
                humidity: Optional[np.ndarray] = None,
                horizon: int = 30) -> Tuple[float, float]:
        """Predict temperature for given horizon."""
        if not self.is_trained:
            return self._fallback_predict(temperatures, horizon)

        if horizon not in self.config.prediction_horizons:
            horizon = min(self.config.prediction_horizons,
                         key=lambda x: abs(x - horizon))

        if not XGB_AVAILABLE or horizon not in self.models:
            return self._fallback_predict(temperatures, horizon)

        # Need enough data for feature creation
        min_required = max(self.config.lag_features) + 1
        if len(temperatures) < min_required:
            return self._fallback_predict(temperatures, horizon)

        # Create features for last point
        X = self._create_features(temperatures, humidity)
        X_last = X[-1:, :]

        dtest = xgb.DMatrix(X_last, feature_names=self._feature_names)
        prediction = self.models[horizon].predict(dtest)[0]

        # Confidence based on recent variance
        variance = np.var(temperatures[-10:]) if len(temperatures) >= 10 else 1.0
        confidence = max(0.3, min(0.95, 0.9 - variance * 0.05 - horizon / 150))

        return float(prediction), float(confidence)

    def _fallback_predict(self, temperatures: np.ndarray,
                          horizon: int) -> Tuple[float, float]:
        """Simple moving average fallback."""
        if len(temperatures) < 2:
            return float(temperatures[-1]) if len(temperatures) > 0 else 5.0, 0.3

        # Weighted moving average with trend
        recent = temperatures[-min(10, len(temperatures)):]
        weights = np.arange(1, len(recent) + 1)
        weighted_avg = np.average(recent, weights=weights)

        # Simple trend
        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / len(recent)
            predicted = weighted_avg + trend * (horizon / 10)
        else:
            predicted = weighted_avg

        variance = np.var(recent) if len(recent) > 1 else 1.0
        confidence = max(0.3, 0.8 - variance * 0.1 - horizon / 100)

        return float(predicted), float(confidence)

    def predict_all_horizons(self, temperatures: np.ndarray,
                             humidity: Optional[np.ndarray] = None
                            ) -> Dict[int, Tuple[float, float]]:
        """Predict for all configured horizons."""
        return {
            horizon: self.predict(temperatures, humidity, horizon)
            for horizon in self.config.prediction_horizons
        }

    def get_feature_importance(self, horizon: int = 30) -> Dict[str, float]:
        """Get feature importance for a model."""
        if not XGB_AVAILABLE or horizon not in self.models:
            return {}

        importance = self.models[horizon].get_score(importance_type='gain')
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str):
        """Save models and config."""
        if not XGB_AVAILABLE:
            return

        for horizon, model in self.models.items():
            model.save_model(f"{path}_xgb_{horizon}min.json")

        with open(f"{path}_xgb_config.json", 'w') as f:
            json.dump({
                'scalers': self.scalers,
                'feature_names': self._feature_names
            }, f)

    def load(self, path: str):
        """Load models and config."""
        if not XGB_AVAILABLE:
            return

        for horizon in self.config.prediction_horizons:
            try:
                model = xgb.Booster()
                model.load_model(f"{path}_xgb_{horizon}min.json")
                self.models[horizon] = model
            except Exception as e:
                logger.warning(f"Could not load XGBoost model: {e}")

        try:
            with open(f"{path}_xgb_config.json", 'r') as f:
                config = json.load(f)
                self.scalers = config.get('scalers', {})
                self._feature_names = config.get('feature_names', [])
            self.is_trained = True
        except Exception as e:
            logger.warning(f"Could not load config: {e}")


def quick_xgb_forecast(temperatures: List[float], horizon: int = 30) -> Tuple[float, float]:
    """Quick forecast using fallback."""
    forecaster = XGBoostForecaster()
    return forecaster._fallback_predict(np.array(temperatures), horizon)
