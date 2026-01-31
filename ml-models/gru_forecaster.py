"""
GRU Forecaster for Cold Chain Temperature Prediction
Implements Gated Recurrent Unit networks for 30-60 minute ahead forecasts.
GRU is faster than LSTM with comparable performance for time series.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# TensorFlow import with fallback
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. GRU forecaster will use fallback mode.")


@dataclass
class GRUConfig:
    """Configuration for GRU forecaster."""
    # Sequence parameters
    sequence_length: int = 60  # Input sequence length (minutes of history)
    prediction_horizons: Tuple[int, ...] = (30, 60)  # Minutes ahead to predict

    # Model architecture
    gru_units: List[int] = None  # Units per GRU layer
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1
    use_bidirectional: bool = False

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5

    # Feature settings
    use_humidity: bool = True

    def __post_init__(self):
        if self.gru_units is None:
            self.gru_units = [64, 32]


class GRUForecaster:
    """
    GRU-based temperature forecaster for cold chain monitoring.

    GRU (Gated Recurrent Unit) is similar to LSTM but with fewer parameters,
    making it faster to train while maintaining good performance on time series.
    """

    def __init__(self, config: Optional[GRUConfig] = None):
        self.config = config or GRUConfig()
        self.models: Dict[int, any] = {}
        self.scalers: Dict[str, float] = {}
        self.is_trained = False
        self._feature_dim = 1

        if not TF_AVAILABLE:
            logger.warning("Running in fallback mode without TensorFlow")

    def _build_model(self) -> any:
        """Build GRU model."""
        if not TF_AVAILABLE:
            return None

        model = Sequential()
        input_shape = (self.config.sequence_length, self._feature_dim)

        for i, units in enumerate(self.config.gru_units):
            return_sequences = i < len(self.config.gru_units) - 1

            if self.config.use_bidirectional:
                if i == 0:
                    model.add(Bidirectional(
                        GRU(units, return_sequences=return_sequences,
                            recurrent_dropout=self.config.recurrent_dropout),
                        input_shape=input_shape
                    ))
                else:
                    model.add(Bidirectional(
                        GRU(units, return_sequences=return_sequences,
                            recurrent_dropout=self.config.recurrent_dropout)
                    ))
            else:
                if i == 0:
                    model.add(GRU(units, return_sequences=return_sequences,
                                 recurrent_dropout=self.config.recurrent_dropout,
                                 input_shape=input_shape))
                else:
                    model.add(GRU(units, return_sequences=return_sequences,
                                 recurrent_dropout=self.config.recurrent_dropout))

            model.add(Dropout(self.config.dropout_rate))

        model.add(Dense(1))

        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _prepare_sequences(self, data: np.ndarray, horizon: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input sequences and targets."""
        X, y = [], []
        seq_len = self.config.sequence_length

        for i in range(len(data) - seq_len - horizon):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len + horizon - 1, 0])

        return np.array(X), np.array(y)

    def _normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Min-max normalization."""
        if fit or 'min' not in self.scalers:
            self.scalers['min'] = float(data[:, 0].min())
            self.scalers['max'] = float(data[:, 0].max())

        range_val = self.scalers['max'] - self.scalers['min']
        if range_val == 0:
            range_val = 1.0

        normalized = data.copy()
        normalized[:, 0] = (data[:, 0] - self.scalers['min']) / range_val
        return normalized

    def _denormalize(self, value: float) -> float:
        """Denormalize temperature."""
        range_val = self.scalers['max'] - self.scalers['min']
        return value * range_val + self.scalers['min']

    def train(self, temperatures: np.ndarray,
              humidity: Optional[np.ndarray] = None,
              verbose: int = 1) -> Dict:
        """Train GRU models for all horizons."""
        if not TF_AVAILABLE:
            self.is_trained = True
            return {"status": "fallback"}

        # Prepare data
        if humidity is not None and self.config.use_humidity:
            data = np.column_stack([temperatures, humidity])
            self._feature_dim = 2
        else:
            data = temperatures.reshape(-1, 1)
            self._feature_dim = 1

        normalized = self._normalize(data, fit=True)
        histories = {}

        for horizon in self.config.prediction_horizons:
            logger.info(f"Training GRU for {horizon}-minute horizon...")

            X, y = self._prepare_sequences(normalized, horizon)
            if len(X) < self.config.batch_size * 2:
                continue

            model = self._build_model()

            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.reduce_lr_patience
                )
            ]

            history = model.fit(
                X, y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=verbose
            )

            self.models[horizon] = model
            histories[horizon] = {
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_mae': history.history['mae'][-1],
                'final_val_mae': history.history['val_mae'][-1]
            }

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

        if not TF_AVAILABLE or horizon not in self.models:
            return self._fallback_predict(temperatures, horizon)

        if len(temperatures) < self.config.sequence_length:
            return self._fallback_predict(temperatures, horizon)

        temps = temperatures[-self.config.sequence_length:]

        if humidity is not None and self.config.use_humidity:
            hum = humidity[-self.config.sequence_length:]
            data = np.column_stack([temps, hum])
        else:
            data = temps.reshape(-1, 1)

        normalized = self._normalize(data)
        X = normalized.reshape(1, self.config.sequence_length, self._feature_dim)

        pred = self.models[horizon].predict(X, verbose=0)[0, 0]
        predicted_temp = self._denormalize(pred)

        # Confidence calculation
        variance = np.var(temps[-10:]) if len(temps) >= 10 else 1.0
        confidence = max(0.3, min(0.95, 0.9 - variance * 0.05 - horizon / 150))

        return float(predicted_temp), float(confidence)

    def _fallback_predict(self, temperatures: np.ndarray,
                          horizon: int) -> Tuple[float, float]:
        """Exponential smoothing fallback."""
        if len(temperatures) < 2:
            return float(temperatures[-1]) if len(temperatures) > 0 else 5.0, 0.3

        alpha, beta = 0.3, 0.1
        level = temperatures[0]
        trend = temperatures[1] - temperatures[0]

        for temp in temperatures[1:]:
            prev_level = level
            level = alpha * temp + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        predicted = level + horizon * trend
        variance = np.var(temperatures[-10:]) if len(temperatures) >= 10 else 1.0
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

    def save(self, path: str):
        """Save models and scalers."""
        if not TF_AVAILABLE:
            return

        import json
        for horizon, model in self.models.items():
            model.save(f"{path}_gru_{horizon}min.h5")

        with open(f"{path}_gru_scalers.json", 'w') as f:
            json.dump(self.scalers, f)

    def load(self, path: str):
        """Load models and scalers."""
        if not TF_AVAILABLE:
            return

        import json
        for horizon in self.config.prediction_horizons:
            try:
                self.models[horizon] = load_model(f"{path}_gru_{horizon}min.h5")
            except Exception as e:
                logger.warning(f"Could not load GRU model for {horizon}min: {e}")

        try:
            with open(f"{path}_gru_scalers.json", 'r') as f:
                self.scalers = json.load(f)
            self.is_trained = True
        except Exception as e:
            logger.warning(f"Could not load scalers: {e}")


def quick_gru_forecast(temperatures: List[float], horizon: int = 30) -> Tuple[float, float]:
    """Quick forecast using fallback method."""
    forecaster = GRUForecaster()
    return forecaster._fallback_predict(np.array(temperatures), horizon)
