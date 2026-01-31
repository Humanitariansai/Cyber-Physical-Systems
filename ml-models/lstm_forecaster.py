"""
LSTM Forecaster for Cold Chain Temperature Prediction
Implements Long Short-Term Memory networks for 30-60 minute ahead forecasts.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# TensorFlow import with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM forecaster will use fallback mode.")


@dataclass
class LSTMConfig:
    """Configuration for LSTM forecaster."""
    # Sequence parameters
    sequence_length: int = 60  # Input sequence length (minutes of history)
    prediction_horizons: Tuple[int, ...] = (30, 60)  # Minutes ahead to predict

    # Model architecture
    lstm_units: List[int] = None  # Units per LSTM layer
    dropout_rate: float = 0.2
    use_bidirectional: bool = False

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10

    # Feature settings
    use_humidity: bool = True
    use_time_features: bool = True

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [64, 32]


class LSTMForecaster:
    """
    LSTM-based temperature forecaster for cold chain monitoring.

    Supports multiple prediction horizons (30, 60 minutes) and provides
    confidence estimates based on prediction variance.
    """

    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.models: Dict[int, any] = {}  # One model per horizon
        self.scalers: Dict[str, any] = {}
        self.is_trained = False
        self._feature_dim = 1  # Will be updated based on features

        if not TF_AVAILABLE:
            logger.warning("Running in fallback mode without TensorFlow")

    def _build_model(self, horizon: int) -> any:
        """Build LSTM model for a specific prediction horizon."""
        if not TF_AVAILABLE:
            return None

        model = Sequential()

        # Input shape: (sequence_length, features)
        input_shape = (self.config.sequence_length, self._feature_dim)

        # Add LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1

            if self.config.use_bidirectional:
                model.add(Bidirectional(
                    LSTM(units, return_sequences=return_sequences),
                    input_shape=input_shape if i == 0 else None
                ))
            else:
                if i == 0:
                    model.add(LSTM(units, return_sequences=return_sequences,
                                  input_shape=input_shape))
                else:
                    model.add(LSTM(units, return_sequences=return_sequences))

            model.add(Dropout(self.config.dropout_rate))

        # Output layer - predict single temperature value
        model.add(Dense(1))

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _prepare_sequences(self, data: np.ndarray, horizon: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input sequences and targets for training."""
        X, y = [], []
        seq_len = self.config.sequence_length

        # Calculate steps for horizon (assuming 1-minute intervals)
        steps_ahead = horizon

        for i in range(len(data) - seq_len - steps_ahead):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len + steps_ahead - 1, 0])  # Temperature only

        return np.array(X), np.array(y)

    def _normalize_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using min-max scaling."""
        if fit or 'temp_min' not in self.scalers:
            self.scalers['temp_min'] = data[:, 0].min()
            self.scalers['temp_max'] = data[:, 0].max()

        temp_range = self.scalers['temp_max'] - self.scalers['temp_min']
        if temp_range == 0:
            temp_range = 1.0

        normalized = data.copy()
        normalized[:, 0] = (data[:, 0] - self.scalers['temp_min']) / temp_range

        return normalized

    def _denormalize_temp(self, normalized_temp: float) -> float:
        """Denormalize temperature prediction."""
        temp_range = self.scalers['temp_max'] - self.scalers['temp_min']
        return normalized_temp * temp_range + self.scalers['temp_min']

    def train(self, temperatures: np.ndarray, humidity: Optional[np.ndarray] = None,
              verbose: int = 1) -> Dict[str, any]:
        """
        Train LSTM models for all prediction horizons.

        Args:
            temperatures: Array of temperature readings
            humidity: Optional array of humidity readings
            verbose: Verbosity level (0, 1, 2)

        Returns:
            Training history and metrics
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, using fallback training")
            self.is_trained = True
            return {"status": "fallback", "message": "Using statistical fallback"}

        # Prepare features
        if humidity is not None and self.config.use_humidity:
            data = np.column_stack([temperatures, humidity])
            self._feature_dim = 2
        else:
            data = temperatures.reshape(-1, 1)
            self._feature_dim = 1

        # Normalize
        normalized_data = self._normalize_data(data, fit=True)

        histories = {}

        # Train a model for each horizon
        for horizon in self.config.prediction_horizons:
            logger.info(f"Training LSTM for {horizon}-minute horizon...")

            # Prepare sequences
            X, y = self._prepare_sequences(normalized_data, horizon)

            if len(X) < self.config.batch_size * 2:
                logger.warning(f"Insufficient data for {horizon}-min model")
                continue

            # Build model
            model = self._build_model(horizon)

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                )
            ]

            # Train
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
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae']
            }

            logger.info(f"  Final validation MAE: {history.history['val_mae'][-1]:.4f}")

        self.is_trained = True
        return histories

    def predict(self, recent_temperatures: np.ndarray,
                recent_humidity: Optional[np.ndarray] = None,
                horizon: int = 30) -> Tuple[float, float]:
        """
        Predict future temperature.

        Args:
            recent_temperatures: Recent temperature readings (at least sequence_length)
            recent_humidity: Optional humidity readings
            horizon: Minutes ahead to predict (30 or 60)

        Returns:
            (predicted_temperature, confidence)
        """
        if not self.is_trained:
            logger.warning("Model not trained, using fallback prediction")
            return self._fallback_predict(recent_temperatures, horizon)

        if horizon not in self.config.prediction_horizons:
            logger.warning(f"Horizon {horizon} not supported, using closest")
            horizon = min(self.config.prediction_horizons,
                         key=lambda x: abs(x - horizon))

        if not TF_AVAILABLE or horizon not in self.models:
            return self._fallback_predict(recent_temperatures, horizon)

        # Prepare input
        if len(recent_temperatures) < self.config.sequence_length:
            return self._fallback_predict(recent_temperatures, horizon)

        # Take last sequence_length readings
        temps = recent_temperatures[-self.config.sequence_length:]

        if recent_humidity is not None and self.config.use_humidity:
            humidity = recent_humidity[-self.config.sequence_length:]
            data = np.column_stack([temps, humidity])
        else:
            data = temps.reshape(-1, 1)

        # Normalize
        normalized = self._normalize_data(data, fit=False)

        # Reshape for LSTM: (1, sequence_length, features)
        X = normalized.reshape(1, self.config.sequence_length, self._feature_dim)

        # Predict
        model = self.models[horizon]
        normalized_pred = model.predict(X, verbose=0)[0, 0]

        # Denormalize
        predicted_temp = self._denormalize_temp(normalized_pred)

        # Calculate confidence based on recent variance
        variance = np.var(temps[-10:]) if len(temps) >= 10 else np.var(temps)
        base_confidence = 0.9 - (variance * 0.05)
        horizon_penalty = horizon / 150.0
        confidence = max(0.3, min(0.95, base_confidence - horizon_penalty))

        return float(predicted_temp), float(confidence)

    def _fallback_predict(self, temperatures: np.ndarray,
                          horizon: int) -> Tuple[float, float]:
        """Fallback prediction using exponential smoothing."""
        if len(temperatures) < 2:
            return temperatures[-1] if len(temperatures) > 0 else 5.0, 0.3

        # Double exponential smoothing
        alpha, beta = 0.3, 0.1
        level = temperatures[0]
        trend = temperatures[1] - temperatures[0]

        for temp in temperatures[1:]:
            prev_level = level
            level = alpha * temp + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        predicted = level + horizon * trend

        # Confidence
        variance = np.var(temperatures[-10:]) if len(temperatures) >= 10 else 1.0
        confidence = max(0.3, 0.8 - variance * 0.1 - horizon / 100)

        return float(predicted), float(confidence)

    def predict_multi_horizon(self, recent_temperatures: np.ndarray,
                              recent_humidity: Optional[np.ndarray] = None
                             ) -> Dict[int, Tuple[float, float]]:
        """
        Predict for all configured horizons.

        Returns:
            Dict mapping horizon -> (predicted_temp, confidence)
        """
        predictions = {}
        for horizon in self.config.prediction_horizons:
            predictions[horizon] = self.predict(
                recent_temperatures, recent_humidity, horizon
            )
        return predictions

    def save_model(self, path: str):
        """Save trained models to disk."""
        if not TF_AVAILABLE:
            logger.warning("Cannot save models - TensorFlow not available")
            return

        for horizon, model in self.models.items():
            model_path = f"{path}_horizon_{horizon}.h5"
            model.save(model_path)
            logger.info(f"Saved model: {model_path}")

        # Save scalers
        import json
        with open(f"{path}_scalers.json", 'w') as f:
            json.dump(self.scalers, f)

    def load_model(self, path: str):
        """Load trained models from disk."""
        if not TF_AVAILABLE:
            logger.warning("Cannot load models - TensorFlow not available")
            return

        import json

        for horizon in self.config.prediction_horizons:
            model_path = f"{path}_horizon_{horizon}.h5"
            try:
                self.models[horizon] = load_model(model_path)
                logger.info(f"Loaded model: {model_path}")
            except Exception as e:
                logger.warning(f"Could not load {model_path}: {e}")

        # Load scalers
        try:
            with open(f"{path}_scalers.json", 'r') as f:
                self.scalers = json.load(f)
            self.is_trained = True
        except Exception as e:
            logger.warning(f"Could not load scalers: {e}")

    def get_model_summary(self, horizon: int = 30) -> str:
        """Get model architecture summary."""
        if not TF_AVAILABLE or horizon not in self.models:
            return "Model not available"

        import io
        stream = io.StringIO()
        self.models[horizon].summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


# Convenience function for quick predictions
def quick_forecast(temperatures: List[float], horizon: int = 30) -> Tuple[float, float]:
    """
    Quick temperature forecast without training.

    Uses exponential smoothing for immediate predictions.
    """
    forecaster = LSTMForecaster()
    return forecaster._fallback_predict(np.array(temperatures), horizon)
