"""
Predictor Agent for Cold Chain Monitoring System
Responsible for ML-based temperature forecasting with 30-60 minute horizons.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple
import logging
import numpy as np

from .agent_base import BaseAgent, AgentConfig
from .message_bus import (
    Message, MessageBus, MessageType,
    create_prediction_message, create_alert_message
)

logger = logging.getLogger(__name__)


@dataclass
class PredictorAgentConfig(AgentConfig):
    """Configuration for Predictor Agent."""
    # Prediction horizons in minutes
    prediction_horizons: Tuple[int, ...] = (30, 60)

    # Model parameters
    sequence_length: int = 60  # Number of historical points for prediction
    min_data_points: int = 30  # Minimum data points before making predictions

    # Thresholds for predictive alerts
    predicted_temp_min: float = 2.0
    predicted_temp_max: float = 8.0

    # Update interval (how often to make predictions)
    prediction_interval: float = 60.0  # seconds

    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.5


@dataclass
class PredictionResult:
    """Single prediction result."""
    sensor_id: str
    current_temp: float
    predicted_temp: float
    horizon_minutes: int
    confidence: float
    timestamp: datetime
    trend: str  # "rising", "falling", "stable"


class SimpleForecaster:
    """
    Simple forecasting model using exponential smoothing and trend analysis.
    This is a lightweight model suitable for real-time predictions.
    For production, this would be replaced with LSTM/GRU models.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha  # Level smoothing
        self.beta = beta    # Trend smoothing

    def predict(self, data: np.ndarray, horizon_minutes: int,
                sample_interval_seconds: float = 60.0) -> Tuple[float, float]:
        """
        Predict future temperature using double exponential smoothing.

        Args:
            data: Historical temperature readings
            horizon_minutes: Minutes ahead to predict
            sample_interval_seconds: Time between samples

        Returns:
            (predicted_value, confidence)
        """
        if len(data) < 2:
            return data[-1] if len(data) > 0 else 5.0, 0.3

        # Initialize
        level = data[0]
        trend = data[1] - data[0]

        # Apply double exponential smoothing
        for i in range(1, len(data)):
            prev_level = level
            level = self.alpha * data[i] + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend

        # Calculate how many steps ahead
        steps_ahead = int((horizon_minutes * 60) / sample_interval_seconds)

        # Predict
        predicted = level + steps_ahead * trend

        # Calculate confidence based on data variance and horizon
        variance = np.var(data)
        base_confidence = 0.9 - (variance * 0.1)  # Lower confidence with higher variance
        horizon_penalty = horizon_minutes / 120.0  # Penalty increases with horizon
        confidence = max(0.3, min(0.95, base_confidence - horizon_penalty))

        return float(predicted), float(confidence)

    def detect_trend(self, data: np.ndarray, window: int = 10) -> str:
        """Detect temperature trend."""
        if len(data) < window:
            return "stable"

        recent = data[-window:]
        slope = (recent[-1] - recent[0]) / window

        if slope > 0.05:
            return "rising"
        elif slope < -0.05:
            return "falling"
        return "stable"


class PredictorAgent(BaseAgent):
    """
    Predictor Agent: ML-based temperature forecasting.

    Responsibilities:
    - Collect sensor data for model input
    - Generate 30-60 minute ahead forecasts
    - Detect temperature trends
    - Issue predictive alerts for anticipated threshold violations
    - Track prediction accuracy
    """

    def __init__(self, agent_id: str, bus: MessageBus,
                 config: Optional[PredictorAgentConfig] = None):
        super().__init__(agent_id, bus, config or PredictorAgentConfig())
        self.config: PredictorAgentConfig = self.config

        # Data buffers per sensor
        self._sensor_data: Dict[str, Deque[Tuple[datetime, float]]] = {}

        # Forecaster model
        self._forecaster = SimpleForecaster()

        # Prediction history for accuracy tracking
        self._predictions: Dict[str, List[PredictionResult]] = {}

        # Last prediction time per sensor
        self._last_prediction_time: Dict[str, datetime] = {}

    async def _subscribe_to_messages(self):
        """Subscribe to sensor data messages."""
        self.bus.subscribe(MessageType.SENSOR_DATA, self._on_sensor_data)

    async def _on_sensor_data(self, message: Message):
        """Callback for sensor data - handled in main loop."""
        pass

    async def _handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.message_type == MessageType.SENSOR_DATA:
            await self._process_sensor_data(message)

    async def _process_sensor_data(self, message: Message):
        """Process incoming sensor data and generate predictions."""
        payload = message.payload
        sensor_id = payload.get("sensor_id")
        temperature = payload.get("temperature")

        if sensor_id is None or temperature is None:
            return

        # Initialize buffer if needed
        if sensor_id not in self._sensor_data:
            self._sensor_data[sensor_id] = deque(maxlen=self.config.sequence_length)
            self._predictions[sensor_id] = []

        # Add data point
        self._sensor_data[sensor_id].append((datetime.now(), temperature))

        # Check if we should make a prediction
        await self._maybe_predict(sensor_id, temperature)

    async def _maybe_predict(self, sensor_id: str, current_temp: float):
        """Make predictions if enough data and time has passed."""
        data_buffer = self._sensor_data[sensor_id]

        # Check minimum data requirement
        if len(data_buffer) < self.config.min_data_points:
            return

        # Check time since last prediction
        last_time = self._last_prediction_time.get(sensor_id)
        now = datetime.now()

        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.config.prediction_interval:
                return

        # Generate predictions for each horizon
        temps = np.array([t[1] for t in data_buffer])

        for horizon in self.config.prediction_horizons:
            predicted_temp, confidence = self._forecaster.predict(temps, horizon)
            trend = self._forecaster.detect_trend(temps)

            result = PredictionResult(
                sensor_id=sensor_id,
                current_temp=current_temp,
                predicted_temp=predicted_temp,
                horizon_minutes=horizon,
                confidence=confidence,
                timestamp=now,
                trend=trend
            )

            # Store prediction
            self._predictions[sensor_id].append(result)
            if len(self._predictions[sensor_id]) > 100:
                self._predictions[sensor_id].pop(0)

            # Send prediction message
            pred_msg = create_prediction_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                predicted_value=predicted_temp,
                horizon_minutes=horizon,
                confidence=confidence
            )
            await self.send_message(pred_msg)

            # Check for predictive alerts
            await self._check_predictive_alerts(result)

        self._last_prediction_time[sensor_id] = now
        logger.debug(f"Predictions generated for {sensor_id}")

    async def _check_predictive_alerts(self, prediction: PredictionResult):
        """Generate alerts for predicted threshold violations."""
        # Only alert on high confidence predictions
        if prediction.confidence < self.config.low_confidence_threshold:
            return

        predicted = prediction.predicted_temp

        if predicted < self.config.predicted_temp_min:
            severity = "WARNING" if prediction.confidence < self.config.high_confidence_threshold else "ERROR"
            alert = create_alert_message(
                source=self.agent_id,
                sensor_id=prediction.sensor_id,
                alert_type="PREDICTED_LOW_TEMPERATURE",
                severity=severity,
                message=(f"Predicted temperature {predicted:.1f}°C in "
                        f"{prediction.horizon_minutes} minutes "
                        f"(confidence: {prediction.confidence:.0%})")
            )
            await self.send_message(alert)

        elif predicted > self.config.predicted_temp_max:
            severity = "WARNING" if prediction.confidence < self.config.high_confidence_threshold else "ERROR"
            alert = create_alert_message(
                source=self.agent_id,
                sensor_id=prediction.sensor_id,
                alert_type="PREDICTED_HIGH_TEMPERATURE",
                severity=severity,
                message=(f"Predicted temperature {predicted:.1f}°C in "
                        f"{prediction.horizon_minutes} minutes "
                        f"(confidence: {prediction.confidence:.0%})")
            )
            await self.send_message(alert)

    def get_latest_prediction(self, sensor_id: str,
                             horizon: int = 30) -> Optional[PredictionResult]:
        """Get the latest prediction for a sensor and horizon."""
        predictions = self._predictions.get(sensor_id, [])
        for pred in reversed(predictions):
            if pred.horizon_minutes == horizon:
                return pred
        return None

    def get_prediction_accuracy(self, sensor_id: str) -> Dict[str, float]:
        """
        Calculate prediction accuracy by comparing past predictions
        with actual values.
        """
        predictions = self._predictions.get(sensor_id, [])
        data = self._sensor_data.get(sensor_id, [])

        if not predictions or not data:
            return {"accuracy": 0.0, "samples": 0}

        accurate_count = 0
        total_count = 0
        errors = []

        for pred in predictions:
            # Find actual temperature at predicted time
            target_time = pred.timestamp
            # This is simplified - in production would need proper time matching
            for ts, temp in data:
                time_diff = abs((ts - target_time).total_seconds())
                if time_diff < 120:  # Within 2 minutes
                    error = abs(pred.predicted_temp - temp)
                    errors.append(error)
                    if error < 1.0:  # Within 1°C
                        accurate_count += 1
                    total_count += 1
                    break

        if total_count == 0:
            return {"accuracy": 0.0, "samples": 0}

        return {
            "accuracy": accurate_count / total_count,
            "mae": sum(errors) / len(errors) if errors else 0.0,
            "samples": total_count
        }

    def get_trend(self, sensor_id: str) -> str:
        """Get current temperature trend for a sensor."""
        data = self._sensor_data.get(sensor_id)
        if not data or len(data) < 10:
            return "unknown"

        temps = np.array([t[1] for t in data])
        return self._forecaster.detect_trend(temps)
