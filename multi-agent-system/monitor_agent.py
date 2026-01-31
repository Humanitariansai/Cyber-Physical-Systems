"""
Monitor Agent for Cold Chain Monitoring System
Responsible for ingesting sensor data and detecting anomalies in real-time.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional
import logging
import statistics

from .agent_base import BaseAgent, AgentConfig
from .message_bus import (
    Message, MessageBus, MessageType, MessagePriority,
    create_sensor_message, create_alert_message
)

logger = logging.getLogger(__name__)


@dataclass
class MonitorAgentConfig(AgentConfig):
    """Configuration for Monitor Agent."""
    # Temperature thresholds for cold chain (2-8°C typical for pharmaceuticals)
    temp_min: float = 2.0
    temp_max: float = 8.0
    temp_critical_min: float = 0.0
    temp_critical_max: float = 10.0

    # Humidity thresholds
    humidity_min: float = 30.0
    humidity_max: float = 70.0

    # Anomaly detection
    window_size: int = 60  # readings to keep for trend analysis
    rate_of_change_threshold: float = 0.5  # °C per minute
    std_deviation_threshold: float = 2.0  # for anomaly detection

    # Sampling
    sample_interval: float = 1.0  # seconds


@dataclass
class SensorReading:
    """Single sensor reading."""
    sensor_id: str
    temperature: float
    humidity: float
    timestamp: datetime


class MonitorAgent(BaseAgent):
    """
    Monitor Agent: Ingests sensor data and detects anomalies.

    Responsibilities:
    - Receive and validate sensor readings
    - Detect threshold violations
    - Detect anomalous patterns (rapid changes, unusual values)
    - Forward data to predictor agent
    - Generate immediate alerts for critical conditions
    """

    def __init__(self, agent_id: str, bus: MessageBus,
                 config: Optional[MonitorAgentConfig] = None):
        super().__init__(agent_id, bus, config or MonitorAgentConfig())
        self.config: MonitorAgentConfig = self.config

        # Sensor data buffers (per sensor)
        self._sensor_buffers: Dict[str, Deque[SensorReading]] = {}

        # Statistics per sensor
        self._sensor_stats: Dict[str, Dict[str, float]] = {}

    async def _subscribe_to_messages(self):
        """Subscribe to sensor data messages."""
        self.bus.subscribe(MessageType.SENSOR_DATA, self._on_sensor_data)
        self.bus.subscribe(MessageType.COMMAND, self._on_command)

    async def _on_sensor_data(self, message: Message):
        """Callback for sensor data messages."""
        # This is handled in the main loop
        pass

    async def _on_command(self, message: Message):
        """Handle command messages."""
        pass

    async def _handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.message_type == MessageType.SENSOR_DATA:
            await self._process_sensor_data(message)
        elif message.message_type == MessageType.COMMAND:
            await self._handle_command(message)

    async def _handle_command(self, message: Message):
        """Handle command messages."""
        command = message.payload.get("command")
        if command == "reset_buffers":
            self._sensor_buffers.clear()
            self._sensor_stats.clear()
            logger.info(f"Monitor agent buffers reset")

    async def _process_sensor_data(self, message: Message):
        """Process incoming sensor data."""
        payload = message.payload
        sensor_id = payload.get("sensor_id")
        temperature = payload.get("temperature")
        humidity = payload.get("humidity", 50.0)

        if sensor_id is None or temperature is None:
            logger.warning("Invalid sensor data received")
            return

        # Create reading
        reading = SensorReading(
            sensor_id=sensor_id,
            temperature=temperature,
            humidity=humidity,
            timestamp=datetime.now()
        )

        # Initialize buffer if needed
        if sensor_id not in self._sensor_buffers:
            self._sensor_buffers[sensor_id] = deque(maxlen=self.config.window_size)

        # Add to buffer
        self._sensor_buffers[sensor_id].append(reading)

        # Update statistics
        self._update_statistics(sensor_id)

        # Check for anomalies and threshold violations
        alerts = self._check_conditions(sensor_id, reading)

        # Send alerts if any
        for alert in alerts:
            await self.send_message(alert)

        # Forward to predictor (broadcast sensor data for other agents)
        forward_msg = create_sensor_message(
            source=self.agent_id,
            sensor_id=sensor_id,
            temperature=temperature,
            humidity=humidity
        )
        await self.send_message(forward_msg)

    def _update_statistics(self, sensor_id: str):
        """Update rolling statistics for a sensor."""
        buffer = self._sensor_buffers[sensor_id]
        if len(buffer) < 2:
            return

        temps = [r.temperature for r in buffer]

        self._sensor_stats[sensor_id] = {
            "mean": statistics.mean(temps),
            "std": statistics.stdev(temps) if len(temps) > 1 else 0,
            "min": min(temps),
            "max": max(temps),
            "count": len(temps)
        }

    def _check_conditions(self, sensor_id: str, reading: SensorReading) -> List[Message]:
        """Check for threshold violations and anomalies."""
        alerts = []
        temp = reading.temperature
        humidity = reading.humidity

        # Critical temperature violations
        if temp <= self.config.temp_critical_min:
            alerts.append(create_alert_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                alert_type="TEMPERATURE_CRITICAL_LOW",
                severity="CRITICAL",
                message=f"Critical: Temperature {temp:.1f}°C below {self.config.temp_critical_min}°C"
            ))
        elif temp >= self.config.temp_critical_max:
            alerts.append(create_alert_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                alert_type="TEMPERATURE_CRITICAL_HIGH",
                severity="CRITICAL",
                message=f"Critical: Temperature {temp:.1f}°C above {self.config.temp_critical_max}°C"
            ))
        # Warning temperature violations
        elif temp < self.config.temp_min:
            alerts.append(create_alert_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                alert_type="TEMPERATURE_LOW",
                severity="WARNING",
                message=f"Warning: Temperature {temp:.1f}°C below {self.config.temp_min}°C"
            ))
        elif temp > self.config.temp_max:
            alerts.append(create_alert_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                alert_type="TEMPERATURE_HIGH",
                severity="WARNING",
                message=f"Warning: Temperature {temp:.1f}°C above {self.config.temp_max}°C"
            ))

        # Humidity violations
        if humidity < self.config.humidity_min:
            alerts.append(create_alert_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                alert_type="HUMIDITY_LOW",
                severity="WARNING",
                message=f"Warning: Humidity {humidity:.1f}% below {self.config.humidity_min}%"
            ))
        elif humidity > self.config.humidity_max:
            alerts.append(create_alert_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                alert_type="HUMIDITY_HIGH",
                severity="WARNING",
                message=f"Warning: Humidity {humidity:.1f}% above {self.config.humidity_max}%"
            ))

        # Check for rapid temperature change
        rate_of_change = self._calculate_rate_of_change(sensor_id)
        if abs(rate_of_change) > self.config.rate_of_change_threshold:
            direction = "rising" if rate_of_change > 0 else "falling"
            alerts.append(create_alert_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                alert_type="RAPID_TEMPERATURE_CHANGE",
                severity="WARNING",
                message=f"Rapid temperature {direction}: {abs(rate_of_change):.2f}°C/min"
            ))

        # Check for statistical anomaly
        if self._is_statistical_anomaly(sensor_id, temp):
            alerts.append(create_alert_message(
                source=self.agent_id,
                sensor_id=sensor_id,
                alert_type="STATISTICAL_ANOMALY",
                severity="INFO",
                message=f"Statistical anomaly detected: {temp:.1f}°C"
            ))

        return alerts

    def _calculate_rate_of_change(self, sensor_id: str) -> float:
        """Calculate temperature rate of change (°C per minute)."""
        buffer = self._sensor_buffers.get(sensor_id)
        if not buffer or len(buffer) < 2:
            return 0.0

        # Use last 10 readings or all available
        recent = list(buffer)[-10:]
        if len(recent) < 2:
            return 0.0

        first = recent[0]
        last = recent[-1]

        time_diff = (last.timestamp - first.timestamp).total_seconds() / 60.0
        if time_diff == 0:
            return 0.0

        temp_diff = last.temperature - first.temperature
        return temp_diff / time_diff

    def _is_statistical_anomaly(self, sensor_id: str, temperature: float) -> bool:
        """Check if temperature is a statistical anomaly."""
        stats = self._sensor_stats.get(sensor_id)
        if not stats or stats["std"] == 0:
            return False

        z_score = abs(temperature - stats["mean"]) / stats["std"]
        return z_score > self.config.std_deviation_threshold

    def get_sensor_statistics(self, sensor_id: str) -> Optional[Dict[str, float]]:
        """Get current statistics for a sensor."""
        return self._sensor_stats.get(sensor_id)

    def get_recent_readings(self, sensor_id: str, count: int = 10) -> List[SensorReading]:
        """Get recent readings for a sensor."""
        buffer = self._sensor_buffers.get(sensor_id)
        if not buffer:
            return []
        return list(buffer)[-count:]

    async def inject_sensor_reading(self, sensor_id: str, temperature: float,
                                   humidity: float = 50.0):
        """Inject a sensor reading directly (for simulation/testing)."""
        message = create_sensor_message(
            source="external",
            sensor_id=sensor_id,
            temperature=temperature,
            humidity=humidity
        )
        await self._process_sensor_data(message)
