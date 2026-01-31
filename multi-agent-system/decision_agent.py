"""
Decision Agent for Cold Chain Monitoring System
Responsible for making decisions based on sensor data, predictions, and alerts.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from .agent_base import BaseAgent, AgentConfig
from .message_bus import (
    Message, MessageBus, MessageType,
    create_decision_message, create_alert_message
)

logger = logging.getLogger(__name__)


@dataclass
class DecisionAgentConfig(AgentConfig):
    """Configuration for Decision Agent."""
    # Temperature thresholds
    temp_warning_low: float = 2.0
    temp_warning_high: float = 8.0
    temp_critical_low: float = 0.0
    temp_critical_high: float = 10.0

    # Decision cooldown (prevent rapid repeated decisions)
    decision_cooldown_seconds: float = 60.0

    # Alert aggregation window
    alert_window_seconds: float = 300.0  # 5 minutes

    # Thresholds for escalation
    alerts_before_escalation: int = 3
    predictions_confidence_threshold: float = 0.7


class DecisionContext:
    """Context for making decisions about a sensor."""

    def __init__(self):
        self.recent_alerts: List[Dict] = []
        self.recent_predictions: List[Dict] = []
        self.current_temperature: Optional[float] = None
        self.last_decision_time: Optional[datetime] = None
        self.active_actions: Dict[str, datetime] = {}

    def add_alert(self, alert: Dict):
        """Add an alert to context."""
        self.recent_alerts.append(alert)
        # Keep only recent alerts
        cutoff = datetime.now() - timedelta(minutes=10)
        self.recent_alerts = [
            a for a in self.recent_alerts
            if datetime.fromisoformat(a.get("timestamp", datetime.now().isoformat())) > cutoff
        ]

    def add_prediction(self, prediction: Dict):
        """Add a prediction to context."""
        self.recent_predictions.append(prediction)
        # Keep only recent predictions
        if len(self.recent_predictions) > 20:
            self.recent_predictions = self.recent_predictions[-20:]

    def get_alert_count(self, severity: Optional[str] = None) -> int:
        """Get count of recent alerts, optionally filtered by severity."""
        if severity:
            return sum(1 for a in self.recent_alerts if a.get("severity") == severity)
        return len(self.recent_alerts)


class DecisionAgent(BaseAgent):
    """
    Decision Agent: Makes operational decisions based on system state.

    Responsibilities:
    - Aggregate alerts and predictions
    - Determine appropriate responses
    - Issue commands to other agents
    - Escalate issues when thresholds are exceeded
    - Prevent alert fatigue through intelligent suppression
    """

    def __init__(self, agent_id: str, bus: MessageBus,
                 config: Optional[DecisionAgentConfig] = None):
        super().__init__(agent_id, bus, config or DecisionAgentConfig())
        self.config: DecisionAgentConfig = self.config

        # Decision context per sensor
        self._contexts: Dict[str, DecisionContext] = {}

        # Global state
        self._escalation_active = False
        self._maintenance_mode = False

    async def _subscribe_to_messages(self):
        """Subscribe to relevant message types."""
        self.bus.subscribe(MessageType.ALERT, self._on_alert)
        self.bus.subscribe(MessageType.PREDICTION, self._on_prediction)
        self.bus.subscribe(MessageType.SENSOR_DATA, self._on_sensor_data)
        self.bus.subscribe(MessageType.COMMAND, self._on_command)

    async def _on_alert(self, message: Message):
        pass

    async def _on_prediction(self, message: Message):
        pass

    async def _on_sensor_data(self, message: Message):
        pass

    async def _on_command(self, message: Message):
        pass

    async def _handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.message_type == MessageType.ALERT:
            await self._process_alert(message)
        elif message.message_type == MessageType.PREDICTION:
            await self._process_prediction(message)
        elif message.message_type == MessageType.SENSOR_DATA:
            await self._process_sensor_data(message)
        elif message.message_type == MessageType.COMMAND:
            await self._handle_command(message)

    def _get_context(self, sensor_id: str) -> DecisionContext:
        """Get or create decision context for sensor."""
        if sensor_id not in self._contexts:
            self._contexts[sensor_id] = DecisionContext()
        return self._contexts[sensor_id]

    async def _process_alert(self, message: Message):
        """Process incoming alert."""
        payload = message.payload
        sensor_id = payload.get("sensor_id")
        if not sensor_id:
            return

        context = self._get_context(sensor_id)
        context.add_alert({
            "alert_type": payload.get("alert_type"),
            "severity": payload.get("severity"),
            "message": payload.get("message"),
            "timestamp": payload.get("timestamp", datetime.now().isoformat())
        })

        # Make decision based on updated context
        await self._evaluate_and_decide(sensor_id)

    async def _process_prediction(self, message: Message):
        """Process incoming prediction."""
        payload = message.payload
        sensor_id = payload.get("sensor_id")
        if not sensor_id:
            return

        context = self._get_context(sensor_id)
        context.add_prediction({
            "predicted_value": payload.get("predicted_value"),
            "horizon_minutes": payload.get("horizon_minutes"),
            "confidence": payload.get("confidence"),
            "timestamp": payload.get("timestamp", datetime.now().isoformat())
        })

        # Check if high-confidence predictions warrant action
        confidence = payload.get("confidence", 0)
        predicted = payload.get("predicted_value")

        if confidence >= self.config.predictions_confidence_threshold:
            if predicted and (predicted < self.config.temp_critical_low or
                            predicted > self.config.temp_critical_high):
                await self._issue_preemptive_action(sensor_id, predicted, confidence)

    async def _process_sensor_data(self, message: Message):
        """Process sensor data for context."""
        payload = message.payload
        sensor_id = payload.get("sensor_id")
        temperature = payload.get("temperature")

        if sensor_id and temperature is not None:
            context = self._get_context(sensor_id)
            context.current_temperature = temperature

    async def _handle_command(self, message: Message):
        """Handle command messages."""
        command = message.payload.get("command")

        if command == "enter_maintenance":
            self._maintenance_mode = True
            logger.info("Entered maintenance mode")
        elif command == "exit_maintenance":
            self._maintenance_mode = False
            logger.info("Exited maintenance mode")
        elif command == "reset_escalation":
            self._escalation_active = False
            logger.info("Escalation reset")

    async def _evaluate_and_decide(self, sensor_id: str):
        """Evaluate context and make decisions."""
        if self._maintenance_mode:
            return  # Suppress decisions in maintenance mode

        context = self._get_context(sensor_id)
        now = datetime.now()

        # Check cooldown
        if context.last_decision_time:
            elapsed = (now - context.last_decision_time).total_seconds()
            if elapsed < self.config.decision_cooldown_seconds:
                return

        # Count alerts by severity
        critical_count = context.get_alert_count("CRITICAL")
        error_count = context.get_alert_count("ERROR")
        warning_count = context.get_alert_count("WARNING")

        decisions = []

        # Critical alerts require immediate action
        if critical_count > 0:
            decisions.append(self._create_action(
                sensor_id, "EMERGENCY_RESPONSE",
                "immediate_investigation",
                {"priority": "critical", "alert_count": critical_count}
            ))

            # Escalate if multiple criticals
            if critical_count >= 2 and not self._escalation_active:
                self._escalation_active = True
                decisions.append(self._create_action(
                    sensor_id, "ESCALATION",
                    "notify_supervisor",
                    {"reason": "multiple_critical_alerts", "count": critical_count}
                ))

        # Error alerts need attention
        elif error_count >= self.config.alerts_before_escalation:
            decisions.append(self._create_action(
                sensor_id, "MAINTENANCE_REQUEST",
                "schedule_inspection",
                {"priority": "high", "alert_count": error_count}
            ))

        # Multiple warnings indicate trend
        elif warning_count >= self.config.alerts_before_escalation:
            decisions.append(self._create_action(
                sensor_id, "MONITORING_INCREASE",
                "increase_sample_rate",
                {"priority": "medium", "warning_count": warning_count}
            ))

        # Send decisions
        for decision in decisions:
            await self.send_message(decision)
            context.active_actions[decision.payload["action"]] = now

        if decisions:
            context.last_decision_time = now

    async def _issue_preemptive_action(self, sensor_id: str, predicted_temp: float,
                                       confidence: float):
        """Issue preemptive action based on prediction."""
        context = self._get_context(sensor_id)

        # Check if we already have an active preemptive action
        if "preemptive_alert" in context.active_actions:
            last_action = context.active_actions["preemptive_alert"]
            if (datetime.now() - last_action).total_seconds() < 300:
                return  # Already acted recently

        action_type = "PREEMPTIVE_COOLING" if predicted_temp > self.config.temp_warning_high else "PREEMPTIVE_HEATING"

        decision = self._create_action(
            sensor_id, action_type,
            "adjust_hvac",
            {
                "predicted_temperature": predicted_temp,
                "confidence": confidence,
                "action": "increase_cooling" if predicted_temp > self.config.temp_warning_high else "increase_heating"
            }
        )

        await self.send_message(decision)
        context.active_actions["preemptive_alert"] = datetime.now()

        # Also send an informational alert
        alert = create_alert_message(
            source=self.agent_id,
            sensor_id=sensor_id,
            alert_type="PREEMPTIVE_ACTION",
            severity="INFO",
            message=f"Preemptive action taken: {action_type} (predicted: {predicted_temp:.1f}°C)"
        )
        await self.send_message(alert)

    def _create_action(self, sensor_id: str, decision_type: str,
                       action: str, parameters: Dict) -> Message:
        """Create a decision message."""
        parameters["sensor_id"] = sensor_id
        return create_decision_message(
            source=self.agent_id,
            decision_type=decision_type,
            action=action,
            parameters=parameters
        )

    def get_sensor_context(self, sensor_id: str) -> Optional[Dict]:
        """Get current context for a sensor."""
        context = self._contexts.get(sensor_id)
        if not context:
            return None

        return {
            "current_temperature": context.current_temperature,
            "alert_count": len(context.recent_alerts),
            "critical_alerts": context.get_alert_count("CRITICAL"),
            "active_actions": list(context.active_actions.keys()),
            "last_decision": context.last_decision_time.isoformat()
                if context.last_decision_time else None
        }

    def is_escalated(self) -> bool:
        """Check if system is in escalation state."""
        return self._escalation_active

    def is_maintenance_mode(self) -> bool:
        """Check if system is in maintenance mode."""
        return self._maintenance_mode
