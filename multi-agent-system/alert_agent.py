"""
Alert Agent for Cold Chain Monitoring System
Responsible for managing alerts, notifications, and alert lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import uuid

from .agent_base import BaseAgent, AgentConfig
from .message_bus import Message, MessageBus, MessageType

logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    """Record of an alert."""
    alert_id: str
    sensor_id: str
    alert_type: str
    severity: str
    message: str
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    escalated: bool = False
    notification_sent: bool = False


@dataclass
class AlertAgentConfig(AgentConfig):
    """Configuration for Alert Agent."""
    # Alert retention
    max_active_alerts: int = 100
    alert_history_hours: int = 168  # 1 week

    # Notification settings
    notification_cooldown_seconds: float = 60.0
    auto_acknowledge_info_alerts: bool = True

    # Escalation
    escalation_timeout_minutes: int = 15
    max_unacknowledged_before_escalation: int = 5

    # Deduplication
    dedup_window_seconds: float = 30.0


class AlertAgent(BaseAgent):
    """
    Alert Agent: Manages alert lifecycle and notifications.

    Responsibilities:
    - Receive and catalog alerts
    - Deduplicate similar alerts
    - Track alert lifecycle (triggered -> acknowledged -> resolved)
    - Send notifications
    - Escalate unacknowledged alerts
    - Maintain alert history
    """

    def __init__(self, agent_id: str, bus: MessageBus,
                 config: Optional[AlertAgentConfig] = None):
        super().__init__(agent_id, bus, config or AlertAgentConfig())
        self.config: AlertAgentConfig = self.config

        # Active alerts
        self._active_alerts: Dict[str, AlertRecord] = {}

        # Alert history
        self._alert_history: List[AlertRecord] = []

        # Deduplication tracking
        self._recent_alert_keys: Dict[str, datetime] = {}

        # Notification tracking
        self._last_notification_time: Dict[str, datetime] = {}

        # Statistics
        self._stats = {
            "total_alerts": 0,
            "critical_alerts": 0,
            "acknowledged_alerts": 0,
            "resolved_alerts": 0,
            "escalated_alerts": 0
        }

    async def _subscribe_to_messages(self):
        """Subscribe to alert messages."""
        self.bus.subscribe(MessageType.ALERT, self._on_alert)
        self.bus.subscribe(MessageType.COMMAND, self._on_command)
        self.bus.subscribe(MessageType.DECISION, self._on_decision)

    async def _on_alert(self, message: Message):
        pass

    async def _on_command(self, message: Message):
        pass

    async def _on_decision(self, message: Message):
        pass

    async def _handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.message_type == MessageType.ALERT:
            await self._process_alert(message)
        elif message.message_type == MessageType.COMMAND:
            await self._handle_command(message)
        elif message.message_type == MessageType.DECISION:
            await self._handle_decision(message)

    async def _process_alert(self, message: Message):
        """Process incoming alert."""
        payload = message.payload

        # Extract alert details
        alert_id = payload.get("alert_id", str(uuid.uuid4()))
        sensor_id = payload.get("sensor_id", "unknown")
        alert_type = payload.get("alert_type", "UNKNOWN")
        severity = payload.get("severity", "INFO")
        alert_message = payload.get("message", "")

        # Check for duplicate
        dedup_key = f"{sensor_id}:{alert_type}:{severity}"
        if self._is_duplicate(dedup_key):
            logger.debug(f"Duplicate alert suppressed: {dedup_key}")
            return

        # Create alert record
        alert = AlertRecord(
            alert_id=alert_id,
            sensor_id=sensor_id,
            alert_type=alert_type,
            severity=severity,
            message=alert_message,
            triggered_at=datetime.now()
        )

        # Auto-acknowledge INFO alerts if configured
        if severity == "INFO" and self.config.auto_acknowledge_info_alerts:
            alert.acknowledged_at = datetime.now()

        # Store alert
        self._active_alerts[alert_id] = alert
        self._stats["total_alerts"] += 1

        if severity == "CRITICAL":
            self._stats["critical_alerts"] += 1

        # Update dedup tracking
        self._recent_alert_keys[dedup_key] = datetime.now()

        # Send notification if needed
        await self._maybe_notify(alert)

        # Check for escalation conditions
        await self._check_escalation()

        # Cleanup old alerts
        self._cleanup_old_alerts()

        logger.info(f"Alert processed: {alert_id} - {severity}: {alert_type}")

    def _is_duplicate(self, dedup_key: str) -> bool:
        """Check if alert is a duplicate within dedup window."""
        if dedup_key not in self._recent_alert_keys:
            return False

        last_time = self._recent_alert_keys[dedup_key]
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed < self.config.dedup_window_seconds

    async def _maybe_notify(self, alert: AlertRecord):
        """Send notification if cooldown has passed."""
        # Only notify for WARNING and above
        if alert.severity not in ["WARNING", "ERROR", "CRITICAL"]:
            return

        notify_key = f"{alert.sensor_id}:{alert.severity}"
        last_notify = self._last_notification_time.get(notify_key)

        if last_notify:
            elapsed = (datetime.now() - last_notify).total_seconds()
            if elapsed < self.config.notification_cooldown_seconds:
                return

        # Send notification (in real system, this would integrate with
        # email, SMS, Slack, PagerDuty, etc.)
        await self._send_notification(alert)
        self._last_notification_time[notify_key] = datetime.now()
        alert.notification_sent = True

    async def _send_notification(self, alert: AlertRecord):
        """Send notification for alert."""
        # In production, this would integrate with notification services
        # For now, just log it
        logger.warning(
            f"[NOTIFICATION] {alert.severity} Alert: "
            f"Sensor {alert.sensor_id} - {alert.alert_type}: {alert.message}"
        )

        # Publish notification status
        status_msg = Message(
            priority=2,
            message_type=MessageType.STATUS,
            source=self.agent_id,
            payload={
                "type": "notification_sent",
                "alert_id": alert.alert_id,
                "sensor_id": alert.sensor_id,
                "severity": alert.severity,
                "timestamp": datetime.now().isoformat()
            }
        )
        await self.send_message(status_msg)

    async def _check_escalation(self):
        """Check if escalation is needed."""
        now = datetime.now()
        unacknowledged_count = 0
        alerts_to_escalate = []

        for alert in self._active_alerts.values():
            if alert.acknowledged_at is None and alert.resolved_at is None:
                unacknowledged_count += 1

                # Check timeout
                age_minutes = (now - alert.triggered_at).total_seconds() / 60
                if (age_minutes > self.config.escalation_timeout_minutes and
                    not alert.escalated and
                    alert.severity in ["ERROR", "CRITICAL"]):
                    alerts_to_escalate.append(alert)

        # Escalate individual alerts
        for alert in alerts_to_escalate:
            await self._escalate_alert(alert)

        # Check bulk escalation
        if unacknowledged_count >= self.config.max_unacknowledged_before_escalation:
            logger.warning(f"High unacknowledged alert count: {unacknowledged_count}")

    async def _escalate_alert(self, alert: AlertRecord):
        """Escalate an alert."""
        alert.escalated = True
        self._stats["escalated_alerts"] += 1

        logger.warning(f"[ESCALATION] Alert {alert.alert_id} escalated")

        # Send escalation message
        escalation_msg = Message(
            priority=4,  # Critical priority
            message_type=MessageType.ALERT,
            source=self.agent_id,
            payload={
                "alert_id": f"ESC-{alert.alert_id}",
                "sensor_id": alert.sensor_id,
                "alert_type": "ESCALATION",
                "severity": "CRITICAL",
                "message": f"Escalated: {alert.message} (unacknowledged for "
                          f"{self.config.escalation_timeout_minutes} minutes)",
                "original_alert_id": alert.alert_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        await self.send_message(escalation_msg)

    async def _handle_command(self, message: Message):
        """Handle command messages."""
        command = message.payload.get("command")
        alert_id = message.payload.get("alert_id")

        if command == "acknowledge" and alert_id:
            await self.acknowledge_alert(alert_id)
        elif command == "resolve" and alert_id:
            notes = message.payload.get("notes", "")
            await self.resolve_alert(alert_id, notes)
        elif command == "clear_all":
            await self._clear_all_alerts()

    async def _handle_decision(self, message: Message):
        """Handle decision messages that might affect alerts."""
        # Decisions might trigger automatic acknowledgment
        decision_type = message.payload.get("decision_type")
        if decision_type == "EMERGENCY_RESPONSE":
            # Auto-acknowledge related alerts
            sensor_id = message.payload.get("parameters", {}).get("sensor_id")
            if sensor_id:
                for alert in self._active_alerts.values():
                    if (alert.sensor_id == sensor_id and
                        alert.acknowledged_at is None):
                        alert.acknowledged_at = datetime.now()
                        self._stats["acknowledged_alerts"] += 1

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        alert = self._active_alerts.get(alert_id)
        if not alert:
            return False

        if alert.acknowledged_at is None:
            alert.acknowledged_at = datetime.now()
            self._stats["acknowledged_alerts"] += 1
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False

    async def resolve_alert(self, alert_id: str, notes: str = "") -> bool:
        """Resolve an alert."""
        alert = self._active_alerts.get(alert_id)
        if not alert:
            return False

        alert.resolved_at = datetime.now()
        alert.resolution_notes = notes
        self._stats["resolved_alerts"] += 1

        # Move to history
        self._alert_history.append(alert)
        del self._active_alerts[alert_id]

        logger.info(f"Alert resolved: {alert_id}")
        return True

    async def _clear_all_alerts(self):
        """Clear all active alerts (admin function)."""
        for alert in self._active_alerts.values():
            alert.resolved_at = datetime.now()
            alert.resolution_notes = "Bulk cleared"
            self._alert_history.append(alert)

        count = len(self._active_alerts)
        self._active_alerts.clear()
        logger.info(f"Cleared {count} alerts")

    def _cleanup_old_alerts(self):
        """Remove old alerts from history."""
        cutoff = datetime.now() - timedelta(hours=self.config.alert_history_hours)
        self._alert_history = [
            a for a in self._alert_history
            if a.triggered_at > cutoff
        ]

        # Cleanup dedup keys
        dedup_cutoff = datetime.now() - timedelta(
            seconds=self.config.dedup_window_seconds * 2
        )
        self._recent_alert_keys = {
            k: v for k, v in self._recent_alert_keys.items()
            if v > dedup_cutoff
        }

    # Public query methods

    def get_active_alerts(self, sensor_id: Optional[str] = None,
                         severity: Optional[str] = None) -> List[Dict]:
        """Get active alerts with optional filtering."""
        alerts = []
        for alert in self._active_alerts.values():
            if sensor_id and alert.sensor_id != sensor_id:
                continue
            if severity and alert.severity != severity:
                continue

            alerts.append({
                "alert_id": alert.alert_id,
                "sensor_id": alert.sensor_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged": alert.acknowledged_at is not None,
                "escalated": alert.escalated
            })

        return sorted(alerts, key=lambda x: x["triggered_at"], reverse=True)

    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get alert history."""
        cutoff = datetime.now() - timedelta(hours=hours)
        history = []

        for alert in self._alert_history:
            if alert.triggered_at > cutoff:
                history.append({
                    "alert_id": alert.alert_id,
                    "sensor_id": alert.sensor_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "resolution_notes": alert.resolution_notes
                })

        return sorted(history, key=lambda x: x["triggered_at"], reverse=True)

    def get_statistics(self) -> Dict:
        """Get alert statistics."""
        return {
            **self._stats,
            "active_alerts": len(self._active_alerts),
            "history_size": len(self._alert_history),
            "unacknowledged": sum(
                1 for a in self._active_alerts.values()
                if a.acknowledged_at is None
            )
        }
