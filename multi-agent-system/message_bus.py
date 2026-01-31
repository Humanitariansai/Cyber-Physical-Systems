"""
Message Bus for Multi-Agent System
Provides asynchronous inter-agent communication with message routing and prioritization.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the system."""
    SENSOR_DATA = auto()
    PREDICTION = auto()
    ALERT = auto()
    DECISION = auto()
    COMMAND = auto()
    STATUS = auto()
    HEARTBEAT = auto()
    ERROR = auto()


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(order=True)
class Message:
    """Message structure for inter-agent communication."""
    priority: int = field(compare=True)
    timestamp: datetime = field(compare=False, default_factory=datetime.now)
    message_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = field(compare=False, default=MessageType.STATUS)
    source: str = field(compare=False, default="")
    target: Optional[str] = field(compare=False, default=None)  # None = broadcast
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)

    def __post_init__(self):
        if isinstance(self.priority, MessagePriority):
            self.priority = self.priority.value


class MessageBus:
    """
    Asynchronous message bus for multi-agent communication.
    Supports publish-subscribe pattern with topic-based routing.
    """

    def __init__(self):
        self._subscribers: Dict[MessageType, List[Callable]] = {
            msg_type: [] for msg_type in MessageType
        }
        self._agent_queues: Dict[str, asyncio.PriorityQueue] = {}
        self._running = False
        self._message_count = 0
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the message bus."""
        self._running = True
        logger.info("Message bus started")

    async def stop(self):
        """Stop the message bus."""
        self._running = False
        logger.info("Message bus stopped")

    def register_agent(self, agent_id: str, queue_size: int = 100):
        """Register an agent with its own message queue."""
        if agent_id not in self._agent_queues:
            self._agent_queues[agent_id] = asyncio.PriorityQueue(maxsize=queue_size)
            logger.info(f"Agent registered: {agent_id}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self._agent_queues:
            del self._agent_queues[agent_id]
            logger.info(f"Agent unregistered: {agent_id}")

    def subscribe(self, message_type: MessageType, callback: Callable):
        """Subscribe to a message type."""
        if callback not in self._subscribers[message_type]:
            self._subscribers[message_type].append(callback)

    def unsubscribe(self, message_type: MessageType, callback: Callable):
        """Unsubscribe from a message type."""
        if callback in self._subscribers[message_type]:
            self._subscribers[message_type].remove(callback)

    async def publish(self, message: Message):
        """Publish a message to the bus."""
        if not self._running:
            logger.warning("Message bus is not running")
            return

        async with self._lock:
            self._message_count += 1

        # Route to specific target or broadcast
        if message.target:
            if message.target in self._agent_queues:
                try:
                    self._agent_queues[message.target].put_nowait(
                        (-message.priority, message)
                    )
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for agent: {message.target}")
        else:
            # Broadcast to all agents except source
            for agent_id, queue in self._agent_queues.items():
                if agent_id != message.source:
                    try:
                        queue.put_nowait((-message.priority, message))
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for agent: {agent_id}")

        # Notify subscribers
        for callback in self._subscribers[message.message_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")

    async def receive(self, agent_id: str, timeout: float = 1.0) -> Optional[Message]:
        """Receive a message for a specific agent."""
        if agent_id not in self._agent_queues:
            return None

        try:
            _, message = await asyncio.wait_for(
                self._agent_queues[agent_id].get(),
                timeout=timeout
            )
            return message
        except asyncio.TimeoutError:
            return None

    def get_pending_count(self, agent_id: str) -> int:
        """Get number of pending messages for an agent."""
        if agent_id in self._agent_queues:
            return self._agent_queues[agent_id].qsize()
        return 0

    def get_total_message_count(self) -> int:
        """Get total messages processed."""
        return self._message_count

    def get_registered_agents(self) -> List[str]:
        """Get list of registered agents."""
        return list(self._agent_queues.keys())


# Convenience functions for creating messages
def create_sensor_message(source: str, sensor_id: str, temperature: float,
                         humidity: float, target: Optional[str] = None) -> Message:
    """Create a sensor data message."""
    return Message(
        priority=MessagePriority.NORMAL,
        message_type=MessageType.SENSOR_DATA,
        source=source,
        target=target,
        payload={
            "sensor_id": sensor_id,
            "temperature": temperature,
            "humidity": humidity,
            "timestamp": datetime.now().isoformat()
        }
    )


def create_prediction_message(source: str, sensor_id: str, predicted_value: float,
                             horizon_minutes: int, confidence: float,
                             target: Optional[str] = None) -> Message:
    """Create a prediction message."""
    return Message(
        priority=MessagePriority.NORMAL,
        message_type=MessageType.PREDICTION,
        source=source,
        target=target,
        payload={
            "sensor_id": sensor_id,
            "predicted_value": predicted_value,
            "horizon_minutes": horizon_minutes,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    )


def create_alert_message(source: str, sensor_id: str, alert_type: str,
                        severity: str, message: str,
                        target: Optional[str] = None) -> Message:
    """Create an alert message."""
    priority = MessagePriority.CRITICAL if severity == "CRITICAL" else MessagePriority.HIGH
    return Message(
        priority=priority,
        message_type=MessageType.ALERT,
        source=source,
        target=target,
        payload={
            "alert_id": str(uuid.uuid4()),
            "sensor_id": sensor_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    )


def create_decision_message(source: str, decision_type: str, action: str,
                           parameters: Dict[str, Any],
                           target: Optional[str] = None) -> Message:
    """Create a decision message."""
    return Message(
        priority=MessagePriority.HIGH,
        message_type=MessageType.DECISION,
        source=source,
        target=target,
        payload={
            "decision_type": decision_type,
            "action": action,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        }
    )
