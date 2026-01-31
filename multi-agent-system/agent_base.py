"""
Base Agent Class for Multi-Agent System
Provides abstract base class for all agents with common functionality.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import logging

from .message_bus import Message, MessageBus, MessageType

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states."""
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class AgentConfig:
    """Base configuration for agents."""
    heartbeat_interval: float = 5.0  # seconds
    message_timeout: float = 1.0  # seconds
    max_retries: int = 3
    enable_logging: bool = True


@dataclass
class AgentMetrics:
    """Metrics tracking for agents."""
    messages_received: int = 0
    messages_sent: int = 0
    messages_processed: int = 0
    errors_count: int = 0
    last_activity: Optional[datetime] = None
    start_time: Optional[datetime] = None
    processing_times: List[float] = field(default_factory=list)

    def get_avg_processing_time(self) -> float:
        """Get average processing time in seconds."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def record_processing_time(self, time_seconds: float):
        """Record a processing time, keeping last 100."""
        self.processing_times.append(time_seconds)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.

    Provides:
    - Lifecycle management (start, stop, pause, resume)
    - Message handling infrastructure
    - Metrics collection
    - Heartbeat monitoring
    """

    def __init__(self, agent_id: str, bus: MessageBus,
                 config: Optional[AgentConfig] = None):
        self.agent_id = agent_id
        self.bus = bus
        self.config = config or AgentConfig()
        self.state = AgentState.CREATED
        self.metrics = AgentMetrics()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running and self.state == AgentState.RUNNING

    async def start(self):
        """Start the agent."""
        if self.state in [AgentState.RUNNING, AgentState.STARTING]:
            logger.warning(f"Agent {self.agent_id} already running/starting")
            return

        self.state = AgentState.STARTING
        self._running = True
        self.metrics.start_time = datetime.now()

        # Register with message bus
        self.bus.register_agent(self.agent_id)

        # Subscribe to relevant message types
        await self._subscribe_to_messages()

        # Start main processing loop
        self._task = asyncio.create_task(self._run_loop())

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self.state = AgentState.RUNNING
        logger.info(f"Agent {self.agent_id} started")

    async def stop(self):
        """Stop the agent."""
        if self.state in [AgentState.STOPPED, AgentState.STOPPING]:
            return

        self.state = AgentState.STOPPING
        self._running = False

        # Cancel tasks
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Unregister from message bus
        self.bus.unregister_agent(self.agent_id)

        self.state = AgentState.STOPPED
        logger.info(f"Agent {self.agent_id} stopped")

    async def pause(self):
        """Pause the agent."""
        if self.state == AgentState.RUNNING:
            self.state = AgentState.PAUSED
            logger.info(f"Agent {self.agent_id} paused")

    async def resume(self):
        """Resume the agent."""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            logger.info(f"Agent {self.agent_id} resumed")

    async def _run_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                if self.state == AgentState.PAUSED:
                    await asyncio.sleep(0.1)
                    continue

                # Receive message with timeout
                message = await self.bus.receive(
                    self.agent_id,
                    timeout=self.config.message_timeout
                )

                if message:
                    self.metrics.messages_received += 1
                    self.metrics.last_activity = datetime.now()

                    start_time = datetime.now()
                    await self._handle_message(message)
                    processing_time = (datetime.now() - start_time).total_seconds()

                    self.metrics.messages_processed += 1
                    self.metrics.record_processing_time(processing_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.errors_count += 1
                logger.error(f"Agent {self.agent_id} error: {e}")
                if self.metrics.errors_count > self.config.max_retries * 10:
                    self.state = AgentState.ERROR
                    break

    async def _heartbeat_loop(self):
        """Heartbeat loop for monitoring."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if self.state == AgentState.RUNNING:
                    await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error for {self.agent_id}: {e}")

    async def _send_heartbeat(self):
        """Send heartbeat message."""
        heartbeat = Message(
            priority=1,
            message_type=MessageType.HEARTBEAT,
            source=self.agent_id,
            payload={
                "state": self.state.name,
                "metrics": {
                    "messages_processed": self.metrics.messages_processed,
                    "errors": self.metrics.errors_count,
                    "avg_processing_time": self.metrics.get_avg_processing_time()
                }
            }
        )
        await self.bus.publish(heartbeat)

    async def send_message(self, message: Message):
        """Send a message through the bus."""
        message.source = self.agent_id
        await self.bus.publish(message)
        self.metrics.messages_sent += 1

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.name,
            "messages_processed": self.metrics.messages_processed,
            "messages_sent": self.metrics.messages_sent,
            "errors_count": self.metrics.errors_count,
            "avg_processing_time": self.metrics.get_avg_processing_time(),
            "uptime_seconds": (datetime.now() - self.metrics.start_time).total_seconds()
                if self.metrics.start_time else 0
        }

    @abstractmethod
    async def _subscribe_to_messages(self):
        """Subscribe to relevant message types. Override in subclass."""
        pass

    @abstractmethod
    async def _handle_message(self, message: Message):
        """Handle incoming message. Override in subclass."""
        pass
