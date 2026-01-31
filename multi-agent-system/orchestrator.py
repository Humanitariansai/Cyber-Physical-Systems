"""
System Orchestrator for Multi-Agent Cold Chain Monitoring
Coordinates all agents and provides unified system management.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from .message_bus import MessageBus, Message, MessageType, create_sensor_message
from .agent_base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class SystemOrchestrator:
    """
    System Orchestrator: Coordinates the multi-agent system.

    Responsibilities:
    - Start/stop all agents
    - Monitor agent health
    - Provide unified system status
    - Inject external data into the system
    - Handle system-wide commands
    """

    def __init__(self, bus: Optional[MessageBus] = None):
        self.bus = bus or MessageBus()
        self._agents: Dict[str, BaseAgent] = {}
        self._running = False
        self._start_time: Optional[datetime] = None
        self._health_check_task: Optional[asyncio.Task] = None

    def add_agent(self, name: str, agent: BaseAgent):
        """Add an agent to the system."""
        self._agents[name] = agent
        logger.info(f"Agent added: {name} ({agent.agent_id})")

    def remove_agent(self, name: str):
        """Remove an agent from the system."""
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Agent removed: {name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)

    async def start(self):
        """Start the entire multi-agent system."""
        if self._running:
            logger.warning("System already running")
            return

        logger.info("=" * 50)
        logger.info("Starting Multi-Agent Cold Chain Monitoring System")
        logger.info("=" * 50)

        self._start_time = datetime.now()

        # Start message bus
        await self.bus.start()

        # Start all agents
        for name, agent in self._agents.items():
            try:
                await agent.start()
                logger.info(f"  Started: {name}")
            except Exception as e:
                logger.error(f"  Failed to start {name}: {e}")

        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._running = True
        logger.info("System started successfully")
        logger.info("=" * 50)

    async def stop(self):
        """Stop the entire multi-agent system."""
        if not self._running:
            return

        logger.info("Stopping Multi-Agent System...")

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop all agents
        for name, agent in self._agents.items():
            try:
                await agent.stop()
                logger.info(f"  Stopped: {name}")
            except Exception as e:
                logger.error(f"  Error stopping {name}: {e}")

        # Stop message bus
        await self.bus.stop()

        self._running = False
        logger.info("System stopped")

    async def _health_check_loop(self):
        """Periodic health check for all agents."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                unhealthy = []
                for name, agent in self._agents.items():
                    if agent.state == AgentState.ERROR:
                        unhealthy.append(name)
                    elif agent.state not in [AgentState.RUNNING, AgentState.PAUSED]:
                        unhealthy.append(name)

                if unhealthy:
                    logger.warning(f"Unhealthy agents: {unhealthy}")
                    # Could implement auto-restart here

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def inject_sensor_data(self, sensor_id: str, temperature: float,
                                 humidity: float = 50.0):
        """Inject sensor data into the system."""
        if not self._running:
            logger.warning("Cannot inject data - system not running")
            return

        message = create_sensor_message(
            source="external",
            sensor_id=sensor_id,
            temperature=temperature,
            humidity=humidity
        )
        await self.bus.publish(message)

    async def send_command(self, command: str, target: Optional[str] = None,
                          parameters: Optional[Dict] = None):
        """Send a command to agents."""
        message = Message(
            priority=3,
            message_type=MessageType.COMMAND,
            source="orchestrator",
            target=target,
            payload={
                "command": command,
                **(parameters or {})
            }
        )
        await self.bus.publish(message)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        agent_statuses = {}
        for name, agent in self._agents.items():
            agent_statuses[name] = agent.get_status()

        uptime = 0
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "agent_count": len(self._agents),
            "message_bus": {
                "total_messages": self.bus.get_total_message_count(),
                "registered_agents": self.bus.get_registered_agents()
            },
            "agents": agent_statuses
        }

    def get_agent_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        agent = self._agents.get(name)
        if agent:
            return agent.get_status()
        return None

    async def pause_agent(self, name: str) -> bool:
        """Pause a specific agent."""
        agent = self._agents.get(name)
        if agent:
            await agent.pause()
            return True
        return False

    async def resume_agent(self, name: str) -> bool:
        """Resume a paused agent."""
        agent = self._agents.get(name)
        if agent:
            await agent.resume()
            return True
        return False

    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._running

    @property
    def agent_names(self) -> List[str]:
        """Get list of agent names."""
        return list(self._agents.keys())


async def create_default_system() -> SystemOrchestrator:
    """Create a default multi-agent system with all agents."""
    from .monitor_agent import MonitorAgent, MonitorAgentConfig
    from .predictor_agent import PredictorAgent, PredictorAgentConfig
    from .decision_agent import DecisionAgent, DecisionAgentConfig
    from .alert_agent import AlertAgent, AlertAgentConfig

    bus = MessageBus()
    orchestrator = SystemOrchestrator(bus)

    # Create agents
    monitor = MonitorAgent(
        agent_id="monitor-01",
        bus=bus,
        config=MonitorAgentConfig()
    )

    predictor = PredictorAgent(
        agent_id="predictor-01",
        bus=bus,
        config=PredictorAgentConfig(
            prediction_horizons=(30, 60),
            prediction_interval=30.0  # Faster predictions for demo
        )
    )

    decision = DecisionAgent(
        agent_id="decision-01",
        bus=bus,
        config=DecisionAgentConfig()
    )

    alert = AlertAgent(
        agent_id="alert-01",
        bus=bus,
        config=AlertAgentConfig()
    )

    # Add agents to orchestrator
    orchestrator.add_agent("monitor", monitor)
    orchestrator.add_agent("predictor", predictor)
    orchestrator.add_agent("decision", decision)
    orchestrator.add_agent("alert", alert)

    return orchestrator
