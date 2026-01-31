"""
Multi-Agent System for Cold Chain Monitoring

This module provides a distributed multi-agent architecture for real-time
cold chain monitoring in pharmaceutical supply chains.

Agents:
- MonitorAgent: Sensor data ingestion and anomaly detection
- PredictorAgent: ML-based temperature forecasting (30-60 minute horizons)
- DecisionAgent: Threshold-based decision making and escalation
- AlertAgent: Alert lifecycle management and notifications
- SystemOrchestrator: Agent coordination and system management

Message Bus:
- Asynchronous inter-agent communication
- Priority-based message routing
- Pub/sub pattern for event distribution
"""

__version__ = "0.1.0"

from .message_bus import (
    Message,
    MessageBus,
    MessageType,
    MessagePriority,
    create_sensor_message,
    create_prediction_message,
    create_alert_message,
    create_decision_message,
)

from .agent_base import (
    BaseAgent,
    AgentConfig,
    AgentState,
    AgentMetrics,
)

from .monitor_agent import (
    MonitorAgent,
    MonitorAgentConfig,
)

from .predictor_agent import (
    PredictorAgent,
    PredictorAgentConfig,
    PredictionResult,
)

from .decision_agent import (
    DecisionAgent,
    DecisionAgentConfig,
)

from .alert_agent import (
    AlertAgent,
    AlertAgentConfig,
    AlertRecord,
)

from .orchestrator import (
    SystemOrchestrator,
    create_default_system,
)

__all__ = [
    # Message Bus
    "Message",
    "MessageBus",
    "MessageType",
    "MessagePriority",
    "create_sensor_message",
    "create_prediction_message",
    "create_alert_message",
    "create_decision_message",
    # Base
    "BaseAgent",
    "AgentConfig",
    "AgentState",
    "AgentMetrics",
    # Agents
    "MonitorAgent",
    "MonitorAgentConfig",
    "PredictorAgent",
    "PredictorAgentConfig",
    "PredictionResult",
    "DecisionAgent",
    "DecisionAgentConfig",
    "AlertAgent",
    "AlertAgentConfig",
    "AlertRecord",
    # Orchestrator
    "SystemOrchestrator",
    "create_default_system",
]
