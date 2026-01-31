"""
MCP Server for Cold Chain Monitoring System

Provides LLM advisory layer with access to sensor data, alerts, and analysis tools.
Integrates with Claude Desktop for natural language querying of cold chain data.

Components:
- server.py: FastMCP server with tools for data access and analysis
- database.py: Database query interface
- db_logger.py: Real-time data logging to SQLite
- initialize_db.py: Database schema initialization

Tools exposed via MCP:
- get_recent_alerts: Retrieve recent alerts with filtering
- get_sensor_data: Query sensor readings
- analyze_incident: Deep analysis of incidents
- get_system_status: Overall system health
- get_predictions: ML model predictions
"""

__version__ = "0.1.0"

from .database import (
    get_db_connection,
    get_recent_alerts,
    get_sensor_data,
    get_agent_status,
    get_recent_incidents,
    get_system_summary,
)

from .db_logger import DatabaseLogger

__all__ = [
    "get_db_connection",
    "get_recent_alerts",
    "get_sensor_data",
    "get_agent_status",
    "get_recent_incidents",
    "get_system_summary",
    "DatabaseLogger",
]
