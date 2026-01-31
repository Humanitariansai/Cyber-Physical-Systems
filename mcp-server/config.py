"""
Configuration for MCP server
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _default_db_path() -> str:
    return str(Path(__file__).parent.parent.absolute() / "data" / "cold_chain.db")


@dataclass
class MCPServerConfig:
    """MCP server configuration."""

    # Database
    db_path: str = field(default_factory=_default_db_path)
    
    # Server
    server_name: str = "cold-chain-monitor"
    version: str = "0.1.0"
    
    # LLM API (for future use)
    llm_provider: Optional[str] = None  # "anthropic" or "openai"
    api_key: Optional[str] = None
    model: str = "claude-3-5-sonnet-20241022"
    
    # Analysis settings
    default_time_window_hours: int = 24
    similar_incidents_limit: int = 5
    prediction_accuracy_days: int = 7
    
    # Feature flags
    enable_root_cause_analysis: bool = True
    enable_maintenance_recommendations: bool = True
    enable_natural_language_queries: bool = True


# Default configuration
default_config = MCPServerConfig()
