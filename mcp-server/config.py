"""
Configuration for MCP server
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MCPServerConfig:
    """MCP server configuration."""
    
    # Database
    db_path: str = "../data/cold_chain.db"
    
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
