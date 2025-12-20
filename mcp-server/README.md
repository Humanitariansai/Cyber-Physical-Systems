# MCP Server for Cold Chain Monitoring

Model Context Protocol server providing LLM advisory layer for the cold chain monitoring system.

## Overview

This MCP server exposes sensor data, alerts, and analysis tools to LLM clients (Claude Desktop, VS Code, etc.) for intelligent post-incident analysis and natural language queries.

## Features

### Resources (Read-Only Data)
- `sensor://current-readings` - Real-time sensor data
- `alerts://active` - Currently active alerts
- `alerts://history` - Historical alert records
- `predictions://accuracy` - Model performance metrics
- `system://health` - Multi-agent system status

### Tools (Actions)
- `analyze_incident` - Root cause analysis for temperature breaches
- `query_sensor_history` - Time-series data queries
- `find_similar_incidents` - Pattern matching across alerts
- `generate_maintenance_report` - Preventive maintenance recommendations

### Prompts (Templates)
- `analyze_latest_incident` - Quick incident analysis
- `system_performance_summary` - Weekly performance report
- `troubleshoot_sensor` - Sensor-specific diagnostics

## Installation

```bash
cd mcp-server
pip install -r requirements.txt
```

## Quick Start

### 1. Initialize Database

```bash
python mcp-server/initialize_db.py
```

This creates the database schema with tables for:
- `sensor_data` - Temperature/humidity readings
- `alerts` - Alert lifecycle tracking
- `predictions` - Forecast results
- `system_metrics` - Performance metrics
- `agent_status` - Multi-agent health

### 2. Run Integrated Demo

```bash
python examples/integrated_demo.py
```

This runs a 20-second simulation that:
- Starts the multi-agent system
- Logs all data to the database
- Simulates a temperature breach scenario
- Records alerts and predictions

### 3. Query with MCP Server

```bash
python mcp-server/server.py
```

Then use Claude Desktop to ask questions about the data.

## Configuration

Edit `config.py` to customize:
- Database path
- LLM provider (Anthropic/OpenAI)
- Analysis parameters
- Feature flags

## Database Logging

Use `db_logger.py` to persist multi-agent system data:

```python
from mcp_server.db_logger import DatabaseLogger

logger = DatabaseLogger("data/cold_chain.db")

# Log sensor reading
logger.log_sensor_data("sensor-01", temperature=5.2, humidity=65.0)

# Log alert
logger.log_alert("alert-001", "sensor-01", "TEMPERATURE_HIGH", 
                 "CRITICAL", "Temperature exceeded 8°C")

# Update alert status
logger.update_alert_status("alert-001", resolved_at="2025-12-12T10:30:00")

# Log prediction
logger.log_prediction("sensor-01", predicted_value=7.5, 
                     confidence=0.85, horizon_minutes=30)
```

## Usage with Claude Desktop

1. Add to Claude Desktop config:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "cold-chain": {
      "command": "python",
      "args": [
        "C:/path/to/mcp-server/server.py"
      ],
      "env": {
        "DB_PATH": "../data/cold_chain.db"
      }
    }
  }
}
```

2. Restart Claude Desktop

3. Ask questions like:
   - "What caused the temperature breach at 2 PM?"
   - "Which sensors need maintenance?"
   - "How accurate were predictions this week?"

## Architecture

```
Real-Time Core (Multi-Agent) → SQLite Database → MCP Server → LLM Client
                                                              ↓
                                                   Natural Language Analysis
```

## Development Status

- [x] Database schema and queries
- [x] MCP server implementation
- [x] Basic resources and tools
- [ ] Advanced incident analysis algorithms
- [ ] LLM-powered root cause analysis
- [ ] Integration with multi-agent system logs
- [ ] Real-time monitoring integration

## Future Enhancements

- Advanced pattern recognition with ML
- Predictive maintenance scoring
- Multi-facility analytics
- Custom report generation
- Real-time streaming updates
