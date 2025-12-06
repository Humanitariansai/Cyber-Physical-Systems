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

## Configuration

Edit `config.py` to customize:
- Database path
- LLM provider (Anthropic/OpenAI)
- Analysis parameters
- Feature flags

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
