# Cyber-Physical Systems - Predictive Cold Chain Monitoring

A distributed multi-agent system for predictive pharmaceutical cold chain monitoring, combining real-time control with AI-powered advisory capabilities.

## Overview

This project addresses the $35 billion annual problem of pharmaceutical cold chain failures through:
- **30-60 minute predictive early warning** before temperature breaches
- **Sub-100ms real-time control** for safety-critical decisions
- **Multi-agent architecture** with autonomous, coordinated agents
- **Hybrid AI approach** separating fast control from intelligent advisory
- **Edge deployment** capability on Raspberry Pi 4

## Key Features

- **Predictive Analytics**: ML-powered forecasting (GRU: 93.8% accuracy, LSTM: 94.2% accuracy)
- **Real-time Monitoring**: <100ms response time for control decisions
- **Multi-Agent System**: 4 specialized agents (Monitor, Predictor, Decision, Alert)
- **Event-Driven Architecture**: Priority-based async message bus
- **Interactive Dashboard**: Streamlit-based real-time visualization
- **MLflow Integration**: Experiment tracking and model management
- **CI/CD Pipelines**: Automated testing and deployment
- **MCP Server**: LLM advisory layer for post-incident analysis

## Architecture

### Hybrid Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           REAL-TIME CONTROL CORE (<100ms)                 │  │
│  │                    [Offline Capable]                      │  │
│  │                                                            │  │
│  │   Sensors → Monitor → Predictor → Decision → Alert        │  │
│  │                          ↓                                 │  │
│  │                    Message Bus (Priority Queue)           │  │
│  │                          ↓                                 │  │
│  │                    Orchestrator                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │        LLM ADVISORY LAYER (1-5s, Post-Incident)          │  │
│  │                                                            │  │
│  │   MCP Server → Root Cause Analysis → Recommendations     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/Humanitariansai/Cyber-Physical-Systems.git
cd Cyber-Physical-Systems

# Install dependencies
pip install -r requirements.txt

# Initialize MLflow
mlflow ui
```

### Run Multi-Agent Demo

```bash
cd multi-agent-system
python demo.py
```

### Run Dashboard

```bash
streamlit run Home.py
```

## Project Structure

```
Cyber-Physical-Systems/
├── multi-agent-system/        # Core multi-agent architecture
│   ├── agent_base.py          # Base agent class
│   ├── monitor_agent.py       # Data validation
│   ├── predictor_agent.py     # Time-series forecasting
│   ├── decision_agent.py      # Control logic
│   ├── alert_agent.py         # Alert management
│   ├── message_bus.py         # Event-driven communication
│   ├── orchestrator.py        # System coordination
│   └── demo.py                # 30-second simulation
│
├── mcp-server/                # MCP server for LLM advisory
│   ├── server.py              # MCP implementation
│   ├── database.py            # Data access layer
│   └── config.py              # Configuration
│
├── ml-models/                 # Machine learning models
│   ├── lstm_forecaster.py     # LSTM (94.2% accuracy)
│   ├── gru_forecaster.py      # GRU (93.8% accuracy)
│   ├── xgboost_forecaster.py  # XGBoost baseline
│   └── basic_forecaster.py    # Simple forecaster
│
├── data-collection/           # Data ingestion pipeline
│   ├── sensor_simulator.py    # Sensor data generation
│   └── data_collector.py      # Data collection
│
├── streamlit-dashboard/       # Interactive dashboard
│   ├── pages/
│   │   ├── 1_Data_Analytics.py
│   │   ├── 2_ML_Models.py
│   │   └── 3_System_Health.py
│   └── components/
│
├── .github/workflows/         # CI/CD pipelines
│   ├── ci.yml                 # Continuous integration
│   ├── cd.yml                 # Continuous deployment
│   └── weekly-health-check.yml
│
└── docs/                      # Documentation
```

## Multi-Agent System

### Components

**Monitor Agent**: Validates sensor data, detects anomalies (Z-score method, 3σ)
- Performance: <5ms per reading, 98.5% accuracy

**Predictor Agent**: Forecasts temperature 30-60 minutes ahead
- Performance: <50ms latency, 85-90% accuracy

**Decision Agent**: Makes control decisions based on rules and predictions
- 4 decision types: MAINTAIN, ADJUST, ALERT, EMERGENCY
- Breach tolerance: 5 minutes

**Alert Agent**: Manages alert lifecycle with auto-escalation
- 5 severity levels: INFO → WARNING → ERROR → CRITICAL → EMERGENCY
- Auto-escalation: 15 minutes, Auto-resolve: 60 minutes

**Message Bus**: Event-driven communication with priority queuing
- 4 priority levels, 13 message types
- Performance: <10ms latency, 1000+ msg/sec

**Orchestrator**: Coordinates agents, monitors health, auto-recovery
- Health checks every 60s, graceful shutdown

### Message Types

- **Sensor**: SENSOR_DATA, VALIDATED_SENSOR_DATA, SENSOR_ANOMALY
- **Prediction**: PREDICTION_RESULT, PREDICTION_WARNING
- **Decision**: CONTROL_DECISION, DECISION_OVERRIDE
- **Alert**: ALERT_TRIGGERED, ALERT_ACKNOWLEDGED, ALERT_RESOLVED, ALERT_ESCALATED
- **System**: AGENT_STATUS, SYSTEM_HEALTH

## Machine Learning Models

| Model | Accuracy | Edge Deploy | Latency | Use Case |
|-------|----------|-------------|---------|----------|
| LSTM | 94.2% | No | ~200ms | High accuracy |
| GRU | 93.8% | Yes | ~150ms | Edge deployment |
| XGBoost | 91.5% | Yes | ~50ms | Fast inference |
| ARIMA | 88.7% | Yes | ~30ms | Statistical baseline |

## MCP Server (LLM Advisory Layer)

Provides intelligent post-incident analysis via Model Context Protocol:

**Resources**:
- Current sensor readings
- Active and historical alerts
- Prediction accuracy metrics
- System health status

**Tools**:
- `analyze_incident`: Root cause analysis
- `query_sensor_history`: Time-series queries
- `find_similar_incidents`: Pattern matching
- `generate_maintenance_report`: Preventive recommendations

**Usage with Claude Desktop**:
```json
{
  "mcpServers": {
    "cold-chain": {
      "command": "python",
      "args": ["path/to/mcp-server/server.py"]
    }
  }
}
```

## Dashboard Features

- **Real-time Monitoring**: Live sensor data with trend visualization
- **Predictive Analytics**: 30-minute forecast display
- **Alert Management**: Active alerts with acknowledgment
- **Model Comparison**: Performance metrics across ML models
- **System Health**: Agent status and message bus metrics
- **MLflow Integration**: Experiment tracking and model registry

## Performance Metrics

```
Message Bus:           <10ms latency (99th percentile)
Agent Processing:      <50ms per message
End-to-End Alert:      <100ms (sensor to alert)
Throughput:            1000+ messages/second
Memory Footprint:      ~50MB (all agents)
CPU Usage:             <5% idle, <20% under load
Prediction Accuracy:   85-90% (30 min ahead)
```

## Use Case: Pharmaceutical Cold Chain

### Problem
- $35B annual pharmaceutical losses
- 25% of vaccines arrive degraded
- 50% failures from temperature excursions
- Manual monitoring is reactive

### Solution
- 30-60 minute early warning before breaches
- Proactive intervention prevents damage
- Edge deployment in resource-constrained environments
- Compliance with FDA 21 CFR Part 11, WHO PQS standards

### ROI (100-zone facility)
- Average incident cost: $50,000
- Incidents prevented/year: 10-15
- Annual savings: $500K - $750K
- System cost: ~$50K
- **Payback period: 1-2 months**

## Development Roadmap

### Phase 1: Foundation ✅ (Completed)
- Multi-agent system implementation
- ML model training and evaluation
- Streamlit dashboard
- CI/CD pipelines

### Phase 2: Integration ✅ (Completed)
- MCP server for LLM advisory
- Database schema for persistence
- Multi-agent README documentation

### Phase 3: Enhancement (In Progress)
- Dashboard cold chain monitoring page
- Real sensor data integration
- Alert persistence and analytics
- ML model integration with predictor agent

### Phase 4: Deployment (Upcoming)
- Raspberry Pi 4 edge deployment
- DHT22 sensor integration
- Real-world validation
- Production hardening

## CI/CD Pipelines

- **ci.yml**: Multi-Python testing (3.10, 3.11, 3.12), linting, type checking
- **cd.yml**: Docker builds, staging/production deployment
- **weekly-health-check.yml**: Scheduled maintenance, security audits
- **pr-checks.yml**: Pre-merge quality gates
- **test.yml**: Comprehensive test automation

## Testing

```bash
# Run multi-agent system tests
cd multi-agent-system/tests
pytest test_multi_agent_system.py -v

# Run ML model tests
cd ml-models
pytest test_*.py -v

# Test coverage: ~85%
```

## Configuration

### Cold Chain Monitoring
```python
# Temperature range: 2-8°C (pharmaceutical standard)
target_temp_min = 2.0
target_temp_max = 8.0

# Prediction window: 30-60 minutes
forecast_horizon_minutes = 30
prediction_interval_seconds = 60

# Alert thresholds
consecutive_warnings_threshold = 2
breach_tolerance_seconds = 300
```

## Technology Stack

- **Language**: Python 3.11
- **Async Runtime**: asyncio
- **ML**: TensorFlow 2.20.0, scikit-learn
- **Dashboard**: Streamlit 1.27.2
- **Tracking**: MLflow 2.8.0
- **Testing**: pytest, pytest-asyncio
- **CI/CD**: GitHub Actions
- **Edge**: Raspberry Pi 4, DHT22 sensors

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style
- Python 3.11+ with type hints
- Black formatter (line length: 100)
- Docstrings for public methods
- Async/await for I/O operations

## Documentation

- [Multi-Agent System](multi-agent-system/README.md)
- [MCP Server](mcp-server/README.md)
- [ML Models](ml-models/README.md)
- [MLflow Guide](ml-models/MLFLOW_GUIDE.md)
- [Local Deployment](LOCAL_DEPLOYMENT.md)
- [Hyperparameter Optimization](HYPERPARAMETER_OPTIMIZATION_GUIDE.md)

## License

MIT License - See [LICENSE](LICENSE) file for details

## References

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [WHO PQS Cold Chain Standards](https://www.who.int/pqs)
- [FDA 21 CFR Part 11](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application)

## Acknowledgments

Built for humanitarian impact in pharmaceutical supply chain management.

## Contact

GitHub: [Humanitariansai/Cyber-Physical-Systems](https://github.com/Humanitariansai/Cyber-Physical-Systems)

---

**Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: December 2025
