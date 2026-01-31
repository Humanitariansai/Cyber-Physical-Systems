# Predictive Cold Chain Monitoring with Multi-Agent AI

A distributed multi-agent system for predictive cold chain monitoring in pharmaceutical supply chains. Uses LSTM/GRU neural networks for 30-60 minute ahead temperature forecasts, integrated with an LLM advisory layer via Model Context Protocol (MCP).

## Features

- **Multi-Agent Architecture** — Distributed system with specialized Monitor, Predictor, Decision, and Alert agents communicating via priority message bus
- **ML-Based Forecasting** — LSTM, GRU, and XGBoost models with ensemble averaging for 30/60-minute ahead temperature predictions
- **LLM Advisory Layer** — MCP server providing natural language root cause analysis and maintenance recommendations via Claude Desktop
- **Real-Time Dashboard** — Streamlit + Plotly visualization with 5 pages: overview, sensor data, predictions, alerts, system health
- **MLflow Experiment Tracking** — Model versioning, metric logging, and hyperparameter comparison
- **CI/CD Pipeline** — GitHub Actions with flake8 linting, bandit security scanning, pytest, Docker build, and automated deployment
- **Pharmaceutical Compliance** — Temperature thresholds aligned with WHO cold chain standards (2-8°C)

## Architecture

```
+-------------------------------------------------------------------+
|                    Cold Chain Monitoring System                     |
+-------------------------------------------------------------------+
|                                                                     |
|  +-------------+   +-------------+   +-------------+               |
|  |   Monitor   |   |  Predictor  |   |  Decision   |               |
|  |   Agent     |   |   Agent     |   |   Agent     |               |
|  | - Ingestion |   | - LSTM/GRU  |   | - Threshold |               |
|  | - Validation|   | - XGBoost   |   | - Escalation|               |
|  | - Anomaly   |   | - 30-60 min |   | - Actions   |               |
|  +------+------+   +------+------+   +------+------+               |
|         |                 |                 |                       |
|         +--------+--------+---------+-------+                       |
|                  |                                                   |
|         +--------v--------+                                         |
|         |   Message Bus   |                                         |
|         | (Priority Queue)|                                         |
|         +--------+--------+                                         |
|                  |                                                   |
|     +------------+------------+                                     |
|     |            |            |                                     |
| +---v---+  +----v----+  +----v----+                                |
| | Alert |  | Orchest-|  |Database |                                |
| | Agent |  |  rator  |  |(SQLite) |                                |
| +-------+  +---------+  +----+----+                                |
|                               |                                     |
|              +----------------+----------------+                    |
|              |                                 |                    |
|       +------v------+                  +-------v------+            |
|       |  MCP Server |                  |  Streamlit   |            |
|       | (LLM Layer) |                  |  Dashboard   |            |
|       +-------------+                  +--------------+            |
+-------------------------------------------------------------------+
```

## Project Structure

```
Cyber-Physical-Systems/
|
|-- multi-agent-system/              # Distributed agent architecture
|   |-- message_bus.py               # Priority-based async pub/sub messaging
|   |-- agent_base.py                # Abstract base class for all agents
|   |-- monitor_agent.py             # Sensor ingestion and anomaly detection
|   |-- predictor_agent.py           # ML-based temperature forecasting
|   |-- decision_agent.py            # Threshold logic and escalation
|   |-- alert_agent.py               # Alert lifecycle management
|   |-- orchestrator.py              # System coordination and agent management
|   |-- demo.py                      # Standalone multi-agent demo
|   |-- test_multi_agent_system.py   # Unit tests
|   |-- __init__.py
|
|-- ml-models/                       # Machine learning forecasters
|   |-- lstm_forecaster.py           # LSTM neural network (TensorFlow)
|   |-- gru_forecaster.py            # GRU neural network (TensorFlow)
|   |-- xgboost_forecaster.py        # Gradient boosting with feature engineering
|   |-- basic_forecaster.py          # Baseline: Moving Average + Exponential Smoothing
|   |-- simple_arima_forecaster.py   # ARIMA forecaster (statsmodels)
|   |-- advanced_mlflow_models.py    # Ensemble forecaster (weighted averaging)
|   |-- mlflow_tracking.py           # MLflow experiment tracking wrapper
|   |-- mlflow_experiment_runner.py  # Train all models and log to MLflow
|   |-- mlflow_dashboard.py          # Streamlit page for MLflow results
|   |-- compare_models.py            # Side-by-side model comparison script
|   |-- test_lstm_forecaster.py      # LSTM unit tests
|   |-- test_gru_forecaster.py       # GRU unit tests
|   |-- test_xgboost_forecaster.py   # XGBoost unit tests
|   |-- test_basic_forecaster.py     # Baseline model unit tests
|   |-- test_moving_average_forecaster.py
|   |-- __init__.py
|
|-- mcp-server/                      # LLM advisory layer via MCP
|   |-- server.py                    # MCP server with tools, resources, prompts
|   |-- database.py                  # Database query interface
|   |-- db_logger.py                 # Real-time sensor/alert logging
|   |-- initialize_db.py            # SQLite schema initialization
|   |-- config.py                    # Server configuration
|   |-- __init__.py
|
|-- streamlit-dashboard/             # Real-time visualization
|   |-- app.py                       # Main multi-page dashboard
|   |-- pages/
|   |   |-- 1_Data_Analytics.py      # Temperature distribution, hourly patterns
|   |   |-- 2_ML_Models.py           # Model comparison and accuracy metrics
|   |   |-- 3_System_Health.py       # Agent status and system metrics
|   |-- components/
|   |   |-- metrics_cards.py         # Reusable KPI and status cards
|   |   |-- sidebar.py              # Navigation sidebar component
|   |-- config/
|   |   |-- dashboard_config.py     # Thresholds, colors, settings
|   |-- utils/
|       |-- data_loader.py          # SQLite data loading utilities
|       |-- ml_integration.py       # ML model loading for dashboard
|       |-- path_setup.py           # Project path utilities
|
|-- .github/workflows/
|   |-- ci.yml                       # CI/CD: lint, security, test, docker, deploy
|
|-- docker/                          # Container configuration
|-- docker-compose.yml               # Multi-service orchestration
|-- Dockerfile                       # Production container build
|-- examples/
|   |-- integrated_demo.py           # Full system demo with DB logging
|-- data/                            # SQLite database storage
|-- docs/                            # Documentation and reports
|-- requirements.txt
|-- LICENSE
```

## Installation

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### Setup

```bash
# Clone
git clone https://github.com/udishadc/Cyber-Physical-Systems.git
cd Cyber-Physical-Systems

# Virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# Full ML model support
pip install tensorflow>=2.10.0 xgboost>=1.7.0

# Dashboard
pip install streamlit>=1.28.0 plotly>=5.18.0

# Experiment tracking
pip install mlflow>=2.0.0

# ARIMA models
pip install statsmodels>=0.14.0

# MCP server
pip install mcp
```

## Quick Start

### 1. Generate Sample Data

```bash
python generate_sample_data.py
```

Populates the SQLite database with 7 days of sensor readings, alerts, and predictions.

### 2. Run Streamlit Dashboard

```bash
python -m streamlit run streamlit-dashboard/app.py
```

Open http://localhost:8501 — navigate through Dashboard, Sensor Data, Predictions, Alerts, and System Health pages.

### 3. Run Multi-Agent Demo

```bash
python multi-agent-system/demo.py
```

Simulates 10 seconds of sensor data flowing through the agent pipeline with live console output.

### 4. Run Integrated Demo (Full Pipeline)

```bash
python examples/integrated_demo.py
```

Starts multi-agent system, simulates 20 seconds of data with temperature anomalies, logs everything to the database.

### 5. Start MCP Server (Claude Desktop Integration)

```bash
python mcp-server/server.py
```

Add to Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cold-chain": {
      "command": "python",
      "args": ["C:/path/to/Cyber-Physical-Systems/mcp-server/server.py"]
    }
  }
}
```

Then ask Claude: *"What alerts were triggered recently?"* or *"Analyze the latest temperature incident."*

## Multi-Agent System

### Agents

| Agent | Role | Key Logic |
|-------|------|-----------|
| **MonitorAgent** | First line of defense | Validates data, detects threshold breaches, statistical anomalies, rate-of-change alerts |
| **PredictorAgent** | Forecasting | Runs LSTM/GRU/XGBoost ensemble, produces 30/60-min predictions with confidence scores |
| **DecisionAgent** | Decision making | Applies pharmaceutical business rules, escalation logic, preemptive action triggers |
| **AlertAgent** | Alert management | Lifecycle: triggered → acknowledged → resolved. Deduplication, cooldown, auto-escalation |

### Message Bus

Asynchronous priority-based routing:

| Priority | Level | Use Case |
|----------|-------|----------|
| CRITICAL | 4 | Emergency — immediate action |
| HIGH | 3 | Threshold violations, escalations |
| NORMAL | 2 | Regular sensor data, predictions |
| LOW | 1 | Heartbeats, status updates |

## ML Models

| Model | Architecture | 30-min MAE | 60-min MAE | Training Time |
|-------|-------------|-----------|-----------|---------------|
| **LSTM** | 2-layer (64, 32), dropout 0.2 | ~0.31°C | ~0.58°C | ~2 min |
| **GRU** | 64 units, bidirectional option | ~0.35°C | ~0.63°C | ~1.2 min |
| **XGBoost** | 100 estimators, depth 6, lag features | ~0.42°C | ~0.74°C | ~10 sec |
| **Ensemble** | Weighted: LSTM 0.4, GRU 0.35, XGB 0.25 | ~0.28°C | ~0.55°C | Combined |
| **Baseline** | Moving Average + Exponential Smoothing | ~0.48°C | ~0.85°C | <1 sec |

All models fall back gracefully if optional dependencies (TensorFlow, XGBoost, statsmodels) are not installed.

### MLflow Tracking

```bash
# Run experiments
python ml-models/mlflow_experiment_runner.py

# Compare models
python ml-models/compare_models.py

# View MLflow UI
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

## MCP Server — LLM Advisory Layer

### Resources (Read-Only Data)

| URI | Description |
|-----|-------------|
| `sensor://current-readings` | Real-time temperature/humidity |
| `alerts://active` | Active alerts needing attention |
| `alerts://history` | 30-day alert history |
| `predictions://accuracy` | Model performance stats |
| `system://health` | Agent and message bus health |

### Tools (Actions LLM Can Execute)

| Tool | Description |
|------|-------------|
| `analyze_incident` | Root cause analysis with trend detection and similar incident matching |
| `query_sensor_history` | Historical data with computed statistics |
| `find_similar_incidents` | Pattern matching against past events |
| `generate_maintenance_report` | Preventive maintenance recommendations |

### Prompts (Pre-Built Templates)

- `analyze_latest_incident` — Analyze most recent temperature breach
- `system_performance_summary` — Weekly system performance overview
- `troubleshoot_sensor` — Diagnose issues with a specific sensor

## Streamlit Dashboard

| Page | Content |
|------|---------|
| **Dashboard** | KPI cards, live temperature chart with thresholds, humidity overlay, predictions, active alerts |
| **Sensor Data** | Statistics, temperature histogram, raw data table, CSV export |
| **Predictions** | 30/60-min forecasts with confidence bars, historical vs predicted chart |
| **Alerts** | Severity filter, acknowledge button, expandable alert details |
| **System Health** | Agent status cards, message bus throughput, prediction accuracy, MLflow link |

Additional Streamlit pages available at:
- `pages/1_Data_Analytics.py` — Hourly patterns, distribution analysis, correlation heatmap
- `pages/2_ML_Models.py` — Model cards, comparison table, accuracy bar chart
- `pages/3_System_Health.py` — Agent metrics, message type distribution

## CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
Stages:
  1. Lint      → flake8 code quality
  2. Security  → bandit vulnerability scanning
  3. Test      → pytest with coverage
  4. Docker    → Multi-stage build and push
  5. Deploy    → Automated staging/production
```

Triggers on push to `main`/`develop` and all pull requests.

## Docker

```bash
# Build
docker build -t cold-chain-monitor .

# Run
docker run -p 8501:8501 -p 8000:8000 cold-chain-monitor

# Or use Docker Compose
docker-compose up -d
```

## Configuration

### Temperature Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temp_min` | 2.0°C | Minimum safe (WHO standard) |
| `temp_max` | 8.0°C | Maximum safe |
| `critical_min` | 0.0°C | Critical low — freezing risk |
| `critical_max` | 10.0°C | Critical high |

### Prediction Horizons

| Horizon | Use Case |
|---------|----------|
| 30 min | Immediate operational decisions |
| 60 min | Logistics planning, preemptive actions |

## Testing

```bash
# Multi-agent system tests
python -m pytest multi-agent-system/test_multi_agent_system.py -v

# ML model tests
python -m pytest ml-models/test_*.py -v

# All tests
python -m pytest --tb=short
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | Python asyncio |
| Communication | Custom Priority Message Bus |
| Neural Networks | TensorFlow / Keras |
| Gradient Boosting | XGBoost |
| LLM Integration | MCP (Model Context Protocol) |
| Dashboard | Streamlit + Plotly |
| Experiment Tracking | MLflow |
| Database | SQLite |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Security Scanning | Bandit |

## Future Scope

- **Real IoT Sensor Integration** — Connect to physical DHT22/DS18B20 sensors via MQTT for live cold storage monitoring
- **Cloud Deployment** — Deploy to Azure IoT Hub or AWS IoT Core for multi-site scalability
- **Edge Inference** — Run lightweight TFLite models on Raspberry Pi at sensor locations for low-latency predictions
- **Advanced Anomaly Detection** — Implement Isolation Forest and autoencoder-based anomaly detection for unsupervised pattern discovery
- **Mobile Alerts** — Push notifications via Twilio SMS/WhatsApp for on-call operators
- **Regulatory Reporting** — Automated FDA 21 CFR Part 211 compliant temperature excursion reports
- **Digital Twin** — 3D visualization of cold storage facilities with real-time sensor overlay
- **Multi-Site Monitoring** — Scale to monitor multiple warehouses, transport vehicles, and pharmacy locations from a single dashboard
- **Federated Learning** — Train models across distributed sites without centralizing sensitive data
- **Reinforcement Learning** — Optimize HVAC control actions based on predicted temperature trajectories

## License

MIT License — see [LICENSE](LICENSE).
