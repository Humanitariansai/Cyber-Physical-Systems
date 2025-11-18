# Cyber-Physical Systems with Multi-Agent AI and Predictive Cold Chain Monitoring

![CI Pipeline](https://github.com/Humanitariansai/Cyber-Physical-Systems/workflows/CI%20Pipeline/badge.svg)
![CD Pipeline](https://github.com/Humanitariansai/Cyber-Physical-Systems/workflows/CD%20Pipeline/badge.svg)

**Author:** Udisha Dutta Chowdhury  
**Supervisor:** Prof. Rolando Herrero

A distributed multi-agent cyber-physical system for predictive cold chain monitoring, featuring real-time ML forecasting, edge inference, and LLM-based advisory intelligence via MCP protocol.

## Project Overview

This project implements a novel hybrid architecture combining:
- **Real-time multi-agent coordination** for autonomous monitoring and control
- **Predictive ML models** (LSTM/GRU) for 30-60 minute ahead forecasting
- **Edge deployment** on resource-constrained devices (Raspberry Pi)
- **LLM advisory layer** via Model Context Protocol (MCP)
- **Automated CI/CD** with comprehensive testing and security scanning

### Use Case: Predictive Cold Chain Monitoring
Pharmaceutical and food supply chain monitoring with proactive temperature breach prevention, addressing a $35 billion annual global problem.

## Architecture

### Core Real-Time System 
```
┌─────────────────────────────────────────┐
│  Multi-Agent System                     │
│  ┌──────────┐  ┌──────────┐            │
│  │ Monitor  │→→│Predictor │            │
│  │  Agent   │  │  Agent   │            │
│  └──────────┘  └──────────┘            │
│       ↓              ↓                  │
│  ┌──────────┐  ┌──────────┐            │
│  │ Decision │→→│  Alert   │            │
│  │  Agent   │  │  Agent   │            │
│  └──────────┘  └──────────┘            │
│                                         │
│  Response: <100ms | Offline-Capable    │
└─────────────────────────────────────────┘
         │ MCP Protocol
         ↓
┌─────────────────────────────────────────┐
│  LLM Advisory Layer                     │
│  • Post-incident analysis               │
│  • Maintenance recommendations          │
│  • Natural language queries             │
│  Response: 1-5s | Advisory Only         │
└─────────────────────────────────────────┘
```

## Key Features

### Real-Time Control
- **Multi-Agent Coordination**: 4 specialized agents with event-driven communication
- **Predictive Forecasting**: 30-60 minute ahead temperature/humidity predictions
- **Proactive Alerts**: Warning before threshold breach (not reactive)
- **Edge Inference**: <100ms latency on Raspberry Pi
- **Offline-Capable**: No cloud dependency for critical decisions
- **Deterministic**: Regulatory compliant, auditable decisions

### LLM Advisory Layer
- **Post-Incident Analysis**: Root cause identification
- **Maintenance Prediction**: Equipment health recommendations
- **Natural Language Interface**: Operator queries and insights
- **Configuration Optimization**: Data-driven threshold tuning

### DevOps & Quality
- **Automated CI/CD**: GitHub Actions for testing and deployment
- **Security Scanning**: Bandit vulnerability detection
- **Code Coverage**: Automated test coverage reporting
- **Docker Support**: Containerized deployment
- **MLflow Integration**: Experiment tracking and model versioning

## Getting Started

### Prerequisites
```bash
Python 3.10+
Git
Docker (optional)
Raspberry Pi 4 (optional, for hardware deployment)
```

### Installation
```bash
# Clone repository
git clone https://github.com/Humanitariansai/Cyber-Physical-Systems.git
cd Cyber-Physical-Systems

# Install dependencies
pip install -r requirements.txt

# Run dashboard
cd streamlit-dashboard
streamlit run app.py
```

### Quick Demo
```bash
# Simulate cold chain monitoring with predictive alerts
python data-collection/demo_advanced_simulation.py

# Train GRU model for forecasting
cd ml-models
python gru_forecaster.py

# Run test suite
pytest
```

## Implementation Roadmap

### Phase 1: Infrastructure (Completed)
- Data collection framework
- ML forecasting models (LSTM, GRU, XGBoost, ARIMA)
- Streamlit dashboard with multi-page interface
- CI/CD pipeline with automated testing

### Phase 2: Multi-Agent System (In Progress)
- Agent coordination framework
- Message bus implementation
- Cold chain simulation scenarios
- Agent dashboard integration

### Phase 3: MCP & LLM Integration (In Progress)
- MCP server implementation
- Claude/GPT-4 advisory integration
- Post-incident analysis features
- Natural language query interface

### Phase 4: Hardware Deployment (Planned)
- Raspberry Pi + DHT22 sensor integration
- Single-zone validation
- Edge GRU model deployment
- Multi-zone expansion

## Technologies Used

- **Programming**: Python 3.11
- **ML/DL**: TensorFlow, Scikit-learn, XGBoost, skforecast
- **Hardware**: Raspberry Pi 4, Arduino, DHT22 sensors
- **Visualization**: Streamlit, Plotly, Matplotlib
- **AI Integration**: Model Context Protocol (MCP), Claude API
- **DevOps**: GitHub Actions, Docker, MLflow
- **Testing**: pytest, flake8, bandit
- **Protocols**: MQTT, REST API, MCP

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request (use provided template)

All PRs automatically run CI checks (linting, tests, security scans).

## License

This project is part of academic research under the supervision of Prof. Rolando Herrero.

## Contact

**Udisha Dutta Chowdhury**  
GitHub: [@Humanitariansai](https://github.com/Humanitariansai)

---

**Note**: This is an active research project. The multi-agent system and MCP integration are under development.
