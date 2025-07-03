# Cyber-Physical Systems with Time-Series Forecasting

**Author:** Udisha Dutta Chowdhury  
**Supervisor:** Prof. Rolando Herrero

A full-stack cyber-physical system for capturing, processing, and analyzing time-series sensor data with ML-powered forecasting and real-time edge inference.

## Project Overview

This project implements a comprehensive cyber-physical pipeline that:
- Captures time-series sensor data from microcontrollers (Arduino/Raspberry Pi)
- Processes data using ML models (Scikit-learn, XGBoost, skforecast)
- Provides real-time edge inference capabilities
- Visualizes data through cloud-based dashboards
- Ensures security and privacy compliance
- Implements CI/CD workflows for reproducibility

## Project Structure

```
├── data-collection/    # Sensor data acquisition from microcontrollers
├── ml-models/         # Time-series forecasting models and training
├── edge-inference/    # Lightweight ML inference for edge devices
├── cloud-dashboard/   # Visualization dashboards (Streamlit, Ubidots)
├── security/          # Encryption protocols and privacy compliance
├── ci-cd/            # CI/CD workflows and version control
├── data/             # Raw and processed datasets
└── docs/             # Project documentation and guides
```

## Key Features

- **Real-time Data Collection**: Temperature, humidity, pressure sensors
- **ML Forecasting**: Predictive models for critical system variables
- **Edge Computing**: Low-latency predictions on edge devices
- **Cloud Dashboards**: Historical data and forecast visualization
- **Security**: Encrypted protocols and privacy compliance
- **DevOps**: Automated CI/CD with Docker and MLflow

## Getting Started

1. Clone the repository
2. Follow setup instructions in each component's README
3. Configure your sensor devices
4. Deploy models and dashboards
5. Monitor system health and predictions

## Technologies Used

- **Hardware**: Arduino, Raspberry Pi
- **ML**: Scikit-learn, XGBoost, skforecast
- **Visualization**: Streamlit, Ubidots
- **DevOps**: Docker, MLflow, GitHub Actions
- **Security**: Encrypted protocols, privacy frameworks
