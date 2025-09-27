# Local Deployment Guide

## Overview

This guide demonstrates the local deployment of the cyber-physical systems ML pipeline using Docker containers.

## System Architecture

The deployment consists of 5 containerized services:
- MLflow Server: Experiment tracking and model registry
- Data Collector: Sensor data simulation
- Model Trainer: Hyperparameter optimization
- Prediction API: REST endpoints for forecasting
- PostgreSQL: Production database backend

## Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0+
- At least 4GB RAM available
- Ports 5000 and 8080 available

## Quick Start

### 1. Build the System

```cmd
docker-compose build
```

### 2. Start All Services

```cmd
docker-compose up -d
```

### 3. Verify Services

```cmd
docker-compose ps
```

### 4. Access Interfaces

- MLflow Dashboard: http://localhost:5000
- Prediction API: http://localhost:8080
- Health Check: http://localhost:8080/health

## Testing the Deployment

### Health Check
```cmd
curl http://localhost:8080/health
```

### Make a Prediction
```cmd
curl -X POST http://localhost:8080/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"temperature_history\": [22.5, 23.1, 22.8, 23.3, 22.9]}"
```

## Monitoring

### View Logs
```cmd
docker-compose logs -f
```

### Check Service Status
```cmd
docker-compose ps
```

### Resource Usage
```cmd
docker stats
```

## Stopping the System

```cmd
docker-compose down
```

## Troubleshooting

### Port Conflicts
If ports 5000 or 8080 are in use:
1. Stop conflicting services
2. Or modify ports in docker-compose.yml

### Memory Issues
Ensure Docker Desktop has at least 4GB RAM allocated.

### Container Startup Issues
Check logs:
```cmd
docker-compose logs [service-name]
```

## Success Criteria

Deployment is successful when:
- All 5 containers are running
- MLflow UI loads at localhost:5000
- API health check returns "healthy" status
- Prediction endpoint accepts requests
- No error logs in container output

## Performance Expectations

- Container startup: 2-3 minutes
- API response time: < 1 second
- Model loading time: < 30 seconds
- Memory usage: < 2GB total