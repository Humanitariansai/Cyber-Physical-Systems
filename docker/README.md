# Docker Implementation for Cyber-Physical Systems

## **Overview**

This Docker implementation provides a containerized environment for the entire cyber-physical systems ML pipeline, including MLflow tracking, model training, and prediction API services.

## **Prerequisites**

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- At least 4GB RAM available for containers
- Internet connection for initial image building

## **Quick Start**

### 1. **Build and Start Development Environment**

**Windows:**
```cmd
docker\manage.bat build
docker\manage.bat start
```

**Linux/Mac:**
```bash
chmod +x docker/manage.sh
./docker/manage.sh build
./docker/manage.sh start
```

### 2. **Access Services**

- **MLflow Dashboard**: http://localhost:5000
- **Prediction API**: http://localhost:8080
- **API Health Check**: http://localhost:8080/health

## **Architecture**

### **Services Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │  Data           │    │  Model          │
│   Server        │◄───┤  Collector      │◄───┤  Trainer        │
│   (Port 5000)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         │                                              ▼
┌─────────────────┐                          ┌─────────────────┐
│  Prediction     │                          │  PostgreSQL     │
│  API            │                          │  (Production)   │
│  (Port 8080)    │                          │                 │
└─────────────────┘                          └─────────────────┘
```

### **Container Details**

| Service | Container Name | Port | Purpose |
|---------|---------------|------|---------|
| MLflow Server | `cyber-physical-mlflow` | 5000 | Experiment tracking and model registry |
| Data Collector | `cyber-physical-data-collector` | - | Generate and collect sensor data |
| Model Trainer | `cyber-physical-model-trainer` | - | Run hyperparameter optimization |
| Prediction API | `cyber-physical-prediction-api` | 8080 | Serve model predictions via REST API |
| PostgreSQL | `cyber-physical-postgres` | 5432 | Production database (optional) |

## **Usage Instructions**

### **Development Workflow**

1. **Start Services:**
   ```bash
   ./docker/manage.sh start
   ```

2. **View Logs:**
   ```bash
   ./docker/manage.sh logs              # All services
   ./docker/manage.sh logs mlflow-server # Specific service
   ```

3. **Check Status:**
   ```bash
   ./docker/manage.sh status
   ```

4. **Make Predictions:**
   ```bash
   curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"temperature_history": [22.5, 23.1, 22.8, 23.3, 22.9]}'
   ```

### **Production Deployment**

1. **Start with PostgreSQL Backend:**
   ```bash
   ./docker/manage.sh start-prod
   ```

2. **Environment Variables:**
   ```bash
   export MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow123@postgres:5432/mlflow
   export POSTGRES_PASSWORD=your_secure_password
   ```

##  **Configuration**

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `file:///app/mlruns` | MLflow backend store URI |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | `/app/mlruns` | Artifact storage location |
| `PYTHONPATH` | `/app:/app/ml-models:/app/data-collection` | Python module search paths |
| `PORT` | `8080` | Prediction API port |

### **Volume Mounts**

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./mlruns` | `/app/mlruns` | MLflow experiments and artifacts |
| `./data` | `/app/data` | Generated sensor data |
| `./logs` | `/app/logs` | Application logs |

## 🧪 **API Usage Examples**

### **Health Check**
```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-05T10:30:00",
  "model_loaded": true,
  "model_metadata": {
    "run_id": "abc123",
    "rmse": 5.9981,
    "n_lags": 12
  }
}
```

### **Temperature Prediction**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature_history": [22.5, 23.1, 22.8, 23.3, 22.9, 23.0]
  }'
```

**Response:**
```json
{
  "predicted_temperature": 23.15,
  "input_length": 6,
  "model_used": "mlflow_model",
  "timestamp": "2025-09-05T10:30:00",
  "model_metadata": {
    "run_id": "abc123",
    "rmse": 5.9981,
    "n_lags": 12
  }
}
```

### **Reload Model**
```bash
curl -X POST http://localhost:8080/model/reload
```

##  **Management Commands**

### **Windows (manage.bat)**
```cmd
docker\manage.bat build        # Build images
docker\manage.bat start        # Start development
docker\manage.bat start-prod   # Start production
docker\manage.bat stop         # Stop all services
docker\manage.bat clean        # Clean up resources
docker\manage.bat logs         # View logs
docker\manage.bat status       # Check status
docker\manage.bat backup       # Create backup
```

### **Linux/Mac (manage.sh)**
```bash
./docker/manage.sh build       # Build images
./docker/manage.sh start       # Start development
./docker/manage.sh start-prod  # Start production
./docker/manage.sh stop        # Stop all services
./docker/manage.sh clean       # Clean up resources
./docker/manage.sh logs        # View logs
./docker/manage.sh status      # Check status
./docker/manage.sh backup      # Create backup
```

##  **Security Considerations**

### **Development Environment**
- Uses file-based MLflow storage
- No authentication required
- Default passwords for demonstration

### **Production Environment**
- PostgreSQL database with authentication
- Change default passwords:
  ```bash
  export POSTGRES_PASSWORD=your_secure_password
  ```
- Consider adding SSL/TLS certificates
- Implement API authentication

##  **Monitoring & Logs**

### **Log Locations**
- MLflow Server: `docker-compose logs mlflow-server`
- Prediction API: `docker-compose logs prediction-api`
- All services: `docker-compose logs`

### **Health Checks**
- MLflow: http://localhost:5000
- Prediction API: http://localhost:8080/health
- Status script: `./docker/manage.sh status`

## 🚧 **Troubleshooting**

### **Common Issues**

1. **Port Already in Use:**
   ```bash
   # Stop existing services
   ./docker/manage.sh stop
   # Or change ports in docker-compose.yml
   ```

2. **Model Not Loading:**
   ```bash
   # Check MLflow experiments
   curl http://localhost:5000
   # Reload model manually
   curl -X POST http://localhost:8080/model/reload
   ```

3. **Container Health Issues:**
   ```bash
   # Check container status
   docker-compose ps
   # View detailed logs
   ./docker/manage.sh logs
   ```

### **Reset Everything**
```bash
./docker/manage.sh clean   # Removes all containers and images
```

##  **Development Workflow**

1. **Code Changes:** Make changes to Python files
2. **Rebuild:** `./docker/manage.sh build`
3. **Restart:** `./docker/manage.sh stop && ./docker/manage.sh start`
4. **Test:** Access MLflow and API endpoints
5. **Monitor:** `./docker/manage.sh logs`

##  **Performance Optimization**

### **Resource Limits**
Add to `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

### **Caching**
- Docker layer caching enabled
- Python package caching in Dockerfile
- Model artifacts cached in volumes

## 🔮 **Next Steps**

1. **Add Kubernetes deployment**
2. **Implement API authentication**
3. **Add metrics collection (Prometheus)**
4. **Set up CI/CD pipeline**
5. **Add distributed training support**

---

**Your cyber-physical systems project is now fully containerized and ready for development and production deployment!** 
