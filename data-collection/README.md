# Data Collection Module

**Author**: Udisha Dutta Chowdhury  
**Supervisor**: Prof. Rolando Herrero

This module provides advanced sensor data simulation capabilities for cyber-physical systems, including realistic patterns, correlations, anomalies, and ML-ready data generation.

## 📋 Overview

The data collection module simulates realistic sensor data with:
- **Seasonal and daily variations** based on physical phenomena
- **Weather effects** that influence sensor readings
- **Cross-sensor correlations** for realistic behavior
- **Configurable anomalies** for robustness testing
- **ML-ready datasets** with engineered features

## 🏗️ Architecture

### Core Components

1. **Enhanced Configuration System** (`enhanced_config.py`)
   - Environmental presets (office, outdoor, industrial)
   - Configurable sensor parameters
   - ML training configurations

2. **Advanced Pattern Generation** (`advanced_patterns.py`)
   - Seasonal and weather pattern simulation
   - Anomaly injection capabilities
   - Correlation modeling

3. **Advanced Sensor Simulator** (`advanced_sensor_simulator.py`)
   - Realistic sensor behavior
   - Quality metrics and drift simulation
   - Network-level coordination

4. **Demonstration Scripts**
   - `demo_advanced_simulation.py` - Comprehensive showcase
   - `demo_sensor_data.py` - Basic simulation demo

## 🚀 Quick Start

### Basic Usage

```python
from advanced_sensor_simulator import AdvancedDataCollectionManager

# Create manager and network
manager = AdvancedDataCollectionManager()
network = manager.create_network("my_network", "office")

# Generate realistic sensor readings
readings = network.read_all_sensors()
print(f"Generated {len(readings)} sensor readings")

# Create ML-ready dataset
ml_data = network.generate_ml_dataset(
    duration_days=30,
    interval_minutes=15,
    include_features=True
)
print(f"ML dataset shape: {ml_data.shape}")
```

### Environment Presets

```python
from enhanced_config import config_manager

# List available presets
presets = config_manager.list_presets()
print(f"Available environments: {presets}")

# Create networks for different environments
office_net = manager.create_network("office", "office")
outdoor_net = manager.create_network("weather", "outdoor")
industrial_net = manager.create_network("factory", "industrial")
```

### Advanced Features

```python
# Generate comparative dataset across environments
comparative_data = manager.generate_comparative_dataset(
    duration_days=90,
    interval_minutes=15
)

# Export ML-ready data
network.export_ml_data(
    filepath="data/training_data.parquet",
    duration_days=365,
    interval_minutes=15,
    format='parquet'
)
```

## 📊 Generated Data Features

### Sensor Readings
- **Temperature**: Seasonal cycles, daily patterns, weather effects
- **Humidity**: Inverse temperature correlation, morning peaks
- **Pressure**: Weather system simulation, atmospheric patterns

### ML Features
- **Temporal**: Hour, day of week, seasonality, business hours
- **Lag Features**: 1, 2, 3, 6, 12, 24, 48, 168-step lags
- **Rolling Statistics**: Mean and standard deviation over multiple windows
- **Quality Metrics**: Sensor health and reading confidence scores

### Data Formats
- **CSV**: Human-readable, universal compatibility
- **Parquet**: Efficient storage, fast loading for ML
- **JSON**: Structured data with metadata

## 🔧 Configuration

### Environment Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `office` | Indoor HVAC environment | Building automation |
| `outdoor` | Weather station data | Environmental monitoring |
| `industrial` | Manufacturing facility | Process control |

### Customization

```python
from enhanced_config import SensorConfig, SeasonalParams, DailyParams

# Create custom sensor configuration
custom_config = SensorConfig(
    base_value=25.0,
    valid_range=(10.0, 50.0),
    seasonal=SeasonalParams(amplitude=8.0),
    daily=DailyParams(amplitude=5.0, peak_hour=15.0),
    units="°C"
)

# Add to network
network.add_custom_sensor("CUSTOM_TEMP_001", "temperature", custom_config)
```

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest test_advanced_simulation.py -v
```

### Run Demonstrations
```bash
python demo_advanced_simulation.py
```

### Performance Testing
```bash
python -m pytest test_advanced_simulation.py::TestPerformance -v
```

## 📈 Data Quality

### Realistic Patterns
- **Seasonal**: Temperature varies ±12°C, humidity ±15%, pressure ±20hPa
- **Daily**: Peak temperature at 2 PM, humidity at 6 AM
- **Weather**: Storm systems, precipitation effects, pressure changes

### Anomaly Simulation
- **Spikes**: Sensor malfunction (2-5x normal values)
- **Drops**: Sensor failure (0.1-0.5x normal values)
- **Drift**: Calibration errors (gradual bias accumulation)
- **Noise**: Interference (high-frequency disturbances)

### Correlations
- **Temperature-Humidity**: -0.6 correlation (higher temp → lower humidity)
- **Pressure-Weather**: Weather systems affect pressure patterns
- **Temporal**: Consistent daily and seasonal cycles

## 📁 File Structure

```
data-collection/
├── advanced_patterns.py          # Weather and pattern simulation
├── advanced_sensor_simulator.py  # Core simulation engine
├── enhanced_config.py            # Configuration management
├── demo_advanced_simulation.py   # Comprehensive demonstration
├── test_advanced_simulation.py   # Unit tests
├── requirements.txt              # Dependencies
├── sensor_simulator.py           # Basic sensor classes
├── demo_sensor_data.py           # Basic demonstration
├── test_sensors.py               # Basic tests
└── config.py                     # Basic configuration
```

## 🔄 Data Pipeline Integration

### For ML Model Training
```python
# Generate training data
train_data = network.generate_ml_dataset(
    duration_days=365,
    interval_minutes=15
)

# Split for ML pipeline
train_size = int(0.7 * len(train_data))
val_size = int(0.2 * len(train_data))

train_set = train_data[:train_size]
val_set = train_data[train_size:train_size+val_size]
test_set = train_data[train_size+val_size:]
```

### For Real-time Simulation
```python
# Continuous data generation
import time

while True:
    readings = network.read_all_sensors()
    # Process readings...
    time.sleep(60)  # 1-minute intervals
```

## 📋 Requirements

### Core Dependencies
- `numpy >= 1.21.0` - Numerical computations
- `pandas >= 1.5.0` - Data manipulation
- `matplotlib >= 3.5.0` - Visualization
- `seaborn >= 0.11.0` - Statistical plotting
- `scikit-learn >= 1.0.0` - ML utilities
- `pyarrow >= 10.0.0` - Parquet support

### Optional Dependencies
- `jupyter` - Interactive development
- `plotly` - Interactive visualizations
- `pytest >= 6.0.0` - Testing framework

## 🎯 Use Cases

### 1. ML Model Development
- **Time Series Forecasting**: LSTM, ARIMA, Prophet models
- **Anomaly Detection**: Isolation Forest, One-Class SVM
- **Classification**: Normal vs. abnormal sensor behavior
- **Regression**: Predict future sensor values

### 2. System Testing
- **Edge Computing**: Test inference algorithms
- **Dashboard Development**: Realistic data for UI testing
- **Alert Systems**: Validate threshold-based alerts
- **Data Pipeline**: Test ETL processes

### 3. Research Applications
- **Pattern Analysis**: Study sensor behavior patterns
- **Correlation Studies**: Multi-sensor relationship analysis
- **Robustness Testing**: System response to anomalies
- **Comparative Studies**: Different environment behaviors

## 🚧 Development Roadmap

### Current Features ✅
- [x] Advanced pattern simulation
- [x] Multiple environment presets
- [x] ML-ready dataset generation
- [x] Comprehensive testing suite
- [x] Data export capabilities

### Next Steps 🔄
- [ ] Real-time data streaming
- [ ] Database integration
- [ ] REST API endpoints
- [ ] Cloud deployment support
- [ ] Custom pattern designer UI

## 📞 Support

For questions, issues, or contributions:
- **Author**: Udisha Dutta Chowdhury
- **Supervisor**: Prof. Rolando Herrero
- **Project**: Cyber-Physical Systems for Time-Series Analysis

## 📄 License

This project is developed as part of academic research under the supervision of Prof. Rolando Herrero.

---

**Next Phase**: The generated datasets from this module will be used to train time-series forecasting models using LSTM, ARIMA, and ensemble methods in the `ml-models/` module.
