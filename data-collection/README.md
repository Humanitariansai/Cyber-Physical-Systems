# Data Collection Module

**Author:** Udisha Dutta Chowdhury  
**Supervisor:** Prof. Rolando Herrero

This module provides comprehensive sensor data simulation and collection capabilities for cyber-physical systems research and development.

## Overview

The data collection module generates realistic time-series sensor data that mimics real-world environmental sensors including temperature, humidity, and atmospheric pressure. The simulation includes realistic patterns like:

- **Seasonal variations** (temperature changes throughout the year)
- **Daily cycles** (temperature and humidity patterns during 24-hour periods)
- **Weather patterns** (pressure variations due to weather systems)
- **Realistic noise** (sensor measurement uncertainties)

## Components

### Core Classes

- **`BaseSensor`** - Abstract base class for all sensor types
- **`TemperatureSensor`** - Simulates temperature sensors with seasonal and daily variations
- **`HumiditySensor`** - Simulates humidity sensors with daily patterns
- **`PressureSensor`** - Simulates atmospheric pressure with weather variations
- **`SensorNetwork`** - Manages multiple sensors and coordinates data collection
- **`DataCollectionManager`** - High-level manager for sensor networks

### Configuration

- **`config.py`** - Configuration settings and sensor presets for different environments
- **Environment presets**: Laboratory, Outdoor, Industrial
- **Configurable parameters**: Base values, variation ranges, noise levels

### Utilities

- **`demo_sensor_data.py`** - Demonstration script showing all functionality
- **`test_sensors.py`** - Unit tests for validating sensor behavior
- **`requirements.txt`** - Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from sensor_simulator import TemperatureSensor, SensorNetwork

# Create a temperature sensor
temp_sensor = TemperatureSensor(
    sensor_id="TEMP_001",
    location="Laboratory Room A",
    base_temp=22.0,
    daily_variation=3.0
)

# Get a single reading
reading = temp_sensor.read_value()
print(f"Temperature: {reading['value']}°C")

# Create a sensor network
network = SensorNetwork("lab_network")
network.add_sensor(temp_sensor)

# Generate time series data
df = network.generate_time_series(
    duration_hours=24,
    interval_seconds=300  # 5 minutes
)

# Export data
network.export_data("sensor_data.csv", format="csv")
```

### 3. Run Demonstration

```bash
python demo_sensor_data.py
```

## Sensor Features

### Temperature Sensor
- **Base temperature**: Configurable baseline temperature
- **Seasonal variation**: Annual temperature cycles
- **Daily variation**: 24-hour temperature patterns
- **Noise simulation**: Realistic measurement uncertainty
- **Range limiting**: Configurable min/max temperatures

### Humidity Sensor
- **Base humidity**: Configurable baseline humidity percentage
- **Daily variation**: Humidity changes during day/night cycles
- **Range limiting**: 0-100% humidity with configurable bounds
- **Realistic patterns**: Higher humidity in early morning

### Pressure Sensor
- **Base pressure**: Standard atmospheric pressure baseline
- **Weather simulation**: Slow pressure changes simulating weather systems
- **Trend tracking**: Persistent weather pattern simulation
- **Range limiting**: Realistic atmospheric pressure bounds

## Data Export Formats

- **CSV**: Standard comma-separated values
- **JSON**: JavaScript Object Notation
- **Parquet**: High-performance columnar format (optional)

## Environment Presets

### Laboratory Environment
- Controlled temperature variations (±2°C daily)
- Moderate humidity (45% ±8%)
- Stable pressure (minimal weather effects)
- Low noise levels

### Outdoor Environment  
- Large temperature variations (±8°C daily, ±12°C seasonal)
- Variable humidity (60% ±20%)
- Full weather pressure variations
- Higher noise levels

### Industrial Environment
- Elevated baseline temperature (35°C)
- Controlled humidity (40% ±10%)
- Moderate pressure variations
- Medium noise levels

## Time Series Generation

Generate realistic time series data with:
- Configurable duration (minutes to weeks)
- Flexible sampling intervals
- Multiple sensors simultaneously
- Synchronized timestamps
- Pandas DataFrame output

## Testing

Run unit tests to validate functionality:

```bash
python test_sensors.py
```

Tests cover:
- Sensor initialization and configuration
- Reading generation and validation
- Value range compliance
- Time series generation
- Network management
- Data export functionality

## Advanced Features

### Custom Sensor Types
Extend `BaseSensor` to create custom sensor types:

```python
class CustomSensor(BaseSensor):
    def read_value(self, timestamp=None):
        # Implement custom sensor logic
        return {
            'timestamp': timestamp.isoformat(),
            'sensor_id': self.sensor_id,
            'sensor_type': 'custom',
            'value': custom_calculation(),
            'unit': 'custom_unit',
            'location': self.location
        }
```

### Network Monitoring
- Real-time sensor status tracking
- Data buffer management
- Network health monitoring
- Configurable sampling rates per sensor

### Data Pipeline Integration
- Designed for integration with ML model training
- Compatible with cloud upload utilities
- Structured for time series forecasting
- Ready for real-time streaming applications

## Future Enhancements

- **Cloud integration**: Direct upload to AWS, Azure, Google Cloud
- **MQTT protocol**: Real-time data streaming
- **Additional sensors**: Light, sound, vibration, gas sensors
- **Fault simulation**: Sensor failure and anomaly injection
- **Real device integration**: Bridge to actual Arduino/Raspberry Pi sensors

## Files Structure

```
data-collection/
├── sensor_simulator.py      # Core sensor simulation classes
├── config.py               # Configuration and presets
├── demo_sensor_data.py     # Demonstration script
├── test_sensors.py         # Unit tests
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

## Integration Points

This module interfaces with other project components:

- **`../ml-models/`** - Provides training data for forecasting models
- **`../data/raw/`** - Stores generated sensor datasets
- **`../cloud-dashboard/`** - Supplies real-time data for visualization
- **`../edge-inference/`** - Provides test data for edge model validation
