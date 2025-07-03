"""
Configuration settings for sensor data collection
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os


@dataclass
class SensorConfig:
    """Configuration for individual sensors."""
    sensor_id: str
    sensor_type: str
    location: str
    sampling_rate: float = 1.0
    noise_level: float = 0.1
    custom_params: Dict = None


@dataclass
class TemperatureConfig(SensorConfig):
    """Configuration specific to temperature sensors."""
    base_temp: float = 22.0
    temp_range: Tuple[float, float] = (-10.0, 40.0)
    seasonal_variation: float = 5.0
    daily_variation: float = 3.0


@dataclass
class HumidityConfig(SensorConfig):
    """Configuration specific to humidity sensors."""
    base_humidity: float = 50.0
    humidity_range: Tuple[float, float] = (20.0, 90.0)
    daily_variation: float = 15.0


@dataclass
class PressureConfig(SensorConfig):
    """Configuration specific to pressure sensors."""
    base_pressure: float = 1013.25
    pressure_range: Tuple[float, float] = (980.0, 1050.0)
    weather_variation: float = 20.0


class DataCollectionConfig:
    """Main configuration class for data collection system."""
    
    # File paths
    DATA_DIR = os.path.join("..", "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
    
    # Sampling settings
    DEFAULT_SAMPLING_RATE = 1.0  # Hz
    DEFAULT_INTERVAL_SECONDS = 60
    DEFAULT_NOISE_LEVEL = 0.1
    
    # Time series generation
    DEFAULT_DURATION_HOURS = 24
    MAX_DURATION_HOURS = 168  # 1 week
    
    # Export settings
    SUPPORTED_FORMATS = ['csv', 'json', 'parquet']
    DEFAULT_EXPORT_FORMAT = 'csv'
    
    # Network settings
    MAX_SENSORS_PER_NETWORK = 50
    MAX_DATA_BUFFER_SIZE = 10000
    
    # Sensor presets for different environments
    SENSOR_PRESETS = {
        'laboratory': {
            'temperature': {
                'base_temp': 22.0,
                'daily_variation': 2.0,
                'seasonal_variation': 1.0,
                'noise_level': 0.05
            },
            'humidity': {
                'base_humidity': 45.0,
                'daily_variation': 8.0,
                'noise_level': 0.1
            },
            'pressure': {
                'base_pressure': 1013.25,
                'weather_variation': 5.0,
                'noise_level': 0.05
            }
        },
        'outdoor': {
            'temperature': {
                'base_temp': 15.0,
                'daily_variation': 8.0,
                'seasonal_variation': 12.0,
                'noise_level': 0.3
            },
            'humidity': {
                'base_humidity': 60.0,
                'daily_variation': 20.0,
                'noise_level': 0.2
            },
            'pressure': {
                'base_pressure': 1013.25,
                'weather_variation': 25.0,
                'noise_level': 0.1
            }
        },
        'industrial': {
            'temperature': {
                'base_temp': 35.0,
                'daily_variation': 5.0,
                'seasonal_variation': 3.0,
                'noise_level': 0.15
            },
            'humidity': {
                'base_humidity': 40.0,
                'daily_variation': 10.0,
                'noise_level': 0.15
            },
            'pressure': {
                'base_pressure': 1013.25,
                'weather_variation': 8.0,
                'noise_level': 0.08
            }
        }
    }
    
    @classmethod
    def get_preset_config(cls, environment: str, sensor_type: str) -> Dict:
        """Get preset configuration for specific environment and sensor type."""
        if environment not in cls.SENSOR_PRESETS:
            raise ValueError(f"Unknown environment: {environment}")
        
        if sensor_type not in cls.SENSOR_PRESETS[environment]:
            raise ValueError(f"Unknown sensor type for {environment}: {sensor_type}")
        
        return cls.SENSOR_PRESETS[environment][sensor_type]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for data collection."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.EXPORTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Example network configurations
EXAMPLE_NETWORKS = {
    'demo_lab': {
        'network_id': 'demo_lab_network',
        'description': 'Demonstration laboratory setup',
        'sensors': [
            {
                'sensor_id': 'TEMP_LAB_001',
                'type': 'temperature',
                'location': 'Lab Room A',
                'config': DataCollectionConfig.get_preset_config('laboratory', 'temperature')
            },
            {
                'sensor_id': 'HUM_LAB_001',
                'type': 'humidity', 
                'location': 'Lab Room A',
                'config': DataCollectionConfig.get_preset_config('laboratory', 'humidity')
            },
            {
                'sensor_id': 'PRESS_LAB_001',
                'type': 'pressure',
                'location': 'Lab Outdoor',
                'config': DataCollectionConfig.get_preset_config('laboratory', 'pressure')
            }
        ]
    },
    'outdoor_station': {
        'network_id': 'outdoor_weather_station',
        'description': 'Outdoor weather monitoring station',
        'sensors': [
            {
                'sensor_id': 'TEMP_OUT_001',
                'type': 'temperature',
                'location': 'Outdoor Station',
                'config': DataCollectionConfig.get_preset_config('outdoor', 'temperature')
            },
            {
                'sensor_id': 'HUM_OUT_001',
                'type': 'humidity',
                'location': 'Outdoor Station',
                'config': DataCollectionConfig.get_preset_config('outdoor', 'humidity')
            },
            {
                'sensor_id': 'PRESS_OUT_001',
                'type': 'pressure',
                'location': 'Outdoor Station',
                'config': DataCollectionConfig.get_preset_config('outdoor', 'pressure')
            }
        ]
    }
}
