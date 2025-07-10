"""
Sensor Data Simulator for Cyber-Physical Systems
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

This module provides classes for simulating realistic sensor data
including temperature, humidity, pressure, and other environmental sensors.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import time
import random


class BaseSensor:
    """Base class for all sensor types with common functionality."""
    
    def __init__(self, 
                 sensor_id: str,
                 location: str,
                 sampling_rate: float = 1.0,
                 noise_level: float = 0.1):
        """
        Initialize base sensor.
        
        Args:
            sensor_id: Unique identifier for the sensor
            location: Physical location of the sensor
            sampling_rate: Samples per second
            noise_level: Amount of random noise (0.0-1.0)
            
        Raises:
            ValueError: If input parameters are invalid
            TypeError: If input types are incorrect
        """
        # Input validation
        self._validate_inputs(sensor_id, location, sampling_rate, noise_level)
        
        self.sensor_id = sensor_id
        self.location = location
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        self.last_reading = None
        self.created_at = datetime.now()
        self._reading_count = 0  # Track number of readings taken
        
    def _validate_inputs(self, sensor_id: str, location: str, 
                        sampling_rate: float, noise_level: float) -> None:
        """Validate sensor initialization parameters."""
        if not isinstance(sensor_id, str) or not sensor_id.strip():
            raise ValueError("sensor_id must be a non-empty string")
            
        if not isinstance(location, str) or not location.strip():
            raise ValueError("location must be a non-empty string")
            
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive number")
            
        if not isinstance(noise_level, (int, float)) or not (0.0 <= noise_level <= 1.0):
            raise ValueError("noise_level must be between 0.0 and 1.0")
    
    def add_noise(self, value: float, noise_factor: float = None) -> float:
        """
        Add realistic noise to sensor readings.
        
        Args:
            value: Original sensor value
            noise_factor: Custom noise factor (defaults to sensor's noise_level)
            
        Returns:
            float: Value with added noise
            
        Raises:
            TypeError: If value is not numeric
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be numeric")
            
        if noise_factor is None:
            noise_factor = self.noise_level
        elif not isinstance(noise_factor, (int, float)) or not (0.0 <= noise_factor <= 1.0):
            raise ValueError("noise_factor must be between 0.0 and 1.0")
            
        noise = np.random.normal(0, noise_factor * abs(value))
        return value + noise
    
    def get_reading_count(self) -> int:
        """Get the total number of readings taken by this sensor."""
        return self._reading_count
        
    def reset_reading_count(self) -> None:
        """Reset the reading counter to zero."""
        self._reading_count = 0
    
    def get_metadata(self) -> Dict:
        """Get sensor metadata."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.__class__.__name__,
            'location': self.location,
            'sampling_rate': self.sampling_rate,
            'noise_level': self.noise_level,
            'created_at': self.created_at.isoformat()
        }


class TemperatureSensor(BaseSensor):
    """Simulates realistic temperature sensor data."""
    
    def __init__(self, 
                 sensor_id: str,
                 location: str,
                 base_temp: float = 22.0,
                 temp_range: Tuple[float, float] = (-10.0, 40.0),
                 seasonal_variation: float = 5.0,
                 daily_variation: float = 3.0,
                 **kwargs):
        """
        Initialize temperature sensor.
        
        Args:
            base_temp: Base temperature in Celsius
            temp_range: Min and max temperature limits
            seasonal_variation: Seasonal temperature swing
            daily_variation: Daily temperature variation
        """
        super().__init__(sensor_id, location, **kwargs)
        self.base_temp = base_temp
        self.temp_range = temp_range
        self.seasonal_variation = seasonal_variation
        self.daily_variation = daily_variation
        
    def read_value(self, timestamp: datetime = None) -> Dict:
        """Generate realistic temperature reading."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Calculate time-based variations
        day_of_year = timestamp.timetuple().tm_yday
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        
        # Seasonal variation (sine wave over year)
        seasonal = self.seasonal_variation * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Daily variation (cosine wave, peak at 2 PM)
        daily = self.daily_variation * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
        
        # Calculate base temperature with variations
        temperature = self.base_temp + seasonal + daily
        
        # Add random noise
        temperature = self.add_noise(temperature)
        
        # Clamp to realistic range
        temperature = np.clip(temperature, self.temp_range[0], self.temp_range[1])
        
        # Increment reading counter
        self._reading_count += 1
        
        self.last_reading = temperature
        
        return {
            'timestamp': timestamp.isoformat(),
            'sensor_id': self.sensor_id,
            'sensor_type': 'temperature',
            'value': round(temperature, 2),
            'unit': 'celsius',
            'location': self.location,
            'reading_number': self._reading_count
        }


class HumiditySensor(BaseSensor):
    """Simulates realistic humidity sensor data."""
    
    def __init__(self,
                 sensor_id: str,
                 location: str,
                 base_humidity: float = 50.0,
                 humidity_range: Tuple[float, float] = (20.0, 90.0),
                 daily_variation: float = 15.0,
                 **kwargs):
        """
        Initialize humidity sensor.
        
        Args:
            base_humidity: Base humidity percentage
            humidity_range: Min and max humidity limits
            daily_variation: Daily humidity swing
        """
        super().__init__(sensor_id, location, **kwargs)
        self.base_humidity = base_humidity
        self.humidity_range = humidity_range
        self.daily_variation = daily_variation
        
    def read_value(self, timestamp: datetime = None) -> Dict:
        """Generate realistic humidity reading."""
        if timestamp is None:
            timestamp = datetime.now()
            
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        
        # Daily variation (higher humidity in early morning)
        daily = self.daily_variation * np.cos(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Calculate humidity with variation
        humidity = self.base_humidity + daily
        
        # Add random noise
        humidity = self.add_noise(humidity)
        
        # Clamp to realistic range
        humidity = np.clip(humidity, self.humidity_range[0], self.humidity_range[1])
        
        self.last_reading = humidity
        
        return {
            'timestamp': timestamp.isoformat(),
            'sensor_id': self.sensor_id,
            'sensor_type': 'humidity',
            'value': round(humidity, 2),
            'unit': 'percentage',
            'location': self.location
        }


class PressureSensor(BaseSensor):
    """Simulates realistic atmospheric pressure sensor data."""
    
    def __init__(self,
                 sensor_id: str,
                 location: str,
                 base_pressure: float = 1013.25,
                 pressure_range: Tuple[float, float] = (980.0, 1050.0),
                 weather_variation: float = 20.0,
                 **kwargs):
        """
        Initialize pressure sensor.
        
        Args:
            base_pressure: Base pressure in hPa
            pressure_range: Min and max pressure limits
            weather_variation: Weather-based pressure variation
        """
        super().__init__(sensor_id, location, **kwargs)
        self.base_pressure = base_pressure
        self.pressure_range = pressure_range
        self.weather_variation = weather_variation
        self.weather_trend = 0.0  # Current weather trend
        
    def read_value(self, timestamp: datetime = None) -> Dict:
        """Generate realistic pressure reading."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Simulate weather systems (slow changes)
        self.weather_trend += np.random.normal(0, 0.1)
        self.weather_trend = np.clip(self.weather_trend, -1.0, 1.0)
        
        # Weather-based variation
        weather = self.weather_variation * self.weather_trend
        
        # Calculate pressure with weather effects
        pressure = self.base_pressure + weather
        
        # Add random noise
        pressure = self.add_noise(pressure, self.noise_level * 2)
        
        # Clamp to realistic range
        pressure = np.clip(pressure, self.pressure_range[0], self.pressure_range[1])
        
        self.last_reading = pressure
        
        return {
            'timestamp': timestamp.isoformat(),
            'sensor_id': self.sensor_id,
            'sensor_type': 'pressure',
            'value': round(pressure, 2),
            'unit': 'hPa',
            'location': self.location
        }


class SensorNetwork:
    """Manages multiple sensors and coordinates data collection."""
    
    def __init__(self, network_id: str):
        """
        Initialize sensor network.
        
        Args:
            network_id: Unique identifier for the sensor network
        """
        self.network_id = network_id
        self.sensors: Dict[str, BaseSensor] = {}
        self.data_buffer: List[Dict] = []
        self.created_at = datetime.now()
        
    def add_sensor(self, sensor: BaseSensor) -> None:
        """Add a sensor to the network."""
        self.sensors[sensor.sensor_id] = sensor
        
    def remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor from the network."""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            
    def read_all_sensors(self, timestamp: datetime = None) -> List[Dict]:
        """Read data from all sensors in the network."""
        readings = []
        for sensor in self.sensors.values():
            reading = sensor.read_value(timestamp)
            readings.append(reading)
            self.data_buffer.append(reading)
        return readings
        
    def generate_time_series(self, 
                           duration_hours: float,
                           interval_seconds: float = 60) -> pd.DataFrame:
        """
        Generate time series data for all sensors.
        
        Args:
            duration_hours: How many hours of data to generate
            interval_seconds: Interval between readings in seconds
            
        Returns:
            DataFrame with time series data
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        all_data = []
        current_time = start_time
        
        while current_time <= end_time:
            readings = self.read_all_sensors(current_time)
            all_data.extend(readings)
            current_time += timedelta(seconds=interval_seconds)
            
        return pd.DataFrame(all_data)
        
    def export_data(self, 
                   filename: str,
                   format: str = 'csv',
                   clear_buffer: bool = True) -> None:
        """
        Export collected data to file.
        
        Args:
            filename: Output filename
            format: Export format ('csv', 'json')
            clear_buffer: Whether to clear the data buffer after export
        """
        if not self.data_buffer:
            print("No data to export")
            return
            
        df = pd.DataFrame(self.data_buffer)
        
        if format.lower() == 'csv':
            df.to_csv(filename, index=False)
        elif format.lower() == 'json':
            df.to_json(filename, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Exported {len(self.data_buffer)} readings to {filename}")
        
        if clear_buffer:
            self.data_buffer.clear()
            
    def get_network_status(self) -> Dict:
        """Get network status and statistics."""
        return {
            'network_id': self.network_id,
            'sensor_count': len(self.sensors),
            'sensors': [sensor.get_metadata() for sensor in self.sensors.values()],
            'data_points': len(self.data_buffer),
            'created_at': self.created_at.isoformat()
        }


class DataCollectionManager:
    """High-level manager for sensor data collection operations."""
    
    def __init__(self):
        """Initialize data collection manager."""
        self.networks: Dict[str, SensorNetwork] = {}
        
    def create_network(self, network_id: str) -> SensorNetwork:
        """Create a new sensor network."""
        network = SensorNetwork(network_id)
        self.networks[network_id] = network
        return network
        
    def setup_demo_network(self, location: str = "Demo Lab") -> SensorNetwork:
        """Set up a demonstration sensor network."""
        network = self.create_network("demo_network")
        
        # Add temperature sensors
        temp_sensor = TemperatureSensor(
            sensor_id="TEMP_001",
            location=f"{location} - Room A",
            base_temp=22.5,
            seasonal_variation=3.0,
            daily_variation=2.0
        )
        network.add_sensor(temp_sensor)
        
        # Add humidity sensor
        humidity_sensor = HumiditySensor(
            sensor_id="HUM_001",
            location=f"{location} - Room A",
            base_humidity=45.0,
            daily_variation=12.0
        )
        network.add_sensor(humidity_sensor)
        
        # Add pressure sensor
        pressure_sensor = PressureSensor(
            sensor_id="PRESS_001",
            location=f"{location} - Outdoor",
            base_pressure=1015.0,
            weather_variation=15.0
        )
        network.add_sensor(pressure_sensor)
        
        return network
        
    def run_continuous_collection(self, 
                                network_id: str,
                                duration_minutes: int = 60,
                                interval_seconds: int = 30) -> None:
        """
        Run continuous data collection for specified duration.
        
        Args:
            network_id: ID of the network to collect from
            duration_minutes: How long to collect data
            interval_seconds: Interval between collections
        """
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
            
        network = self.networks[network_id]
        end_time = time.time() + (duration_minutes * 60)
        
        print(f"Starting continuous collection for {duration_minutes} minutes...")
        
        while time.time() < end_time:
            readings = network.read_all_sensors()
            print(f"Collected {len(readings)} readings at {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(interval_seconds)
            
        print("Data collection completed")
        
    def get_network_statistics(self) -> Dict:
        """
        Get comprehensive statistics for all networks.
        
        Returns:
            Dict containing network statistics and sensor performance metrics
        """
        stats = {
            'total_networks': len(self.networks),
            'networks': {},
            'overall_readings': 0,
            'total_sensors': 0
        }
        
        for network_id, network in self.networks.items():
            network_readings = 0
            sensor_stats = {}
            
            for sensor_id, sensor in network.sensors.items():
                reading_count = sensor.get_reading_count()
                network_readings += reading_count
                sensor_stats[sensor_id] = {
                    'type': sensor.__class__.__name__,
                    'location': sensor.location,
                    'readings_taken': reading_count,
                    'last_reading': sensor.last_reading,
                    'created_at': sensor.created_at.isoformat()
                }
            
            stats['networks'][network_id] = {
                'sensor_count': len(network.sensors),
                'total_readings': network_readings,
                'data_buffer_size': len(network.data_buffer),
                'sensors': sensor_stats,
                'created_at': network.created_at.isoformat()
            }
            
            stats['overall_readings'] += network_readings
            stats['total_sensors'] += len(network.sensors)
            
        return stats
        
    def print_status_report(self) -> None:
        """Print a formatted status report of all networks and sensors."""
        stats = self.get_network_statistics()
        
        print("=" * 60)
        print("SENSOR NETWORK STATUS REPORT")
        print("=" * 60)
        print(f"Total Networks: {stats['total_networks']}")
        print(f"Total Sensors: {stats['total_sensors']}")
        print(f"Overall Readings: {stats['overall_readings']}")
        print()
        
        for network_id, network_data in stats['networks'].items():
            print(f"Network: {network_id}")
            print(f"  Sensors: {network_data['sensor_count']}")
            print(f"  Total Readings: {network_data['total_readings']}")
            print(f"  Buffer Size: {network_data['data_buffer_size']}")
            print("  Sensor Details:")
            
            for sensor_id, sensor_data in network_data['sensors'].items():
                print(f"    {sensor_id} ({sensor_data['type']}):")
                print(f"      Location: {sensor_data['location']}")
                print(f"      Readings: {sensor_data['readings_taken']}")
                if sensor_data['last_reading']:
                    print(f"      Last Value: {sensor_data['last_reading']:.2f}")
            print()
        
        print("=" * 60)
