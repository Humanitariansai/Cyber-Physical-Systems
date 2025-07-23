"""
Advanced Sensor Simulator with Enhanced Patterns
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

This module provides advanced sensor simulation capabilities with realistic patterns,
correlations, anomalies, and ML-ready data generation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import random
from dataclasses import asdict

from .enhanced_config import config_manager, EnvironmentPreset, SensorConfig
from .advanced_patterns import AdvancedPatternGenerator, EnhancedTimeSeries, WeatherCondition


class AdvancedSensor:
    """Enhanced sensor with realistic pattern generation."""
    
    def __init__(self, 
                 sensor_id: str,
                 sensor_type: str,
                 config: SensorConfig,
                 location: str = "Unknown"):
        """
        Initialize advanced sensor.
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor (temperature, humidity, pressure)
            config: Sensor configuration parameters
            location: Physical location
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.config = config
        self.location = location
        self.created_at = datetime.now()
        
        # Internal state
        self._reading_count = 0
        self._last_reading = None
        self._drift_accumulator = 0.0
        self._calibration_drift = 0.0
        self._history = []
        
        # Pattern generator
        self.pattern_generator = AdvancedPatternGenerator(location)
        
    def read_value(self, timestamp: datetime = None) -> Dict:
        """
        Generate realistic sensor reading with advanced patterns.
        
        Args:
            timestamp: Target timestamp for reading
            
        Returns:
            Dictionary containing sensor reading and metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Base value calculation
        base_value = self._calculate_base_value(timestamp)
        
        # Apply seasonal variations
        seasonal_effect = self._calculate_seasonal_effect(timestamp)
        
        # Apply daily variations
        daily_effect = self._calculate_daily_effect(timestamp)
        
        # Combine base effects
        sensor_value = base_value + seasonal_effect + daily_effect
        
        # Apply weather effects
        weather_effects = self.pattern_generator.get_weather_effects(timestamp)
        if self.sensor_type in weather_effects:
            sensor_value += weather_effects[self.sensor_type]
        
        # Apply noise
        sensor_value = self._apply_noise(sensor_value, timestamp)
        
        # Apply drift and calibration errors
        sensor_value = self._apply_drift(sensor_value)
        
        # Apply anomalies
        sensor_value = self._apply_anomalies(sensor_value)
        
        # Clamp to valid range
        sensor_value = np.clip(sensor_value, 
                             self.config.valid_range[0], 
                             self.config.valid_range[1])
        
        # Create reading record
        reading = {
            'timestamp': timestamp.isoformat(),
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'value': round(sensor_value, 3),
            'unit': self.config.units,
            'location': self.location,
            'reading_number': self._reading_count + 1,
            'quality_score': self._calculate_quality_score(sensor_value),
            'metadata': {
                'seasonal_effect': round(seasonal_effect, 3),
                'daily_effect': round(daily_effect, 3),
                'weather_effect': round(weather_effects.get(self.sensor_type, 0.0), 3),
                'drift': round(self._calibration_drift, 3)
            }
        }
        
        # Update internal state
        self._reading_count += 1
        self._last_reading = sensor_value
        self._history.append((timestamp, sensor_value))
        
        # Keep history manageable
        if len(self._history) > 1000:
            self._history = self._history[-500:]
        
        return reading
    
    def _calculate_base_value(self, timestamp: datetime) -> float:
        """Calculate base sensor value."""
        return self.config.base_value
    
    def _calculate_seasonal_effect(self, timestamp: datetime) -> float:
        """Calculate seasonal variation effect."""
        day_of_year = timestamp.timetuple().tm_yday
        
        # Get seasonal factor from pattern generator
        seasonal_factor = self.pattern_generator.get_seasonal_factor(
            timestamp, self.sensor_type
        )
        
        return seasonal_factor * self.config.seasonal.amplitude
    
    def _calculate_daily_effect(self, timestamp: datetime) -> float:
        """Calculate daily variation effect."""
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        
        # Weekend modifier
        weekend_modifier = (self.config.daily.weekend_modifier 
                          if timestamp.weekday() >= 5 
                          else 1.0)
        
        # Daily cycle
        phase = 2 * np.pi * (hour_of_day - self.config.daily.peak_hour) / 24
        daily_effect = self.config.daily.amplitude * np.cos(phase)
        
        return daily_effect * weekend_modifier
    
    def _apply_noise(self, value: float, timestamp: datetime) -> float:
        """Apply realistic noise to sensor reading."""
        # Base noise
        base_noise = np.random.normal(0, abs(value) * self.config.noise.base_noise)
        
        # Random walk drift
        self._drift_accumulator += np.random.normal(0, self.config.noise.random_walk)
        
        # Noise bursts
        if np.random.random() < self.config.noise.burst_probability:
            burst_noise = np.random.normal(0, abs(value) * self.config.noise.burst_amplitude)
            base_noise += burst_noise
        
        return value + base_noise + self._drift_accumulator
    
    def _apply_drift(self, value: float) -> float:
        """Apply sensor drift and calibration errors."""
        # Gradual calibration drift
        self._calibration_drift += np.random.normal(0, 0.001)
        return value + self._calibration_drift
    
    def _apply_anomalies(self, value: float) -> float:
        """Apply sensor anomalies."""
        if np.random.random() < self.config.anomaly.probability:
            anomaly_type = np.random.choice([
                'spike', 'drop', 'drift', 'noise'
            ], p=[
                self.config.anomaly.spike_probability,
                self.config.anomaly.drop_probability,
                self.config.anomaly.drift_probability,
                self.config.anomaly.noise_probability
            ])
            
            if anomaly_type == 'spike':
                multiplier = np.random.uniform(*self.config.anomaly.spike_multiplier)
                return value * multiplier
                
            elif anomaly_type == 'drop':
                multiplier = np.random.uniform(*self.config.anomaly.drop_multiplier)
                return value * multiplier
                
            elif anomaly_type == 'drift':
                drift = np.random.normal(0, abs(value) * self.config.anomaly.drift_std)
                return value + drift
                
            elif anomaly_type == 'noise':
                noise = np.random.normal(0, abs(value) * self.config.anomaly.noise_multiplier)
                return value + noise
        
        return value
    
    def _calculate_quality_score(self, value: float) -> float:
        """Calculate reading quality score (0-1)."""
        # Quality based on how close to expected range
        range_center = sum(self.config.valid_range) / 2
        range_width = self.config.valid_range[1] - self.config.valid_range[0]
        
        distance_from_center = abs(value - range_center)
        normalized_distance = distance_from_center / (range_width / 2)
        
        # Quality decreases with distance from center
        quality = max(0.0, 1.0 - normalized_distance * 0.5)
        
        # Add random quality variations
        quality += np.random.normal(0, 0.05)
        
        return np.clip(quality, 0.0, 1.0)
    
    def get_statistics(self) -> Dict:
        """Get sensor statistics and performance metrics."""
        if not self._history:
            return {}
        
        values = [reading[1] for reading in self._history]
        
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'reading_count': self._reading_count,
            'last_reading': self._last_reading,
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'min_value': np.min(values),
            'max_value': np.max(values),
            'drift_accumulator': self._drift_accumulator,
            'calibration_drift': self._calibration_drift,
            'created_at': self.created_at.isoformat()
        }
    
    def reset_state(self) -> None:
        """Reset sensor internal state."""
        self._reading_count = 0
        self._last_reading = None
        self._drift_accumulator = 0.0
        self._calibration_drift = 0.0
        self._history.clear()


class AdvancedSensorNetwork:
    """Advanced sensor network with correlation and ML features."""
    
    def __init__(self, 
                 network_id: str,
                 environment_preset: str = "office"):
        """
        Initialize advanced sensor network.
        
        Args:
            network_id: Unique network identifier
            environment_preset: Environmental preset name
        """
        self.network_id = network_id
        self.environment_preset = environment_preset
        self.sensors: Dict[str, AdvancedSensor] = {}
        self.created_at = datetime.now()
        
        # Get preset configuration
        self.preset = config_manager.get_preset(environment_preset)
        if self.preset is None:
            raise ValueError(f"Environment preset '{environment_preset}' not found")
        
        # Data collection
        self.data_buffer: List[Dict] = []
        self.correlation_matrix = None
        
        # Setup sensors from preset
        self._setup_sensors_from_preset()
    
    def _setup_sensors_from_preset(self) -> None:
        """Setup sensors based on environment preset."""
        for sensor_type, sensor_config in self.preset.sensors.items():
            sensor_id = f"{sensor_type.upper()}_{self.network_id}_001"
            
            sensor = AdvancedSensor(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                config=sensor_config,
                location=self.preset.location
            )
            
            self.sensors[sensor_id] = sensor
    
    def add_custom_sensor(self, 
                         sensor_id: str,
                         sensor_type: str,
                         config: SensorConfig) -> None:
        """Add a custom sensor to the network."""
        sensor = AdvancedSensor(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            config=config,
            location=self.preset.location
        )
        self.sensors[sensor_id] = sensor
    
    def read_all_sensors(self, timestamp: datetime = None) -> List[Dict]:
        """Read all sensors with correlation effects."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get individual readings
        readings = {}
        for sensor_id, sensor in self.sensors.items():
            reading = sensor.read_value(timestamp)
            readings[sensor.sensor_type] = reading
        
        # Apply cross-sensor correlations
        self._apply_correlations(readings, timestamp)
        
        # Convert to list and store in buffer
        reading_list = list(readings.values())
        self.data_buffer.extend(reading_list)
        
        return reading_list
    
    def _apply_correlations(self, readings: Dict[str, Dict], timestamp: datetime) -> None:
        """Apply cross-sensor correlations."""
        correlation_params = self.preset.correlations
        
        # Temperature-Humidity correlation
        if 'temperature' in readings and 'humidity' in readings:
            temp_value = readings['temperature']['value']
            temp_normalized = (temp_value - 20) / 20
            
            humidity_adjustment = (correlation_params.temp_humidity_correlation * 
                                 temp_normalized * 5.0)
            readings['humidity']['value'] += humidity_adjustment
            
            # Clamp humidity to valid range
            humidity_config = self.preset.sensors['humidity']
            readings['humidity']['value'] = np.clip(
                readings['humidity']['value'],
                humidity_config.valid_range[0],
                humidity_config.valid_range[1]
            )
        
        # Additional correlations can be implemented here
    
    def generate_ml_dataset(self, 
                          duration_days: int = 30,
                          interval_minutes: int = 15,
                          include_features: bool = True) -> pd.DataFrame:
        """
        Generate ML-ready dataset with time series features.
        
        Args:
            duration_days: Days of data to generate
            interval_minutes: Minutes between readings
            include_features: Whether to include engineered features
            
        Returns:
            DataFrame ready for ML training
        """
        start_time = datetime.now() - timedelta(days=duration_days)
        end_time = datetime.now()
        
        # Generate timestamps
        timestamps = pd.date_range(start_time, end_time, freq=f'{interval_minutes}min')
        
        all_data = []
        
        for timestamp in timestamps:
            readings = self.read_all_sensors(timestamp)
            
            # Create combined record
            record = {
                'timestamp': timestamp,
                'network_id': self.network_id,
                'environment': self.environment_preset
            }
            
            # Add sensor readings
            for reading in readings:
                prefix = reading['sensor_type']
                record[f'{prefix}_value'] = reading['value']
                record[f'{prefix}_quality'] = reading['quality_score']
                
                # Add metadata
                for key, value in reading['metadata'].items():
                    record[f'{prefix}_{key}'] = value
            
            all_data.append(record)
        
        df = pd.DataFrame(all_data)
        
        if include_features:
            df = self._add_ml_features(df)
        
        return df
    
    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for ML training."""
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        df['is_business_hours'] = (df['hour'] >= 9) & (df['hour'] <= 17)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Lag features for each sensor type
        sensor_types = [col.split('_')[0] for col in df.columns 
                       if col.endswith('_value')]
        sensor_types = list(set(sensor_types))
        
        for sensor_type in sensor_types:
            value_col = f'{sensor_type}_value'
            if value_col in df.columns:
                # Lag features
                for lag in [1, 2, 3, 6, 12, 24]:
                    df[f'{sensor_type}_lag_{lag}'] = df[value_col].shift(lag)
                
                # Rolling statistics
                for window in [6, 24, 168]:  # 1.5h, 6h, 1 week
                    df[f'{sensor_type}_rolling_mean_{window}'] = (
                        df[value_col].rolling(window=window).mean()
                    )
                    df[f'{sensor_type}_rolling_std_{window}'] = (
                        df[value_col].rolling(window=window).std()
                    )
                
                # Difference features
                df[f'{sensor_type}_diff_1'] = df[value_col].diff(1)
                df[f'{sensor_type}_diff_24'] = df[value_col].diff(24)
        
        return df
    
    def export_ml_data(self, 
                      filepath: str,
                      duration_days: int = 365,
                      interval_minutes: int = 15,
                      format: str = 'parquet') -> None:
        """Export ML-ready dataset to file."""
        df = self.generate_ml_dataset(duration_days, interval_minutes)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"ML dataset exported to {filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
    
    def get_network_summary(self) -> Dict:
        """Get comprehensive network summary."""
        sensor_stats = {}
        for sensor_id, sensor in self.sensors.items():
            sensor_stats[sensor_id] = sensor.get_statistics()
        
        return {
            'network_id': self.network_id,
            'environment_preset': self.environment_preset,
            'sensor_count': len(self.sensors),
            'total_readings': sum(s.get('reading_count', 0) for s in sensor_stats.values()),
            'data_buffer_size': len(self.data_buffer),
            'sensors': sensor_stats,
            'created_at': self.created_at.isoformat(),
            'preset_config': {
                'name': self.preset.name,
                'description': self.preset.description,
                'location': self.preset.location
            }
        }


class AdvancedDataCollectionManager:
    """Enhanced data collection manager with ML capabilities."""
    
    def __init__(self):
        """Initialize advanced data collection manager."""
        self.networks: Dict[str, AdvancedSensorNetwork] = {}
        self.global_config = config_manager
    
    def create_network(self, 
                      network_id: str,
                      environment_preset: str = "office") -> AdvancedSensorNetwork:
        """Create a new advanced sensor network."""
        if network_id in self.networks:
            raise ValueError(f"Network '{network_id}' already exists")
        
        network = AdvancedSensorNetwork(network_id, environment_preset)
        self.networks[network_id] = network
        return network
    
    def list_available_presets(self) -> List[str]:
        """List available environment presets."""
        return self.global_config.list_presets()
    
    def generate_comparative_dataset(self, 
                                   duration_days: int = 90,
                                   interval_minutes: int = 15) -> pd.DataFrame:
        """Generate comparative dataset across all networks."""
        all_data = []
        
        for network_id, network in self.networks.items():
            network_data = network.generate_ml_dataset(duration_days, interval_minutes)
            network_data['source_network'] = network_id
            all_data.append(network_data)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        network_summaries = {}
        total_sensors = 0
        total_readings = 0
        
        for network_id, network in self.networks.items():
            summary = network.get_network_summary()
            network_summaries[network_id] = summary
            total_sensors += summary['sensor_count']
            total_readings += summary['total_readings']
        
        return {
            'total_networks': len(self.networks),
            'total_sensors': total_sensors,
            'total_readings': total_readings,
            'available_presets': self.list_available_presets(),
            'networks': network_summaries,
            'system_timestamp': datetime.now().isoformat()
        }
