"""
Advanced Pattern Simulation for Cyber-Physical Systems
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

This module provides enhanced pattern simulation capabilities including
complex seasonal cycles, weather effects, and anomaly injection.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import random
from enum import Enum


class WeatherCondition(Enum):
    """Weather condition types for realistic simulation."""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"


@dataclass
class SeasonalConfig:
    """Configuration for seasonal patterns."""
    winter_temp_offset: float = -5.0
    spring_temp_offset: float = 2.0
    summer_temp_offset: float = 8.0
    autumn_temp_offset: float = 0.0
    humidity_seasonal_range: float = 20.0
    pressure_seasonal_range: float = 15.0


@dataclass
class WeatherEvent:
    """Represents a weather event with duration and intensity."""
    condition: WeatherCondition
    start_time: datetime
    duration_hours: float
    intensity: float  # 0.0 to 1.0
    

class AdvancedPatternGenerator:
    """Advanced pattern generator for realistic sensor simulation."""
    
    def __init__(self, base_location: str = "Default Location"):
        """
        Initialize advanced pattern generator.
        
        Args:
            base_location: Base location for weather patterns
        """
        self.base_location = base_location
        self.seasonal_config = SeasonalConfig()
        self.active_weather_events: List[WeatherEvent] = []
        self.random_seed = None
        
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducible patterns."""
        self.random_seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def get_seasonal_factor(self, timestamp: datetime, 
                          parameter: str = "temperature") -> float:
        """
        Calculate seasonal factor for given timestamp and parameter.
        
        Args:
            timestamp: Target timestamp
            parameter: Parameter type (temperature, humidity, pressure)
            
        Returns:
            Seasonal adjustment factor
        """
        day_of_year = timestamp.timetuple().tm_yday
        
        # Calculate season (0=winter, 1=spring, 2=summer, 3=autumn)
        season_cycle = (day_of_year - 1) / 365.25 * 4
        
        if parameter == "temperature":
            # Temperature follows sine wave with peak in summer
            return np.sin(2 * np.pi * (day_of_year - 81) / 365.25)  # Peak around June 21
            
        elif parameter == "humidity":
            # Humidity tends to be higher in winter, lower in summer
            return -0.3 * np.sin(2 * np.pi * (day_of_year - 81) / 365.25)
            
        elif parameter == "pressure":
            # Pressure variations with seasonal weather patterns
            return 0.2 * np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
            
        return 0.0
    
    def generate_weather_event(self, timestamp: datetime) -> Optional[WeatherEvent]:
        """
        Generate random weather events based on seasonal probability.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Weather event or None
        """
        # Seasonal weather probabilities
        day_of_year = timestamp.timetuple().tm_yday
        season_factor = np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Base probability of weather events (5% per hour)
        base_prob = 0.05
        
        # Adjust probability based on season
        if season_factor > 0.5:  # Summer - more storms
            weather_prob = base_prob * 1.5
            likely_conditions = [WeatherCondition.SUNNY, WeatherCondition.STORMY, WeatherCondition.CLOUDY]
        elif season_factor < -0.5:  # Winter - more snow/rain
            weather_prob = base_prob * 1.2
            likely_conditions = [WeatherCondition.CLOUDY, WeatherCondition.RAINY, WeatherCondition.SNOWY]
        else:  # Spring/Autumn - moderate weather
            weather_prob = base_prob
            likely_conditions = [WeatherCondition.SUNNY, WeatherCondition.CLOUDY, WeatherCondition.RAINY]
            
        if np.random.random() < weather_prob:
            condition = np.random.choice(likely_conditions)
            duration = np.random.exponential(4.0) + 1.0  # 1-12 hours typical
            intensity = np.random.beta(2, 3)  # Skewed toward lower intensity
            
            return WeatherEvent(
                condition=condition,
                start_time=timestamp,
                duration_hours=duration,
                intensity=intensity
            )
        return None
    
    def get_weather_effects(self, timestamp: datetime) -> Dict[str, float]:
        """
        Calculate current weather effects on sensor readings.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary of weather effect multipliers
        """
        effects = {
            'temperature': 0.0,
            'humidity': 0.0,
            'pressure': 0.0
        }
        
        # Clean up expired events
        self.active_weather_events = [
            event for event in self.active_weather_events
            if timestamp < event.start_time + timedelta(hours=event.duration_hours)
        ]
        
        # Generate new weather events
        new_event = self.generate_weather_event(timestamp)
        if new_event:
            self.active_weather_events.append(new_event)
        
        # Apply effects from active events
        for event in self.active_weather_events:
            time_into_event = (timestamp - event.start_time).total_seconds() / 3600
            event_progress = min(time_into_event / event.duration_hours, 1.0)
            
            # Weather intensity factor (peak in middle of event)
            intensity_factor = event.intensity * np.sin(np.pi * event_progress)
            
            if event.condition == WeatherCondition.RAINY:
                effects['temperature'] -= 3.0 * intensity_factor
                effects['humidity'] += 20.0 * intensity_factor
                effects['pressure'] -= 8.0 * intensity_factor
                
            elif event.condition == WeatherCondition.STORMY:
                effects['temperature'] -= 5.0 * intensity_factor
                effects['humidity'] += 15.0 * intensity_factor
                effects['pressure'] -= 15.0 * intensity_factor
                
            elif event.condition == WeatherCondition.SNOWY:
                effects['temperature'] -= 8.0 * intensity_factor
                effects['humidity'] += 10.0 * intensity_factor
                effects['pressure'] -= 5.0 * intensity_factor
                
            elif event.condition == WeatherCondition.SUNNY:
                effects['temperature'] += 2.0 * intensity_factor
                effects['humidity'] -= 5.0 * intensity_factor
                effects['pressure'] += 2.0 * intensity_factor
                
        return effects
    
    def generate_anomaly(self, sensor_type: str, 
                        base_value: float, 
                        anomaly_probability: float = 0.01) -> float:
        """
        Generate sensor anomalies for testing robustness.
        
        Args:
            sensor_type: Type of sensor (temperature, humidity, pressure)
            base_value: Normal sensor value
            anomaly_probability: Probability of anomaly occurrence
            
        Returns:
            Value with potential anomaly
        """
        if np.random.random() < anomaly_probability:
            anomaly_types = ['spike', 'drop', 'drift', 'noise']
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'spike':
                # Sudden spike (sensor malfunction)
                multiplier = np.random.uniform(1.5, 3.0)
                return base_value * multiplier
                
            elif anomaly_type == 'drop':
                # Sudden drop (sensor failure)
                multiplier = np.random.uniform(0.1, 0.5)
                return base_value * multiplier
                
            elif anomaly_type == 'drift':
                # Gradual drift (calibration issue)
                drift = np.random.normal(0, abs(base_value) * 0.2)
                return base_value + drift
                
            elif anomaly_type == 'noise':
                # High noise (interference)
                noise = np.random.normal(0, abs(base_value) * 0.3)
                return base_value + noise
                
        return base_value
    
    def generate_correlation_effects(self, sensor_readings: Dict[str, float]) -> Dict[str, float]:
        """
        Apply cross-sensor correlations for realistic behavior.
        
        Args:
            sensor_readings: Current sensor readings
            
        Returns:
            Adjusted sensor readings with correlations
        """
        adjusted = sensor_readings.copy()
        
        # Temperature-Humidity inverse correlation
        if 'temperature' in adjusted and 'humidity' in adjusted:
            temp_normalized = (adjusted['temperature'] - 20) / 20  # Normalize around 20°C
            humidity_adjustment = -5.0 * temp_normalized
            adjusted['humidity'] += humidity_adjustment
            
        # Pressure-Weather correlation (already handled in weather effects)
        # Additional correlations can be added here
        
        return adjusted


class EnhancedTimeSeries:
    """Enhanced time series generation with realistic patterns."""
    
    def __init__(self, pattern_generator: AdvancedPatternGenerator):
        """
        Initialize enhanced time series generator.
        
        Args:
            pattern_generator: Advanced pattern generator instance
        """
        self.pattern_generator = pattern_generator
        
    def generate_training_dataset(self, 
                                start_date: datetime,
                                duration_days: int,
                                interval_minutes: int = 15,
                                sensor_config: Dict = None) -> pd.DataFrame:
        """
        Generate comprehensive training dataset for ML models.
        
        Args:
            start_date: Starting date for data generation
            duration_days: Number of days to generate
            interval_minutes: Interval between readings in minutes
            sensor_config: Configuration for sensor parameters
            
        Returns:
            DataFrame with enhanced time series data
        """
        if sensor_config is None:
            sensor_config = {
                'temperature': {'base': 22.0, 'seasonal_amp': 8.0, 'daily_amp': 3.0},
                'humidity': {'base': 50.0, 'seasonal_amp': 15.0, 'daily_amp': 20.0},
                'pressure': {'base': 1013.25, 'seasonal_amp': 20.0, 'daily_amp': 5.0}
            }
        
        # Generate time index
        end_date = start_date + timedelta(days=duration_days)
        timestamps = pd.date_range(start_date, end_date, freq=f'{interval_minutes}min')
        
        data_points = []
        
        for timestamp in timestamps:
            # Get base patterns
            day_of_year = timestamp.timetuple().tm_yday
            hour_of_day = timestamp.hour + timestamp.minute / 60.0
            
            # Generate readings for each sensor type
            readings = {}
            
            for sensor_type, config in sensor_config.items():
                # Base value
                base_value = config['base']
                
                # Seasonal variation
                seasonal_factor = self.pattern_generator.get_seasonal_factor(timestamp, sensor_type)
                seasonal_effect = seasonal_factor * config['seasonal_amp']
                
                # Daily variation
                if sensor_type == 'temperature':
                    daily_effect = config['daily_amp'] * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
                elif sensor_type == 'humidity':
                    daily_effect = config['daily_amp'] * np.cos(2 * np.pi * (hour_of_day - 6) / 24)
                else:  # pressure
                    daily_effect = config['daily_amp'] * np.sin(2 * np.pi * hour_of_day / 24)
                
                # Calculate base reading
                sensor_value = base_value + seasonal_effect + daily_effect
                
                # Add noise
                noise = np.random.normal(0, abs(sensor_value) * 0.02)
                sensor_value += noise
                
                readings[sensor_type] = sensor_value
            
            # Apply weather effects
            weather_effects = self.pattern_generator.get_weather_effects(timestamp)
            for sensor_type in readings:
                readings[sensor_type] += weather_effects[sensor_type]
            
            # Apply correlations
            readings = self.pattern_generator.generate_correlation_effects(readings)
            
            # Apply anomalies
            for sensor_type in readings:
                readings[sensor_type] = self.pattern_generator.generate_anomaly(
                    sensor_type, readings[sensor_type]
                )
            
            # Create data point
            data_point = {
                'timestamp': timestamp,
                'hour_of_day': hour_of_day,
                'day_of_year': day_of_year,
                'month': timestamp.month,
                'day_of_week': timestamp.weekday(),
                'is_weekend': timestamp.weekday() >= 5,
                **readings
            }
            
            # Add lag features for ML training
            data_points.append(data_point)
        
        df = pd.DataFrame(data_points)
        
        # Add lag features
        for sensor_type in sensor_config.keys():
            for lag in [1, 2, 3, 6, 12, 24]:  # Various lag periods
                df[f'{sensor_type}_lag_{lag}'] = df[sensor_type].shift(lag)
        
        # Add rolling statistics
        for sensor_type in sensor_config.keys():
            df[f'{sensor_type}_rolling_mean_24'] = df[sensor_type].rolling(window=24).mean()
            df[f'{sensor_type}_rolling_std_24'] = df[sensor_type].rolling(window=24).std()
            df[f'{sensor_type}_rolling_mean_168'] = df[sensor_type].rolling(window=168).mean()  # Weekly
        
        return df
    
    def export_training_data(self, df: pd.DataFrame, 
                           export_path: str, 
                           format: str = 'csv') -> None:
        """
        Export training data in specified format.
        
        Args:
            df: DataFrame to export
            export_path: Path for export file
            format: Export format (csv, parquet, json)
        """
        if format.lower() == 'csv':
            df.to_csv(export_path, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(export_path, index=False)
        elif format.lower() == 'json':
            df.to_json(export_path, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Training data exported to {export_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
