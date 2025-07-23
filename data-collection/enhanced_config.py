"""
Enhanced Configuration for Advanced Sensor Simulation
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

This module provides advanced configuration options for realistic sensor simulation
including environmental presets, seasonal configurations, and ML training parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


@dataclass
class SeasonalParams:
    """Seasonal variation parameters for sensors."""
    winter_offset: float = -5.0
    spring_offset: float = 2.0
    summer_offset: float = 8.0
    autumn_offset: float = 0.0
    amplitude: float = 5.0
    phase_shift: float = 0.0  # Days to shift seasonal peak


@dataclass
class DailyParams:
    """Daily variation parameters for sensors."""
    amplitude: float = 3.0
    peak_hour: float = 14.0  # 2 PM for temperature
    phase_shift: float = 0.0
    weekend_modifier: float = 1.0  # Multiplier for weekends


@dataclass
class NoiseParams:
    """Noise characteristics for realistic sensor readings."""
    base_noise: float = 0.02  # Base noise level (% of reading)
    random_walk: float = 0.001  # Random walk drift
    burst_probability: float = 0.001  # Probability of noise bursts
    burst_amplitude: float = 0.1  # Amplitude of noise bursts


@dataclass
class CorrelationParams:
    """Cross-sensor correlation parameters."""
    temp_humidity_correlation: float = -0.6  # Inverse correlation
    pressure_humidity_correlation: float = 0.3
    temp_pressure_correlation: float = 0.1
    correlation_lag: int = 1  # Time lag for correlations


@dataclass
class AnomalyParams:
    """Anomaly simulation parameters."""
    probability: float = 0.001  # Base anomaly probability
    spike_probability: float = 0.3  # Fraction that are spikes
    drop_probability: float = 0.2  # Fraction that are drops
    drift_probability: float = 0.3  # Fraction that are drifts
    noise_probability: float = 0.2  # Fraction that are noise
    spike_multiplier: Tuple[float, float] = (2.0, 5.0)
    drop_multiplier: Tuple[float, float] = (0.1, 0.5)
    drift_std: float = 0.2
    noise_multiplier: float = 5.0


@dataclass
class SensorConfig:
    """Complete configuration for a sensor type."""
    base_value: float
    valid_range: Tuple[float, float]
    seasonal: SeasonalParams = field(default_factory=SeasonalParams)
    daily: DailyParams = field(default_factory=DailyParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    anomaly: AnomalyParams = field(default_factory=AnomalyParams)
    units: str = ""
    description: str = ""


@dataclass
class EnvironmentPreset:
    """Environmental preset containing multiple sensor configurations."""
    name: str
    description: str
    location: str
    sensors: Dict[str, SensorConfig]
    correlations: CorrelationParams = field(default_factory=CorrelationParams)
    metadata: Dict = field(default_factory=dict)


class EnhancedConfigManager:
    """Manager for advanced sensor simulation configurations."""
    
    def __init__(self):
        """Initialize configuration manager with default presets."""
        self.presets = {}
        self._create_default_presets()
    
    def _create_default_presets(self) -> None:
        """Create default environmental presets."""
        
        # Indoor Office Environment
        office_temp = SensorConfig(
            base_value=22.0,
            valid_range=(-10.0, 45.0),
            seasonal=SeasonalParams(
                winter_offset=-2.0,
                spring_offset=0.0,
                summer_offset=3.0,
                autumn_offset=1.0,
                amplitude=2.0
            ),
            daily=DailyParams(
                amplitude=1.5,
                peak_hour=15.0,  # Peak after lunch
                weekend_modifier=0.7
            ),
            units="°C",
            description="Indoor office temperature"
        )
        
        office_humidity = SensorConfig(
            base_value=45.0,
            valid_range=(20.0, 80.0),
            seasonal=SeasonalParams(
                winter_offset=10.0,  # Higher in winter (heating)
                spring_offset=0.0,
                summer_offset=-5.0,  # Lower in summer (AC)
                autumn_offset=5.0,
                amplitude=8.0
            ),
            daily=DailyParams(
                amplitude=8.0,
                peak_hour=7.0,  # Higher in morning
                weekend_modifier=1.2
            ),
            units="%",
            description="Indoor office humidity"
        )
        
        office_pressure = SensorConfig(
            base_value=1013.25,
            valid_range=(980.0, 1050.0),
            seasonal=SeasonalParams(amplitude=5.0),
            daily=DailyParams(amplitude=2.0, peak_hour=10.0),
            units="hPa",
            description="Indoor office pressure"
        )
        
        self.presets["office"] = EnvironmentPreset(
            name="Office Environment",
            description="Typical indoor office environment with HVAC",
            location="Indoor Office",
            sensors={
                "temperature": office_temp,
                "humidity": office_humidity,
                "pressure": office_pressure
            }
        )
        
        # Outdoor Weather Station
        outdoor_temp = SensorConfig(
            base_value=18.0,
            valid_range=(-30.0, 50.0),
            seasonal=SeasonalParams(
                winter_offset=-12.0,
                spring_offset=0.0,
                summer_offset=15.0,
                autumn_offset=-2.0,
                amplitude=12.0
            ),
            daily=DailyParams(
                amplitude=8.0,
                peak_hour=14.0,
                weekend_modifier=1.0
            ),
            noise=NoiseParams(base_noise=0.05),  # More outdoor variability
            units="°C",
            description="Outdoor ambient temperature"
        )
        
        outdoor_humidity = SensorConfig(
            base_value=65.0,
            valid_range=(10.0, 100.0),
            seasonal=SeasonalParams(
                winter_offset=-15.0,
                spring_offset=10.0,
                summer_offset=5.0,
                autumn_offset=0.0,
                amplitude=15.0
            ),
            daily=DailyParams(
                amplitude=25.0,
                peak_hour=6.0,  # Higher at dawn
                weekend_modifier=1.0
            ),
            noise=NoiseParams(base_noise=0.08),
            units="%",
            description="Outdoor relative humidity"
        )
        
        outdoor_pressure = SensorConfig(
            base_value=1013.25,
            valid_range=(960.0, 1060.0),
            seasonal=SeasonalParams(amplitude=15.0),
            daily=DailyParams(amplitude=5.0, peak_hour=10.0),
            noise=NoiseParams(base_noise=0.03),
            units="hPa",
            description="Outdoor atmospheric pressure"
        )
        
        self.presets["outdoor"] = EnvironmentPreset(
            name="Outdoor Weather Station",
            description="Outdoor weather monitoring station",
            location="Outdoor Weather Station",
            sensors={
                "temperature": outdoor_temp,
                "humidity": outdoor_humidity,
                "pressure": outdoor_pressure
            },
            correlations=CorrelationParams(
                temp_humidity_correlation=-0.7,
                pressure_humidity_correlation=0.4
            )
        )
        
        # Industrial Environment
        industrial_temp = SensorConfig(
            base_value=35.0,
            valid_range=(10.0, 80.0),
            seasonal=SeasonalParams(amplitude=8.0),
            daily=DailyParams(
                amplitude=12.0,
                peak_hour=13.0,
                weekend_modifier=0.3  # Lower on weekends
            ),
            noise=NoiseParams(base_noise=0.08),
            anomaly=AnomalyParams(probability=0.005),  # Higher anomaly rate
            units="°C",
            description="Industrial facility temperature"
        )
        
        industrial_humidity = SensorConfig(
            base_value=40.0,
            valid_range=(15.0, 85.0),
            seasonal=SeasonalParams(amplitude=10.0),
            daily=DailyParams(amplitude=15.0, peak_hour=11.0, weekend_modifier=0.5),
            noise=NoiseParams(base_noise=0.1),
            units="%",
            description="Industrial facility humidity"
        )
        
        industrial_pressure = SensorConfig(
            base_value=1015.0,
            valid_range=(990.0, 1040.0),
            seasonal=SeasonalParams(amplitude=8.0),
            daily=DailyParams(amplitude=3.0, peak_hour=12.0, weekend_modifier=0.8),
            noise=NoiseParams(base_noise=0.05),
            units="hPa",
            description="Industrial facility pressure"
        )
        
        self.presets["industrial"] = EnvironmentPreset(
            name="Industrial Facility",
            description="Industrial manufacturing environment",
            location="Industrial Facility",
            sensors={
                "temperature": industrial_temp,
                "humidity": industrial_humidity,
                "pressure": industrial_pressure
            },
            correlations=CorrelationParams(
                temp_humidity_correlation=-0.4,
                correlation_lag=2
            )
        )
    
    def get_preset(self, preset_name: str) -> Optional[EnvironmentPreset]:
        """Get environmental preset by name."""
        return self.presets.get(preset_name)
    
    def list_presets(self) -> List[str]:
        """Get list of available preset names."""
        return list(self.presets.keys())
    
    def add_preset(self, preset: EnvironmentPreset) -> None:
        """Add a new environmental preset."""
        self.presets[preset.name.lower().replace(" ", "_")] = preset
    
    def export_preset(self, preset_name: str, filepath: str) -> None:
        """Export preset configuration to JSON file."""
        preset = self.get_preset(preset_name)
        if preset is None:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        # Convert to dict for JSON serialization
        preset_dict = self._preset_to_dict(preset)
        
        with open(filepath, 'w') as f:
            json.dump(preset_dict, f, indent=2, default=str)
        
        print(f"Preset '{preset_name}' exported to {filepath}")
    
    def import_preset(self, filepath: str) -> EnvironmentPreset:
        """Import preset configuration from JSON file."""
        with open(filepath, 'r') as f:
            preset_dict = json.load(f)
        
        preset = self._dict_to_preset(preset_dict)
        self.add_preset(preset)
        return preset
    
    def _preset_to_dict(self, preset: EnvironmentPreset) -> Dict:
        """Convert preset to dictionary for serialization."""
        # This would implement conversion logic
        # Simplified for brevity
        return {
            "name": preset.name,
            "description": preset.description,
            "location": preset.location,
            "sensors": {k: self._sensor_config_to_dict(v) for k, v in preset.sensors.items()},
            "correlations": self._correlation_params_to_dict(preset.correlations),
            "metadata": preset.metadata
        }
    
    def _dict_to_preset(self, preset_dict: Dict) -> EnvironmentPreset:
        """Convert dictionary to preset object."""
        # This would implement conversion logic
        # Simplified for brevity - would need full implementation
        pass
    
    def _sensor_config_to_dict(self, config: SensorConfig) -> Dict:
        """Convert sensor config to dictionary."""
        # Implementation would convert all dataclass fields
        pass
    
    def _correlation_params_to_dict(self, params: CorrelationParams) -> Dict:
        """Convert correlation params to dictionary."""
        # Implementation would convert all dataclass fields
        pass
    
    def create_ml_training_config(self, 
                                preset_name: str,
                                duration_days: int = 365,
                                interval_minutes: int = 15,
                                validation_split: float = 0.2,
                                test_split: float = 0.1) -> Dict:
        """
        Create configuration for ML training data generation.
        
        Args:
            preset_name: Environmental preset to use
            duration_days: Days of data to generate
            interval_minutes: Minutes between readings
            validation_split: Fraction for validation set
            test_split: Fraction for test set
            
        Returns:
            Configuration dictionary for ML training
        """
        preset = self.get_preset(preset_name)
        if preset is None:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        return {
            "preset": preset_name,
            "duration_days": duration_days,
            "interval_minutes": interval_minutes,
            "validation_split": validation_split,
            "test_split": test_split,
            "total_samples": int((duration_days * 24 * 60) / interval_minutes),
            "features": {
                "temporal": ["hour_of_day", "day_of_year", "day_of_week", "is_weekend"],
                "lag_features": [1, 2, 3, 6, 12, 24, 48, 168],  # Various time lags
                "rolling_windows": [6, 24, 168],  # Rolling statistics windows
                "seasonal": ["seasonal_factor", "daily_factor"]
            },
            "target_sensors": list(preset.sensors.keys()),
            "anomaly_detection": True,
            "correlation_features": True
        }


# Global configuration manager instance
config_manager = EnhancedConfigManager()
