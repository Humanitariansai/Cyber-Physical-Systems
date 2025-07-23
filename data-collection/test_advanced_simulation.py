"""
Advanced Unit Tests for Enhanced Sensor Simulation
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

Comprehensive test suite for advanced sensor simulation capabilities.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_sensor_simulator import (
    AdvancedSensor, AdvancedSensorNetwork, AdvancedDataCollectionManager
)
from enhanced_config import config_manager, SensorConfig, SeasonalParams, DailyParams
from advanced_patterns import AdvancedPatternGenerator, WeatherCondition


class TestEnhancedConfig:
    """Test enhanced configuration management."""
    
    def test_config_manager_initialization(self):
        """Test that config manager initializes with default presets."""
        presets = config_manager.list_presets()
        assert len(presets) > 0
        assert "office" in presets
        assert "outdoor" in presets
        assert "industrial" in presets
    
    def test_preset_retrieval(self):
        """Test preset retrieval functionality."""
        office_preset = config_manager.get_preset("office")
        assert office_preset is not None
        assert office_preset.name == "Office Environment"
        assert "temperature" in office_preset.sensors
        assert "humidity" in office_preset.sensors
        assert "pressure" in office_preset.sensors
    
    def test_sensor_config_validation(self):
        """Test sensor configuration parameters."""
        office_preset = config_manager.get_preset("office")
        temp_config = office_preset.sensors["temperature"]
        
        assert temp_config.base_value > 0
        assert len(temp_config.valid_range) == 2
        assert temp_config.valid_range[0] < temp_config.valid_range[1]
        assert temp_config.units == "°C"
    
    def test_ml_training_config_generation(self):
        """Test ML training configuration generation."""
        ml_config = config_manager.create_ml_training_config(
            preset_name="office",
            duration_days=30,
            interval_minutes=15
        )
        
        assert ml_config["preset"] == "office"
        assert ml_config["duration_days"] == 30
        assert ml_config["interval_minutes"] == 15
        assert "features" in ml_config
        assert "lag_features" in ml_config["features"]
        assert ml_config["total_samples"] > 0


class TestAdvancedPatternGenerator:
    """Test advanced pattern generation capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pattern_gen = AdvancedPatternGenerator("Test Location")
        self.pattern_gen.set_random_seed(42)  # For reproducible tests
    
    def test_seasonal_factor_calculation(self):
        """Test seasonal factor calculation."""
        # Test winter (January 1st)
        winter_date = datetime(2024, 1, 1)
        winter_factor = self.pattern_gen.get_seasonal_factor(winter_date, "temperature")
        
        # Test summer (July 1st)
        summer_date = datetime(2024, 7, 1)
        summer_factor = self.pattern_gen.get_seasonal_factor(summer_date, "temperature")
        
        # Summer should have higher temperature factor than winter
        assert summer_factor > winter_factor
        
        # Factors should be in reasonable range
        assert -1.5 <= winter_factor <= 1.5
        assert -1.5 <= summer_factor <= 1.5
    
    def test_weather_event_generation(self):
        """Test weather event generation."""
        test_timestamp = datetime(2024, 6, 15, 12, 0)  # Summer day
        
        # Generate multiple weather events to test probability
        events = []
        for _ in range(1000):
            event = self.pattern_gen.generate_weather_event(test_timestamp)
            if event:
                events.append(event)
        
        # Should generate some events but not too many
        assert 0 <= len(events) <= 100  # Roughly 5% probability
        
        # Check event properties if any were generated
        if events:
            event = events[0]
            assert isinstance(event.condition, WeatherCondition)
            assert event.duration_hours > 0
            assert 0.0 <= event.intensity <= 1.0
    
    def test_weather_effects_calculation(self):
        """Test weather effects on sensor readings."""
        test_timestamp = datetime(2024, 6, 15, 12, 0)
        
        effects = self.pattern_gen.get_weather_effects(test_timestamp)
        
        assert "temperature" in effects
        assert "humidity" in effects
        assert "pressure" in effects
        
        # Effects should be numeric
        assert isinstance(effects["temperature"], (int, float))
        assert isinstance(effects["humidity"], (int, float))
        assert isinstance(effects["pressure"], (int, float))


class TestAdvancedSensor:
    """Test advanced sensor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.office_preset = config_manager.get_preset("office")
        self.temp_config = self.office_preset.sensors["temperature"]
        
        self.sensor = AdvancedSensor(
            sensor_id="TEST_TEMP_001",
            sensor_type="temperature",
            config=self.temp_config,
            location="Test Lab"
        )
    
    def test_sensor_initialization(self):
        """Test sensor initialization."""
        assert self.sensor.sensor_id == "TEST_TEMP_001"
        assert self.sensor.sensor_type == "temperature"
        assert self.sensor.location == "Test Lab"
        assert self.sensor._reading_count == 0
    
    def test_basic_reading_generation(self):
        """Test basic sensor reading generation."""
        reading = self.sensor.read_value()
        
        assert "timestamp" in reading
        assert "sensor_id" in reading
        assert "sensor_type" in reading
        assert "value" in reading
        assert "unit" in reading
        assert "quality_score" in reading
        assert "metadata" in reading
        
        # Value should be within valid range
        value = reading["value"]
        assert self.temp_config.valid_range[0] <= value <= self.temp_config.valid_range[1]
        
        # Quality score should be between 0 and 1
        assert 0.0 <= reading["quality_score"] <= 1.0
    
    def test_reading_counter(self):
        """Test reading counter functionality."""
        initial_count = self.sensor._reading_count
        
        # Take several readings
        for _ in range(5):
            self.sensor.read_value()
        
        assert self.sensor._reading_count == initial_count + 5
        
        # Reset counter
        self.sensor.reset_state()
        assert self.sensor._reading_count == 0
    
    def test_temporal_consistency(self):
        """Test that readings show consistent temporal patterns."""
        readings = []
        base_time = datetime(2024, 6, 15, 0, 0)  # Start at midnight
        
        # Generate readings every hour for 24 hours
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)
            reading = self.sensor.read_value(timestamp)
            readings.append((hour, reading["value"]))
        
        # Extract values
        hours, values = zip(*readings)
        
        # Should show some daily variation pattern
        daily_mean = np.mean(values)
        daily_std = np.std(values)
        
        assert daily_std > 0  # Should have some variation
        assert daily_std < daily_mean * 0.5  # But not too extreme
    
    def test_sensor_statistics(self):
        """Test sensor statistics generation."""
        # Generate some readings
        for _ in range(10):
            self.sensor.read_value()
        
        stats = self.sensor.get_statistics()
        
        assert "sensor_id" in stats
        assert "reading_count" in stats
        assert "mean_value" in stats
        assert "std_value" in stats
        assert stats["reading_count"] == 10


class TestAdvancedSensorNetwork:
    """Test advanced sensor network functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.network = AdvancedSensorNetwork("TEST_NETWORK", "office")
    
    def test_network_initialization(self):
        """Test network initialization."""
        assert self.network.network_id == "TEST_NETWORK"
        assert self.network.environment_preset == "office"
        assert len(self.network.sensors) > 0
        
        # Should have sensors for each type in preset
        sensor_types = [sensor.sensor_type for sensor in self.network.sensors.values()]
        assert "temperature" in sensor_types
        assert "humidity" in sensor_types
        assert "pressure" in sensor_types
    
    def test_all_sensors_reading(self):
        """Test reading from all sensors."""
        readings = self.network.read_all_sensors()
        
        assert len(readings) == len(self.network.sensors)
        
        # Each reading should have required fields
        for reading in readings:
            assert "sensor_type" in reading
            assert "value" in reading
            assert "timestamp" in reading
    
    def test_ml_dataset_generation(self):
        """Test ML dataset generation."""
        df = self.network.generate_ml_dataset(
            duration_days=2,
            interval_minutes=60,
            include_features=True
        )
        
        assert len(df) > 0
        assert "timestamp" in df.columns
        
        # Should have sensor value columns
        sensor_cols = [col for col in df.columns if "_value" in col]
        assert len(sensor_cols) > 0
        
        # Should have engineered features
        assert "hour" in df.columns
        assert "day_of_week" in df.columns
        
        # Should have lag features
        lag_cols = [col for col in df.columns if "_lag_" in col]
        assert len(lag_cols) > 0
    
    def test_data_export(self):
        """Test data export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, "test_data.parquet")
            
            self.network.export_ml_data(
                filepath=export_path,
                duration_days=1,
                interval_minutes=60,
                format='parquet'
            )
            
            assert os.path.exists(export_path)
            
            # Load and verify exported data
            df = pd.read_parquet(export_path)
            assert len(df) > 0
    
    def test_network_summary(self):
        """Test network summary generation."""
        # Generate some readings first
        for _ in range(5):
            self.network.read_all_sensors()
        
        summary = self.network.get_network_summary()
        
        assert "network_id" in summary
        assert "sensor_count" in summary
        assert "total_readings" in summary
        assert "sensors" in summary
        assert summary["sensor_count"] > 0
        assert summary["total_readings"] > 0


class TestAdvancedDataCollectionManager:
    """Test advanced data collection manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdvancedDataCollectionManager()
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert len(self.manager.networks) == 0
        assert self.manager.global_config is not None
    
    def test_network_creation(self):
        """Test network creation."""
        network = self.manager.create_network("TEST_NET", "office")
        
        assert network.network_id == "TEST_NET"
        assert "TEST_NET" in self.manager.networks
        
        # Should not allow duplicate network IDs
        with pytest.raises(ValueError):
            self.manager.create_network("TEST_NET", "office")
    
    def test_preset_listing(self):
        """Test available presets listing."""
        presets = self.manager.list_available_presets()
        assert len(presets) > 0
        assert "office" in presets
    
    def test_comparative_dataset_generation(self):
        """Test comparative dataset generation across networks."""
        # Create multiple networks
        self.manager.create_network("NET1", "office")
        self.manager.create_network("NET2", "outdoor")
        
        df = self.manager.generate_comparative_dataset(
            duration_days=1,
            interval_minutes=120
        )
        
        assert len(df) > 0
        assert "source_network" in df.columns
        
        # Should have data from both networks
        networks_in_data = df["source_network"].unique()
        assert "NET1" in networks_in_data
        assert "NET2" in networks_in_data
    
    def test_system_status(self):
        """Test system status reporting."""
        # Create some networks
        self.manager.create_network("NET1", "office")
        self.manager.create_network("NET2", "industrial")
        
        status = self.manager.get_system_status()
        
        assert "total_networks" in status
        assert "total_sensors" in status
        assert "networks" in status
        assert status["total_networks"] == 2
        assert status["total_sensors"] > 0


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from setup to data export."""
        # 1. Create manager and network
        manager = AdvancedDataCollectionManager()
        network = manager.create_network("E2E_TEST", "office")
        
        # 2. Generate some readings
        readings = []
        for i in range(10):
            timestamp = datetime.now() + timedelta(minutes=i*15)
            reading_batch = network.read_all_sensors(timestamp)
            readings.extend(reading_batch)
        
        assert len(readings) > 0
        
        # 3. Generate ML dataset
        df = network.generate_ml_dataset(
            duration_days=1,
            interval_minutes=30,
            include_features=True
        )
        
        assert len(df) > 0
        
        # 4. Test export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, "e2e_test.csv")
            df.to_csv(export_path, index=False)
            assert os.path.exists(export_path)
        
        # 5. Get system status
        status = manager.get_system_status()
        assert status["total_networks"] == 1
        assert status["total_sensors"] > 0
    
    def test_multiple_environment_comparison(self):
        """Test comparison across different environments."""
        manager = AdvancedDataCollectionManager()
        
        # Create networks for different environments
        environments = ["office", "outdoor", "industrial"]
        for env in environments:
            manager.create_network(f"{env}_test", env)
        
        # Generate comparative dataset
        df = manager.generate_comparative_dataset(
            duration_days=1,
            interval_minutes=60
        )
        
        # Should have data from all environments
        env_data = df["source_network"].unique()
        for env in environments:
            assert f"{env}_test" in env_data
        
        # Different environments should show different patterns
        env_means = df.groupby("source_network")["temperature_value"].mean()
        assert len(env_means.unique()) > 1  # Different means


# Performance tests
class TestPerformance:
    """Performance tests for the simulation system."""
    
    def test_large_dataset_generation(self):
        """Test generation of large datasets."""
        network = AdvancedSensorNetwork("PERF_TEST", "office")
        
        start_time = datetime.now()
        
        # Generate a month of data at 15-minute intervals
        df = network.generate_ml_dataset(
            duration_days=30,
            interval_minutes=15,
            include_features=True
        )
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time (less than 30 seconds)
        assert generation_time < 30
        
        # Should generate expected amount of data
        expected_points = (30 * 24 * 60) // 15  # 30 days, 15-min intervals
        assert len(df) >= expected_points * 0.9  # Allow 10% tolerance
    
    def test_memory_usage(self):
        """Test memory usage for large datasets."""
        network = AdvancedSensorNetwork("MEM_TEST", "office")
        
        # Generate dataset
        df = network.generate_ml_dataset(
            duration_days=7,
            interval_minutes=15,
            include_features=True
        )
        
        # Check memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        
        # Should use reasonable amount of memory (less than 100 MB for a week)
        assert memory_mb < 100


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
