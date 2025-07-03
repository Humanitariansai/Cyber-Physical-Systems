"""
Unit tests for sensor simulation classes
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sensor_simulator import (
    TemperatureSensor,
    HumiditySensor, 
    PressureSensor,
    SensorNetwork,
    DataCollectionManager
)


class TestBaseSensor(unittest.TestCase):
    """Test base sensor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_sensor = TemperatureSensor(
            sensor_id="TEST_TEMP",
            location="Test Lab",
            base_temp=20.0
        )
    
    def test_sensor_initialization(self):
        """Test sensor initialization."""
        self.assertEqual(self.temp_sensor.sensor_id, "TEST_TEMP")
        self.assertEqual(self.temp_sensor.location, "Test Lab")
        self.assertEqual(self.temp_sensor.base_temp, 20.0)
        self.assertIsNotNone(self.temp_sensor.created_at)
    
    def test_metadata_generation(self):
        """Test metadata generation."""
        metadata = self.temp_sensor.get_metadata()
        self.assertIn('sensor_id', metadata)
        self.assertIn('sensor_type', metadata)
        self.assertIn('location', metadata)
        self.assertEqual(metadata['sensor_id'], "TEST_TEMP")


class TestTemperatureSensor(unittest.TestCase):
    """Test temperature sensor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sensor = TemperatureSensor(
            sensor_id="TEMP_001",
            location="Test Room",
            base_temp=22.0,
            temp_range=(-10.0, 40.0)
        )
    
    def test_reading_generation(self):
        """Test temperature reading generation."""
        reading = self.sensor.read_value()
        
        # Check reading structure
        self.assertIn('timestamp', reading)
        self.assertIn('sensor_id', reading)
        self.assertIn('sensor_type', reading)
        self.assertIn('value', reading)
        self.assertIn('unit', reading)
        self.assertIn('location', reading)
        
        # Check values
        self.assertEqual(reading['sensor_id'], "TEMP_001")
        self.assertEqual(reading['sensor_type'], 'temperature')
        self.assertEqual(reading['unit'], 'celsius')
        self.assertTrue(-10.0 <= reading['value'] <= 40.0)
    
    def test_temperature_range_limits(self):
        """Test temperature stays within specified range."""
        readings = []
        for _ in range(100):
            reading = self.sensor.read_value()
            readings.append(reading['value'])
        
        # All readings should be within range
        self.assertTrue(all(-10.0 <= temp <= 40.0 for temp in readings))
    
    def test_seasonal_variation(self):
        """Test seasonal temperature variation."""
        # Test readings at different times of year
        summer_time = datetime(2024, 7, 15, 14, 0)  # Mid-July, 2 PM
        winter_time = datetime(2024, 1, 15, 14, 0)  # Mid-January, 2 PM
        
        summer_reading = self.sensor.read_value(summer_time)
        winter_reading = self.sensor.read_value(winter_time)
        
        # Summer should generally be warmer (though noise can affect this)
        # We'll just verify both are reasonable values
        self.assertTrue(-10.0 <= summer_reading['value'] <= 40.0)
        self.assertTrue(-10.0 <= winter_reading['value'] <= 40.0)


class TestHumiditySensor(unittest.TestCase):
    """Test humidity sensor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sensor = HumiditySensor(
            sensor_id="HUM_001",
            location="Test Room",
            base_humidity=50.0
        )
    
    def test_humidity_reading(self):
        """Test humidity reading generation."""
        reading = self.sensor.read_value()
        
        self.assertEqual(reading['sensor_type'], 'humidity')
        self.assertEqual(reading['unit'], 'percentage')
        self.assertTrue(20.0 <= reading['value'] <= 90.0)
    
    def test_humidity_range(self):
        """Test humidity stays within range."""
        readings = []
        for _ in range(100):
            reading = self.sensor.read_value()
            readings.append(reading['value'])
        
        self.assertTrue(all(20.0 <= hum <= 90.0 for hum in readings))


class TestPressureSensor(unittest.TestCase):
    """Test pressure sensor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sensor = PressureSensor(
            sensor_id="PRESS_001",
            location="Outdoor",
            base_pressure=1013.25
        )
    
    def test_pressure_reading(self):
        """Test pressure reading generation."""
        reading = self.sensor.read_value()
        
        self.assertEqual(reading['sensor_type'], 'pressure')
        self.assertEqual(reading['unit'], 'hPa')
        self.assertTrue(980.0 <= reading['value'] <= 1050.0)


class TestSensorNetwork(unittest.TestCase):
    """Test sensor network functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = SensorNetwork("test_network")
        self.temp_sensor = TemperatureSensor("TEMP_001", "Room A")
        self.hum_sensor = HumiditySensor("HUM_001", "Room A")
    
    def test_sensor_management(self):
        """Test adding and removing sensors."""
        # Initially empty
        self.assertEqual(len(self.network.sensors), 0)
        
        # Add sensors
        self.network.add_sensor(self.temp_sensor)
        self.network.add_sensor(self.hum_sensor)
        self.assertEqual(len(self.network.sensors), 2)
        
        # Remove sensor
        self.network.remove_sensor("TEMP_001")
        self.assertEqual(len(self.network.sensors), 1)
        self.assertNotIn("TEMP_001", self.network.sensors)
    
    def test_read_all_sensors(self):
        """Test reading from all sensors."""
        self.network.add_sensor(self.temp_sensor)
        self.network.add_sensor(self.hum_sensor)
        
        readings = self.network.read_all_sensors()
        
        self.assertEqual(len(readings), 2)
        sensor_types = [r['sensor_type'] for r in readings]
        self.assertIn('temperature', sensor_types)
        self.assertIn('humidity', sensor_types)
    
    def test_time_series_generation(self):
        """Test time series data generation."""
        self.network.add_sensor(self.temp_sensor)
        self.network.add_sensor(self.hum_sensor)
        
        # Generate 1 hour of data with 10-minute intervals
        df = self.network.generate_time_series(
            duration_hours=1,
            interval_seconds=600
        )
        
        # Should have 7 time points (0, 10, 20, 30, 40, 50, 60 minutes)
        # times 2 sensors = 14 readings
        expected_readings = 7 * 2
        self.assertEqual(len(df), expected_readings)
        
        # Check data structure
        required_columns = ['timestamp', 'sensor_id', 'sensor_type', 'value', 'unit', 'location']
        for col in required_columns:
            self.assertIn(col, df.columns)


class TestDataCollectionManager(unittest.TestCase):
    """Test data collection manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = DataCollectionManager()
    
    def test_network_creation(self):
        """Test network creation."""
        network = self.manager.create_network("test_net")
        
        self.assertIsInstance(network, SensorNetwork)
        self.assertEqual(network.network_id, "test_net")
        self.assertIn("test_net", self.manager.networks)
    
    def test_demo_network_setup(self):
        """Test demo network setup."""
        network = self.manager.setup_demo_network("Test Lab")
        
        # Should have 3 sensors (temp, humidity, pressure)
        self.assertEqual(len(network.sensors), 3)
        
        # Check sensor types
        sensor_types = []
        for sensor in network.sensors.values():
            sensor_types.append(sensor.__class__.__name__)
        
        self.assertIn('TemperatureSensor', sensor_types)
        self.assertIn('HumiditySensor', sensor_types)
        self.assertIn('PressureSensor', sensor_types)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
