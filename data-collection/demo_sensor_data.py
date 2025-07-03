"""
Demonstration script for sensor data generation
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

This script demonstrates how to use the sensor simulation classes
to generate realistic time-series data for cyber-physical systems.
"""

from sensor_simulator import (
    TemperatureSensor, 
    HumiditySensor, 
    PressureSensor,
    SensorNetwork,
    DataCollectionManager
)
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def demo_single_sensor():
    """Demonstrate individual sensor functionality."""
    print("=== Single Sensor Demo ===")
    
    # Create a temperature sensor
    temp_sensor = TemperatureSensor(
        sensor_id="TEMP_DEMO",
        location="Laboratory Room 1",
        base_temp=23.0,
        seasonal_variation=4.0,
        daily_variation=3.0,
        noise_level=0.2
    )
    
    # Generate some readings
    print("Temperature sensor readings:")
    for i in range(5):
        reading = temp_sensor.read_value()
        print(f"  {reading['timestamp']}: {reading['value']}°C")
    
    print(f"\nSensor metadata: {temp_sensor.get_metadata()}")
    print()


def demo_sensor_network():
    """Demonstrate sensor network functionality."""
    print("=== Sensor Network Demo ===")
    
    # Create sensor network
    network = SensorNetwork("demo_network")
    
    # Add multiple sensors
    sensors = [
        TemperatureSensor("TEMP_001", "Room A", base_temp=22.0),
        TemperatureSensor("TEMP_002", "Room B", base_temp=24.0),
        HumiditySensor("HUM_001", "Room A", base_humidity=50.0),
        HumiditySensor("HUM_002", "Room B", base_humidity=45.0),
        PressureSensor("PRESS_001", "Outdoor", base_pressure=1013.0)
    ]
    
    for sensor in sensors:
        network.add_sensor(sensor)
    
    # Read from all sensors
    print("Network readings:")
    readings = network.read_all_sensors()
    for reading in readings:
        print(f"  {reading['sensor_id']}: {reading['value']} {reading['unit']}")
    
    print(f"\nNetwork status: {network.get_network_status()}")
    print()


def demo_time_series_generation():
    """Demonstrate time series data generation."""
    print("=== Time Series Generation Demo ===")
    
    # Create data collection manager
    manager = DataCollectionManager()
    
    # Set up demo network
    network = manager.setup_demo_network("Cyber-Physical Lab")
    
    # Generate 24 hours of data with 5-minute intervals
    print("Generating 24 hours of sensor data...")
    df = network.generate_time_series(
        duration_hours=24,
        interval_seconds=300  # 5 minutes
    )
    
    print(f"Generated {len(df)} data points")
    print(f"Data shape: {df.shape}")
    print(f"Sensors: {df['sensor_type'].unique()}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save data to CSV
    output_dir = "../data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_filename = f"{output_dir}/demo_sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to: {csv_filename}")
    
    return df


def demo_realistic_patterns():
    """Demonstrate realistic sensor patterns."""
    print("=== Realistic Patterns Demo ===")
    
    # Create sensors with different characteristics
    indoor_temp = TemperatureSensor(
        "TEMP_INDOOR", 
        "Indoor",
        base_temp=22.0,
        daily_variation=2.0,
        noise_level=0.1
    )
    
    outdoor_temp = TemperatureSensor(
        "TEMP_OUTDOOR",
        "Outdoor", 
        base_temp=15.0,
        daily_variation=8.0,
        seasonal_variation=10.0,
        noise_level=0.3
    )
    
    # Generate readings over a week
    start_time = datetime.now()
    data = []
    
    for hour in range(24 * 7):  # One week
        timestamp = start_time + timedelta(hours=hour)
        
        indoor_reading = indoor_temp.read_value(timestamp)
        outdoor_reading = outdoor_temp.read_value(timestamp)
        
        data.extend([indoor_reading, outdoor_reading])
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Show statistics
    print("Temperature statistics:")
    for sensor_type in ['TEMP_INDOOR', 'TEMP_OUTDOOR']:
        sensor_data = df[df['sensor_id'] == sensor_type]['value']
        print(f"  {sensor_type}:")
        print(f"    Mean: {sensor_data.mean():.2f}°C")
        print(f"    Std:  {sensor_data.std():.2f}°C")
        print(f"    Min:  {sensor_data.min():.2f}°C")
        print(f"    Max:  {sensor_data.max():.2f}°C")
    
    print()


def demo_data_export():
    """Demonstrate data export functionality."""
    print("=== Data Export Demo ===")
    
    network = SensorNetwork("export_demo")
    
    # Add sensors
    network.add_sensor(TemperatureSensor("TEMP_001", "Lab"))
    network.add_sensor(HumiditySensor("HUM_001", "Lab"))
    network.add_sensor(PressureSensor("PRESS_001", "Lab"))
    
    # Collect some data
    for i in range(10):
        network.read_all_sensors()
    
    # Export to different formats
    output_dir = "../data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    csv_file = f"{output_dir}/sensor_export_{timestamp}.csv"
    json_file = f"{output_dir}/sensor_export_{timestamp}.json"
    
    network.export_data(csv_file, format='csv')
    network.export_data(json_file, format='json', clear_buffer=False)
    
    print(f"Data exported to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")
    print()


def main():
    """Run all demonstrations."""
    print("Sensor Data Simulation Demonstration")
    print("=" * 50)
    print()
    
    # Run demonstrations
    demo_single_sensor()
    demo_sensor_network()
    df = demo_time_series_generation()
    demo_realistic_patterns()
    demo_data_export()
    
    print("All demonstrations completed successfully!")
    print("\nNext steps:")
    print("1. Check the generated data files in ../data/raw/")
    print("2. Use the SensorNetwork class to create custom sensor configurations")
    print("3. Integrate with cloud upload utilities for real-time data streaming")
    print("4. Implement data preprocessing for ML model training")


if __name__ == "__main__":
    main()
