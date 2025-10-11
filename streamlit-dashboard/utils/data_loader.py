"""
Data Loader Utility for Streamlit Dashboard
Handles data loading from various sources including CSV files, databases,
and real-time sensor data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "data-collection"))

try:
    from sensor_simulator import SensorDataGenerator
    from advanced_sensor_simulator import AdvancedSensorSimulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False
    print("Sensor simulators not available")

class DataLoader:
    """Handles loading data from various sources for the dashboard"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize simulators if available
        if SIMULATOR_AVAILABLE:
            try:
                self.sensor_simulator = SensorDataGenerator()
                self.advanced_simulator = AdvancedSensorSimulator()
            except Exception as e:
                print(f"Error initializing simulators: {e}")
                self.sensor_simulator = None
                self.advanced_simulator = None
        else:
            self.sensor_simulator = None
            self.advanced_simulator = None
    
    def load_historical_data(self, days_back=30, data_type="all"):
        """
        Load historical sensor data
        
        Args:
            days_back (int): Number of days to load
            data_type (str): Type of data to load ('temperature', 'humidity', 'all')
        
        Returns:
            pd.DataFrame: Historical data
        """
        # Check if historical data file exists
        historical_file = self.data_dir / "historical_sensor_data.csv"
        
        if historical_file.exists():
            df = pd.read_csv(historical_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            # Filter by data type
            if data_type != "all":
                if data_type in df.columns:
                    df = df[['timestamp', data_type]]
            
            return df
        else:
            # Generate simulated historical data
            return self._generate_simulated_historical_data(days_back, data_type)
    
    def _generate_simulated_historical_data(self, days_back=30, data_type="all"):
        """Generate simulated historical data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Create timestamp range (hourly data)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        data = {
            'timestamp': timestamps,
            'temperature': 22 + 5 * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.normal(0, 1, len(timestamps)),
            'humidity': 60 + 15 * np.cos(np.linspace(0, 3*np.pi, len(timestamps))) + np.random.normal(0, 2, len(timestamps)),
            'pressure': 1013 + 10 * np.sin(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 0.5, len(timestamps)),
            'vibration': 0.5 + 0.3 * np.random.random(len(timestamps)),
            'power_consumption': 150 + 30 * np.sin(np.linspace(0, 6*np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
        }
        
        df = pd.DataFrame(data)
        
        # Filter by data type
        if data_type != "all" and data_type in df.columns:
            df = df[['timestamp', data_type]]
        
        return df
    
    def get_real_time_data(self):
        """
        Get current real-time sensor readings
        
        Returns:
            dict: Current sensor readings
        """
        if self.sensor_simulator:
            try:
                # Generate current readings using simulator
                current_time = datetime.now()
                data = self.sensor_simulator.generate_sample_data(1)
                
                return {
                    'timestamp': current_time,
                    'temperature': data.get('temperature', [22.0])[0],
                    'humidity': data.get('humidity', [60.0])[0],
                    'pressure': data.get('pressure', [1013.0])[0],
                    'status': 'active'
                }
            except Exception as e:
                print(f"Error getting real-time data: {e}")
                return self._get_simulated_real_time_data()
        else:
            return self._get_simulated_real_time_data()
    
    def _get_simulated_real_time_data(self):
        """Generate simulated real-time data"""
        return {
            'timestamp': datetime.now(),
            'temperature': 22 + np.random.normal(0, 2),
            'humidity': 60 + np.random.normal(0, 5),
            'pressure': 1013 + np.random.normal(0, 1),
            'vibration': 0.5 + np.random.random() * 0.3,
            'power_consumption': 150 + np.random.normal(0, 10),
            'status': 'simulated'
        }
    
    def load_ml_results(self):
        """
        Load ML model results and performance metrics
        
        Returns:
            dict: ML model performance data
        """
        results_files = {
            'basic_forecaster': self.results_dir / "basic_forecaster_results.csv",
            'xgboost': self.results_dir / "xgboost_results.csv",
            'arima': self.results_dir / "arima_results.csv"
        }
        
        results = {}
        
        for model_name, file_path in results_files.items():
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    results[model_name] = df
                except Exception as e:
                    print(f"Error loading {model_name} results: {e}")
                    results[model_name] = self._generate_dummy_results(model_name)
            else:
                results[model_name] = self._generate_dummy_results(model_name)
        
        return results
    
    def _generate_dummy_results(self, model_name):
        """Generate dummy ML results for demonstration"""
        metrics = {
            'basic_forecaster': {'rmse': 2.34, 'mae': 1.82, 'r2': 0.89},
            'xgboost': {'rmse': 1.87, 'mae': 1.43, 'r2': 0.94},
            'arima': {'rmse': 2.91, 'mae': 2.15, 'r2': 0.83}
        }
        
        return pd.DataFrame([metrics.get(model_name, {'rmse': 2.0, 'mae': 1.5, 'r2': 0.85})])
    
    def save_data(self, data, filename):
        """
        Save data to file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Filename to save to
        """
        try:
            filepath = self.data_dir / filename
            data.to_csv(filepath, index=False)
            return True
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")
            return False
    
    def get_system_metrics(self):
        """
        Get system performance metrics
        
        Returns:
            dict: System metrics
        """
        import psutil
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'memory_total': memory.total,
                'memory_available': memory.available,
                'timestamp': datetime.now()
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                'cpu_usage': np.random.uniform(20, 80),
                'memory_usage': np.random.uniform(30, 70),
                'disk_usage': np.random.uniform(40, 60),
                'memory_total': 8 * 1024**3,  # 8GB
                'memory_available': 6 * 1024**3,  # 6GB
                'timestamp': datetime.now()
            }
    
    def get_data_summary(self):
        """
        Get summary statistics of available data
        
        Returns:
            dict: Data summary
        """
        try:
            historical_data = self.load_historical_data(days_back=30)
            
            summary = {
                'total_records': len(historical_data),
                'date_range': {
                    'start': historical_data['timestamp'].min(),
                    'end': historical_data['timestamp'].max()
                },
                'sensors_active': len([col for col in historical_data.columns if col != 'timestamp']),
                'data_quality': np.random.uniform(95, 99),  # Simulated data quality score
                'missing_values': historical_data.isnull().sum().sum(),
                'last_updated': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error generating data summary: {e}")
            return {
                'total_records': 0,
                'date_range': {'start': None, 'end': None},
                'sensors_active': 0,
                'data_quality': 0,
                'missing_values': 0,
                'last_updated': datetime.now()
            }