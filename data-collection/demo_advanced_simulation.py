"""
Advanced Sensor Simulation Demo
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

This script demonstrates the advanced sensor simulation capabilities including
realistic patterns, correlations, anomalies, and ML-ready data generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add the data-collection path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_sensor_simulator import AdvancedDataCollectionManager, AdvancedSensorNetwork
    from enhanced_config import config_manager
    from advanced_patterns import AdvancedPatternGenerator, EnhancedTimeSeries
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are in the same directory")
    sys.exit(1)


def demo_environment_presets():
    """Demonstrate different environment presets."""
    print("=" * 60)
    print("ENVIRONMENT PRESETS DEMONSTRATION")
    print("=" * 60)
    
    # List available presets
    presets = config_manager.list_presets()
    print(f"Available presets: {presets}")
    print()
    
    manager = AdvancedDataCollectionManager()
    
    # Create networks for different environments
    environments = ["office", "outdoor", "industrial"]
    networks = {}
    
    for env in environments:
        try:
            network_id = f"{env}_demo"
            network = manager.create_network(network_id, env)
            networks[env] = network
            print(f"✓ Created {env} network with {len(network.sensors)} sensors")
        except Exception as e:
            print(f"✗ Failed to create {env} network: {e}")
    
    print()
    return manager, networks


def demo_realistic_patterns(networks, duration_hours=48):
    """Demonstrate realistic pattern generation."""
    print("=" * 60)
    print(f"REALISTIC PATTERN SIMULATION ({duration_hours} hours)")
    print("=" * 60)
    
    # Generate data for each environment
    datasets = {}
    
    for env_name, network in networks.items():
        print(f"Generating data for {env_name} environment...")
        
        # Generate 48 hours of data with 15-minute intervals
        df = network.generate_ml_dataset(
            duration_days=duration_hours/24,
            interval_minutes=15,
            include_features=True
        )
        
        datasets[env_name] = df
        print(f"  Generated {len(df)} data points")
        print(f"  Features: {len(df.columns)} columns")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print()
    
    return datasets


def demo_advanced_features(networks):
    """Demonstrate advanced simulation features."""
    print("=" * 60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # Use office environment for detailed demo
    office_network = networks.get("office")
    if not office_network:
        print("Office network not available for demo")
        return
    
    print("1. Correlation Effects:")
    print("   Reading sensors multiple times to show correlation...")
    
    readings_data = []
    for i in range(100):
        timestamp = datetime.now() + timedelta(minutes=i*15)
        readings = office_network.read_all_sensors(timestamp)
        
        # Extract temperature and humidity values
        temp_reading = next((r for r in readings if r['sensor_type'] == 'temperature'), None)
        humid_reading = next((r for r in readings if r['sensor_type'] == 'humidity'), None)
        
        if temp_reading and humid_reading:
            readings_data.append({
                'timestamp': timestamp,
                'temperature': temp_reading['value'],
                'humidity': humid_reading['value']
            })
    
    if readings_data:
        df_corr = pd.DataFrame(readings_data)
        correlation = df_corr['temperature'].corr(df_corr['humidity'])
        print(f"   Temperature-Humidity correlation: {correlation:.3f}")
        print(f"   Expected: negative correlation (higher temp → lower humidity)")
        print()
    
    print("2. Sensor Quality Metrics:")
    for sensor_id, sensor in office_network.sensors.items():
        stats = sensor.get_statistics()
        print(f"   {sensor_id}:")
        print(f"     Readings taken: {stats.get('reading_count', 0)}")
        print(f"     Mean value: {stats.get('mean_value', 0):.2f}")
        print(f"     Std deviation: {stats.get('std_value', 0):.2f}")
        print(f"     Calibration drift: {stats.get('calibration_drift', 0):.4f}")
    print()
    
    print("3. Network Summary:")
    summary = office_network.get_network_summary()
    print(f"   Network ID: {summary['network_id']}")
    print(f"   Environment: {summary['environment_preset']}")
    print(f"   Total sensors: {summary['sensor_count']}")
    print(f"   Total readings: {summary['total_readings']}")
    print(f"   Buffer size: {summary['data_buffer_size']}")
    print()


def demo_ml_ready_data(networks):
    """Demonstrate ML-ready dataset generation."""
    print("=" * 60)
    print("ML-READY DATASET GENERATION")
    print("=" * 60)
    
    manager = AdvancedDataCollectionManager()
    
    # Add networks to manager
    for env_name, network in networks.items():
        manager.networks[f"{env_name}_demo"] = network
    
    print("1. Generating comprehensive training dataset...")
    
    # Generate a week of data for ML training
    training_data = manager.generate_comparative_dataset(
        duration_days=7,
        interval_minutes=30
    )
    
    print(f"   Dataset shape: {training_data.shape}")
    print(f"   Memory usage: {training_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    print("2. Dataset Features:")
    sensor_features = [col for col in training_data.columns if '_value' in col]
    lag_features = [col for col in training_data.columns if '_lag_' in col]
    rolling_features = [col for col in training_data.columns if '_rolling_' in col]
    time_features = [col for col in training_data.columns if col in ['hour', 'day_of_week', 'month']]
    
    print(f"   Sensor values: {len(sensor_features)} features")
    print(f"   Lag features: {len(lag_features)} features")
    print(f"   Rolling statistics: {len(rolling_features)} features")
    print(f"   Time features: {len(time_features)} features")
    print()
    
    print("3. Data Export Demonstration:")
    export_dir = "data/exports"
    os.makedirs(export_dir, exist_ok=True)
    
    # Export in different formats
    for env_name, network in networks.items():
        try:
            # Export as Parquet (efficient for ML)
            parquet_path = f"{export_dir}/{env_name}_ml_data.parquet"
            network.export_ml_data(
                filepath=parquet_path,
                duration_days=7,
                interval_minutes=30,
                format='parquet'
            )
            print(f"   ✓ Exported {env_name} data to {parquet_path}")
            
        except Exception as e:
            print(f"   ✗ Failed to export {env_name} data: {e}")
    
    print()
    return training_data


def demo_pattern_visualization(datasets):
    """Create visualizations of the generated patterns."""
    print("=" * 60)
    print("PATTERN VISUALIZATION")
    print("=" * 60)
    
    try:
        # Create visualization directory
        viz_dir = "data/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Plot 1: Time series for each environment
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Sensor Patterns Across Different Environments', fontsize=16)
        
        sensor_types = ['temperature', 'humidity', 'pressure']
        colors = ['red', 'blue', 'green']
        
        for i, sensor_type in enumerate(sensor_types):
            ax = axes[i]
            
            for env_name, df in datasets.items():
                value_col = f'{sensor_type}_value'
                if value_col in df.columns:
                    # Sample data for cleaner visualization
                    sample_df = df.iloc[::4]  # Every 4th point
                    ax.plot(sample_df['timestamp'], sample_df[value_col], 
                           label=f'{env_name}', alpha=0.7, linewidth=1)
            
            ax.set_title(f'{sensor_type.title()} Patterns')
            ax.set_ylabel(f'{sensor_type.title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            if i == len(sensor_types) - 1:
                ax.set_xlabel('Time')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/sensor_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved sensor patterns visualization to {viz_dir}/sensor_patterns.png")
        
        # Plot 2: Correlation heatmap for office environment
        if 'office' in datasets:
            office_df = datasets['office']
            sensor_cols = [col for col in office_df.columns if '_value' in col]
            
            if len(sensor_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = office_df[sensor_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5)
                plt.title('Sensor Correlation Matrix (Office Environment)')
                plt.tight_layout()
                plt.savefig(f'{viz_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Saved correlation matrix to {viz_dir}/correlation_matrix.png")
        
        # Plot 3: Daily patterns
        if 'office' in datasets:
            office_df = datasets['office']
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Daily Patterns (Office Environment)', fontsize=16)
            
            for i, sensor_type in enumerate(['temperature', 'humidity', 'pressure']):
                value_col = f'{sensor_type}_value'
                if value_col in office_df.columns:
                    # Group by hour and calculate mean
                    hourly_pattern = office_df.groupby('hour')[value_col].mean()
                    
                    axes[i].plot(hourly_pattern.index, hourly_pattern.values, 
                               marker='o', linewidth=2, markersize=4, color=colors[i])
                    axes[i].set_title(f'{sensor_type.title()} Daily Pattern')
                    axes[i].set_xlabel('Hour of Day')
                    axes[i].set_ylabel(f'{sensor_type.title()}')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].set_xticks(range(0, 24, 4))
            
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/daily_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved daily patterns to {viz_dir}/daily_patterns.png")
        
        print("   Visualizations completed successfully!")
        
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
        print("   Note: Install matplotlib and seaborn for visualizations")
    
    print()


def demo_configuration_export():
    """Demonstrate configuration export/import capabilities."""
    print("=" * 60)
    print("CONFIGURATION MANAGEMENT")
    print("=" * 60)
    
    config_dir = "data/configs"
    os.makedirs(config_dir, exist_ok=True)
    
    print("1. Available Presets:")
    presets = config_manager.list_presets()
    for preset_name in presets:
        preset = config_manager.get_preset(preset_name)
        print(f"   - {preset.name}: {preset.description}")
    print()
    
    print("2. Exporting Configurations:")
    for preset_name in presets:
        try:
            export_path = f"{config_dir}/{preset_name}_config.json"
            config_manager.export_preset(preset_name, export_path)
        except Exception as e:
            print(f"   ✗ Failed to export {preset_name}: {e}")
    print()
    
    print("3. ML Training Configuration:")
    try:
        ml_config = config_manager.create_ml_training_config(
            preset_name="office",
            duration_days=30,
            interval_minutes=15
        )
        
        config_path = f"{config_dir}/ml_training_config.json"
        with open(config_path, 'w') as f:
            import json
            json.dump(ml_config, f, indent=2, default=str)
        
        print(f"   ✓ ML training config saved to {config_path}")
        print(f"   Total samples: {ml_config['total_samples']}")
        print(f"   Features: {len(ml_config['features']['temporal'] + ml_config['features']['lag_features'])}")
        
    except Exception as e:
        print(f"   ✗ Failed to create ML config: {e}")
    
    print()


def main():
    """Main demonstration function."""
    print("ADVANCED SENSOR SIMULATION DEMONSTRATION")
    print("Author: Udisha Dutta Chowdhury")
    print("Supervisor: Prof. Rolando Herrero")
    print()
    
    try:
        # Create output directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/exports", exist_ok=True)
        os.makedirs("data/visualizations", exist_ok=True)
        os.makedirs("data/configs", exist_ok=True)
        
        # Run demonstrations
        manager, networks = demo_environment_presets()
        datasets = demo_realistic_patterns(networks, duration_hours=24)
        demo_advanced_features(networks)
        training_data = demo_ml_ready_data(networks)
        demo_pattern_visualization(datasets)
        demo_configuration_export()
        
        # Final summary
        print("=" * 60)
        print("DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✓ Environment presets demonstrated")
        print("✓ Realistic patterns generated")
        print("✓ Advanced features showcased")
        print("✓ ML-ready datasets created")
        print("✓ Visualizations generated")
        print("✓ Configuration management shown")
        print()
        print("Files generated:")
        print("  - data/exports/: ML training datasets")
        print("  - data/visualizations/: Pattern plots")
        print("  - data/configs/: Configuration files")
        print()
        print("Next steps for ML model development:")
        print("  1. Use generated datasets for time-series forecasting")
        print("  2. Implement LSTM/ARIMA models for prediction")
        print("  3. Develop anomaly detection algorithms")
        print("  4. Create real-time inference pipeline")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
