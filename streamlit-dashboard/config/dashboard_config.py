"""
Dashboard Configuration Module
Manages configuration settings for the Streamlit dashboard.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class DashboardConfig:
    """Configuration manager for the dashboard"""
    
    def __init__(self, config_file=None):
        self.config_dir = Path(__file__).parent
        self.config_file = config_file or (self.config_dir / "dashboard_config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'dashboard': {
                'title': 'CPS Cloud Dashboard',
                'description': 'Cyber-Physical Systems Monitoring and Control',
                'version': '1.0.0',
                'auto_refresh': True,
                'refresh_interval': 60,  # seconds
                'theme': 'light'
            },
            'data': {
                'default_time_range': '24h',
                'max_data_points': 10000,
                'cache_timeout': 300,  # seconds
                'data_sources': {
                    'sensors': True,
                    'ml_models': True,
                    'system_metrics': True
                }
            },
            'ml_models': {
                'enabled_models': ['basic_forecaster', 'xgboost', 'arima'],
                'default_model': 'xgboost',
                'prediction_horizon': 7,  # days
                'confidence_interval': 0.95,
                'auto_retrain': False
            },
            'visualization': {
                'chart_height': 400,
                'color_scheme': 'blue',
                'show_confidence_intervals': True,
                'animation_enabled': True
            },
            'alerts': {
                'enabled': True,
                'email_notifications': False,
                'thresholds': {
                    'cpu_usage': 80,
                    'memory_usage': 85,
                    'disk_usage': 90,
                    'model_accuracy': 0.8
                }
            },
            'security': {
                'require_authentication': False,
                'session_timeout': 3600,  # seconds
                'allowed_ips': [],
                'rate_limiting': True
            },
            'export': {
                'formats': ['csv', 'json', 'excel'],
                'max_export_size': 1000000,  # rows
                'include_metadata': True
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration to file"""
        self.config_dir.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        env_mappings = {
            'DASHBOARD_TITLE': 'dashboard.title',
            'DASHBOARD_AUTO_REFRESH': 'dashboard.auto_refresh',
            'DASHBOARD_REFRESH_INTERVAL': 'dashboard.refresh_interval',
            'ML_DEFAULT_MODEL': 'ml_models.default_model',
            'ALERT_CPU_THRESHOLD': 'alerts.thresholds.cpu_usage',
            'ALERT_MEMORY_THRESHOLD': 'alerts.thresholds.memory_usage'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert string values to appropriate types
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)
                
                self.set(config_key, value)
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return {
            'enabled': model_name in self.get('ml_models.enabled_models', []),
            'prediction_horizon': self.get('ml_models.prediction_horizon', 7),
            'confidence_interval': self.get('ml_models.confidence_interval', 0.95),
            'auto_retrain': self.get('ml_models.auto_retrain', False)
        }
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return self.get('visualization', {})
    
    def get_alert_thresholds(self) -> Dict[str, Any]:
        """Get alert threshold configuration"""
        return self.get('alerts.thresholds', {})
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled"""
        enabled_models = self.get('ml_models.enabled_models', [])
        return model_name in enabled_models
    
    def get_color_scheme(self):
        """Get color scheme for visualizations"""
        schemes = {
            'blue': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'green': ['#2ca02c', '#98df8a', '#1f77b4', '#aec7e8', '#ff7f0e'],
            'purple': ['#9467bd', '#c5b0d5', '#1f77b4', '#aec7e8', '#ff7f0e'],
            'orange': ['#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4']
        }
        
        color_scheme = self.get('visualization.color_scheme', 'blue')
        return schemes.get(color_scheme, schemes['blue'])

# Global configuration instance
config = DashboardConfig()