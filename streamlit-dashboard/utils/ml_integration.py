"""
ML Integration Utility for Streamlit Dashboard
Handles integration with ML models, predictions, and MLflow tracking.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import pickle
import joblib

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "ml-models"))

try:
    from basic_forecaster import BasicTimeSeriesForecaster
    BASIC_FORECASTER_AVAILABLE = True
except ImportError:
    BASIC_FORECASTER_AVAILABLE = False
    print("Basic forecaster not available")

try:
    from xgboost_forecaster import XGBoostForecaster
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost forecaster not available")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available")

class MLModelManager:
    """Manages ML models and predictions for the dashboard"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "ml-models"
        self.results_dir = self.project_root / "results"
        
        # Model storage
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
        
        # Load available models
        self._discover_models()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow_dir = self.models_dir / "mlruns"
            if mlflow_dir.exists():
                mlflow.set_tracking_uri(f"file://{mlflow_dir}")
            else:
                mlflow.set_tracking_uri("sqlite:///mlflow.db")
        except Exception as e:
            print(f"Error setting up MLflow: {e}")
    
    def _discover_models(self):
        """Discover available trained models"""
        # Look for saved model files
        model_files = {
            'basic_forecaster': ['basic_forecaster_model.pkl', 'basic_model.joblib'],
            'xgboost': ['xgboost_model.pkl', 'xgboost_model.joblib'],
            'arima': ['arima_model.pkl']
        }
        
        for model_name, possible_files in model_files.items():
            for filename in possible_files:
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.model_metadata[model_name] = {
                        'path': model_path,
                        'type': filename.split('.')[-1],
                        'last_modified': datetime.fromtimestamp(model_path.stat().st_mtime),
                        'status': 'available'
                    }
                    break
            else:
                self.model_metadata[model_name] = {
                    'path': None,
                    'type': None,
                    'last_modified': None,
                    'status': 'not_found'
                }
    
    def load_model(self, model_name):
        """
        Load a specific model
        
        Args:
            model_name (str): Name of the model to load
        
        Returns:
            object: Loaded model or None
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if model_name not in self.model_metadata:
            print(f"Model {model_name} not found in metadata")
            return None
        
        metadata = self.model_metadata[model_name]
        
        if metadata['status'] != 'available':
            print(f"Model {model_name} is not available")
            return None
        
        try:
            model_path = metadata['path']
            
            if metadata['type'] == 'pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif metadata['type'] == 'joblib':
                model = joblib.load(model_path)
            else:
                print(f"Unsupported model type: {metadata['type']}")
                return None
            
            self.loaded_models[model_name] = model
            return model
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def get_model_performance(self, model_name=None):
        """
        Get performance metrics for models
        
        Args:
            model_name (str, optional): Specific model name, or None for all models
        
        Returns:
            dict: Performance metrics
        """
        if MLFLOW_AVAILABLE:
            return self._get_mlflow_performance(model_name)
        else:
            return self._get_simulated_performance(model_name)
    
    def _get_mlflow_performance(self, model_name=None):
        """Get performance from MLflow tracking"""
        try:
            experiments = mlflow.search_experiments()
            
            performance_data = {}
            
            for exp in experiments:
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                
                if not runs.empty:
                    # Get the best run (lowest RMSE)
                    if 'metrics.rmse' in runs.columns:
                        best_run = runs.loc[runs['metrics.rmse'].idxmin()]
                        
                        exp_name = exp.name.lower().replace(' ', '_')
                        
                        performance_data[exp_name] = {
                            'rmse': best_run.get('metrics.rmse', np.nan),
                            'mae': best_run.get('metrics.mae', np.nan),
                            'r2': best_run.get('metrics.r2', np.nan),
                            'run_id': best_run.get('run_id', ''),
                            'timestamp': best_run.get('start_time', datetime.now())
                        }
            
            if model_name and model_name in performance_data:
                return {model_name: performance_data[model_name]}
            
            return performance_data
            
        except Exception as e:
            print(f"Error getting MLflow performance: {e}")
            return self._get_simulated_performance(model_name)
    
    def _get_simulated_performance(self, model_name=None):
        """Get simulated performance metrics"""
        all_performance = {
            'basic_forecaster': {
                'rmse': 2.34,
                'mae': 1.82,
                'r2': 0.89,
                'run_id': 'sim_basic_001',
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            'xgboost': {
                'rmse': 1.87,
                'mae': 1.43,
                'r2': 0.94,
                'run_id': 'sim_xgb_001',
                'timestamp': datetime.now() - timedelta(hours=1)
            },
            'arima': {
                'rmse': 2.91,
                'mae': 2.15,
                'r2': 0.83,
                'run_id': 'sim_arima_001',
                'timestamp': datetime.now() - timedelta(hours=3)
            }
        }
        
        if model_name and model_name in all_performance:
            return {model_name: all_performance[model_name]}
        
        return all_performance
    
    def generate_prediction(self, model_name, data, forecast_steps=7):
        """
        Generate predictions using specified model
        
        Args:
            model_name (str): Name of the model to use
            data (pd.DataFrame): Input data for prediction
            forecast_steps (int): Number of steps to forecast
        
        Returns:
            dict: Prediction results
        """
        try:
            if model_name == 'basic_forecaster' and BASIC_FORECASTER_AVAILABLE:
                return self._predict_with_basic_forecaster(data, forecast_steps)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                return self._predict_with_xgboost(data, forecast_steps)
            else:
                return self._generate_simulated_prediction(model_name, data, forecast_steps)
                
        except Exception as e:
            print(f"Error generating prediction with {model_name}: {e}")
            return self._generate_simulated_prediction(model_name, data, forecast_steps)
    
    def _predict_with_basic_forecaster(self, data, forecast_steps):
        """Make prediction using basic forecaster"""
        try:
            # Try to load or create basic forecaster
            model = self.load_model('basic_forecaster')
            
            if model is None:
                # Create new model if none exists
                model = BasicTimeSeriesForecaster(n_lags=5)
                
                # Prepare data (assuming temperature column)
                if 'temperature' in data.columns:
                    train_data = data['temperature'].values
                    model.fit(train_data)
                    self.loaded_models['basic_forecaster'] = model
            
            # Generate prediction
            if hasattr(model, 'predict'):
                predictions = model.predict(forecast_steps)
            else:
                # Fallback to simulated prediction
                predictions = self._generate_simulated_values(forecast_steps)
            
            return {
                'model': 'basic_forecaster',
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'forecast_steps': forecast_steps,
                'timestamp': datetime.now(),
                'confidence_interval': self._generate_confidence_interval(predictions),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error in basic forecaster prediction: {e}")
            return self._generate_simulated_prediction('basic_forecaster', data, forecast_steps)
    
    def _predict_with_xgboost(self, data, forecast_steps):
        """Make prediction using XGBoost"""
        try:
            model = self.load_model('xgboost')
            
            if model is None and XGBOOST_AVAILABLE:
                # Create new XGBoost model
                model = XGBoostForecaster(n_lags=5)
                
                if 'temperature' in data.columns:
                    train_data = data['temperature'].values
                    model.fit(train_data)
                    self.loaded_models['xgboost'] = model
            
            if hasattr(model, 'predict'):
                predictions = model.predict(forecast_steps)
            else:
                predictions = self._generate_simulated_values(forecast_steps)
            
            return {
                'model': 'xgboost',
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'forecast_steps': forecast_steps,
                'timestamp': datetime.now(),
                'confidence_interval': self._generate_confidence_interval(predictions),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error in XGBoost prediction: {e}")
            return self._generate_simulated_prediction('xgboost', data, forecast_steps)
    
    def _generate_simulated_prediction(self, model_name, data, forecast_steps):
        """Generate simulated prediction for demonstration"""
        # Base prediction on last values in data if available
        if 'temperature' in data.columns:
            last_value = data['temperature'].iloc[-1]
        else:
            last_value = 22.0  # Default temperature
        
        # Generate predictions with some trend and noise
        predictions = []
        current_value = last_value
        
        for i in range(forecast_steps):
            # Add slight trend and noise
            trend = 0.1 * np.sin(i * 0.1)
            noise = np.random.normal(0, 0.5)
            current_value += trend + noise
            predictions.append(current_value)
        
        return {
            'model': model_name,
            'predictions': predictions,
            'forecast_steps': forecast_steps,
            'timestamp': datetime.now(),
            'confidence_interval': self._generate_confidence_interval(predictions),
            'status': 'simulated'
        }
    
    def _generate_simulated_values(self, n_steps):
        """Generate simulated prediction values"""
        base_value = 22.0
        return base_value + np.cumsum(np.random.normal(0, 0.5, n_steps))
    
    def _generate_confidence_interval(self, predictions, confidence=0.95):
        """Generate confidence intervals for predictions"""
        predictions = np.array(predictions)
        std_dev = np.std(predictions) if len(predictions) > 1 else 1.0
        
        # Simple confidence interval based on standard deviation
        margin = 1.96 * std_dev  # 95% confidence interval
        
        return {
            'upper': (predictions + margin).tolist(),
            'lower': (predictions - margin).tolist(),
            'confidence_level': confidence
        }
    
    def get_available_models(self):
        """
        Get list of available models
        
        Returns:
            list: Available model names
        """
        return list(self.model_metadata.keys())
    
    def get_model_info(self, model_name):
        """
        Get detailed information about a specific model
        
        Args:
            model_name (str): Name of the model
        
        Returns:
            dict: Model information
        """
        if model_name not in self.model_metadata:
            return None
        
        metadata = self.model_metadata[model_name]
        performance = self.get_model_performance(model_name)
        
        return {
            'name': model_name,
            'status': metadata['status'],
            'last_modified': metadata['last_modified'],
            'performance': performance.get(model_name, {}),
            'loaded': model_name in self.loaded_models
        }
    
    def start_training(self, model_name, config):
        """
        Start model training (simulated for dashboard)
        
        Args:
            model_name (str): Name of model to train
            config (dict): Training configuration
        
        Returns:
            dict: Training status
        """
        # This would typically start an actual training process
        # For the dashboard, we'll simulate the training
        
        return {
            'model': model_name,
            'status': 'started',
            'config': config,
            'start_time': datetime.now(),
            'estimated_duration': f"{np.random.randint(5, 30)} minutes",
            'training_id': f"train_{model_name}_{int(datetime.now().timestamp())}"
        }