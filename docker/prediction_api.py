"""
Simple Prediction API for Cyber-Physical Systems
==============================================

Flask API service for serving temperature forecasting predictions
with MLflow model loading and health checks.
"""

import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
from pathlib import Path
import logging
import traceback
from datetime import datetime

# Add project paths
sys.path.append('/app')
sys.path.append('/app/ml-models')
sys.path.append('/app/data-collection')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global model storage
loaded_model = None
model_metadata = {}

def load_latest_model():
    """Load the latest trained model from MLflow"""
    global loaded_model, model_metadata
    
    try:
        # Set MLflow tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:///app/mlruns')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Get the latest model from the hyperparameter tuning experiment
        experiment = mlflow.get_experiment_by_name("Hyperparameter_Tuning_Demo")
        
        if experiment is None:
            logger.warning("No Hyperparameter_Tuning_Demo experiment found")
            return False
            
        # Get all runs from the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.cv_mean_rmse ASC"],
            max_results=1
        )
        
        if len(runs) == 0:
            logger.warning("No runs found in experiment")
            return False
            
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        
        # Load the model
        model_uri = f"runs:/{run_id}/best_model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Store metadata
        model_metadata = {
            'run_id': run_id,
            'rmse': best_run.get('metrics.cv_mean_rmse', 'N/A'),
            'n_lags': best_run.get('params.n_lags', 'N/A'),
            'loaded_at': datetime.now().isoformat()
        }
        
        logger.info(f"Model loaded successfully: {model_metadata}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return False

def fallback_model_prediction(data):
    """Fallback prediction using simple moving average"""
    if len(data) < 3:
        return 22.0  # Default temperature
    
    # Simple moving average of last 3 points
    return np.mean(data[-3:])

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': loaded_model is not None,
        'model_metadata': model_metadata
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data or 'temperature_history' not in data:
            return jsonify({
                'error': 'Missing temperature_history in request'
            }), 400
        
        temperature_history = data['temperature_history']
        
        if not isinstance(temperature_history, list) or len(temperature_history) == 0:
            return jsonify({
                'error': 'temperature_history must be a non-empty list'
            }), 400
        
        # Convert to numpy array
        temp_array = np.array(temperature_history)
        
        # Make prediction
        if loaded_model is not None:
            try:
                # Create a simple DataFrame for prediction
                df = pd.DataFrame({
                    'temperature': temp_array
                })
                
                # Use the model's predict method
                prediction = loaded_model.predict(df)
                
                # Handle prediction format
                if np.isscalar(prediction):
                    predicted_temp = float(prediction)
                else:
                    predicted_temp = float(prediction[0])
                    
            except Exception as e:
                logger.warning(f"Model prediction failed, using fallback: {e}")
                predicted_temp = fallback_model_prediction(temp_array)
        else:
            logger.info("No model loaded, using fallback prediction")
            predicted_temp = fallback_model_prediction(temp_array)
        
        # Prepare response
        response = {
            'predicted_temperature': predicted_temp,
            'input_length': len(temperature_history),
            'model_used': 'mlflow_model' if loaded_model is not None else 'fallback',
            'timestamp': datetime.now().isoformat()
        }
        
        if model_metadata:
            response['model_metadata'] = model_metadata
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload the model from MLflow"""
    try:
        success = load_latest_model()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model reloaded successfully',
                'model_metadata': model_metadata
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to reload model'
            }), 500
            
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Model reload failed: {str(e)}'
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Cyber-Physical Systems Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Temperature prediction (POST)',
            '/model/reload': 'Reload model (POST)',
        },
        'model_loaded': loaded_model is not None
    })

if __name__ == '__main__':
    # Try to load model on startup
    logger.info("Starting Prediction API...")
    load_latest_model()
    
    # Start Flask app
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
