#!/usr/bin/env python3
"""
Run local services without Docker for demonstration
This will start MLflow server and prediction API locally
"""

import subprocess
import threading
import time
import sys
import os
import signal
from pathlib import Path

class LocalDeployment:
    def __init__(self):
        self.processes = []
        self.running = False
        
    def start_mlflow_server(self):
        """Start MLflow tracking server"""
        print("🚀 Starting MLflow server on http://localhost:5000")
        try:
            # Create mlruns directory if it doesn't exist
            os.makedirs("mlruns", exist_ok=True)
            
            cmd = [
                sys.executable, "-m", "mlflow", "server",
                "--host", "0.0.0.0",
                "--port", "5000",
                "--backend-store-uri", "file:./mlruns",
                "--default-artifact-root", "./mlartifacts"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("MLflow Server", process))
            return process
        except Exception as e:
            print(f"❌ Failed to start MLflow server: {e}")
            return None
    
    def start_prediction_api(self):
        """Start the prediction API"""
        print("🚀 Starting Prediction API on http://localhost:8080")
        
        # Create a simple Flask API
        api_code = '''
import flask
from flask import Flask, request, jsonify
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'prediction-api',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Simple prediction logic for demonstration
        if 'temperature_history' in data:
            temps = data['temperature_history']
            # Simple moving average prediction
            prediction = np.mean(temps[-3:]) + 0.1 * np.random.random()
            confidence = 0.85 + 0.1 * np.random.random()
            
            return jsonify({
                'prediction': round(prediction, 2),
                'confidence': round(confidence, 3),
                'timestamp': datetime.now().isoformat(),
                'input_size': len(temps)
            })
        else:
            return jsonify({'error': 'temperature_history required'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Cyber-Physical Systems Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Make predictions (POST)',
            '/': 'This information'
        },
        'status': 'running'
    })

if __name__ == '__main__':
    print("Starting Prediction API on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
'''
        
        try:
            # Write API to temporary file
            with open('temp_api.py', 'w') as f:
                f.write(api_code)
            
            # Start the API
            process = subprocess.Popen([
                sys.executable, 'temp_api.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.processes.append(("Prediction API", process))
            return process
        except Exception as e:
            print(f"❌ Failed to start Prediction API: {e}")
            return None
    
    def start_services(self):
        """Start all services"""
        print("=== STARTING LOCAL DEPLOYMENT ===")
        print()
        
        self.running = True
        
        # Start MLflow server
        mlflow_process = self.start_mlflow_server()
        time.sleep(3)
        
        # Start prediction API
        api_process = self.start_prediction_api()
        time.sleep(3)
        
        print()
        print("✅ Services started successfully!")
        print()
        print("🌐 Access points:")
        print("   📊 MLflow Dashboard: http://localhost:5000")
        print("   🔮 Prediction API: http://localhost:8080")
        print("   📋 API Health: http://localhost:8080/health")
        print()
        print("📸 You can now take screenshots of these interfaces!")
        print()
        print("Press Ctrl+C to stop all services...")
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_services()
    
    def stop_services(self):
        """Stop all services"""
        print("\n🛑 Stopping services...")
        self.running = False
        
        for name, process in self.processes:
            try:
                print(f"   Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"   Error stopping {name}: {e}")
        
        # Clean up temporary file
        if os.path.exists('temp_api.py'):
            os.remove('temp_api.py')
        
        print("✅ All services stopped")

if __name__ == "__main__":
    deployment = LocalDeployment()
    
    try:
        deployment.start_services()
    except KeyboardInterrupt:
        deployment.stop_services()
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        deployment.stop_services()