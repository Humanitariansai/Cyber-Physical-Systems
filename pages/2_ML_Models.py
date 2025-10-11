"""
ML Models Page
============

This page provides access to machine learning model training,
evaluation, and prediction functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add module paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml_models.basic_forecaster import BasicTimeSeriesForecaster
try:
    from ml_models.mlflow_tracking import MLflowConfig, ExperimentTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="ML Models",
    page_icon="🤖",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    mlflow_config = MLflowConfig() if MLFLOW_AVAILABLE else None
    experiment_tracker = ExperimentTracker(mlflow_config) if MLFLOW_AVAILABLE else None
    return experiment_tracker

experiment_tracker = init_components()

# Main content
st.title("Machine Learning Models")
st.write("""
Train, evaluate, and deploy machine learning models for time series forecasting.
""")

# Model configuration
st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Basic Forecaster"]
)

if model_type == "Basic Forecaster":
    n_lags = st.sidebar.slider("Number of Lags", 1, 10, 5)
    method = st.sidebar.selectbox(
        "Forecasting Method",
        ["Simple Moving Average", "Exponential Moving Average"]
    )

# Data selection
st.header("Data Selection")
data_path = project_root / "data-collection/data/processed"
available_data = [f.stem for f in data_path.glob("*.csv") if f.is_file()]

if available_data:
    selected_data = st.selectbox(
        "Select Data Source",
        available_data
    )
    
    try:
        # Load data
        df = pd.read_csv(data_path / f"{selected_data}.csv")
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Training section
        st.header("Model Training")
        
        with st.form("training_form"):
            test_size = st.slider("Test Set Size (%)", 10, 50, 20)
            
            if st.form_submit_button("Train Model"):
                # Prepare data
                if 'value' in df.columns:
                    data = df['value'].values
                    train_size = int(len(data) * (1 - test_size/100))
                    train_data = data[:train_size]
                    test_data = data[train_size:]
                    
                    # Initialize and train model
                    model = BasicTimeSeriesForecaster(
                        n_lags=n_lags,
                        enable_mlflow=MLFLOW_AVAILABLE
                    )
                    
                    # Train and predict
                    model.fit(train_data)
                    predictions = model.predict(test_data)
                    
                    # Plot results
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=train_data,
                        name="Training Data",
                        line=dict(color="blue")
                    ))
                    fig.add_trace(go.Scatter(
                        y=np.concatenate([
                            np.full(train_size, np.nan),
                            test_data
                        ]),
                        name="Test Data",
                        line=dict(color="green")
                    ))
                    fig.add_trace(go.Scatter(
                        y=np.concatenate([
                            np.full(train_size, np.nan),
                            predictions
                        ]),
                        name="Predictions",
                        line=dict(color="red", dash="dash")
                    ))
                    
                    st.plotly_chart(fig)
                    
                    # Show metrics
                    metrics = model.get_metrics()
                    st.write("### Model Metrics")
                    st.write(metrics)
                    
                    # Save model if MLflow is available
                    if MLFLOW_AVAILABLE and experiment_tracker:
                        st.write("### MLflow Tracking")
                        st.write("Model and metrics saved to MLflow")
                        
                else:
                    st.error("Data must contain a 'value' column")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        
else:
    st.info("""
    No processed data available. Please process some data first!
    
    Go to the Data Analytics page to process raw data.
    """)