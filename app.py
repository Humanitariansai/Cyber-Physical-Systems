"""
Cyber-Physical Systems Main Application
====================================

Main Streamlit application integrating data collection,
forecasting models, and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add module paths
project_root = Path(__file__).parent
paths_to_add = [
    str(project_root),
    str(project_root / "ml-models"),
    str(project_root / "data-collection")
]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)

# Import project modules
from data_collection.data_collector import DataCollector
from ml_models.basic_forecaster import BasicTimeSeriesForecaster
try:
    from ml_models.mlflow_tracking import MLflowConfig, ExperimentTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    st.warning("MLflow not available. Some features will be disabled.")

# Configure page
st.set_page_config(
    page_title="Cyber-Physical Systems Dashboard",
    page_icon="🌟",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize main application components."""
    data_collector = DataCollector()
    mlflow_config = MLflowConfig() if MLFLOW_AVAILABLE else None
    experiment_tracker = ExperimentTracker(mlflow_config) if MLFLOW_AVAILABLE else None
    forecaster = BasicTimeSeriesForecaster(
        enable_mlflow=MLFLOW_AVAILABLE
    )
    return data_collector, forecaster, experiment_tracker

data_collector, forecaster, experiment_tracker = init_components()

# Main interface
st.title("Cyber-Physical Systems Dashboard")
st.write("""
This dashboard provides real-time monitoring, data collection,
and forecasting capabilities for cyber-physical systems.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Data collection settings
    st.subheader("Data Collection")
    source_id = st.text_input("Source ID", "sensor_1")
    
    # Model settings
    st.subheader("Forecasting")
    n_lags = st.slider("Number of Lags", 1, 10, 5)
    
    # MLflow tracking
    if MLFLOW_AVAILABLE:
        st.subheader("Experiment Tracking")
        track_experiment = st.checkbox("Track with MLflow", True)

# Main content tabs
tab1, tab2, tab3 = st.tabs([
    "Data Collection",
    "Forecasting",
    "Experiment Tracking"
])

# Tab 1: Data Collection
with tab1:
    st.header("Data Collection")
    
    # Manual data input (for testing)
    with st.form("data_input"):
        st.write("Add new data point:")
        value = st.number_input("Value", value=0.0)
        submit = st.form_submit_button("Submit")
        
        if submit:
            success = data_collector.collect_data(
                source_id=source_id,
                data={"value": value}
            )
            if success:
                st.success("Data point added successfully!")
            else:
                st.error("Failed to add data point.")
    
    # Display collected data
    if source_id in data_collector.current_data:
        st.write("### Collected Data")
        st.dataframe(data_collector.current_data[source_id])

# Tab 2: Forecasting
with tab2:
    st.header("Forecasting")
    
    if source_id in data_collector.current_data:
        df = data_collector.current_data[source_id]
        
        if len(df) > n_lags:
            # Train forecaster
            X = df['value'].values
            forecaster.n_lags = n_lags
            predictions = forecaster.fit_predict(X)
            
            # Plot results
            st.write("### Forecasting Results")
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=X,
                name="Actual",
                line=dict(color="blue")
            ))
            fig.add_trace(go.Scatter(
                y=predictions,
                name="Predicted",
                line=dict(color="red", dash="dash")
            ))
            fig.update_layout(
                title="Time Series Forecast",
                xaxis_title="Time Step",
                yaxis_title="Value"
            )
            st.plotly_chart(fig)
            
            # Show metrics
            metrics = forecaster.get_metrics()
            st.write("### Model Metrics")
            st.write(metrics)
        else:
            st.warning(f"Need at least {n_lags + 1} data points for forecasting.")
    else:
        st.info("No data available for forecasting. Add some data points first!")

# Tab 3: Experiment Tracking
with tab3:
    st.header("Experiment Tracking")
    
    if MLFLOW_AVAILABLE and experiment_tracker:
        st.write("MLflow tracking is enabled.")
        # Add MLflow UI link if available
        if os.path.exists(mlflow_config.tracking_uri):
            st.write(f"MLflow tracking URI: {mlflow_config.tracking_uri}")
    else:
        st.warning("""
        MLflow is not available. Install it with:
        ```
        pip install mlflow
        ```
        """)