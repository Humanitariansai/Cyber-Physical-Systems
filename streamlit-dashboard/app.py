"""
Cyber-Physical Systems Cloud Dashboard
Author: Udisha Dutta Chowdhury
Supervisor: Prof. Rolando Herrero

A comprehensive Streamlit dashboard for monitoring, visualizing, and managing
cyber-physical systems data with ML model integration.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "ml-models"))
sys.path.append(str(project_root / "data-collection"))
sys.path.append(str(project_root / "streamlit-dashboard"))

# Import dashboard components
from utils.data_loader import DataLoader
from utils.ml_integration import MLModelManager
from components.sidebar import render_sidebar
from components.metrics_cards import render_metrics_cards

# Page configuration
st.set_page_config(
    page_title="CPS Cloud Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .dashboard-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    if 'ml_manager' not in st.session_state:
        st.session_state.ml_manager = MLModelManager()
    if 'refresh_data' not in st.session_state:
        st.session_state.refresh_data = False

def main():
    """Main dashboard application"""
    initialize_session_state()
    
    # Dashboard header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🌐 Cyber-Physical Systems Cloud Dashboard</h1>
        <p>Real-time monitoring, ML predictions, and system analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    # Metrics cards
    with col1:
        st.metric(
            label="Active Sensors",
            value="12",
            delta="2"
        )
    
    with col2:
        st.metric(
            label="ML Models",
            value="3",
            delta="1"
        )
    
    with col3:
        st.metric(
            label="Predictions Today",
            value="847",
            delta="125"
        )
    
    with col4:
        st.metric(
            label="System Health",
            value="98.5%",
            delta="0.3%"
        )
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔮 ML Predictions", 
        "📈 Historical Data", 
        "⚙️ System Health",
        "🚀 Model Training"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_ml_predictions_tab()
    
    with tab3:
        render_historical_data_tab()
    
    with tab4:
        render_system_health_tab()
    
    with tab5:
        render_model_training_tab()

def render_overview_tab():
    """Render the overview tab"""
    st.subheader("📊 System Overview")
    
    # Real-time data simulation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Real-time Sensor Data")
        
        # Generate sample real-time data
        current_time = datetime.now()
        time_range = [current_time - timedelta(minutes=x) for x in range(60, 0, -1)]
        
        # Sample sensor data
        temperature_data = 20 + 5 * np.random.sin(np.linspace(0, 4*np.pi, 60)) + np.random.normal(0, 0.5, 60)
        humidity_data = 50 + 10 * np.random.cos(np.linspace(0, 3*np.pi, 60)) + np.random.normal(0, 1, 60)
        
        # Create plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Temperature (°C)", "Humidity (%)"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=time_range, y=temperature_data, mode='lines', name='Temperature',
                      line=dict(color='#ff6b6b', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_range, y=humidity_data, mode='lines', name='Humidity',
                      line=dict(color='#4ecdc4', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("System Status")
        
        # Status indicators
        status_data = {
            "Component": ["Data Collection", "ML Pipeline", "Database", "API Gateway", "Dashboard"],
            "Status": ["🟢 Online", "🟢 Online", "🟡 Warning", "🟢 Online", "🟢 Online"],
            "Uptime": ["99.9%", "98.7%", "97.2%", "99.8%", "100%"],
            "Last Updated": ["2 min ago", "1 min ago", "5 min ago", "30 sec ago", "Live"]
        }
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        
        # Quick actions
        st.subheader("Quick Actions")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔄 Refresh Data", type="primary"):
                st.session_state.refresh_data = True
                st.rerun()
        
        with col_b:
            if st.button("📊 Generate Report"):
                st.success("Report generation started!")

def render_ml_predictions_tab():
    """Render the ML predictions tab"""
    st.subheader("🔮 ML Model Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Forecast Visualization")
        
        # Generate sample forecast data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        historical_data = 20 + 5 * np.sin(np.linspace(0, 8*np.pi, 70)) + np.random.normal(0, 1, 70)
        forecast_data = 20 + 5 * np.sin(np.linspace(8*np.pi*70/100, 8*np.pi, 30)) + np.random.normal(0, 1.5, 30)
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=dates[:70],
            y=historical_data,
            mode='lines',
            name='Historical Data',
            line=dict(color='#3498db', width=2)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=dates[70:],
            y=forecast_data,
            mode='lines',
            name='Forecast',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        # Confidence interval
        upper_bound = forecast_data + 2
        lower_bound = forecast_data - 2
        
        fig.add_trace(go.Scatter(
            x=list(dates[70:]) + list(dates[70:])[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title="Temperature Forecast - Next 30 Days",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        
        # Model metrics
        metrics_data = {
            "Model": ["Basic Forecaster", "XGBoost", "ARIMA"],
            "RMSE": [2.34, 1.87, 2.91],
            "MAE": [1.82, 1.43, 2.15],
            "R²": [0.89, 0.94, 0.83]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Prediction",
            ["Basic Forecaster", "XGBoost", "ARIMA"]
        )
        
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
        
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Generating prediction..."):
                # Simulate prediction generation
                import time
                time.sleep(2)
                st.success(f"✅ {forecast_days}-day forecast generated using {selected_model}")

def render_historical_data_tab():
    """Render the historical data tab"""
    st.subheader("📈 Historical Data Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Data type selector
    data_types = st.multiselect(
        "Select Data Types",
        ["Temperature", "Humidity", "Pressure", "Vibration", "Power Consumption"],
        default=["Temperature", "Humidity"]
    )
    
    if data_types:
        # Generate sample historical data
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        fig = go.Figure()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, data_type in enumerate(data_types):
            # Generate sample data based on type
            if data_type == "Temperature":
                base_value = 22
                variation = 5
            elif data_type == "Humidity":
                base_value = 60
                variation = 15
            elif data_type == "Pressure":
                base_value = 1013
                variation = 10
            elif data_type == "Vibration":
                base_value = 0.5
                variation = 0.3
            else:  # Power Consumption
                base_value = 150
                variation = 30
            
            data = base_value + variation * np.sin(np.linspace(0, 4*np.pi, len(date_range))) + np.random.normal(0, variation*0.1, len(date_range))
            
            fig.add_trace(go.Scatter(
                x=date_range,
                y=data,
                mode='lines',
                name=data_type,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Historical Data Trends",
            xaxis_title="Time",
            yaxis_title="Values",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        
        # Generate summary statistics
        summary_data = []
        for data_type in data_types:
            # Generate sample statistics
            summary_data.append({
                "Metric": data_type,
                "Mean": f"{np.random.uniform(20, 100):.2f}",
                "Std Dev": f"{np.random.uniform(1, 10):.2f}",
                "Min": f"{np.random.uniform(10, 50):.2f}",
                "Max": f"{np.random.uniform(80, 150):.2f}",
                "Trend": np.random.choice(["📈 Increasing", "📉 Decreasing", "➡️ Stable"])
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

def render_system_health_tab():
    """Render the system health tab"""
    st.subheader("⚙️ System Health Monitor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resource Usage")
        
        # CPU usage gauge
        cpu_usage = np.random.uniform(20, 80)
        fig_cpu = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cpu_usage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU Usage (%)"},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_cpu.update_layout(height=300)
        st.plotly_chart(fig_cpu, use_container_width=True)
        
        # Memory usage
        memory_usage = np.random.uniform(30, 70)
        fig_memory = go.Figure(go.Indicator(
            mode="gauge+number",
            value=memory_usage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memory Usage (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        fig_memory.update_layout(height=300)
        st.plotly_chart(fig_memory, use_container_width=True)
    
    with col2:
        st.subheader("Service Status")
        
        services = [
            {"Service": "API Gateway", "Status": "🟢", "Uptime": "99.9%", "Response Time": "45ms"},
            {"Service": "Database", "Status": "🟡", "Uptime": "97.2%", "Response Time": "120ms"},
            {"Service": "ML Service", "Status": "🟢", "Uptime": "98.7%", "Response Time": "300ms"},
            {"Service": "Data Collector", "Status": "🟢", "Uptime": "99.8%", "Response Time": "25ms"},
            {"Service": "Dashboard", "Status": "🟢", "Uptime": "100%", "Response Time": "80ms"}
        ]
        
        services_df = pd.DataFrame(services)
        st.dataframe(services_df, use_container_width=True, hide_index=True)
        
        st.subheader("Recent Alerts")
        
        alerts = [
            {"Time": "10:30 AM", "Level": "⚠️ Warning", "Message": "High memory usage on DB server"},
            {"Time": "09:15 AM", "Level": "ℹ️ Info", "Message": "ML model retrain completed"},
            {"Time": "08:45 AM", "Level": "✅ Success", "Message": "Daily backup completed"},
            {"Time": "07:30 AM", "Level": "🔴 Error", "Message": "Sensor #7 connection lost (resolved)"}
        ]
        
        alerts_df = pd.DataFrame(alerts)
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)

def render_model_training_tab():
    """Render the model training tab"""
    st.subheader("🚀 Model Training & Experiments")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Progress")
        
        # Simulate training metrics
        epochs = list(range(1, 101))
        train_loss = [1.0 * np.exp(-x/20) + 0.1 + np.random.normal(0, 0.02) for x in epochs]
        val_loss = [1.2 * np.exp(-x/25) + 0.15 + np.random.normal(0, 0.03) for x in epochs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Training Loss', line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss', line=dict(color='#e74c3c')))
        
        fig.update_layout(
            title="Model Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Experiment history
        st.subheader("Recent Experiments")
        
        experiments = [
            {"ID": "exp_001", "Model": "XGBoost", "RMSE": 1.87, "R²": 0.94, "Status": "✅ Completed", "Duration": "12m 34s"},
            {"ID": "exp_002", "Model": "Basic Forecaster", "RMSE": 2.34, "R²": 0.89, "Status": "✅ Completed", "Duration": "8m 15s"},
            {"ID": "exp_003", "Model": "ARIMA", "RMSE": 2.91, "R²": 0.83, "Status": "✅ Completed", "Duration": "15m 22s"},
            {"ID": "exp_004", "Model": "Neural Network", "RMSE": 1.92, "R²": 0.93, "Status": "🔄 Running", "Duration": "25m 10s"}
        ]
        
        experiments_df = pd.DataFrame(experiments)
        st.dataframe(experiments_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Start New Training")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Basic Forecaster", "XGBoost", "ARIMA", "Neural Network", "Random Forest"]
        )
        
        # Hyperparameters based on model type
        if model_type == "XGBoost":
            n_estimators = st.slider("N Estimators", 50, 500, 100)
            max_depth = st.slider("Max Depth", 3, 15, 6)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        elif model_type == "Basic Forecaster":
            n_lags = st.slider("Number of Lags", 1, 20, 5)
        elif model_type == "Neural Network":
            hidden_layers = st.slider("Hidden Layers", 1, 5, 2)
            neurons_per_layer = st.slider("Neurons per Layer", 16, 256, 64)
        
        data_split = st.slider("Train/Validation Split", 0.6, 0.9, 0.8)
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Initializing training..."):
                import time
                time.sleep(3)
                st.success(f"✅ Training started for {model_type}")
                st.info("Check the MLflow dashboard for detailed progress")

if __name__ == "__main__":
    main()