"""
Metrics Cards Component for Streamlit Dashboard
Provides reusable metric display components.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

def render_metrics_cards():
    """Render key metrics cards"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_sensor_metric()
    
    with col2:
        render_model_metric()
    
    with col3:
        render_prediction_metric()
    
    with col4:
        render_health_metric()

def render_sensor_metric():
    """Render active sensors metric"""
    total_sensors = 12
    active_sensors = 12
    offline_sensors = total_sensors - active_sensors
    
    st.metric(
        label="🔌 Active Sensors",
        value=f"{active_sensors}/{total_sensors}",
        delta=f"+{2}" if offline_sensors == 0 else f"-{offline_sensors}",
        delta_color="normal" if offline_sensors == 0 else "inverse"
    )
    
    # Mini chart for sensor status
    if st.button("📊 Details", key="sensor_details"):
        with st.expander("Sensor Details", expanded=True):
            sensor_data = {
                "Sensor": [f"Sensor {i}" for i in range(1, 13)],
                "Status": ["🟢 Active"] * 10 + ["🟡 Warning"] * 1 + ["🟢 Active"] * 1,
                "Last Reading": ["< 1 min"] * 12
            }
            st.dataframe(sensor_data, use_container_width=True, hide_index=True)

def render_model_metric():
    """Render ML models metric"""
    total_models = 3
    active_models = 3
    
    st.metric(
        label="🤖 ML Models",
        value=f"{active_models}",
        delta="+1",
        help="Currently active machine learning models"
    )
    
    if st.button("🔍 View", key="model_details"):
        with st.expander("Model Status", expanded=True):
            model_data = {
                "Model": ["Basic Forecaster", "XGBoost", "ARIMA"],
                "Status": ["🟢 Active", "🟢 Active", "🟡 Idle"],
                "Accuracy": ["89%", "94%", "83%"],
                "Last Trained": ["2 hours ago", "1 hour ago", "1 day ago"]
            }
            st.dataframe(model_data, use_container_width=True, hide_index=True)

def render_prediction_metric():
    """Render predictions metric"""
    predictions_today = 847
    predictions_change = 125
    
    st.metric(
        label="🔮 Predictions Today",
        value=predictions_today,
        delta=f"+{predictions_change}",
        help="Total predictions generated today"
    )
    
    if st.button("📈 Trend", key="prediction_details"):
        with st.expander("Prediction Trends", expanded=True):
            # Create mini trend chart
            hours = list(range(0, 24))
            predictions_per_hour = [
                20 + 15 * np.sin(h * np.pi / 12) + np.random.randint(-5, 6)
                for h in hours
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours,
                y=predictions_per_hour,
                mode='lines+markers',
                name='Predictions/Hour',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Predictions per Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Predictions",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_health_metric():
    """Render system health metric"""
    health_score = 98.5
    health_change = 0.3
    
    # Determine health status color
    if health_score >= 95:
        health_color = "normal"
        health_icon = "🟢"
    elif health_score >= 85:
        health_color = "normal"
        health_icon = "🟡"
    else:
        health_color = "inverse"
        health_icon = "🔴"
    
    st.metric(
        label=f"{health_icon} System Health",
        value=f"{health_score}%",
        delta=f"+{health_change}%",
        delta_color=health_color,
        help="Overall system health score"
    )
    
    if st.button("⚙️ Details", key="health_details"):
        with st.expander("Health Breakdown", expanded=True):
            health_components = {
                "Component": ["CPU", "Memory", "Storage", "Network", "Database"],
                "Score": ["95%", "98%", "92%", "99%", "97%"],
                "Status": ["🟢 Good", "🟢 Good", "🟡 Warning", "🟢 Good", "🟢 Good"]
            }
            st.dataframe(health_components, use_container_width=True, hide_index=True)

def render_custom_metric(title, value, delta=None, help_text=None, icon="📊"):
    """
    Render a custom metric card
    
    Args:
        title (str): Metric title
        value (str/int/float): Metric value
        delta (str, optional): Change indicator
        help_text (str, optional): Help tooltip text
        icon (str): Icon for the metric
    """
    st.metric(
        label=f"{icon} {title}",
        value=value,
        delta=delta,
        help=help_text
    )

def render_kpi_dashboard():
    """Render a comprehensive KPI dashboard"""
    st.subheader("📊 Key Performance Indicators")
    
    # First row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_custom_metric(
            title="Data Points/Day",
            value="24.5K",
            delta="+2.1K",
            help_text="Total data points collected per day",
            icon="📈"
        )
    
    with col2:
        render_custom_metric(
            title="Model Accuracy",
            value="94.2%",
            delta="+1.3%",
            help_text="Average accuracy across all models",
            icon="🎯"
        )
    
    with col3:
        render_custom_metric(
            title="Response Time",
            value="120ms",
            delta="-15ms",
            help_text="Average API response time",
            icon="⚡"
        )
    
    with col4:
        render_custom_metric(
            title="Error Rate",
            value="0.02%",
            delta="-0.01%",
            help_text="System error rate",
            icon="🛡️"
        )
    
    # Second row of metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        render_custom_metric(
            title="Storage Used",
            value="67%",
            delta="+3%",
            help_text="Total storage utilization",
            icon="💾"
        )
    
    with col6:
        render_custom_metric(
            title="Active Users",
            value="156",
            delta="+12",
            help_text="Currently active dashboard users",
            icon="👥"
        )
    
    with col7:
        render_custom_metric(
            title="Alerts Today",
            value="3",
            delta="-2",
            help_text="System alerts generated today",
            icon="🚨"
        )
    
    with col8:
        render_custom_metric(
            title="Uptime",
            value="99.9%",
            delta="+0.1%",
            help_text="System uptime this month",
            icon="🔧"
        )

def render_gauge_metric(title, value, max_value=100, color_ranges=None):
    """
    Render a gauge-style metric
    
    Args:
        title (str): Gauge title
        value (float): Current value
        max_value (float): Maximum value for the gauge
        color_ranges (list): Color ranges for different value ranges
    """
    if color_ranges is None:
        color_ranges = [
            [0, 50, "red"],
            [50, 80, "yellow"],
            [80, 100, "green"]
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [r[0], r[1]], 'color': r[2]}
                for r in color_ranges
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

def render_real_time_metrics():
    """Render real-time updating metrics"""
    # This would typically connect to real-time data sources
    # For demo purposes, we'll simulate real-time updates
    
    placeholder = st.empty()
    
    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_temp = 22.5 + np.random.normal(0, 0.5)
            st.metric(
                label="🌡️ Current Temperature",
                value=f"{current_temp:.1f}°C",
                delta=f"{np.random.uniform(-0.5, 0.5):.1f}°C"
            )
        
        with col2:
            current_humidity = 60 + np.random.normal(0, 2)
            st.metric(
                label="💧 Current Humidity",
                value=f"{current_humidity:.1f}%",
                delta=f"{np.random.uniform(-2, 2):.1f}%"
            )
        
        with col3:
            current_pressure = 1013 + np.random.normal(0, 1)
            st.metric(
                label="📊 Current Pressure",
                value=f"{current_pressure:.1f} hPa",
                delta=f"{np.random.uniform(-1, 1):.1f} hPa"
            )