"""
System Health Monitoring Page for Streamlit Dashboard
Provides real-time system metrics and health monitoring capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "streamlit-dashboard"))

from utils.data_loader import DataLoader

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

st.set_page_config(
    page_title="System Health - CPS Dashboard",
    page_icon=None,  # Removed emoji
    layout="wide"
)

def main():
    """Main system health monitoring page"""
    st.title("System Health Monitoring")
    st.markdown("Monitor system performance and health metrics in real-time")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Monitoring Controls")
        
        # Refresh rate
        refresh_rate = st.slider(
            "Refresh Rate (seconds)",
            min_value=5,
            max_value=60,
            value=30
        )
        
        # Metrics selection
        selected_metrics = st.multiselect(
            "Select Metrics",
            ["CPU Usage", "Memory Usage", "Disk Usage", "Network Traffic"],
            default=["CPU Usage", "Memory Usage"]
        )
        
        # Time window
        time_window = st.selectbox(
            "Time Window",
            ["Last 5 minutes", "Last 15 minutes", "Last hour", "Last 24 hours"]
        )
        
        # Auto refresh
        auto_refresh = st.checkbox("Auto Refresh", value=True)

    # Check if it's time to refresh
    now = datetime.now()
    if auto_refresh and (now - st.session_state.last_refresh).total_seconds() >= refresh_rate:
        st.rerun()
        st.session_state.last_refresh = now

    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()

    # Layout for metrics
    col1, col2 = st.columns(2)

    # CPU Usage
    if "CPU Usage" in selected_metrics:
        with col1:
            display_cpu_metrics()

    # Memory Usage
    if "Memory Usage" in selected_metrics:
        with col2:
            display_memory_metrics()

    # Disk Usage
    if "Disk Usage" in selected_metrics:
        with col1:
            display_disk_metrics()

    # Network Traffic
    if "Network Traffic" in selected_metrics:
        with col2:
            display_network_metrics()

    # Historical trends
    st.subheader("Historical Trends")
    display_historical_trends(selected_metrics, time_window)

def display_cpu_metrics():
    """Display CPU usage metrics"""
    st.subheader("CPU Usage")
    
    # Generate sample CPU data
    cpu_usage = np.random.uniform(20, 80)
    
    # CPU gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cpu_usage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "CPU Usage %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # CPU core details
    st.markdown(f"**Core Count:** {np.random.randint(4, 16)}")
    st.markdown(f"**Clock Speed:** {np.random.uniform(2.0, 4.0):.2f} GHz")

def display_memory_metrics():
    """Display memory usage metrics"""
    st.subheader("Memory Usage")
    
    # Generate sample memory data
    total_memory = 16  # GB
    used_memory = np.random.uniform(4, 12)
    memory_percent = (used_memory / total_memory) * 100
    
    # Memory gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=memory_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Memory Usage %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 85], 'color': "gray"},
                {'range': [85, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Memory details
    st.markdown(f"**Total Memory:** {total_memory:.1f} GB")
    st.markdown(f"**Used Memory:** {used_memory:.1f} GB")
    st.markdown(f"**Available Memory:** {(total_memory - used_memory):.1f} GB")

def display_disk_metrics():
    """Display disk usage metrics"""
    st.subheader("Disk Usage")
    
    # Generate sample disk data
    total_space = 512  # GB
    used_space = np.random.uniform(200, 400)
    disk_percent = (used_space / total_space) * 100
    
    # Disk usage pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Used', 'Free'],
        values=[used_space, total_space - used_space],
        hole=.3
    )])
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Disk details
    st.markdown(f"**Total Space:** {total_space:.1f} GB")
    st.markdown(f"**Used Space:** {used_space:.1f} GB")
    st.markdown(f"**Free Space:** {(total_space - used_space):.1f} GB")

def display_network_metrics():
    """Display network traffic metrics"""
    st.subheader("Network Traffic")
    
    # Generate sample network data
    download_speed = np.random.uniform(1, 100)  # Mbps
    upload_speed = np.random.uniform(1, 50)  # Mbps
    
    # Network traffic bar chart
    fig = go.Figure(data=[
        go.Bar(name='Download', x=['Speed'], y=[download_speed]),
        go.Bar(name='Upload', x=['Speed'], y=[upload_speed])
    ])
    
    fig.update_layout(
        height=250,
        yaxis_title="Speed (Mbps)",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network details
    st.markdown(f"**Download Speed:** {download_speed:.1f} Mbps")
    st.markdown(f"**Upload Speed:** {upload_speed:.1f} Mbps")
    st.markdown(f"**Active Connections:** {np.random.randint(10, 100)}")

def display_historical_trends(metrics, time_window):
    """Display historical trends for selected metrics"""
    # Generate time points based on window
    window_hours = {
        "Last 5 minutes": 5/60,
        "Last 15 minutes": 15/60,
        "Last hour": 1,
        "Last 24 hours": 24
    }
    
    hours = window_hours[time_window]
    points = 100
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=points,
        freq=f"{int(hours * 60 / points)}min"
    )
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for each metric
    if "CPU Usage" in metrics:
        cpu_data = 50 + 30 * np.random.randn(points) / 3
        cpu_data = np.clip(cpu_data, 0, 100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_data, name="CPU %"),
            secondary_y=False
        )
    
    if "Memory Usage" in metrics:
        memory_data = 60 + 20 * np.random.randn(points) / 3
        memory_data = np.clip(memory_data, 0, 100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_data, name="Memory %"),
            secondary_y=False
        )
    
    if "Disk Usage" in metrics:
        disk_data = 70 + 5 * np.random.randn(points) / 3
        disk_data = np.clip(disk_data, 0, 100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=disk_data, name="Disk %"),
            secondary_y=False
        )
    
    if "Network Traffic" in metrics:
        network_data = 50 + 40 * np.random.randn(points) / 3
        network_data = np.clip(network_data, 0, 100)
        fig.add_trace(
            go.Scatter(x=timestamps, y=network_data, name="Network (Mbps)"),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title="System Metrics Over Time",
        height=400
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Usage %", secondary_y=False)
    fig.update_yaxes(title_text="Network Speed (Mbps)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()