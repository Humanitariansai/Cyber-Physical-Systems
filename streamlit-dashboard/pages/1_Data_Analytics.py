"""
Data Analytics Page for Streamlit Dashboard
Provides comprehensive data analysis and visualization capabilities.
"""

# Import path setup first
from utils.path_setup import setup_project_paths
setup_project_paths()

# Now import all other required modules
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

st.set_page_config(
    page_title="Data Analytics - CPS Dashboard",
    page_icon=None,  # Removed emoji
    layout="wide"
)

def main():
    """Main data analytics page"""
    
    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    st.title("Data Analytics")
    st.markdown("Analyze sensor data, patterns, and trends")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Analysis Controls")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Time Series Analysis", "Statistical Analysis", 
             "Correlation Analysis", "Anomaly Detection"]
        )
        
        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 24 Hours", "Last Week", "Last Month", "Custom"]
        )
        
        if time_range == "Custom":
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
            end_date = st.date_input("End Date", datetime.now())
        
        # Data source selector
        data_sources = st.multiselect(
            "Data Sources",
            ["Temperature", "Humidity", "Pressure", "System Load"],
            default=["Temperature"]
        )
        
        # Refresh data
        if st.button("Refresh Data"):
            st.rerun()
    
    # Main content based on analysis type
    if analysis_type == "Time Series Analysis":
        render_time_series_analysis(data_sources, time_range)
    elif analysis_type == "Statistical Analysis":
        render_statistical_analysis(data_sources, time_range)
    elif analysis_type == "Correlation Analysis":
        render_correlation_analysis(data_sources, time_range)
    elif analysis_type == "Anomaly Detection":
        render_anomaly_detection(data_sources, time_range)

def render_time_series_analysis(data_sources, time_range):
    """Render time series analysis section"""
    st.subheader("Time Series Analysis")
    
    # Generate sample time series data
    dates = pd.date_range(
        end=datetime.now(),
        periods={"Last Hour": 60, "Last 24 Hours": 24*60, 
                "Last Week": 7*24, "Last Month": 30*24}.get(time_range, 100),
        freq={"Last Hour": "1min", "Last 24 Hours": "1min",
              "Last Week": "1H", "Last Month": "1H"}.get(time_range, "1H")
    )
    
    # Create multi-line plot
    fig = go.Figure()
    
    for source in data_sources:
        if source == "Temperature":
            base = 25
            amplitude = 5
        elif source == "Humidity":
            base = 60
            amplitude = 15
        elif source == "Pressure":
            base = 1013
            amplitude = 10
        else:  # System Load
            base = 50
            amplitude = 20
        
        # Generate realistic looking data
        data = base + amplitude * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, amplitude/5, len(dates))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=data,
            mode='lines',
            name=source,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Time Series Data",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving averages
    st.subheader("Moving Averages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        window_size = st.slider("Window Size", 5, 50, 20)
    
    with col2:
        ma_type = st.selectbox(
            "Average Type",
            ["Simple", "Exponential"]
        )
    
    fig = go.Figure()
    
    for source in data_sources:
        if source == "Temperature":
            base = 25
            amplitude = 5
        elif source == "Humidity":
            base = 60
            amplitude = 15
        elif source == "Pressure":
            base = 1013
            amplitude = 10
        else:  # System Load
            base = 50
            amplitude = 20
        
        # Original data
        data = base + amplitude * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, amplitude/5, len(dates))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=data,
            mode='lines',
            name=f"{source} (Raw)",
            line=dict(width=1, dash='dot')
        ))
        
        # Moving average
        if ma_type == "Simple":
            ma = pd.Series(data).rolling(window=window_size).mean()
        else:  # Exponential
            ma = pd.Series(data).ewm(span=window_size).mean()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma,
            mode='lines',
            name=f"{source} ({ma_type} MA)",
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f"{ma_type} Moving Average Analysis",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_statistical_analysis(data_sources, time_range):
    """Render statistical analysis section"""
    st.subheader("Statistical Analysis")
    
    # Generate sample data for each source
    stats_data = {}
    
    for source in data_sources:
        if source == "Temperature":
            data = np.random.normal(25, 5, 1000)
        elif source == "Humidity":
            data = np.random.normal(60, 15, 1000)
        elif source == "Pressure":
            data = np.random.normal(1013, 10, 1000)
        else:  # System Load
            data = np.random.normal(50, 20, 1000)
        
        stats_data[source] = data
    
    # Display summary statistics
    for source, data in stats_data.items():
        st.write(f"**{source} Statistics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{np.mean(data):.2f}")
        
        with col2:
            st.metric("Std Dev", f"{np.std(data):.2f}")
        
        with col3:
            st.metric("Min", f"{np.min(data):.2f}")
        
        with col4:
            st.metric("Max", f"{np.max(data):.2f}")
        
        # Distribution plot
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            name=source,
            nbinsx=30,
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"{source} Distribution",
            xaxis_title=source,
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=data,
            name=source,
            boxpoints='outliers'
        ))
        
        fig.update_layout(
            title=f"{source} Box Plot",
            yaxis_title=source,
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

def render_correlation_analysis(data_sources, time_range):
    """Render correlation analysis section"""
    st.subheader("Correlation Analysis")
    
    if len(data_sources) < 2:
        st.warning("Please select at least two data sources for correlation analysis")
        return
    
    # Generate correlated sample data
    n_samples = 1000
    correlation_data = {}
    
    # Base signal
    base = np.sin(np.linspace(0, 4*np.pi, n_samples))
    
    for source in data_sources:
        if source == "Temperature":
            # Temperature correlates positively with base
            correlation_data[source] = 25 + 5 * (base + np.random.normal(0, 0.2, n_samples))
        elif source == "Humidity":
            # Humidity correlates negatively with temperature
            correlation_data[source] = 60 - 15 * (base + np.random.normal(0, 0.2, n_samples))
        elif source == "Pressure":
            # Pressure has weak correlation
            correlation_data[source] = 1013 + 10 * (0.3 * base + np.random.normal(0, 0.8, n_samples))
        else:  # System Load
            # System load is mostly random
            correlation_data[source] = 50 + 20 * np.random.normal(0, 1, n_samples)
    
    # Create correlation matrix
    df = pd.DataFrame(correlation_data)
    correlation_matrix = df.corr()
    
    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    st.subheader("Scatter Plot Matrix")
    
    for i, source1 in enumerate(data_sources):
        for j, source2 in enumerate(data_sources):
            if i < j:  # Only plot upper triangle
                fig = px.scatter(
                    x=correlation_data[source1],
                    y=correlation_data[source2],
                    labels={'x': source1, 'y': source2},
                    trendline="ols"
                )
                
                fig.update_layout(
                    title=f"{source1} vs {source2}",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def render_anomaly_detection(data_sources, time_range):
    """Render anomaly detection section"""
    st.subheader("Anomaly Detection")
    
    # Anomaly detection settings
    col1, col2 = st.columns(2)
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method",
            ["Statistical", "Rolling Z-Score", "IQR Method"]
        )
    
    with col2:
        sensitivity = st.slider(
            "Detection Sensitivity",
            1.0, 5.0, 3.0,
            help="Number of standard deviations for anomaly threshold"
        )
    
    # Generate sample data with anomalies
    dates = pd.date_range(
        end=datetime.now(),
        periods=100,
        freq="1H"
    )
    
    for source in data_sources:
        # Generate base signal
        if source == "Temperature":
            base = 25
            amplitude = 5
        elif source == "Humidity":
            base = 60
            amplitude = 15
        elif source == "Pressure":
            base = 1013
            amplitude = 10
        else:  # System Load
            base = 50
            amplitude = 20
        
        # Generate normal data
        data = base + amplitude * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        noise = np.random.normal(0, amplitude/5, len(dates))
        
        # Add anomalies
        n_anomalies = 5
        anomaly_indices = np.random.choice(len(dates), n_anomalies, replace=False)
        anomalies = np.zeros(len(dates))
        anomalies[anomaly_indices] = np.random.normal(0, amplitude, n_anomalies)
        
        # Combine signals
        final_data = data + noise + anomalies
        
        # Detect anomalies
        if detection_method == "Statistical":
            mean = np.mean(final_data)
            std = np.std(final_data)
            threshold = sensitivity * std
            is_anomaly = np.abs(final_data - mean) > threshold
        
        elif detection_method == "Rolling Z-Score":
            window = 10
            rolling_mean = pd.Series(final_data).rolling(window=window).mean()
            rolling_std = pd.Series(final_data).rolling(window=window).std()
            z_scores = np.abs((final_data - rolling_mean) / rolling_std)
            is_anomaly = z_scores > sensitivity
        
        else:  # IQR Method
            Q1 = np.percentile(final_data, 25)
            Q3 = np.percentile(final_data, 75)
            IQR = Q3 - Q1
            threshold = sensitivity * IQR
            is_anomaly = (final_data < (Q1 - threshold)) | (final_data > (Q3 + threshold))
        
        # Plot results
        fig = go.Figure()
        
        # Normal data
        fig.add_trace(go.Scatter(
            x=dates,
            y=final_data,
            mode='lines',
            name=source,
            line=dict(width=2)
        ))
        
        # Anomalies
        fig.add_trace(go.Scatter(
            x=dates[is_anomaly],
            y=final_data[is_anomaly],
            mode='markers',
            name='Anomalies',
            marker=dict(
                color='red',
                size=10,
                symbol='x'
            )
        ))
        
        if detection_method == "Statistical":
            # Add threshold bounds
            fig.add_trace(go.Scatter(
                x=dates,
                y=[mean + threshold] * len(dates),
                mode='lines',
                name='Upper Threshold',
                line=dict(dash='dash', color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=[mean - threshold] * len(dates),
                mode='lines',
                name='Lower Threshold',
                line=dict(dash='dash', color='red')
            ))
        
        fig.update_layout(
            title=f"{source} Anomaly Detection ({detection_method})",
            xaxis_title="Time",
            yaxis_title=source,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly statistics
        n_detected = np.sum(is_anomaly)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies", str(n_detected))
        
        with col2:
            st.metric("Anomaly Rate", f"{(n_detected/len(dates))*100:.1f}%")
        
        with col3:
            st.metric("Normal Data", f"{((len(dates)-n_detected)/len(dates))*100:.1f}%")
        
        st.markdown("---")

if __name__ == "__main__":
    main()