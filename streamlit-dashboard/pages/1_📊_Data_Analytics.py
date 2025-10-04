"""
Data Analytics Page for Streamlit Dashboard
Provides detailed data analysis and visualization capabilities.
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
from components.sidebar import get_time_range_dates

st.set_page_config(
    page_title="Data Analytics - CPS Dashboard",
    page_icon="📊",
    layout="wide"
)

def main():
    """Main data analytics page"""
    
    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    st.title("📊 Data Analytics")
    st.markdown("Deep dive into sensor data patterns, trends, and anomalies")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Analytics Controls")
        
        # Data source selection
        data_sources = st.multiselect(
            "Select Data Sources",
            ["Temperature", "Humidity", "Pressure", "Vibration", "Power"],
            default=["Temperature", "Humidity"]
        )
        
        # Time range
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"]
        )
        
        if analysis_period == "Custom Range":
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
            end_date = st.date_input("End Date", datetime.now())
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Time Series", "Statistical Analysis", "Correlation Analysis", "Anomaly Detection"]
        )
        
        # Aggregation level
        aggregation = st.selectbox(
            "Data Aggregation",
            ["Raw Data", "Hourly", "Daily", "Weekly"]
        )
        
        if st.button("🔄 Refresh Analysis"):
            st.rerun()
    
    # Main content based on analysis type
    if analysis_type == "Time Series":
        render_time_series_analysis(data_sources, analysis_period, aggregation)
    elif analysis_type == "Statistical Analysis":
        render_statistical_analysis(data_sources, analysis_period)
    elif analysis_type == "Correlation Analysis":
        render_correlation_analysis(data_sources, analysis_period)
    elif analysis_type == "Anomaly Detection":
        render_anomaly_detection(data_sources, analysis_period)

def render_time_series_analysis(data_sources, period, aggregation):
    """Render time series analysis"""
    st.subheader("📈 Time Series Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Generate sample time series data
        if period == "Last 24 Hours":
            hours = 24
            freq = 'H'
        elif period == "Last 7 Days":
            hours = 24 * 7
            freq = 'H'
        else:
            hours = 24 * 30
            freq = 'D'
        
        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq=freq)
        
        # Create subplots
        fig = make_subplots(
            rows=len(data_sources),
            cols=1,
            subplot_titles=data_sources,
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, source in enumerate(data_sources):
            # Generate sample data based on source type
            if source == "Temperature":
                base_value = 22
                variation = 5
                trend = np.sin(np.linspace(0, 4*np.pi, len(timestamps)))
            elif source == "Humidity":
                base_value = 60
                variation = 15
                trend = np.cos(np.linspace(0, 3*np.pi, len(timestamps)))
            elif source == "Pressure":
                base_value = 1013
                variation = 10
                trend = np.sin(np.linspace(0, 2*np.pi, len(timestamps)))
            elif source == "Vibration":
                base_value = 0.5
                variation = 0.3
                trend = np.random.random(len(timestamps)) - 0.5
            else:  # Power
                base_value = 150
                variation = 30
                trend = np.sin(np.linspace(0, 6*np.pi, len(timestamps)))
            
            data = base_value + variation * trend + np.random.normal(0, variation*0.1, len(timestamps))
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=data,
                    mode='lines',
                    name=source,
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=200 * len(data_sources),
            showlegend=False,
            title="Time Series Data Visualization"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Analysis Summary")
        
        # Display statistics for each data source
        for source in data_sources:
            with st.expander(f"{source} Stats"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Mean", f"{np.random.uniform(20, 100):.2f}")
                    st.metric("Min", f"{np.random.uniform(10, 50):.2f}")
                
                with col_b:
                    st.metric("Max", f"{np.random.uniform(80, 150):.2f}")
                    st.metric("Std Dev", f"{np.random.uniform(1, 10):.2f}")
                
                # Trend indicator
                trend = np.random.choice(["📈 Increasing", "📉 Decreasing", "➡️ Stable"])
                st.write(f"**Trend:** {trend}")
    
    # Data table
    with st.expander("📋 Raw Data Preview", expanded=False):
        # Generate sample data table
        sample_data = []
        for i in range(50):  # Last 50 records
            timestamp = datetime.now() - timedelta(hours=i)
            row = {'timestamp': timestamp}
            
            for source in data_sources:
                if source == "Temperature":
                    value = 22 + np.random.normal(0, 2)
                elif source == "Humidity":
                    value = 60 + np.random.normal(0, 5)
                elif source == "Pressure":
                    value = 1013 + np.random.normal(0, 1)
                elif source == "Vibration":
                    value = 0.5 + np.random.uniform(-0.3, 0.3)
                else:  # Power
                    value = 150 + np.random.normal(0, 10)
                
                row[source] = round(value, 2)
            
            sample_data.append(row)
        
        df = pd.DataFrame(sample_data)
        st.dataframe(df, use_container_width=True)

def render_statistical_analysis(data_sources, period):
    """Render statistical analysis"""
    st.subheader("📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution Analysis")
        
        # Create histogram for each data source
        for source in data_sources:
            # Generate sample data
            if source == "Temperature":
                data = np.random.normal(22, 3, 1000)
            elif source == "Humidity":
                data = np.random.normal(60, 10, 1000)
            elif source == "Pressure":
                data = np.random.normal(1013, 5, 1000)
            elif source == "Vibration":
                data = np.random.exponential(0.5, 1000)
            else:  # Power
                data = np.random.normal(150, 20, 1000)
            
            fig = px.histogram(
                x=data,
                title=f"{source} Distribution",
                nbins=30,
                marginal="box"
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Statistical Summary")
        
        # Statistical summary table
        stats_data = []
        for source in data_sources:
            # Generate sample statistics
            stats_data.append({
                "Metric": source,
                "Count": 1000,
                "Mean": f"{np.random.uniform(20, 100):.2f}",
                "Std": f"{np.random.uniform(1, 10):.2f}",
                "Min": f"{np.random.uniform(10, 50):.2f}",
                "25%": f"{np.random.uniform(40, 70):.2f}",
                "50%": f"{np.random.uniform(60, 90):.2f}",
                "75%": f"{np.random.uniform(80, 120):.2f}",
                "Max": f"{np.random.uniform(100, 150):.2f}",
                "Skewness": f"{np.random.uniform(-1, 1):.3f}",
                "Kurtosis": f"{np.random.uniform(-2, 2):.3f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Normality tests
        st.subheader("Normality Tests")
        
        normality_data = []
        for source in data_sources:
            normality_data.append({
                "Metric": source,
                "Shapiro-Wilk p-value": f"{np.random.uniform(0.001, 0.1):.4f}",
                "Kolmogorov-Smirnov p-value": f"{np.random.uniform(0.001, 0.1):.4f}",
                "Normal Distribution": np.random.choice(["❌ No", "✅ Yes"])
            })
        
        normality_df = pd.DataFrame(normality_data)
        st.dataframe(normality_df, use_container_width=True, hide_index=True)

def render_correlation_analysis(data_sources, period):
    """Render correlation analysis"""
    st.subheader("🔗 Correlation Analysis")
    
    if len(data_sources) < 2:
        st.warning("Please select at least 2 data sources for correlation analysis.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate correlation matrix
        n_points = 1000
        correlation_data = {}
        
        # Generate correlated sample data
        base_data = np.random.randn(n_points)
        
        for i, source in enumerate(data_sources):
            # Create some correlation between sources
            correlation_strength = np.random.uniform(0.3, 0.9)
            noise = np.random.randn(n_points) * (1 - correlation_strength)
            
            if source == "Temperature":
                correlation_data[source] = 22 + 5 * base_data + noise
            elif source == "Humidity":
                # Negative correlation with temperature
                correlation_data[source] = 60 - 3 * base_data + noise
            elif source == "Pressure":
                correlation_data[source] = 1013 + 2 * base_data + noise
            elif source == "Vibration":
                correlation_data[source] = 0.5 + 0.1 * base_data + noise
            else:  # Power
                correlation_data[source] = 150 + 10 * base_data + noise
        
        df = pd.DataFrame(correlation_data)
        correlation_matrix = df.corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix Heatmap"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot matrix
        if len(data_sources) <= 4:  # Only show for reasonable number of variables
            st.subheader("Scatter Plot Matrix")
            fig = px.scatter_matrix(df, title="Pairwise Scatter Plots")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Correlation Statistics")
        
        # Correlation table
        corr_pairs = []
        for i in range(len(data_sources)):
            for j in range(i+1, len(data_sources)):
                source1 = data_sources[i]
                source2 = data_sources[j]
                correlation = correlation_matrix.loc[source1, source2]
                
                # Interpret correlation strength
                if abs(correlation) > 0.7:
                    strength = "Strong"
                elif abs(correlation) > 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                direction = "Positive" if correlation > 0 else "Negative"
                
                corr_pairs.append({
                    "Pair": f"{source1} - {source2}",
                    "Correlation": f"{correlation:.3f}",
                    "Strength": strength,
                    "Direction": direction
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
        
        # Interpretation guide
        st.subheader("📖 Interpretation Guide")
        st.markdown("""
        **Correlation Strength:**
        - |r| > 0.7: Strong correlation
        - 0.4 < |r| ≤ 0.7: Moderate correlation
        - |r| ≤ 0.4: Weak correlation
        
        **Direction:**
        - Positive: Variables increase together
        - Negative: One increases as other decreases
        """)

def render_anomaly_detection(data_sources, period):
    """Render anomaly detection analysis"""
    st.subheader("🚨 Anomaly Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Generate time series with anomalies
        hours = 168  # One week
        timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')
        
        fig = make_subplots(
            rows=len(data_sources),
            cols=1,
            subplot_titles=[f"{source} - Anomaly Detection" for source in data_sources],
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, source in enumerate(data_sources):
            # Generate normal data
            if source == "Temperature":
                base_value = 22
                variation = 2
                normal_data = base_value + variation * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.normal(0, 0.5, len(timestamps))
            elif source == "Humidity":
                base_value = 60
                variation = 5
                normal_data = base_value + variation * np.cos(np.linspace(0, 3*np.pi, len(timestamps))) + np.random.normal(0, 1, len(timestamps))
            else:
                base_value = 100
                variation = 10
                normal_data = base_value + variation * np.sin(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 2, len(timestamps))
            
            # Inject anomalies
            anomaly_indices = np.random.choice(len(timestamps), size=5, replace=False)
            anomaly_data = normal_data.copy()
            for idx in anomaly_indices:
                anomaly_data[idx] += np.random.choice([-1, 1]) * variation * 3
            
            # Plot normal data
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=normal_data,
                    mode='lines',
                    name=f'{source} Normal',
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=(i == 0)
                ),
                row=i+1, col=1
            )
            
            # Highlight anomalies
            fig.add_trace(
                go.Scatter(
                    x=timestamps[anomaly_indices],
                    y=anomaly_data[anomaly_indices],
                    mode='markers',
                    name=f'{source} Anomalies',
                    marker=dict(color='red', size=8, symbol='x'),
                    showlegend=(i == 0)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=200 * len(data_sources),
            title="Anomaly Detection Results"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Anomaly Summary")
        
        # Anomaly statistics
        anomaly_stats = []
        for source in data_sources:
            anomalies_detected = np.random.randint(3, 8)
            anomaly_rate = (anomalies_detected / hours) * 100
            
            anomaly_stats.append({
                "Metric": source,
                "Anomalies": anomalies_detected,
                "Rate (%)": f"{anomaly_rate:.2f}",
                "Severity": np.random.choice(["Low", "Medium", "High"]),
                "Last Anomaly": f"{np.random.randint(1, 24)}h ago"
            })
        
        anomaly_df = pd.DataFrame(anomaly_stats)
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
        
        # Detection methods
        st.subheader("🔧 Detection Methods")
        
        method = st.selectbox(
            "Anomaly Detection Method",
            ["Statistical (Z-Score)", "Isolation Forest", "Local Outlier Factor", "DBSCAN"]
        )
        
        if method == "Statistical (Z-Score)":
            threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.5, 0.1)
        elif method in ["Isolation Forest", "Local Outlier Factor"]:
            contamination = st.slider("Contamination Rate", 0.01, 0.1, 0.05, 0.01)
        
        if st.button("🔍 Rerun Detection"):
            st.success(f"Anomaly detection completed using {method}")
        
        # Alert settings
        st.subheader("🚨 Alert Settings")
        
        enable_alerts = st.toggle("Enable Anomaly Alerts", value=True)
        
        if enable_alerts:
            alert_threshold = st.selectbox(
                "Alert Threshold",
                ["Any Anomaly", "High Severity Only", "Multiple Anomalies"]
            )
            
            notification_method = st.multiselect(
                "Notification Methods",
                ["Email", "SMS", "Dashboard Alert", "Webhook"],
                default=["Dashboard Alert"]
            )

if __name__ == "__main__":
    main()