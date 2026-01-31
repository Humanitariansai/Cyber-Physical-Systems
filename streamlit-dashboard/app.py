"""
Streamlit Dashboard for Cold Chain Monitoring System
Real-time visualization and monitoring interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Page configuration
st.set_page_config(
    page_title="Cold Chain Monitor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .status-ok { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-critical { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


def generate_sample_data(hours: int = 24, interval_minutes: int = 1) -> pd.DataFrame:
    """Generate sample sensor data for demonstration."""
    np.random.seed(42)

    n_points = (hours * 60) // interval_minutes
    timestamps = [datetime.now() - timedelta(minutes=i * interval_minutes)
                  for i in range(n_points, 0, -1)]

    # Generate realistic cold chain temperature data (2-8°C range)
    base_temp = 5.0
    noise = np.random.normal(0, 0.3, n_points)
    trend = np.sin(np.linspace(0, 4 * np.pi, n_points)) * 0.5

    # Add some anomalies
    temps = base_temp + noise + trend
    temps[int(n_points * 0.3):int(n_points * 0.35)] += 2  # Warm period
    temps[int(n_points * 0.7):int(n_points * 0.72)] += 3  # Anomaly

    humidity = 55 + np.random.normal(0, 5, n_points)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temps,
        'humidity': humidity,
        'sensor_id': ['sensor-01'] * n_points
    })

    return data


def generate_predictions(current_temp: float) -> dict:
    """Generate sample predictions."""
    trend = np.random.uniform(-0.1, 0.1)
    return {
        30: (current_temp + trend * 30 + np.random.uniform(-0.5, 0.5), 0.85),
        60: (current_temp + trend * 60 + np.random.uniform(-1, 1), 0.72)
    }


def generate_alerts() -> list:
    """Generate sample alerts."""
    return [
        {
            "id": "ALT-001",
            "sensor_id": "sensor-01",
            "type": "TEMPERATURE_HIGH",
            "severity": "WARNING",
            "message": "Temperature exceeded 8°C threshold",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "acknowledged": False
        },
        {
            "id": "ALT-002",
            "sensor_id": "sensor-02",
            "type": "RAPID_CHANGE",
            "severity": "INFO",
            "message": "Rapid temperature change detected",
            "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
            "acknowledged": True
        }
    ]


def main():
    # Sidebar
    st.sidebar.title("🌡️ Cold Chain Monitor")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Sensor Data", "Predictions", "Alerts", "System Health"]
    )

    st.sidebar.markdown("---")

    # Sensor selection
    sensors = ["sensor-01", "sensor-02", "sensor-03"]
    selected_sensor = st.sidebar.selectbox("Select Sensor", sensors)

    # Time range
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"]
    )

    hours_map = {
        "Last Hour": 1,
        "Last 6 Hours": 6,
        "Last 24 Hours": 24,
        "Last 7 Days": 168
    }
    hours = hours_map[time_range]

    # Generate sample data
    data = generate_sample_data(hours=hours)

    if page == "Dashboard":
        show_dashboard(data, selected_sensor)
    elif page == "Sensor Data":
        show_sensor_data(data, selected_sensor)
    elif page == "Predictions":
        show_predictions(data)
    elif page == "Alerts":
        show_alerts()
    elif page == "System Health":
        show_system_health()


def show_dashboard(data: pd.DataFrame, sensor_id: str):
    """Main dashboard view."""
    st.title("Cold Chain Monitoring Dashboard")
    st.markdown("Real-time temperature monitoring for pharmaceutical supply chain")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    current_temp = data['temperature'].iloc[-1]
    avg_temp = data['temperature'].mean()
    min_temp = data['temperature'].min()
    max_temp = data['temperature'].max()

    with col1:
        st.metric(
            "Current Temperature",
            f"{current_temp:.1f}°C",
            f"{current_temp - avg_temp:.2f}°C from avg"
        )

    with col2:
        st.metric(
            "Average (24h)",
            f"{avg_temp:.1f}°C",
            None
        )

    with col3:
        st.metric(
            "Min/Max",
            f"{min_temp:.1f}°C / {max_temp:.1f}°C",
            None
        )

    with col4:
        # Status indicator
        if current_temp < 2 or current_temp > 8:
            status = "⚠️ Warning"
            status_color = "orange"
        elif current_temp < 0 or current_temp > 10:
            status = "🚨 Critical"
            status_color = "red"
        else:
            status = "✅ Normal"
            status_color = "green"
        st.metric("Status", status)

    st.markdown("---")

    # Temperature chart
    st.subheader("Temperature Trend")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Temperature line
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['temperature'],
            name="Temperature",
            line=dict(color='#1f77b4', width=2)
        ),
        secondary_y=False
    )

    # Threshold lines
    fig.add_hline(y=2, line_dash="dash", line_color="blue",
                  annotation_text="Min (2°C)")
    fig.add_hline(y=8, line_dash="dash", line_color="orange",
                  annotation_text="Max (8°C)")

    # Humidity
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['humidity'],
            name="Humidity",
            line=dict(color='#2ca02c', width=1, dash='dot'),
            opacity=0.5
        ),
        secondary_y=True
    )

    fig.update_layout(
        height=400,
        xaxis_title="Time",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Humidity (%)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Predictions and alerts in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔮 Predictions")
        predictions = generate_predictions(current_temp)

        for horizon, (pred, conf) in predictions.items():
            st.write(f"**{horizon}-minute forecast:** {pred:.1f}°C "
                    f"(confidence: {conf:.0%})")

    with col2:
        st.subheader("🔔 Active Alerts")
        alerts = [a for a in generate_alerts() if not a['acknowledged']]

        if alerts:
            for alert in alerts:
                severity_emoji = "⚠️" if alert['severity'] == "WARNING" else "ℹ️"
                st.warning(f"{severity_emoji} {alert['message']}")
        else:
            st.success("No active alerts")


def show_sensor_data(data: pd.DataFrame, sensor_id: str):
    """Detailed sensor data view."""
    st.title("Sensor Data Analysis")

    # Statistics
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{data['temperature'].mean():.2f}°C")
    with col2:
        st.metric("Std Dev", f"{data['temperature'].std():.2f}°C")
    with col3:
        st.metric("Min", f"{data['temperature'].min():.2f}°C")
    with col4:
        st.metric("Max", f"{data['temperature'].max():.2f}°C")

    # Distribution
    st.subheader("Temperature Distribution")
    fig = px.histogram(data, x='temperature', nbins=50,
                       title="Temperature Distribution")
    fig.add_vline(x=2, line_dash="dash", line_color="blue")
    fig.add_vline(x=8, line_dash="dash", line_color="orange")
    st.plotly_chart(fig, use_container_width=True)

    # Raw data
    st.subheader("Raw Data")
    st.dataframe(data.tail(100), use_container_width=True)

    # Download button
    csv = data.to_csv(index=False)
    st.download_button(
        "Download Data (CSV)",
        csv,
        "sensor_data.csv",
        "text/csv"
    )


def show_predictions(data: pd.DataFrame):
    """Predictions view."""
    st.title("Temperature Predictions")

    current_temp = data['temperature'].iloc[-1]
    predictions = generate_predictions(current_temp)

    st.subheader("Current Forecasts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 30-Minute Forecast")
        pred_30, conf_30 = predictions[30]
        st.metric("Predicted Temperature", f"{pred_30:.1f}°C")
        st.progress(conf_30)
        st.caption(f"Confidence: {conf_30:.0%}")

    with col2:
        st.markdown("### 60-Minute Forecast")
        pred_60, conf_60 = predictions[60]
        st.metric("Predicted Temperature", f"{pred_60:.1f}°C")
        st.progress(conf_60)
        st.caption(f"Confidence: {conf_60:.0%}")

    # Forecast visualization
    st.subheader("Forecast Visualization")

    # Create forecast line
    forecast_times = [
        datetime.now(),
        datetime.now() + timedelta(minutes=30),
        datetime.now() + timedelta(minutes=60)
    ]
    forecast_temps = [current_temp, pred_30, pred_60]

    fig = go.Figure()

    # Historical data (last 2 hours)
    recent = data.tail(120)
    fig.add_trace(go.Scatter(
        x=recent['timestamp'],
        y=recent['temperature'],
        name="Historical",
        line=dict(color='blue')
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast_temps,
        name="Forecast",
        line=dict(color='red', dash='dash')
    ))

    # Thresholds
    fig.add_hline(y=2, line_dash="dot", line_color="lightblue")
    fig.add_hline(y=8, line_dash="dot", line_color="orange")

    fig.update_layout(
        title="Temperature Forecast",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Model info
    st.subheader("Model Information")
    st.info("""
    **Prediction Models:**
    - LSTM (Long Short-Term Memory) for sequence modeling
    - GRU (Gated Recurrent Unit) for faster inference
    - XGBoost for feature-based predictions

    Predictions are generated with 30 and 60-minute horizons using
    ensemble methods for improved accuracy.
    """)


def show_alerts():
    """Alerts management view."""
    st.title("Alert Management")

    alerts = generate_alerts()

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["CRITICAL", "WARNING", "INFO"],
            default=["CRITICAL", "WARNING", "INFO"]
        )
    with col2:
        show_acknowledged = st.checkbox("Show Acknowledged", value=True)

    # Filter alerts
    filtered = [
        a for a in alerts
        if a['severity'] in severity_filter
        and (show_acknowledged or not a['acknowledged'])
    ]

    # Display alerts
    st.subheader(f"Alerts ({len(filtered)})")

    for alert in filtered:
        severity_colors = {
            "CRITICAL": "🔴",
            "WARNING": "🟡",
            "INFO": "🔵"
        }

        with st.expander(
            f"{severity_colors.get(alert['severity'], '⚪')} "
            f"{alert['type']} - {alert['sensor_id']}"
        ):
            st.write(f"**Message:** {alert['message']}")
            st.write(f"**Time:** {alert['timestamp']}")
            st.write(f"**Status:** {'Acknowledged' if alert['acknowledged'] else 'Active'}")

            if not alert['acknowledged']:
                if st.button(f"Acknowledge", key=f"ack_{alert['id']}"):
                    st.success("Alert acknowledged!")


def show_system_health():
    """System health monitoring view."""
    st.title("System Health")

    # Agent status
    st.subheader("Agent Status")

    agents = {
        "Monitor Agent": {"status": "Running", "messages": 1523, "errors": 0},
        "Predictor Agent": {"status": "Running", "messages": 847, "errors": 2},
        "Decision Agent": {"status": "Running", "messages": 234, "errors": 0},
        "Alert Agent": {"status": "Running", "messages": 156, "errors": 0}
    }

    cols = st.columns(len(agents))
    for i, (name, info) in enumerate(agents.items()):
        with cols[i]:
            status_color = "green" if info['status'] == "Running" else "red"
            st.markdown(f"**{name}**")
            st.markdown(f"Status: :{status_color}[{info['status']}]")
            st.metric("Messages", info['messages'])
            st.metric("Errors", info['errors'])

    st.markdown("---")

    # System metrics
    st.subheader("System Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Message Bus Throughput", "~100 msg/sec")
        st.metric("Total Messages Today", "45,231")

    with col2:
        st.metric("Prediction Accuracy (30min)", "87%")
        st.metric("Prediction Accuracy (60min)", "79%")

    with col3:
        st.metric("Uptime", "99.9%")
        st.metric("Database Size", "245 MB")

    # MLflow integration info
    st.subheader("MLflow Experiment Tracking")
    st.info("""
    **MLflow Integration:**
    - Experiment tracking for model training
    - Model versioning and registry
    - Hyperparameter optimization logs

    Access MLflow UI at: `http://localhost:5000`
    """)


if __name__ == "__main__":
    main()
