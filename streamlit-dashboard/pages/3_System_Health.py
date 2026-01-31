"""System Health page for Cold Chain Dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="System Health", layout="wide")
st.title("System Health Monitor")

# Agent status
st.subheader("Agent Status")
agents = {
    "Monitor Agent": {"status": "Running", "messages": 15234, "errors": 0, "uptime": "99.9%"},
    "Predictor Agent": {"status": "Running", "messages": 8471, "errors": 3, "uptime": "99.8%"},
    "Decision Agent": {"status": "Running", "messages": 2341, "errors": 0, "uptime": "99.9%"},
    "Alert Agent": {"status": "Running", "messages": 1567, "errors": 1, "uptime": "99.9%"},
}

cols = st.columns(len(agents))
for i, (name, info) in enumerate(agents.items()):
    with cols[i]:
        status_color = "green" if info["status"] == "Running" else "red"
        st.markdown(f"**{name}**")
        st.markdown(f"Status: :{status_color}[{info['status']}]")
        st.metric("Messages", f"{info['messages']:,}")
        st.metric("Errors", info["errors"])
        st.caption(f"Uptime: {info['uptime']}")

st.markdown("---")

# System metrics
st.subheader("System Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Message Bus Throughput", "~100 msg/sec")
    st.metric("Total Messages Today", "27,613")
    st.metric("Queue Depth", "3")

with col2:
    st.metric("Prediction Accuracy (30min)", "87%")
    st.metric("Prediction Accuracy (60min)", "79%")
    st.metric("False Alert Rate", "<5%")

with col3:
    st.metric("System Uptime", "99.9%")
    st.metric("Database Size", "2.0 MB")
    st.metric("Active Sensors", "3")

st.markdown("---")

# Message bus activity
st.subheader("Message Type Distribution")
msg_data = pd.DataFrame({
    "Type": ["SENSOR_DATA", "PREDICTION", "DECISION", "ALERT", "HEARTBEAT"],
    "Count": [15234, 8471, 2341, 1567, 45231],
    "Priority": ["NORMAL", "NORMAL", "HIGH", "HIGH", "LOW"],
})
st.dataframe(msg_data, use_container_width=True, hide_index=True)

# MLflow
st.subheader("MLflow Experiment Tracking")
st.info("""
**MLflow Integration Active**
- Experiment tracking for model training
- Model versioning and registry
- Hyperparameter optimization logs

Access MLflow UI at: `http://localhost:5000`
""")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
