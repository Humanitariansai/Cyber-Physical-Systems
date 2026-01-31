"""Sidebar component for Cold Chain Dashboard."""

import streamlit as st
from typing import Dict


def render_sidebar() -> Dict:
    """Render sidebar and return user selections."""
    st.sidebar.title("Cold Chain Monitor")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Sensor Data", "Predictions", "Alerts", "System Health"]
    )

    st.sidebar.markdown("---")

    sensors = ["sensor-01", "sensor-02", "sensor-03"]
    selected_sensor = st.sidebar.selectbox("Select Sensor", sensors)

    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"]
    )

    hours_map = {
        "Last Hour": 1, "Last 6 Hours": 6,
        "Last 24 Hours": 24, "Last 7 Days": 168
    }

    st.sidebar.markdown("---")
    st.sidebar.caption("Predictive Cold Chain Monitoring v1.0")

    return {
        "page": page,
        "sensor": selected_sensor,
        "time_range": time_range,
        "hours": hours_map[time_range],
    }
