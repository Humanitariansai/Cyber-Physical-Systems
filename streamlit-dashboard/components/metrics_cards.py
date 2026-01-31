"""
Reusable Streamlit metric card components for the Cold Chain Dashboard.
"""

import streamlit as st
from typing import Dict, List, Optional


def render_kpi_card(label: str, value: str, delta: str = None,
                    delta_color: str = "normal") -> None:
    """Render a KPI metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_status_card(agent_name: str, status: str,
                       messages: int = 0, errors: int = 0) -> None:
    """Render an agent status card."""
    status_color = "green" if status == "running" else "red"
    st.markdown(f"**{agent_name}**")
    st.markdown(f"Status: :{status_color}[{status.upper()}]")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", f"{messages:,}")
    with col2:
        st.metric("Errors", errors)


def render_alert_summary_card(alerts: List[Dict]) -> None:
    """Render alert summary with severity breakdown."""
    critical = sum(1 for a in alerts if a.get("severity") == "CRITICAL")
    warning = sum(1 for a in alerts if a.get("severity") == "WARNING")
    info = sum(1 for a in alerts if a.get("severity") == "INFO")

    st.markdown("### Alert Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Critical", critical)
    with col2:
        st.metric("Warning", warning)
    with col3:
        st.metric("Info", info)


def render_temperature_status(temp: float, min_t: float = 2.0,
                               max_t: float = 8.0) -> None:
    """Render temperature status indicator."""
    if temp < min_t or temp > max_t:
        if temp < 0 or temp > 10:
            st.error(f"CRITICAL: {temp:.1f} C")
        else:
            st.warning(f"WARNING: {temp:.1f} C")
    else:
        st.success(f"NORMAL: {temp:.1f} C")


def render_prediction_card(horizon: int, prediction: float,
                            confidence: float) -> None:
    """Render a prediction result card."""
    st.markdown(f"### {horizon}-Minute Forecast")
    st.metric("Predicted Temperature", f"{prediction:.1f} C")
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.0%}")
