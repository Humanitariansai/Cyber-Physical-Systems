"""Data Analytics page for Cold Chain Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Data Analytics", layout="wide")
st.title("Sensor Data Analytics")


def generate_data(hours=24):
    """Generate sample sensor data."""
    np.random.seed(42)
    n = hours * 60
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n, 0, -1)]
    base = 5.0
    noise = np.random.normal(0, 0.3, n)
    cycle = 0.5 * np.sin(np.linspace(0, 4 * np.pi, n))
    temps = base + noise + cycle
    temps[int(n * 0.3):int(n * 0.35)] += 2.5
    humidity = 55 + np.random.normal(0, 5, n)
    return pd.DataFrame({
        "timestamp": timestamps, "temperature": temps,
        "humidity": humidity, "sensor_id": ["sensor-01"] * n
    })


hours = st.sidebar.selectbox("Time Range (hours)", [1, 6, 24, 168], index=2)
data = generate_data(hours)

# Statistics
st.subheader("Summary Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean", f"{data['temperature'].mean():.2f} C")
col2.metric("Std Dev", f"{data['temperature'].std():.2f} C")
col3.metric("Min", f"{data['temperature'].min():.2f} C")
col4.metric("Max", f"{data['temperature'].max():.2f} C")

# Temperature distribution
st.subheader("Temperature Distribution")
fig = px.histogram(data, x="temperature", nbins=50, title="Temperature Histogram")
fig.add_vline(x=2, line_dash="dash", line_color="blue", annotation_text="Min (2 C)")
fig.add_vline(x=8, line_dash="dash", line_color="orange", annotation_text="Max (8 C)")
st.plotly_chart(fig, use_container_width=True)

# Hourly average pattern
st.subheader("Hourly Average Pattern")
data["hour"] = pd.to_datetime(data["timestamp"]).dt.hour
hourly = data.groupby("hour")["temperature"].mean().reset_index()
fig2 = px.bar(hourly, x="hour", y="temperature", title="Average Temperature by Hour")
fig2.add_hline(y=2, line_dash="dot", line_color="blue")
fig2.add_hline(y=8, line_dash="dot", line_color="orange")
st.plotly_chart(fig2, use_container_width=True)

# Temperature vs Humidity correlation
st.subheader("Temperature vs Humidity Correlation")
fig3 = px.scatter(data.sample(min(500, len(data))),
                  x="temperature", y="humidity",
                  title="Temperature-Humidity Correlation",
                  opacity=0.5)
st.plotly_chart(fig3, use_container_width=True)

corr = data[["temperature", "humidity"]].corr().iloc[0, 1]
st.info(f"Pearson correlation coefficient: {corr:.3f}")

# Raw data
st.subheader("Raw Data")
st.dataframe(data.tail(100), use_container_width=True)
csv = data.to_csv(index=False)
st.download_button("Download CSV", csv, "sensor_data.csv", "text/csv")
