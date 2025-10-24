import streamlit as st

st.set_page_config(
    page_title="Home",
    layout="wide"
)

st.title("Cyber-Physical Systems Dashboard")

st.markdown("""
## Welcome to the CPS Dashboard

This dashboard provides comprehensive monitoring and analysis tools for your Cyber-Physical System:

1. **Data Analytics** - Visualize and analyze sensor data
2. **ML Models** - Train and evaluate machine learning models
3. **System Health** - Monitor system performance and health metrics

Use the sidebar to navigate between different sections of the dashboard.
""")