"""
Data Analytics Page
=================

This page provides detailed data analytics and visualization capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add module paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_collection.data_collector import DataCollector

# Page config
st.set_page_config(
    page_title="Data Analytics",
    page_icon="📊",
    layout="wide"
)

# Initialize components
@st.cache_resource
def get_data_collector():
    return DataCollector()

data_collector = get_data_collector()

# Main content
st.title("Data Analytics")
st.write("""
Analyze and visualize collected data from various sources.
""")

# Data source selection
st.sidebar.header("Settings")
available_sources = [
    d.stem for d in (project_root / "data-collection/data/raw").glob("*.csv")
]

if available_sources:
    selected_source = st.sidebar.selectbox(
        "Select Data Source",
        available_sources
    )
    
    # Load data
    try:
        df = pd.read_csv(
            project_root / "data-collection/data/raw" / f"{selected_source}.csv"
        )
        
        # Data overview
        st.header("Data Overview")
        st.write("Shape:", df.shape)
        st.dataframe(df.head())
        
        # Time series visualization
        if 'timestamp' in df.columns and 'value' in df.columns:
            st.header("Time Series Analysis")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['value'],
                mode='lines+markers',
                name='Value'
            ))
            fig.update_layout(
                title="Time Series Data",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                showlegend=True
            )
            st.plotly_chart(fig)
            
            # Statistics
            st.header("Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Basic Statistics")
                st.write(df['value'].describe())
                
            with col2:
                st.write("Rolling Statistics")
                window = st.slider("Rolling Window Size", 2, 20, 5)
                rolling_mean = df['value'].rolling(window=window).mean()
                rolling_std = df['value'].rolling(window=window).std()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=df['value'],
                    name='Original'
                ))
                fig.add_trace(go.Scatter(
                    y=rolling_mean,
                    name=f'Rolling Mean (w={window})',
                    line=dict(dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    y=rolling_std,
                    name=f'Rolling Std (w={window})',
                    line=dict(dash='dot')
                ))
                st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.info("""
    No data sources available. Please collect some data first!
    
    Go to the main page and use the data collection interface
    to add data points.
    """)