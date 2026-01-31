"""ML Models page for Cold Chain Dashboard."""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Models", layout="wide")
st.title("Machine Learning Models")

# Model information cards
st.subheader("Forecasting Models")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### LSTM")
    st.markdown("""
    **Long Short-Term Memory**
    - Architecture: 2-layer (64, 32 units)
    - Input: 60 timesteps
    - Horizons: 30, 60 minutes
    - Dropout: 20%
    - Best for: Sequential patterns
    """)
    st.metric("30-min MAE", "0.31 C")
    st.metric("60-min MAE", "0.58 C")

with col2:
    st.markdown("### GRU")
    st.markdown("""
    **Gated Recurrent Unit**
    - Architecture: 64 units (bidirectional)
    - Input: 60 timesteps
    - Horizons: 30, 60 minutes
    - 40% faster training than LSTM
    - Best for: Fast inference
    """)
    st.metric("30-min MAE", "0.35 C")
    st.metric("60-min MAE", "0.63 C")

with col3:
    st.markdown("### XGBoost")
    st.markdown("""
    **Gradient Boosting**
    - Features: Lag, rolling stats, time
    - Estimators: 100
    - Max depth: 6
    - Best for: Interpretability
    - Feature importance available
    """)
    st.metric("30-min MAE", "0.42 C")
    st.metric("60-min MAE", "0.74 C")

st.markdown("---")

# Comparison table
st.subheader("Model Comparison")
comparison = pd.DataFrame({
    "Model": ["LSTM", "GRU", "XGBoost", "Ensemble", "Baseline (MA+ES)"],
    "MAE 30min": [0.31, 0.35, 0.42, 0.28, 0.48],
    "MAE 60min": [0.58, 0.63, 0.74, 0.55, 0.85],
    "RMSE 30min": [0.38, 0.41, 0.51, 0.34, 0.56],
    "Confidence 30min": ["87%", "84%", "80%", "89%", "72%"],
    "Training Time": ["~2 min", "~1.2 min", "~10 sec", "~3.5 min", "<1 sec"],
    "Ensemble Weight": ["0.40", "0.35", "0.25", "-", "-"],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

# Bar chart comparison
st.subheader("Accuracy Comparison")
chart_data = pd.DataFrame({
    "Model": ["LSTM", "GRU", "XGBoost", "Ensemble"],
    "30-min MAE": [0.31, 0.35, 0.42, 0.28],
    "60-min MAE": [0.58, 0.63, 0.74, 0.55],
}).set_index("Model")
st.bar_chart(chart_data)

# MLflow info
st.subheader("MLflow Experiment Tracking")
st.info("""
**MLflow tracks all model training runs:**
- Parameters: learning rate, epochs, architecture config
- Metrics: MAE, RMSE at each prediction horizon
- Artifacts: Trained model files, prediction plots
- Model Registry: Version control for production models

Access MLflow UI: `mlflow ui --backend-store-uri mlruns` then open http://localhost:5000
""")
