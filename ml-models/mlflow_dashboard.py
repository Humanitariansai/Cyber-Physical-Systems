"""
MLflow Results Dashboard - Streamlit mini-page for viewing experiment results.
"""

import streamlit as st
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def show_mlflow_dashboard():
    """Render MLflow experiment results in Streamlit."""
    st.title("MLflow Experiment Results")

    if not MLFLOW_AVAILABLE:
        st.warning("MLflow is not installed. Install with: pip install mlflow")
        show_sample_results()
        return

    try:
        mlflow.set_tracking_uri("mlruns")
        experiment = mlflow.get_experiment_by_name("cold-chain-forecasting")
        if experiment is None:
            st.info("No experiments found. Run mlflow_experiment_runner.py first.")
            show_sample_results()
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) == 0:
            st.info("No runs found.")
            return

        st.subheader("Experiment Runs")
        display_cols = [c for c in runs.columns if c.startswith("params.") or c.startswith("metrics.")]
        st.dataframe(runs[["run_id", "status"] + display_cols], use_container_width=True)

        st.subheader("Metrics Comparison")
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        if metric_cols:
            chart_data = runs[metric_cols].rename(columns=lambda c: c.replace("metrics.", ""))
            st.bar_chart(chart_data)

    except Exception as e:
        st.error(f"Error loading MLflow data: {e}")
        show_sample_results()


def show_sample_results():
    """Show sample experiment results for demonstration."""
    st.subheader("Sample Experiment Results")
    data = pd.DataFrame({
        "Model": ["LSTM", "GRU", "XGBoost", "BasicForecaster"],
        "MAE (30min)": [0.31, 0.35, 0.42, 0.48],
        "MAE (60min)": [0.58, 0.63, 0.74, 0.85],
        "RMSE (30min)": [0.38, 0.41, 0.51, 0.56],
        "Confidence (30min)": ["87%", "84%", "80%", "72%"],
    })
    st.dataframe(data, use_container_width=True)

    st.subheader("Model Performance Comparison")
    chart_df = pd.DataFrame({
        "Model": ["LSTM", "GRU", "XGBoost", "Baseline"],
        "30-min MAE": [0.31, 0.35, 0.42, 0.48],
        "60-min MAE": [0.58, 0.63, 0.74, 0.85],
    }).set_index("Model")
    st.bar_chart(chart_df)


if __name__ == "__main__":
    show_mlflow_dashboard()
