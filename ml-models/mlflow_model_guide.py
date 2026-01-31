"""
MLflow Integration Guide - Documentation for how MLflow is used in this project.
"""


def print_guide():
    """Print MLflow integration documentation."""
    guide = """
================================================================================
                    MLFLOW INTEGRATION GUIDE
                Cold Chain Monitoring System
================================================================================

1. OVERVIEW
-----------
MLflow tracks all model training experiments, enabling reproducibility,
comparison, and deployment of the best-performing models.

2. TRACKED MODELS
-----------------
  - LSTM Forecaster (lstm_forecaster.py)
    * Parameters: sequence_length, lstm_units, dropout_rate, epochs, batch_size
    * Metrics: MAE, RMSE at 30/60-minute horizons

  - GRU Forecaster (gru_forecaster.py)
    * Parameters: gru_units, bidirectional, recurrent_dropout
    * Metrics: MAE, RMSE at 30/60-minute horizons

  - XGBoost Forecaster (xgboost_forecaster.py)
    * Parameters: n_estimators, max_depth, learning_rate, n_lag_features
    * Metrics: MAE, RMSE, feature importance scores

  - Basic Forecaster (basic_forecaster.py)
    * Parameters: window_size, alpha (smoothing factor)
    * Metrics: MAE, RMSE (baseline comparison)

3. EXPERIMENT TRACKING
----------------------
  Start tracking:
    from mlflow_tracking import MLflowTracker
    tracker = MLflowTracker()
    tracker.start_run(run_name="lstm-v2")
    tracker.log_params({"epochs": 100, "lstm_units": [64, 32]})
    tracker.log_metrics({"mae_30min": 0.31, "rmse_30min": 0.38})
    tracker.end_run()

4. RUNNING EXPERIMENTS
----------------------
  python ml-models/mlflow_experiment_runner.py

5. VIEWING RESULTS
------------------
  Option A: MLflow UI
    mlflow ui --backend-store-uri mlruns
    Open: http://localhost:5000

  Option B: Streamlit Dashboard
    Navigate to ML Models page in the dashboard

6. MODEL REGISTRY
-----------------
  Best models are automatically registered for production deployment.
  Use tracker.get_best_run("mae_30min") to retrieve the best model.

7. DIRECTORY STRUCTURE
----------------------
  mlruns/                    # MLflow tracking directory
    0/                       # Default experiment
      <run-id>/              # Individual run
        artifacts/           # Model files, plots
        metrics/             # Logged metrics
        params/              # Logged parameters
        tags/                # Run tags

================================================================================
"""
    print(guide)


if __name__ == "__main__":
    print_guide()
