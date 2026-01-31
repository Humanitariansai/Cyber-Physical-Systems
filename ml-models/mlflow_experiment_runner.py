"""
MLflow Experiment Runner - Trains all forecasting models and logs results.
"""

import sys
import importlib.util
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def generate_training_data(n=500):
    np.random.seed(42)
    t = np.arange(n)
    temps = 5.0 + 0.5 * np.sin(2 * np.pi * t / 1440) + np.random.normal(0, 0.3, n)
    humidity = 55.0 + np.random.normal(0, 3, n)
    return temps.tolist(), humidity.tolist()


def run_experiments():
    ml_dir = PROJECT_ROOT / "ml-models"
    tracker_mod = load_mod("mlflow_tracking", ml_dir / "mlflow_tracking.py")
    basic_mod = load_mod("basic_forecaster", ml_dir / "basic_forecaster.py")
    lstm_mod = load_mod("lstm_forecaster", ml_dir / "lstm_forecaster.py")
    gru_mod = load_mod("gru_forecaster", ml_dir / "gru_forecaster.py")
    xgb_mod = load_mod("xgboost_forecaster", ml_dir / "xgboost_forecaster.py")

    tracker = tracker_mod.MLflowTracker()
    temps, humidity = generate_training_data()

    models = {
        "BasicForecaster": basic_mod.BasicForecaster(),
        "LSTM": lstm_mod.LSTMForecaster(),
        "GRU": gru_mod.GRUForecaster(),
        "XGBoost": xgb_mod.XGBoostForecaster(),
    }

    print("=" * 60)
    print("COLD CHAIN ML EXPERIMENT RUNNER")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        run_id = tracker.start_run(run_name=name)
        tracker.log_params({"model_type": name, "data_points": len(temps)})

        try:
            result = model.train(temps)
            pred_30, conf_30 = model.predict(30)
            pred_60, conf_60 = model.predict(60)

            metrics = {
                "pred_30min": pred_30,
                "conf_30min": conf_30,
                "pred_60min": pred_60,
                "conf_60min": conf_60,
            }
            tracker.log_metrics(metrics)
            print(f"  30min: {pred_30:.2f}C (conf: {conf_30:.0%})")
            print(f"  60min: {pred_60:.2f}C (conf: {conf_60:.0%})")
        except Exception as e:
            print(f"  Error: {e}")
            tracker.log_metrics({"error": 1.0})

        summary = tracker.end_run()
        print(f"  Run ID: {summary.get('run_id', 'local')}")

    print("\n" + "=" * 60)
    print("All experiments complete.")


if __name__ == "__main__":
    run_experiments()
