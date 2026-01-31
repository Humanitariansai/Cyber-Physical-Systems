"""
Model Comparison Script - Compare all forecasting models side by side.
Generates MAE/RMSE metrics and prints comparison table.
"""

import numpy as np
import math
import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def load_module(name, filepath):
    """Load module from file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def generate_test_data(n=500):
    """Generate test temperature data with patterns."""
    np.random.seed(42)
    t = np.arange(n)
    base = 5.0
    daily_cycle = 0.5 * np.sin(2 * np.pi * t / (24 * 60))
    noise = np.random.normal(0, 0.3, n)
    data = base + daily_cycle + noise
    return data.tolist()


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE."""
    errors = [a - p for a, p in zip(actual, predicted)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    return round(mae, 4), round(rmse, 4)


def run_comparison():
    """Run full model comparison."""
    ml_dir = PROJECT_ROOT / "ml-models"

    # Load models
    basic_mod = load_module("basic_forecaster", ml_dir / "basic_forecaster.py")
    arima_mod = load_module("simple_arima_forecaster", ml_dir / "simple_arima_forecaster.py")
    lstm_mod = load_module("lstm_forecaster", ml_dir / "lstm_forecaster.py")
    gru_mod = load_module("gru_forecaster", ml_dir / "gru_forecaster.py")
    xgb_mod = load_module("xgboost_forecaster", ml_dir / "xgboost_forecaster.py")

    data = generate_test_data(500)
    train_data = data[:400]
    test_data = data[400:]

    models = {
        "BasicForecaster": basic_mod.BasicForecaster(),
        "ARIMA": arima_mod.ARIMAForecaster(),
        "LSTM": lstm_mod.LSTMForecaster(),
        "GRU": gru_mod.GRUForecaster(),
        "XGBoost": xgb_mod.XGBoostForecaster(),
    }

    print("=" * 70)
    print("COLD CHAIN FORECASTING - MODEL COMPARISON")
    print("=" * 70)
    print(f"\nTraining data: {len(train_data)} points")
    print(f"Test data: {len(test_data)} points")
    print()

    results = {}
    for name, model in models.items():
        try:
            model.train(train_data)
            preds_30, preds_60 = [], []

            for i in range(min(20, len(test_data))):
                try:
                    p30, c30 = model.predict(30)
                    p60, c60 = model.predict(60)
                    preds_30.append(p30)
                    preds_60.append(p60)
                except Exception:
                    break

            if preds_30:
                actuals = test_data[:len(preds_30)]
                mae_30, rmse_30 = calculate_metrics(actuals, preds_30)
                mae_60, rmse_60 = calculate_metrics(actuals[:len(preds_60)], preds_60)
                results[name] = {
                    "mae_30": mae_30, "rmse_30": rmse_30,
                    "mae_60": mae_60, "rmse_60": rmse_60
                }
            else:
                results[name] = {"error": "No predictions generated"}
        except Exception as e:
            results[name] = {"error": str(e)}

    # Print results table
    print(f"{'Model':<20} {'MAE(30m)':<12} {'RMSE(30m)':<12} {'MAE(60m)':<12} {'RMSE(60m)':<12}")
    print("-" * 70)

    for name, metrics in results.items():
        if "error" in metrics:
            print(f"{name:<20} ERROR: {metrics['error']}")
        else:
            print(f"{name:<20} {metrics['mae_30']:<12} {metrics['rmse_30']:<12} "
                  f"{metrics['mae_60']:<12} {metrics['rmse_60']:<12}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_comparison()
