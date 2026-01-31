"""
Quick MLflow Test - Verify MLflow tracking works with a dummy experiment.
"""

import sys
import importlib.util
from pathlib import Path

ML_DIR = Path(__file__).parent.absolute()


def load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_mlflow():
    tracker_mod = load_mod("mlflow_tracking", ML_DIR / "mlflow_tracking.py")
    tracker = tracker_mod.MLflowTracker()

    print("MLflow Quick Test")
    print(f"  MLflow available: {tracker.is_available}")

    run_id = tracker.start_run(run_name="quick-test")
    print(f"  Run ID: {run_id}")

    tracker.log_params({
        "model_type": "test",
        "data_points": 100,
        "test_run": True
    })

    tracker.log_metrics({
        "mae_30min": 0.35,
        "rmse_30min": 0.42,
        "mae_60min": 0.65,
        "rmse_60min": 0.78,
    })

    summary = tracker.end_run()
    print(f"  Status: {summary['status']}")
    print(f"  Metrics logged: {len(summary['metrics'])}")
    print(f"  Params logged: {len(summary['params'])}")
    print("\nMLflow test PASSED")


if __name__ == "__main__":
    test_mlflow()
