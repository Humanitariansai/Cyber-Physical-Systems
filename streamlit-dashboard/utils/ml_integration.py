"""ML model integration for the Cold Chain Dashboard."""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
ML_DIR = PROJECT_ROOT / "ml-models"


def _load_mod(name: str, filepath: Path):
    """Load module from file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_forecaster(model_type: str = "lstm"):
    """Load a forecaster model by type."""
    model_map = {
        "lstm": ("lstm_forecaster", "LSTMForecaster"),
        "gru": ("gru_forecaster", "GRUForecaster"),
        "xgboost": ("xgboost_forecaster", "XGBoostForecaster"),
        "basic": ("basic_forecaster", "BasicForecaster"),
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    module_name, class_name = model_map[model_type]
    mod = _load_mod(module_name, ML_DIR / f"{module_name}.py")
    forecaster_class = getattr(mod, class_name)
    return forecaster_class()


def generate_prediction(model, data: List[float],
                        horizon: int = 30) -> Tuple[float, float]:
    """Train model on data and generate prediction."""
    model.train(data)
    return model.predict(horizon)


def get_model_info() -> Dict[str, Dict]:
    """Return information about available models."""
    return {
        "LSTM": {
            "name": "Long Short-Term Memory",
            "architecture": "2-layer (64, 32 units)",
            "horizons": [30, 60],
            "training_time": "~2 min",
            "weight": 0.4,
        },
        "GRU": {
            "name": "Gated Recurrent Unit",
            "architecture": "64 units (bidirectional)",
            "horizons": [30, 60],
            "training_time": "~1.2 min",
            "weight": 0.35,
        },
        "XGBoost": {
            "name": "Gradient Boosting",
            "architecture": "100 estimators, depth 6",
            "horizons": [30, 60],
            "training_time": "~10 sec",
            "weight": 0.25,
        },
    }
