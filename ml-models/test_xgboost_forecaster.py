"""Unit tests for XGBoost Forecaster."""

import unittest
import sys
import importlib.util
from pathlib import Path

ML_DIR = Path(__file__).parent.absolute()
spec = importlib.util.spec_from_file_location("xgboost_forecaster", ML_DIR / "xgboost_forecaster.py")
xf = importlib.util.module_from_spec(spec)
sys.modules["xgboost_forecaster"] = xf
spec.loader.exec_module(xf)


class TestXGBoostConfig(unittest.TestCase):
    def test_default_config(self):
        config = xf.XGBoostConfig()
        self.assertIn(30, config.prediction_horizons)
        self.assertIn(60, config.prediction_horizons)

    def test_custom_config(self):
        config = xf.XGBoostConfig(n_estimators=200, max_depth=8)
        self.assertEqual(config.n_estimators, 200)
        self.assertEqual(config.max_depth, 8)


class TestXGBoostForecaster(unittest.TestCase):
    def setUp(self):
        self.model = xf.XGBoostForecaster()
        self.data = [5.0 + (i % 10) * 0.1 for i in range(200)]

    def test_train(self):
        result = self.model.train(self.data)
        self.assertIn("status", result)
        self.assertTrue(self.model.is_trained)

    def test_predict_after_train(self):
        self.model.train(self.data)
        pred, conf = self.model.predict(30)
        self.assertIsInstance(pred, float)
        self.assertGreater(conf, 0)

    def test_predict_without_train(self):
        with self.assertRaises(ValueError):
            self.model.predict(30)

    def test_feature_importance(self):
        self.model.train(self.data)
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            self.assertIsInstance(importance, dict)


if __name__ == "__main__":
    unittest.main()
