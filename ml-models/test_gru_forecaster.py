"""Unit tests for GRU Forecaster."""

import unittest
import sys
import importlib.util
from pathlib import Path

ML_DIR = Path(__file__).parent.absolute()
spec = importlib.util.spec_from_file_location("gru_forecaster", ML_DIR / "gru_forecaster.py")
gf = importlib.util.module_from_spec(spec)
sys.modules["gru_forecaster"] = gf
spec.loader.exec_module(gf)


class TestGRUConfig(unittest.TestCase):
    def test_default_config(self):
        config = gf.GRUConfig()
        self.assertEqual(config.sequence_length, 60)
        self.assertIn(30, config.prediction_horizons)

    def test_custom_config(self):
        config = gf.GRUConfig(sequence_length=45, gru_units=[32, 16])
        self.assertEqual(config.sequence_length, 45)


class TestGRUForecaster(unittest.TestCase):
    def setUp(self):
        self.model = gf.GRUForecaster()
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

    def test_predict_60min(self):
        self.model.train(self.data)
        pred, conf = self.model.predict(60)
        self.assertIsInstance(pred, float)

    def test_predict_without_train(self):
        with self.assertRaises(ValueError):
            self.model.predict(30)


if __name__ == "__main__":
    unittest.main()
