"""Unit tests for LSTM Forecaster."""

import unittest
import sys
import importlib.util
from pathlib import Path

ML_DIR = Path(__file__).parent.absolute()
spec = importlib.util.spec_from_file_location("lstm_forecaster", ML_DIR / "lstm_forecaster.py")
lf = importlib.util.module_from_spec(spec)
sys.modules["lstm_forecaster"] = lf
spec.loader.exec_module(lf)


class TestLSTMConfig(unittest.TestCase):
    def test_default_config(self):
        config = lf.LSTMConfig()
        self.assertEqual(config.sequence_length, 60)
        self.assertIn(30, config.prediction_horizons)
        self.assertIn(60, config.prediction_horizons)

    def test_custom_config(self):
        config = lf.LSTMConfig(sequence_length=30, prediction_horizons=(15, 30))
        self.assertEqual(config.sequence_length, 30)


class TestLSTMForecaster(unittest.TestCase):
    def setUp(self):
        self.model = lf.LSTMForecaster()
        self.data = [5.0 + (i % 10) * 0.1 for i in range(200)]

    def test_train(self):
        result = self.model.train(self.data)
        self.assertIn("status", result)
        self.assertTrue(self.model.is_trained)

    def test_predict_after_train(self):
        self.model.train(self.data)
        pred, conf = self.model.predict(30)
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0)
        self.assertLess(pred, 20)
        self.assertGreater(conf, 0)
        self.assertLessEqual(conf, 1.0)

    def test_predict_without_train(self):
        with self.assertRaises(ValueError):
            self.model.predict(30)

    def test_fallback_mode(self):
        """Test that fallback works when TensorFlow is unavailable."""
        self.model.train(self.data)
        pred, conf = self.model.predict(60)
        self.assertIsInstance(pred, float)


if __name__ == "__main__":
    unittest.main()
