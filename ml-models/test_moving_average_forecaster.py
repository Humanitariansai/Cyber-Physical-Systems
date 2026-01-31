"""Unit tests for Moving Average Forecaster."""

import unittest
import sys
import importlib.util
from pathlib import Path

ML_DIR = Path(__file__).parent.absolute()
spec = importlib.util.spec_from_file_location("basic_forecaster", ML_DIR / "basic_forecaster.py")
bf = importlib.util.module_from_spec(spec)
sys.modules["basic_forecaster"] = bf
spec.loader.exec_module(bf)


class TestMovingAverage(unittest.TestCase):
    def setUp(self):
        self.model = bf.MovingAverageForecaster(window_size=5)

    def test_train_sets_history(self):
        data = [5.0, 5.1, 5.2, 4.9, 5.0, 5.3]
        self.model.train(data)
        self.assertEqual(len(self.model.history), 6)
        self.assertTrue(self.model.is_trained)

    def test_prediction_is_average(self):
        data = [4.0, 5.0, 6.0, 5.0, 5.0]
        self.model.train(data)
        pred, _ = self.model.predict(30)
        expected = sum(data) / len(data)
        self.assertAlmostEqual(pred, expected, places=1)

    def test_update_extends_history(self):
        data = [5.0, 5.1, 5.2, 4.9, 5.0]
        self.model.train(data)
        self.model.update(5.5)
        self.assertEqual(len(self.model.history), 6)

    def test_confidence_decreases_with_horizon(self):
        data = [5.0, 5.1, 5.2, 4.9, 5.0, 5.1, 5.0, 5.2, 5.1, 5.0]
        self.model.train(data)
        _, conf_30 = self.model.predict(30)
        _, conf_60 = self.model.predict(60)
        self.assertGreater(conf_30, conf_60)

    def test_window_size_respected(self):
        model_large = bf.MovingAverageForecaster(window_size=10)
        data = list(range(20))
        model_large.train(data)
        pred, _ = model_large.predict(30)
        expected = sum(data[-10:]) / 10
        self.assertAlmostEqual(pred, expected, places=1)


if __name__ == "__main__":
    unittest.main()
