"""Unit tests for BasicForecaster, MovingAverage, and ExponentialSmoothing."""

import unittest
import sys
import importlib.util
from pathlib import Path

ML_DIR = Path(__file__).parent.absolute()
spec = importlib.util.spec_from_file_location("basic_forecaster", ML_DIR / "basic_forecaster.py")
bf = importlib.util.module_from_spec(spec)
sys.modules["basic_forecaster"] = bf
spec.loader.exec_module(bf)


class TestMovingAverageForecaster(unittest.TestCase):
    def setUp(self):
        self.model = bf.MovingAverageForecaster(window_size=5)
        self.data = [5.0, 5.1, 5.2, 4.9, 5.0, 5.3, 5.1, 5.0, 4.8, 5.2]

    def test_train(self):
        self.model.train(self.data)
        self.assertTrue(self.model.is_trained)

    def test_predict_returns_tuple(self):
        self.model.train(self.data)
        result = self.model.predict(30)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_predict_in_range(self):
        self.model.train(self.data)
        pred, conf = self.model.predict(30)
        self.assertGreater(pred, 3.0)
        self.assertLess(pred, 8.0)
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_insufficient_data(self):
        self.model.train([5.0])
        with self.assertRaises(ValueError):
            self.model.predict(30)


class TestExponentialSmoothingForecaster(unittest.TestCase):
    def setUp(self):
        self.model = bf.ExponentialSmoothingForecaster(alpha=0.3)
        self.data = [5.0, 5.1, 5.2, 4.9, 5.0, 5.3, 5.1, 5.0, 4.8, 5.2]

    def test_train(self):
        self.model.train(self.data)
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.smoothed)

    def test_predict(self):
        self.model.train(self.data)
        pred, conf = self.model.predict(30)
        self.assertIsInstance(pred, float)
        self.assertGreater(conf, 0.0)

    def test_update(self):
        self.model.train(self.data)
        old = self.model.smoothed
        self.model.update(6.0)
        self.assertNotEqual(self.model.smoothed, old)


class TestBasicForecaster(unittest.TestCase):
    def setUp(self):
        self.model = bf.BasicForecaster()
        self.data = [5.0 + i * 0.01 for i in range(100)]

    def test_train_and_predict(self):
        self.model.train(self.data)
        self.assertTrue(self.model.is_trained)
        pred, conf = self.model.predict(30)
        self.assertIsInstance(pred, float)

    def test_predict_all_horizons(self):
        self.model.train(self.data)
        results = self.model.predict_all_horizons()
        self.assertIn(30, results)
        self.assertIn(60, results)


if __name__ == "__main__":
    unittest.main()
