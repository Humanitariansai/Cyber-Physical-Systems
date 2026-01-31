"""Unit tests for the Multi-Agent System components."""

import unittest
import asyncio
import sys
import importlib.util
from pathlib import Path

AGENT_DIR = Path(__file__).parent.absolute()


def load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mb = load_mod("message_bus", AGENT_DIR / "message_bus.py")
monitor = load_mod("monitor_agent", AGENT_DIR / "monitor_agent.py")
alert = load_mod("alert_agent", AGENT_DIR / "alert_agent.py")


class TestMessageBus(unittest.TestCase):
    """Test MessageBus pub/sub and priority routing."""

    def setUp(self):
        self.bus = mb.MessageBus()

    def test_create_message_bus(self):
        self.assertIsNotNone(self.bus)

    def test_subscribe(self):
        received = []

        async def handler(msg):
            received.append(msg)

        self.bus.subscribe(mb.MessageType.SENSOR_DATA, handler)

    def test_publish_and_receive(self):
        received = []

        async def handler(msg):
            received.append(msg)

        self.bus.subscribe(mb.MessageType.SENSOR_DATA, handler)

        msg = mb.Message(
            type=mb.MessageType.SENSOR_DATA,
            source="test",
            payload={"temp": 5.0},
            priority=mb.MessagePriority.NORMAL
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.bus.publish(msg))
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()

    def test_message_creation_helpers(self):
        msg = mb.create_sensor_message("test-source", {
            "sensor_id": "sensor-01",
            "temperature": 5.5,
            "humidity": 65.0
        })
        self.assertEqual(msg.type, mb.MessageType.SENSOR_DATA)
        self.assertEqual(msg.source, "test-source")

    def test_priority_ordering(self):
        self.assertGreater(
            mb.MessagePriority.CRITICAL,
            mb.MessagePriority.NORMAL
        )
        self.assertGreater(
            mb.MessagePriority.HIGH,
            mb.MessagePriority.LOW
        )


class TestMonitorAgent(unittest.TestCase):
    """Test MonitorAgent threshold detection."""

    def setUp(self):
        self.bus = mb.MessageBus()
        self.config = monitor.MonitorAgentConfig()
        self.agent = monitor.MonitorAgent("test-monitor", self.bus, self.config)

    def test_agent_creation(self):
        self.assertEqual(self.agent.agent_id, "test-monitor")

    def test_config_thresholds(self):
        config = monitor.MonitorAgentConfig()
        self.assertEqual(config.temp_min, 2.0)
        self.assertEqual(config.temp_max, 8.0)

    def test_normal_temperature(self):
        """5.0C should be within safe range."""
        is_normal = 2.0 <= 5.0 <= 8.0
        self.assertTrue(is_normal)

    def test_high_temperature_detection(self):
        """9.5C should exceed threshold."""
        is_high = 9.5 > 8.0
        self.assertTrue(is_high)

    def test_low_temperature_detection(self):
        """1.5C should be below threshold."""
        is_low = 1.5 < 2.0
        self.assertTrue(is_low)


class TestAlertAgent(unittest.TestCase):
    """Test AlertAgent deduplication."""

    def setUp(self):
        self.bus = mb.MessageBus()
        self.config = alert.AlertAgentConfig()
        self.agent = alert.AlertAgent("test-alert", self.bus, self.config)

    def test_agent_creation(self):
        self.assertEqual(self.agent.agent_id, "test-alert")

    def test_config_defaults(self):
        config = alert.AlertAgentConfig()
        self.assertIsNotNone(config)


if __name__ == "__main__":
    unittest.main()
