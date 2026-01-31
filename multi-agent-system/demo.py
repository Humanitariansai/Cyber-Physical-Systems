"""
Multi-Agent System Demo - Standalone demonstration of the cold chain
monitoring agents processing simulated sensor data.
"""

import asyncio
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
import random

AGENT_DIR = Path(__file__).parent.absolute()


def load_mod(name, filepath):
    """Load module from file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load modules
message_bus = load_mod("message_bus", AGENT_DIR / "message_bus.py")
monitor_agent = load_mod("monitor_agent", AGENT_DIR / "monitor_agent.py")
predictor_agent = load_mod("predictor_agent", AGENT_DIR / "predictor_agent.py")
decision_agent = load_mod("decision_agent", AGENT_DIR / "decision_agent.py")
alert_agent = load_mod("alert_agent", AGENT_DIR / "alert_agent.py")
orchestrator = load_mod("orchestrator", AGENT_DIR / "orchestrator.py")

MessageBus = message_bus.MessageBus
MonitorAgent = monitor_agent.MonitorAgent
MonitorAgentConfig = monitor_agent.MonitorAgentConfig
PredictorAgent = predictor_agent.PredictorAgent
PredictorAgentConfig = predictor_agent.PredictorAgentConfig
DecisionAgent = decision_agent.DecisionAgent
DecisionAgentConfig = decision_agent.DecisionAgentConfig
AlertAgent = alert_agent.AlertAgent
AlertAgentConfig = alert_agent.AlertAgentConfig
SystemOrchestrator = orchestrator.SystemOrchestrator


async def run_demo():
    """Run a 10-second multi-agent demo."""
    print("=" * 50)
    print("MULTI-AGENT COLD CHAIN DEMO")
    print("=" * 50)

    bus = MessageBus()

    monitor = MonitorAgent("monitor-01", bus, MonitorAgentConfig())
    predictor = PredictorAgent("predictor-01", bus, PredictorAgentConfig())
    decision = DecisionAgent("decision-01", bus, DecisionAgentConfig())
    alert = AlertAgent("alert-01", bus, AlertAgentConfig())

    orch = SystemOrchestrator(bus)
    orch.add_agent("monitor", monitor)
    orch.add_agent("predictor", predictor)
    orch.add_agent("decision", decision)
    orch.add_agent("alert", alert)

    await orch.start()
    print("\n[Started] All agents running\n")

    for i in range(10):
        # Normal temp with gradual rise on sensor-02
        temp_01 = 5.0 + random.gauss(0, 0.2)
        temp_02 = 5.0 + (i * 0.5) + random.gauss(0, 0.1)

        await orch.inject_sensor_data("sensor-01", temp_01, 65.0)
        await orch.inject_sensor_data("sensor-02", temp_02, 62.0)

        status_02 = "OK" if temp_02 < 8 else "WARNING" if temp_02 < 10 else "CRITICAL"
        print(f"  T+{i}s | sensor-01: {temp_01:.1f}C | sensor-02: {temp_02:.1f}C [{status_02}]")

        await asyncio.sleep(1)

    await orch.stop()

    print("\n" + "=" * 50)
    print("DEMO COMPLETE")
    status = orch.get_system_status()
    for name, data in status.items():
        msgs = data.get("messages_processed", 0)
        print(f"  {name}: {msgs} messages processed")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_demo())
