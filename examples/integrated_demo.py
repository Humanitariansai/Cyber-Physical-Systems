"""
Example: Integrated Multi-Agent System with Database Logging and MCP Server

This demonstrates how to:
1. Initialize the database
2. Run the multi-agent system with database logging
3. Query the data through the MCP server

Note: This demo uses importlib to handle hyphenated directory names.
"""

import asyncio
import sys
import importlib.util
from pathlib import Path
from datetime import datetime

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def load_module_from_path(module_name: str, file_path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load modules from hyphenated directories
multi_agent_dir = PROJECT_ROOT / "multi-agent-system"
mcp_server_dir = PROJECT_ROOT / "mcp-server"

# Load multi-agent system modules
message_bus = load_module_from_path(
    "message_bus",
    multi_agent_dir / "message_bus.py"
)
agent_base = load_module_from_path(
    "agent_base",
    multi_agent_dir / "agent_base.py"
)
monitor_agent = load_module_from_path(
    "monitor_agent",
    multi_agent_dir / "monitor_agent.py"
)
predictor_agent = load_module_from_path(
    "predictor_agent",
    multi_agent_dir / "predictor_agent.py"
)
decision_agent = load_module_from_path(
    "decision_agent",
    multi_agent_dir / "decision_agent.py"
)
alert_agent = load_module_from_path(
    "alert_agent",
    multi_agent_dir / "alert_agent.py"
)
orchestrator = load_module_from_path(
    "orchestrator",
    multi_agent_dir / "orchestrator.py"
)

# Load MCP server modules
db_logger_mod = load_module_from_path(
    "db_logger",
    mcp_server_dir / "db_logger.py"
)
initialize_db = load_module_from_path(
    "initialize_db",
    mcp_server_dir / "initialize_db.py"
)

# Import classes
MessageBus = message_bus.MessageBus
MessageType = message_bus.MessageType
MessagePriority = message_bus.MessagePriority
MonitorAgent = monitor_agent.MonitorAgent
MonitorAgentConfig = monitor_agent.MonitorAgentConfig
PredictorAgent = predictor_agent.PredictorAgent
PredictorAgentConfig = predictor_agent.PredictorAgentConfig
DecisionAgent = decision_agent.DecisionAgent
DecisionAgentConfig = decision_agent.DecisionAgentConfig
AlertAgent = alert_agent.AlertAgent
AlertAgentConfig = alert_agent.AlertAgentConfig
SystemOrchestrator = orchestrator.SystemOrchestrator
DatabaseLogger = db_logger_mod.DatabaseLogger
initialize_database = initialize_db.initialize_database


async def run_integrated_demo():
    """Run multi-agent system with database logging."""

    print("=" * 60)
    print("INTEGRATED COLD CHAIN MONITORING DEMO")
    print("Multi-Agent System + Database Logging + MCP Server Ready")
    print("=" * 60)

    # Step 1: Initialize database
    print("\n[1/4] Initializing database...")
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    db_path = str(data_dir / "cold_chain.db")
    initialize_database(db_path)

    # Step 2: Create database logger
    print("\n[2/4] Creating database logger...")
    db_logger = DatabaseLogger(db_path)

    # Step 3: Set up multi-agent system
    print("\n[3/4] Starting multi-agent system...")

    # Create message bus
    bus = MessageBus()

    # Create agents with database logging
    monitor = MonitorAgent(
        agent_id="monitor-01",
        bus=bus,
        config=MonitorAgentConfig()
    )

    predictor = PredictorAgent(
        agent_id="predictor-01",
        bus=bus,
        config=PredictorAgentConfig()
    )

    decision = DecisionAgent(
        agent_id="decision-01",
        bus=bus,
        config=DecisionAgentConfig()
    )

    alert = AlertAgent(
        agent_id="alert-01",
        bus=bus,
        config=AlertAgentConfig()
    )

    # Create orchestrator
    orch = SystemOrchestrator(bus)
    orch.add_agent("monitor", monitor)
    orch.add_agent("predictor", predictor)
    orch.add_agent("decision", decision)
    orch.add_agent("alert", alert)

    # Start system
    await orch.start()

    # Step 4: Run simulation with database logging
    print("\n[4/4] Running 20-second simulation with database logging...")
    print("-" * 60)

    # Simulate sensor data with logging
    sensors = ["sensor-01", "sensor-02", "sensor-03"]

    for i in range(20):
        timestamp = datetime.now().isoformat()

        for sensor_id in sensors:
            # Normal temperature for most sensors
            if sensor_id == "sensor-02" and i > 10:
                # Sensor 2 rises gradually after 10 seconds
                temp = 5.0 + (i - 10) * 0.4
            else:
                temp = 5.0 + (i * 0.05)  # Slight increase

            humidity = 65.0

            # Log to database
            db_logger.log_sensor_data(sensor_id, temp, humidity, timestamp)

            # Send to multi-agent system
            await orch.inject_sensor_data(sensor_id, temp, humidity)

        await asyncio.sleep(1)

        if i == 10:
            print(f"  T+{i}s: Sensor-02 beginning temperature rise...")
        elif i == 15:
            print(f"  T+{i}s: Predictor should detect trend and issue warning...")

    print("-" * 60)

    # Log final system status
    print("\n[Status] Logging final agent metrics to database...")
    status = orch.get_system_status()

    for agent_name, agent_data in status.items():
        db_logger.log_agent_status(
            agent_id=agent_data.get('agent_id', agent_name),
            agent_name=agent_name,
            status=agent_data.get('state', 'unknown'),
            messages_processed=agent_data.get('messages_processed', 0),
            errors_count=agent_data.get('errors_count', 0),
            timestamp=datetime.now().isoformat()
        )

    # Stop system
    print("\n[Shutdown] Stopping multi-agent system...")
    await orch.stop()

    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nDatabase location: {db_path}")
    print("\nYou can now:")
    print("1. Query the database using MCP server tools")
    print("2. Use Claude Desktop to ask questions like:")
    print("   - 'What alerts were triggered in the simulation?'")
    print("   - 'Show me sensor-02 temperature trend'")
    print("   - 'Analyze the incidents that occurred'")
    print("\nTo start MCP server:")
    print("  python mcp-server/server.py")
    print("\nTo run Streamlit dashboard:")
    print("  streamlit run streamlit-dashboard/app.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(run_integrated_demo())
