"""
Example: Integrated Multi-Agent System with Database Logging and MCP Server

This demonstrates how to:
1. Initialize the database
2. Run the multi-agent system with database logging
3. Query the data through the MCP server
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_agent_system.message_bus import MessageBus, MessageType, MessagePriority
from multi_agent_system.monitor_agent import MonitorAgent, MonitorAgentConfig
from multi_agent_system.predictor_agent import PredictorAgent, PredictorAgentConfig
from multi_agent_system.decision_agent import DecisionAgent, DecisionAgentConfig
from multi_agent_system.alert_agent import AlertAgent, AlertAgentConfig
from multi_agent_system.orchestrator import SystemOrchestrator
from mcp_server.initialize_db import initialize_database
from mcp_server.db_logger import DatabaseLogger
from datetime import datetime


async def run_integrated_demo():
    """Run multi-agent system with database logging."""
    
    print("=" * 60)
    print("INTEGRATED COLD CHAIN MONITORING DEMO")
    print("Multi-Agent System + Database Logging + MCP Server Ready")
    print("=" * 60)
    
    # Step 1: Initialize database
    print("\n[1/4] Initializing database...")
    db_path = "data/cold_chain.db"
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
    orchestrator = SystemOrchestrator(bus)
    orchestrator.add_agent("monitor", monitor)
    orchestrator.add_agent("predictor", predictor)
    orchestrator.add_agent("decision", decision)
    orchestrator.add_agent("alert", alert)
    
    # Start system
    await orchestrator.start()
    
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
            await orchestrator.inject_sensor_data(sensor_id, temp, humidity)
        
        await asyncio.sleep(1)
        
        if i == 10:
            print(f"  T+{i}s: Sensor-02 beginning temperature rise...")
        elif i == 15:
            print(f"  T+{i}s: Predictor should detect trend and issue warning...")
    
    print("-" * 60)
    
    # Log final system status
    print("\n[Status] Logging final agent metrics to database...")
    status = orchestrator.get_system_status()
    
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
    await orchestrator.stop()
    
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
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(run_integrated_demo())
