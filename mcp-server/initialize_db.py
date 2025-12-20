"""
Initialize database schema for cold chain monitoring system.
Run this before starting the multi-agent system to create required tables.
"""

import sqlite3
import os
from datetime import datetime


def initialize_database(db_path: str = "data/cold_chain.db"):
    """Initialize database schema for cold chain monitoring."""
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    print(f"Initializing database at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Sensor data table
    print("Creating sensor_data table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_sensor_timestamp (sensor_id, timestamp)
        )
    """)
    
    # Alerts table
    print("Creating alerts table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT UNIQUE NOT NULL,
            sensor_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT,
            triggered_at TEXT NOT NULL,
            acknowledged_at TEXT,
            resolved_at TEXT,
            resolution_notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_alert_sensor (sensor_id),
            INDEX idx_alert_triggered (triggered_at)
        )
    """)
    
    # Predictions table
    print("Creating predictions table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            actual_value REAL,
            confidence REAL,
            horizon_minutes INTEGER,
            accurate INTEGER,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_pred_timestamp (timestamp)
        )
    """)
    
    # System metrics table
    print("Creating system_metrics table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_metric_name (metric_name)
        )
    """)
    
    # Agent status table
    print("Creating agent_status table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            status TEXT NOT NULL,
            messages_processed INTEGER DEFAULT 0,
            errors_count INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_agent_id (agent_id)
        )
    """)
    
    conn.commit()
    
    # Add sample data for testing
    print("\nAdding sample data...")
    
    # Sample sensor data
    cursor.execute("""
        INSERT INTO sensor_data (sensor_id, temperature, humidity, timestamp)
        VALUES ('sensor-01', 5.2, 65.0, ?)
    """, (datetime.now().isoformat(),))
    
    cursor.execute("""
        INSERT INTO sensor_data (sensor_id, temperature, humidity, timestamp)
        VALUES ('sensor-02', 4.8, 63.5, ?)
    """, (datetime.now().isoformat(),))
    
    # Sample alert
    cursor.execute("""
        INSERT INTO alerts (alert_id, sensor_id, alert_type, severity, message, triggered_at)
        VALUES ('alert-001', 'sensor-01', 'TEMPERATURE_HIGH', 'WARNING', 
                'Temperature approaching upper threshold', ?)
    """, (datetime.now().isoformat(),))
    
    conn.commit()
    
    # Verify tables
    print("\nVerifying tables...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Created tables: {[t[0] for t in tables]}")
    
    # Get row counts
    for table in ['sensor_data', 'alerts', 'predictions', 'system_metrics', 'agent_status']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")
    
    conn.close()
    print(f"\n✓ Database initialized successfully at {db_path}")


if __name__ == "__main__":
    initialize_database()
