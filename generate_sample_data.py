"""
Generate realistic sample data for the cold chain monitoring database.
Run this before taking screenshots to populate the dashboard with data.
"""

import sqlite3
import os
import random
import math
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "cold_chain.db")


def create_tables(conn):
    """Create all required tables."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT UNIQUE NOT NULL,
            sensor_id TEXT NOT NULL,
            type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            triggered_at TEXT NOT NULL,
            acknowledged_at TEXT,
            resolved_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            predicted_temp REAL NOT NULL,
            actual_temp REAL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            status TEXT NOT NULL,
            messages_processed INTEGER DEFAULT 0,
            errors_count INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()


def generate_temperature(base, hour, noise_level=0.3):
    """Generate realistic temperature with daily cycle."""
    daily_cycle = 0.5 * math.sin(2 * math.pi * (hour - 6) / 24)
    noise = random.gauss(0, noise_level)
    return round(base + daily_cycle + noise, 2)


def populate_sensor_data(conn, days=7):
    """Generate 7 days of minute-level sensor data."""
    cursor = conn.cursor()
    sensors = ["sensor-01", "sensor-02", "sensor-03"]
    now = datetime.now()

    print(f"Generating {days} days of sensor data for {len(sensors)} sensors...")

    rows = []
    for minutes_ago in range(days * 24 * 60, 0, -1):
        ts = now - timedelta(minutes=minutes_ago)
        hour = ts.hour + ts.minute / 60.0

        for sensor_id in sensors:
            base = {"sensor-01": 5.0, "sensor-02": 5.2, "sensor-03": 4.8}[sensor_id]
            temp = generate_temperature(base, hour)

            # Add anomalies
            if sensor_id == "sensor-02" and 3 <= (days * 24 * 60 - minutes_ago) / 60 <= 3.5:
                temp += 3.5  # Warm anomaly on day 1
            if sensor_id == "sensor-01" and 48 <= (days * 24 * 60 - minutes_ago) / 60 <= 48.5:
                temp += 4.0  # Critical anomaly on day 2

            humidity = round(55 + random.gauss(0, 5) + 5 * math.sin(2 * math.pi * hour / 24), 1)

            rows.append((sensor_id, temp, max(30, min(90, humidity)), ts.isoformat()))

        if len(rows) >= 5000:
            cursor.executemany(
                "INSERT INTO sensor_data (sensor_id, temperature, humidity, timestamp) VALUES (?, ?, ?, ?)",
                rows
            )
            conn.commit()
            rows = []

    if rows:
        cursor.executemany(
            "INSERT INTO sensor_data (sensor_id, temperature, humidity, timestamp) VALUES (?, ?, ?, ?)",
            rows
        )
        conn.commit()

    total = cursor.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
    print(f"  Inserted {total:,} sensor readings")


def populate_alerts(conn):
    """Generate realistic alerts."""
    cursor = conn.cursor()
    now = datetime.now()

    alerts = [
        ("ALT-001", "sensor-02", "TEMPERATURE_HIGH", "WARNING",
         "Temperature exceeded 8.0C threshold (8.5C detected)",
         (now - timedelta(hours=72)).isoformat(),
         (now - timedelta(hours=71, minutes=45)).isoformat(),
         (now - timedelta(hours=71)).isoformat()),
        ("ALT-002", "sensor-02", "TEMPERATURE_HIGH", "CRITICAL",
         "Temperature exceeded 10.0C critical threshold (10.2C detected)",
         (now - timedelta(hours=71, minutes=30)).isoformat(),
         (now - timedelta(hours=71, minutes=20)).isoformat(),
         (now - timedelta(hours=71)).isoformat()),
        ("ALT-003", "sensor-01", "RAPID_CHANGE", "WARNING",
         "Rapid temperature change detected: +2.1C in 5 minutes",
         (now - timedelta(hours=48)).isoformat(),
         (now - timedelta(hours=47, minutes=50)).isoformat(),
         (now - timedelta(hours=47, minutes=30)).isoformat()),
        ("ALT-004", "sensor-01", "TEMPERATURE_HIGH", "CRITICAL",
         "Temperature exceeded 10.0C critical threshold (11.0C detected)",
         (now - timedelta(hours=47, minutes=55)).isoformat(),
         (now - timedelta(hours=47, minutes=40)).isoformat(),
         (now - timedelta(hours=47)).isoformat()),
        ("ALT-005", "sensor-03", "TEMPERATURE_LOW", "WARNING",
         "Temperature approaching lower threshold (2.3C detected)",
         (now - timedelta(hours=24)).isoformat(),
         (now - timedelta(hours=23, minutes=50)).isoformat(),
         (now - timedelta(hours=23, minutes=30)).isoformat()),
        ("ALT-006", "sensor-02", "PREDICTED_BREACH", "INFO",
         "ML model predicts temperature breach in 30 minutes",
         (now - timedelta(hours=6)).isoformat(),
         (now - timedelta(hours=5, minutes=55)).isoformat(),
         (now - timedelta(hours=5, minutes=30)).isoformat()),
        ("ALT-007", "sensor-01", "TEMPERATURE_HIGH", "WARNING",
         "Temperature exceeded 8.0C threshold (8.2C detected)",
         (now - timedelta(hours=2)).isoformat(),
         None, None),
    ]

    for alert in alerts:
        cursor.execute(
            "INSERT OR IGNORE INTO alerts (alert_id, sensor_id, type, severity, message, triggered_at, acknowledged_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            alert
        )

    conn.commit()
    total = cursor.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    print(f"  Inserted {total} alerts")


def populate_predictions(conn, days=3):
    """Generate prediction history."""
    cursor = conn.cursor()
    now = datetime.now()
    sensors = ["sensor-01", "sensor-02", "sensor-03"]

    print(f"Generating {days} days of predictions...")

    rows = []
    for minutes_ago in range(days * 24 * 60, 0, -10):  # Every 10 minutes
        ts = now - timedelta(minutes=minutes_ago)
        hour = ts.hour + ts.minute / 60.0

        for sensor_id in sensors:
            base = {"sensor-01": 5.0, "sensor-02": 5.2, "sensor-03": 4.8}[sensor_id]
            actual = generate_temperature(base, hour)

            for horizon in [30, 60]:
                future_hour = hour + horizon / 60.0
                predicted = generate_temperature(base, future_hour, noise_level=0.1)
                error = random.gauss(0, 0.2 * (horizon / 30))
                predicted += error
                confidence = round(max(0.5, min(0.98, 0.9 - 0.15 * (horizon / 60) + random.gauss(0, 0.05))), 2)

                rows.append((sensor_id, horizon, round(predicted, 2), round(actual, 2), confidence, ts.isoformat()))

    cursor.executemany(
        "INSERT INTO predictions (sensor_id, horizon_minutes, predicted_temp, actual_temp, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()
    total = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    print(f"  Inserted {total:,} predictions")


def populate_agent_status(conn):
    """Generate agent status entries."""
    cursor = conn.cursor()
    now = datetime.now()

    agents = [
        ("monitor-01", "MonitorAgent", "running", 15234, 0),
        ("predictor-01", "PredictorAgent", "running", 8471, 3),
        ("decision-01", "DecisionAgent", "running", 2341, 0),
        ("alert-01", "AlertAgent", "running", 1567, 1),
    ]

    for agent_id, name, status, msgs, errors in agents:
        cursor.execute(
            "INSERT INTO agent_status (agent_id, agent_name, status, messages_processed, errors_count, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (agent_id, name, status, msgs, errors, now.isoformat())
        )

    conn.commit()
    print(f"  Inserted {len(agents)} agent status records")


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Remove old database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Removed old database")

    conn = sqlite3.connect(DB_PATH)
    print(f"Database: {DB_PATH}\n")

    create_tables(conn)
    populate_sensor_data(conn, days=7)
    populate_alerts(conn)
    populate_predictions(conn, days=3)
    populate_agent_status(conn)

    conn.close()
    print(f"\nDone! Database ready at: {DB_PATH}")
    print(f"File size: {os.path.getsize(DB_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
