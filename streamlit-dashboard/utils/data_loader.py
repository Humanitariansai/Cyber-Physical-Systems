"""
Data loading utilities for the Cold Chain Dashboard.
Loads sensor data, alerts, and predictions from SQLite database.
"""

import sqlite3
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DEFAULT_DB = str(PROJECT_ROOT / "data" / "cold_chain.db")


def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Get database connection."""
    path = db_path or DEFAULT_DB
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def load_sensor_data(db_path: str = None, hours: int = 24,
                     sensor_id: str = None) -> pd.DataFrame:
    """Load sensor data from database."""
    conn = get_db_connection(db_path)
    if conn is None:
        return generate_sample_data(hours)

    try:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        query = "SELECT * FROM sensor_data WHERE timestamp > ?"
        params = [cutoff]

        if sensor_id:
            query += " AND sensor_id = ?"
            params.append(sensor_id)

        query += " ORDER BY timestamp"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            return generate_sample_data(hours)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        conn.close()
        return generate_sample_data(hours)


def load_alerts(db_path: str = None) -> pd.DataFrame:
    """Load alerts from database."""
    conn = get_db_connection(db_path)
    if conn is None:
        return pd.DataFrame()

    try:
        df = pd.read_sql_query(
            "SELECT * FROM alerts ORDER BY triggered_at DESC", conn
        )
        conn.close()
        return df
    except Exception:
        conn.close()
        return pd.DataFrame()


def load_predictions(db_path: str = None, hours: int = 24) -> pd.DataFrame:
    """Load prediction history from database."""
    conn = get_db_connection(db_path)
    if conn is None:
        return pd.DataFrame()

    try:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        df = pd.read_sql_query(
            "SELECT * FROM predictions WHERE timestamp > ? ORDER BY timestamp",
            conn, params=[cutoff]
        )
        conn.close()
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        conn.close()
        return pd.DataFrame()


def generate_sample_data(hours: int = 24, interval_minutes: int = 1) -> pd.DataFrame:
    """Generate sample sensor data when database is unavailable."""
    np.random.seed(42)
    n_points = (hours * 60) // interval_minutes
    timestamps = [datetime.now() - timedelta(minutes=i * interval_minutes)
                  for i in range(n_points, 0, -1)]

    base_temp = 5.0
    noise = np.random.normal(0, 0.3, n_points)
    trend = np.sin(np.linspace(0, 4 * np.pi, n_points)) * 0.5
    temps = base_temp + noise + trend
    temps[int(n_points * 0.3):int(n_points * 0.35)] += 2
    humidity = 55 + np.random.normal(0, 5, n_points)

    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temps,
        'humidity': humidity,
        'sensor_id': ['sensor-01'] * n_points
    })
