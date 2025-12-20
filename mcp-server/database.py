"""
Database interface for MCP server
Provides query methods for sensor data, alerts, and system metrics.
"""

import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json


class ColdChainDatabase:
    """Database interface for cold chain monitoring system."""
    
    def __init__(self, db_path: str = "../data/cold_chain.db"):
        self.db_path = db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_recent_sensor_data(self, sensor_id: Optional[str] = None, 
                               minutes: int = 60) -> List[Dict]:
        """Get recent sensor readings."""
        conn = self._get_connection()
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        if sensor_id:
            query = """
                SELECT sensor_id, temperature, humidity, timestamp 
                FROM sensor_data 
                WHERE sensor_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """
            cursor = conn.execute(query, (sensor_id, cutoff_time.isoformat()))
        else:
            query = """
                SELECT sensor_id, temperature, humidity, timestamp 
                FROM sensor_data 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """
            cursor = conn.execute(query, (cutoff_time.isoformat(),))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts."""
        conn = self._get_connection()
        query = """
            SELECT alert_id, sensor_id, alert_type, severity, 
                   message, triggered_at, acknowledged_at, resolved_at
            FROM alerts 
            WHERE resolved_at IS NULL
            ORDER BY triggered_at DESC
        """
        cursor = conn.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Dict]:
        """Get specific alert details."""
        conn = self._get_connection()
        query = """
            SELECT * FROM alerts WHERE alert_id = ?
        """
        cursor = conn.execute(query, (alert_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_sensor_data_range(self, sensor_id: str, 
                             start_time: datetime, 
                             end_time: datetime) -> List[Dict]:
        """Get sensor data for specific time range."""
        conn = self._get_connection()
        query = """
            SELECT temperature, humidity, timestamp
            FROM sensor_data
            WHERE sensor_id = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """
        cursor = conn.execute(query, (sensor_id, 
                                     start_time.isoformat(), 
                                     end_time.isoformat()))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_alert_history(self, days: int = 30, 
                         sensor_id: Optional[str] = None) -> List[Dict]:
        """Get historical alerts."""
        conn = self._get_connection()
        cutoff_time = datetime.now() - timedelta(days=days)
        
        if sensor_id:
            query = """
                SELECT * FROM alerts 
                WHERE sensor_id = ? AND triggered_at > ?
                ORDER BY triggered_at DESC
            """
            cursor = conn.execute(query, (sensor_id, cutoff_time.isoformat()))
        else:
            query = """
                SELECT * FROM alerts 
                WHERE triggered_at > ?
                ORDER BY triggered_at DESC
            """
            cursor = conn.execute(query, (cutoff_time.isoformat(),))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_prediction_accuracy(self, days: int = 7) -> Dict:
        """Get prediction accuracy metrics."""
        conn = self._get_connection()
        cutoff_time = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT 
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN accurate = 1 THEN 1 ELSE 0 END) as accurate_count
            FROM predictions
            WHERE timestamp > ?
        """
        cursor = conn.execute(query, (cutoff_time.isoformat(),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            result = dict(row)
            result['accuracy_rate'] = (result['accurate_count'] / result['total_predictions'] 
                                      if result['total_predictions'] > 0 else 0)
            return result
        return {}
    
    def find_similar_incidents(self, alert_id: str, limit: int = 5) -> List[Dict]:
        """Find alerts with similar patterns."""
        alert = self.get_alert_by_id(alert_id)
        if not alert:
            return []
        
        conn = self._get_connection()
        query = """
            SELECT * FROM alerts
            WHERE alert_id != ?
            AND sensor_id = ?
            AND alert_type = ?
            ORDER BY triggered_at DESC
            LIMIT ?
        """
        cursor = conn.execute(query, (alert_id, alert['sensor_id'], 
                                     alert['alert_type'], limit))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results


def initialize_database(db_path: str = "../data/cold_chain.db"):
    """Initialize database schema for cold chain monitoring."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Sensor data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Alerts table
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Predictions table
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # System metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp 
        ON sensor_data(timestamp)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_sensor_id 
        ON alerts(sensor_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
        ON predictions(timestamp)
    """)
    
    conn.commit()
    conn.close()
