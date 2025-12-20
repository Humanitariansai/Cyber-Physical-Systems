"""
Database logger for multi-agent system.
Provides methods to log sensor data, alerts, predictions, and metrics to SQLite.
"""

import sqlite3
from typing import Optional
from datetime import datetime
import os


class DatabaseLogger:
    """Logger for persisting multi-agent system data to database."""
    
    def __init__(self, db_path: str = "data/cold_chain.db"):
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def log_sensor_data(self, sensor_id: str, temperature: float, 
                       humidity: Optional[float] = None,
                       timestamp: Optional[str] = None):
        """Log sensor reading to database."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sensor_data (sensor_id, temperature, humidity, timestamp)
            VALUES (?, ?, ?, ?)
        """, (sensor_id, temperature, humidity, timestamp))
        
        conn.commit()
        conn.close()
    
    def log_alert(self, alert_id: str, sensor_id: str, alert_type: str,
                  severity: str, message: str, triggered_at: Optional[str] = None):
        """Log alert to database."""
        if triggered_at is None:
            triggered_at = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO alerts (alert_id, sensor_id, alert_type, severity, 
                                   message, triggered_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (alert_id, sensor_id, alert_type, severity, message, triggered_at))
            
            conn.commit()
        except sqlite3.IntegrityError:
            # Alert already exists, skip
            pass
        finally:
            conn.close()
    
    def update_alert_status(self, alert_id: str, 
                           acknowledged_at: Optional[str] = None,
                           resolved_at: Optional[str] = None,
                           resolution_notes: Optional[str] = None):
        """Update alert status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if acknowledged_at:
            updates.append("acknowledged_at = ?")
            params.append(acknowledged_at)
        
        if resolved_at:
            updates.append("resolved_at = ?")
            params.append(resolved_at)
        
        if resolution_notes:
            updates.append("resolution_notes = ?")
            params.append(resolution_notes)
        
        if updates:
            params.append(alert_id)
            query = f"UPDATE alerts SET {', '.join(updates)} WHERE alert_id = ?"
            cursor.execute(query, params)
            conn.commit()
        
        conn.close()
    
    def log_prediction(self, sensor_id: str, predicted_value: float,
                      confidence: float, horizon_minutes: int,
                      timestamp: Optional[str] = None):
        """Log prediction to database."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (sensor_id, predicted_value, confidence, 
                                    horizon_minutes, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (sensor_id, predicted_value, confidence, horizon_minutes, timestamp))
        
        conn.commit()
        conn.close()
    
    def log_metric(self, metric_name: str, metric_value: float,
                   timestamp: Optional[str] = None):
        """Log system metric to database."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_metrics (metric_name, metric_value, timestamp)
            VALUES (?, ?, ?)
        """, (metric_name, metric_value, timestamp))
        
        conn.commit()
        conn.close()
    
    def log_agent_status(self, agent_id: str, agent_name: str, status: str,
                        messages_processed: int, errors_count: int,
                        timestamp: Optional[str] = None):
        """Log agent status to database."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO agent_status (agent_id, agent_name, status, 
                                     messages_processed, errors_count, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (agent_id, agent_name, status, messages_processed, errors_count, timestamp))
        
        conn.commit()
        conn.close()
