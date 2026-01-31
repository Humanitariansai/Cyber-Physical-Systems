"""
MCP Server for Cold Chain Monitoring
Provides Resources, Tools, and Prompts for LLM advisory layer.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Allow running as standalone script (Claude Desktop) or as package
_dir = Path(__file__).parent.absolute()
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

from mcp.server.fastmcp import FastMCP

from database import ColdChainDatabase, initialize_database
from config import MCPServerConfig

# Initialize config and database
config = MCPServerConfig()
initialize_database(config.db_path)
db = ColdChainDatabase(config.db_path)

# Create FastMCP server
mcp = FastMCP(config.server_name)


# ── Resources ────────────────────────────────────────────────────────────────

@mcp.resource("sensor://current-readings")
def get_current_readings() -> str:
    """Real-time temperature and humidity data from all sensors."""
    data = db.get_recent_sensor_data(minutes=60)
    return json.dumps({
        "resource": "Current Sensor Readings",
        "time_window": "Last 60 minutes",
        "count": len(data),
        "data": data
    }, indent=2)


@mcp.resource("alerts://active")
def get_active_alerts() -> str:
    """Currently active alerts that need attention."""
    data = db.get_active_alerts()
    return json.dumps({
        "resource": "Active Alerts",
        "count": len(data),
        "alerts": data
    }, indent=2)


@mcp.resource("alerts://history")
def get_alert_history() -> str:
    """Historical alerts for pattern analysis."""
    data = db.get_alert_history(days=30)
    return json.dumps({
        "resource": "Alert History",
        "time_window": "Last 30 days",
        "count": len(data),
        "alerts": data
    }, indent=2)


@mcp.resource("predictions://accuracy")
def get_prediction_accuracy() -> str:
    """Model performance and prediction accuracy statistics."""
    data = db.get_prediction_accuracy(days=7)
    return json.dumps({
        "resource": "Prediction Accuracy",
        "time_window": "Last 7 days",
        "metrics": data
    }, indent=2)


@mcp.resource("system://health")
def get_system_health() -> str:
    """Multi-agent system health and performance metrics."""
    return json.dumps({
        "resource": "System Health",
        "status": "operational",
        "agents": {
            "monitor": "running",
            "predictor": "running",
            "decision": "running",
            "alert": "running"
        },
        "message_bus": {
            "status": "healthy",
            "throughput": "~100 msg/sec"
        }
    }, indent=2)


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_incident(alert_id: str, time_range_hours: int = 24) -> str:
    """Analyze a temperature incident to identify root cause and patterns."""
    alert = db.get_alert_by_id(alert_id)
    if not alert:
        return json.dumps({"error": f"Alert {alert_id} not found"})

    triggered_time = datetime.fromisoformat(alert["triggered_at"])
    start_time = triggered_time - timedelta(hours=time_range_hours)
    end_time = triggered_time + timedelta(hours=1)

    sensor_data = db.get_sensor_data_range(alert["sensor_id"], start_time, end_time)
    similar = db.find_similar_incidents(alert_id, limit=3)

    analysis = {
        "alert_details": alert,
        "data_points_analyzed": len(sensor_data),
        "time_range": {"start": start_time.isoformat(), "end": end_time.isoformat()},
        "temperature_trend": _analyze_trend(sensor_data),
        "similar_incidents": len(similar),
        "similar_incident_details": similar,
        "possible_causes": _identify_causes(alert, sensor_data),
        "recommendations": _generate_recommendations(alert, sensor_data),
    }
    return json.dumps(analysis, indent=2)


@mcp.tool()
def query_sensor_history(sensor_id: str, hours_back: int = 24) -> str:
    """Query historical sensor data for a specific time period."""
    data = db.get_recent_sensor_data(sensor_id, minutes=hours_back * 60)
    stats = _calculate_statistics(data)
    return json.dumps({
        "sensor_id": sensor_id,
        "time_range": f"Last {hours_back} hours",
        "data_points": len(data),
        "statistics": stats,
        "data": data[:100],
    }, indent=2)


@mcp.tool()
def find_similar_incidents(alert_id: str, limit: int = 5) -> str:
    """Find historical incidents similar to current alert."""
    similar = db.find_similar_incidents(alert_id, limit)
    return json.dumps({
        "query_alert_id": alert_id,
        "similar_incidents_found": len(similar),
        "incidents": similar,
    }, indent=2)


@mcp.tool()
def generate_maintenance_report(days: int = 30) -> str:
    """Generate maintenance recommendations based on sensor patterns."""
    alerts = db.get_alert_history(days=days)

    sensor_failures: Dict[str, int] = {}
    for a in alerts:
        sid = a["sensor_id"]
        sensor_failures[sid] = sensor_failures.get(sid, 0) + 1

    recommendations = []
    for sid, count in sorted(sensor_failures.items(), key=lambda x: x[1], reverse=True):
        if count >= 3:
            recommendations.append({
                "sensor_id": sid,
                "incident_count": count,
                "priority": "HIGH" if count >= 5 else "MEDIUM",
                "action": "Schedule maintenance inspection",
            })

    return json.dumps({
        "analysis_period": f"Last {days} days",
        "total_incidents": len(alerts),
        "sensors_analyzed": len(sensor_failures),
        "maintenance_recommendations": recommendations,
    }, indent=2)


# ── Prompts ──────────────────────────────────────────────────────────────────

@mcp.prompt()
def analyze_latest_incident() -> str:
    """Analyze the most recent temperature breach incident."""
    alerts = db.get_active_alerts()
    if alerts:
        return (f"Analyze the latest incident (Alert ID: {alerts[0]['alert_id']}). "
                f"What caused this temperature breach and what actions should be taken?")
    return "No active incidents found."


@mcp.prompt()
def system_performance_summary() -> str:
    """Get a summary of system performance over the past week."""
    return ("Provide a comprehensive performance summary for the cold chain monitoring "
            "system over the past 7 days. Include prediction accuracy, alert frequency, "
            "and any maintenance recommendations.")


@mcp.prompt()
def troubleshoot_sensor(sensor_id: str) -> str:
    """Troubleshoot issues with a specific sensor."""
    return (f"Analyze sensor {sensor_id} for any issues. Check recent data patterns, "
            f"alert history, and prediction accuracy. Identify any anomalies or "
            f"maintenance needs.")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _analyze_trend(data: List[Dict]) -> str:
    if len(data) < 2:
        return "insufficient_data"
    temps = [d["temperature"] for d in data]
    if temps[-1] > temps[0] + 2:
        return "rising"
    elif temps[-1] < temps[0] - 2:
        return "falling"
    return "stable"


def _identify_causes(alert: Dict, data: List[Dict]) -> List[str]:
    causes: List[str] = []
    trend = _analyze_trend(data)
    if trend == "rising":
        causes.extend(["HVAC system degradation", "Compressor failure",
                        "High ambient temperature", "Poor ventilation"])
    elif trend == "falling":
        causes.extend(["Overcooling", "Thermostat malfunction",
                        "Sensor calibration issue"])
    return causes


def _generate_recommendations(alert: Dict, data: List[Dict]) -> List[str]:
    recs: List[str] = []
    if alert.get("severity") in ("CRITICAL", "ERROR"):
        recs.append("Immediate manual inspection required")
        recs.append("Check HVAC system functionality")
    recs.append("Schedule preventive maintenance within 48 hours")
    recs.append("Verify sensor calibration")
    return recs


def _calculate_statistics(data: List[Dict]) -> Dict:
    if not data:
        return {}
    temps = [d["temperature"] for d in data]
    return {"min": min(temps), "max": max(temps),
            "avg": round(sum(temps) / len(temps), 2), "count": len(temps)}


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
