"""
MCP Server for Cold Chain Monitoring
Provides Resources, Tools, and Prompts for LLM advisory layer.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import Resource, Tool, TextContent, Prompt, PromptArgument

from .database import ColdChainDatabase, initialize_database
from .config import MCPServerConfig


class ColdChainMCPServer:
    """MCP Server for cold chain monitoring system."""
    
    def __init__(self, config: Optional[MCPServerConfig] = None):
        self.config = config or MCPServerConfig()
        self.db = ColdChainDatabase(self.config.db_path)
        self.server = Server(self.config.server_name)
        
        # Register handlers
        self._register_resources()
        self._register_tools()
        self._register_prompts()
    
    def _register_resources(self):
        """Register MCP resources (read-only data)."""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="sensor://current-readings",
                    name="Current Sensor Readings",
                    mimeType="application/json",
                    description="Real-time temperature and humidity data from all sensors"
                ),
                Resource(
                    uri="alerts://active",
                    name="Active Alerts",
                    mimeType="application/json",
                    description="Currently active alerts that need attention"
                ),
                Resource(
                    uri="alerts://history",
                    name="Alert History",
                    mimeType="application/json",
                    description="Historical alerts for pattern analysis"
                ),
                Resource(
                    uri="predictions://accuracy",
                    name="Prediction Accuracy Metrics",
                    mimeType="application/json",
                    description="Model performance and prediction accuracy statistics"
                ),
                Resource(
                    uri="system://health",
                    name="System Health Status",
                    mimeType="application/json",
                    description="Multi-agent system health and performance metrics"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> TextContent:
            """Read resource data."""
            
            if uri == "sensor://current-readings":
                data = self.db.get_recent_sensor_data(minutes=60)
                return TextContent(
                    type="text",
                    text=json.dumps({
                        "resource": "Current Sensor Readings",
                        "time_window": "Last 60 minutes",
                        "count": len(data),
                        "data": data
                    }, indent=2)
                )
            
            elif uri == "alerts://active":
                data = self.db.get_active_alerts()
                return TextContent(
                    type="text",
                    text=json.dumps({
                        "resource": "Active Alerts",
                        "count": len(data),
                        "alerts": data
                    }, indent=2)
                )
            
            elif uri == "alerts://history":
                data = self.db.get_alert_history(days=30)
                return TextContent(
                    type="text",
                    text=json.dumps({
                        "resource": "Alert History",
                        "time_window": "Last 30 days",
                        "count": len(data),
                        "alerts": data
                    }, indent=2)
                )
            
            elif uri == "predictions://accuracy":
                data = self.db.get_prediction_accuracy(days=7)
                return TextContent(
                    type="text",
                    text=json.dumps({
                        "resource": "Prediction Accuracy",
                        "time_window": "Last 7 days",
                        "metrics": data
                    }, indent=2)
                )
            
            elif uri == "system://health":
                # Placeholder for system health metrics
                return TextContent(
                    type="text",
                    text=json.dumps({
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
                )
            
            else:
                return TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown resource: {uri}"})
                )
    
    def _register_tools(self):
        """Register MCP tools (actions LLM can execute)."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="analyze_incident",
                    description="Analyze a temperature incident to identify root cause and patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "alert_id": {
                                "type": "string",
                                "description": "The alert ID to analyze"
                            },
                            "time_range_hours": {
                                "type": "number",
                                "description": "Hours of data to analyze before/after incident",
                                "default": 24
                            }
                        },
                        "required": ["alert_id"]
                    }
                ),
                Tool(
                    name="query_sensor_history",
                    description="Query historical sensor data for specific time period",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sensor_id": {
                                "type": "string",
                                "description": "Sensor identifier"
                            },
                            "hours_back": {
                                "type": "number",
                                "description": "Number of hours to look back",
                                "default": 24
                            }
                        },
                        "required": ["sensor_id"]
                    }
                ),
                Tool(
                    name="find_similar_incidents",
                    description="Find historical incidents similar to current alert",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "alert_id": {
                                "type": "string",
                                "description": "Current alert ID to find similar incidents for"
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of similar incidents to return",
                                "default": 5
                            }
                        },
                        "required": ["alert_id"]
                    }
                ),
                Tool(
                    name="generate_maintenance_report",
                    description="Generate maintenance recommendations based on sensor patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "number",
                                "description": "Number of days to analyze",
                                "default": 30
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> TextContent:
            """Execute tool."""
            
            if name == "analyze_incident":
                alert_id = arguments["alert_id"]
                hours = arguments.get("time_range_hours", 24)
                
                alert = self.db.get_alert_by_id(alert_id)
                if not alert:
                    return TextContent(
                        type="text",
                        text=json.dumps({"error": f"Alert {alert_id} not found"})
                    )
                
                # Get sensor data around incident
                triggered_time = datetime.fromisoformat(alert["triggered_at"])
                start_time = triggered_time - timedelta(hours=hours)
                end_time = triggered_time + timedelta(hours=1)
                
                sensor_data = self.db.get_sensor_data_range(
                    alert["sensor_id"], start_time, end_time
                )
                
                # Find similar incidents
                similar = self.db.find_similar_incidents(alert_id, limit=3)
                
                analysis = {
                    "alert_details": alert,
                    "data_points_analyzed": len(sensor_data),
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    },
                    "temperature_trend": self._analyze_trend(sensor_data),
                    "similar_incidents": len(similar),
                    "similar_incident_details": similar,
                    "possible_causes": self._identify_causes(alert, sensor_data),
                    "recommendations": self._generate_recommendations(alert, sensor_data)
                }
                
                return TextContent(
                    type="text",
                    text=json.dumps(analysis, indent=2)
                )
            
            elif name == "query_sensor_history":
                sensor_id = arguments["sensor_id"]
                hours = arguments.get("hours_back", 24)
                
                data = self.db.get_recent_sensor_data(sensor_id, minutes=hours*60)
                
                stats = self._calculate_statistics(data)
                
                return TextContent(
                    type="text",
                    text=json.dumps({
                        "sensor_id": sensor_id,
                        "time_range": f"Last {hours} hours",
                        "data_points": len(data),
                        "statistics": stats,
                        "data": data[:100]  # Limit to first 100 for brevity
                    }, indent=2)
                )
            
            elif name == "find_similar_incidents":
                alert_id = arguments["alert_id"]
                limit = arguments.get("limit", 5)
                
                similar = self.db.find_similar_incidents(alert_id, limit)
                
                return TextContent(
                    type="text",
                    text=json.dumps({
                        "query_alert_id": alert_id,
                        "similar_incidents_found": len(similar),
                        "incidents": similar
                    }, indent=2)
                )
            
            elif name == "generate_maintenance_report":
                days = arguments.get("days", 30)
                
                alerts = self.db.get_alert_history(days=days)
                
                # Analyze patterns
                sensor_failures = {}
                for alert in alerts:
                    sensor_id = alert["sensor_id"]
                    sensor_failures[sensor_id] = sensor_failures.get(sensor_id, 0) + 1
                
                # Generate recommendations
                recommendations = []
                for sensor_id, count in sorted(sensor_failures.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True):
                    if count >= 3:
                        recommendations.append({
                            "sensor_id": sensor_id,
                            "incident_count": count,
                            "priority": "HIGH" if count >= 5 else "MEDIUM",
                            "action": "Schedule maintenance inspection"
                        })
                
                return TextContent(
                    type="text",
                    text=json.dumps({
                        "analysis_period": f"Last {days} days",
                        "total_incidents": len(alerts),
                        "sensors_analyzed": len(sensor_failures),
                        "maintenance_recommendations": recommendations
                    }, indent=2)
                )
            
            else:
                return TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"})
                )
    
    def _register_prompts(self):
        """Register MCP prompts (pre-built templates)."""
        
        @self.server.list_prompts()
        async def list_prompts() -> List[Prompt]:
            """List available prompts."""
            return [
                Prompt(
                    name="analyze_latest_incident",
                    description="Analyze the most recent temperature breach incident",
                    arguments=[]
                ),
                Prompt(
                    name="system_performance_summary",
                    description="Get a summary of system performance over the past week",
                    arguments=[]
                ),
                Prompt(
                    name="troubleshoot_sensor",
                    description="Troubleshoot issues with a specific sensor",
                    arguments=[
                        PromptArgument(
                            name="sensor_id",
                            description="The sensor ID to troubleshoot",
                            required=True
                        )
                    ]
                )
            ]
        
        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Dict[str, str]) -> str:
            """Get prompt template."""
            
            if name == "analyze_latest_incident":
                alerts = self.db.get_active_alerts()
                if alerts:
                    return (f"Analyze the latest incident (Alert ID: {alerts[0]['alert_id']}). "
                           f"What caused this temperature breach and what actions should be taken?")
                return "No active incidents found."
            
            elif name == "system_performance_summary":
                return ("Provide a comprehensive performance summary for the cold chain monitoring "
                       "system over the past 7 days. Include prediction accuracy, alert frequency, "
                       "and any maintenance recommendations.")
            
            elif name == "troubleshoot_sensor":
                sensor_id = arguments.get("sensor_id", "unknown")
                return (f"Analyze sensor {sensor_id} for any issues. Check recent data patterns, "
                       f"alert history, and prediction accuracy. Identify any anomalies or "
                       f"maintenance needs.")
            
            return f"Unknown prompt: {name}"
    
    # Helper methods
    
    def _analyze_trend(self, data: List[Dict]) -> str:
        """Analyze temperature trend."""
        if len(data) < 2:
            return "insufficient_data"
        
        temps = [d["temperature"] for d in data]
        if temps[-1] > temps[0] + 2:
            return "rising"
        elif temps[-1] < temps[0] - 2:
            return "falling"
        return "stable"
    
    def _identify_causes(self, alert: Dict, data: List[Dict]) -> List[str]:
        """Identify possible causes of incident."""
        causes = []
        
        trend = self._analyze_trend(data)
        if trend == "rising":
            causes.extend([
                "HVAC system degradation",
                "Compressor failure",
                "High ambient temperature",
                "Poor ventilation"
            ])
        elif trend == "falling":
            causes.extend([
                "Overcooling",
                "Thermostat malfunction",
                "Sensor calibration issue"
            ])
        
        return causes
    
    def _generate_recommendations(self, alert: Dict, data: List[Dict]) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        
        if alert["severity"] in ["CRITICAL", "ERROR"]:
            recommendations.append("Immediate manual inspection required")
            recommendations.append("Check HVAC system functionality")
        
        recommendations.append("Schedule preventive maintenance within 48 hours")
        recommendations.append("Verify sensor calibration")
        
        return recommendations
    
    def _calculate_statistics(self, data: List[Dict]) -> Dict:
        """Calculate statistics for sensor data."""
        if not data:
            return {}
        
        temps = [d["temperature"] for d in data]
        return {
            "min": min(temps),
            "max": max(temps),
            "avg": sum(temps) / len(temps),
            "count": len(temps)
        }
    
    async def run(self):
        """Run the MCP server."""
        # Initialize database if needed
        initialize_database(self.config.db_path)
        
        # Start server
        await self.server.run()


# Entry point
async def main():
    """Main entry point for MCP server."""
    config = MCPServerConfig()
    server = ColdChainMCPServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
