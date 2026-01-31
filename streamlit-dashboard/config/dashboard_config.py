"""Dashboard configuration for Cold Chain Monitoring."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ThresholdConfig:
    """Temperature threshold configuration."""
    temp_min: float = 2.0
    temp_max: float = 8.0
    critical_min: float = 0.0
    critical_max: float = 10.0
    humidity_min: float = 30.0
    humidity_max: float = 80.0


@dataclass
class ChartColors:
    """Color scheme for charts."""
    temperature: str = "#1f77b4"
    humidity: str = "#2ca02c"
    prediction: str = "#ff7f0e"
    threshold_min: str = "#17becf"
    threshold_max: str = "#d62728"
    normal: str = "#28a745"
    warning: str = "#ffc107"
    critical: str = "#dc3545"


@dataclass
class DashboardConfig:
    """Main dashboard configuration."""
    page_title: str = "Cold Chain Monitor"
    layout: str = "wide"
    refresh_interval: int = 5
    sensors: List[str] = field(default_factory=lambda: [
        "sensor-01", "sensor-02", "sensor-03"
    ])
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    colors: ChartColors = field(default_factory=ChartColors)
    db_path: str = "data/cold_chain.db"
    prediction_horizons: List[int] = field(default_factory=lambda: [30, 60])

    def get_status(self, temp: float) -> str:
        """Get status string for a temperature reading."""
        if temp < self.thresholds.critical_min or temp > self.thresholds.critical_max:
            return "CRITICAL"
        elif temp < self.thresholds.temp_min or temp > self.thresholds.temp_max:
            return "WARNING"
        return "NORMAL"

    def get_status_color(self, temp: float) -> str:
        """Get status color for a temperature reading."""
        status = self.get_status(temp)
        return {
            "NORMAL": self.colors.normal,
            "WARNING": self.colors.warning,
            "CRITICAL": self.colors.critical,
        }[status]
