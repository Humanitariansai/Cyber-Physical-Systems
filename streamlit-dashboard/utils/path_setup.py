"""
Path setup utilities for the CPS Dashboard.
This module ensures all project modules can be imported correctly.
"""

import sys
from pathlib import Path

def setup_project_paths():
    """Add all necessary project paths to Python path."""
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    # Add all relevant directories to Python path
    paths_to_add = [
        str(project_root),  # Root directory
        str(project_root / "ml-models"),  # ML models directory
        str(project_root / "data-collection"),  # Data collection directory
        str(project_root / "streamlit-dashboard"),  # Dashboard directory
        str(project_root / "ml-models" / "mlflow-artifacts"),  # MLflow artifacts
        str(project_root / "ml-models" / "mlruns"),  # MLflow runs
        str(project_root / "data-collection" / "simulators"),  # Simulators
    ]
    
    # Add paths if they're not already in sys.path
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)

# Run path setup when this module is imported
setup_project_paths()