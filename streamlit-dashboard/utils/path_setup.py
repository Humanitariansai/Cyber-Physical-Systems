"""Path setup utility for the Streamlit dashboard."""

import sys
from pathlib import Path


def setup_paths():
    """Add project directories to sys.path for imports."""
    project_root = Path(__file__).parent.parent.parent.absolute()

    paths_to_add = [
        str(project_root),
        str(project_root / "ml-models"),
        str(project_root / "mcp-server"),
        str(project_root / "multi-agent-system"),
    ]

    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)

    return project_root


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent.absolute()


def get_db_path() -> str:
    """Return the default database path."""
    return str(get_project_root() / "data" / "cold_chain.db")
