"""Configuration helpers for file locations and environment detection."""

from __future__ import annotations

import os
import platform
from pathlib import Path

APP_NAME = "lekha"


def get_data_root() -> Path:
    """Return the base directory for storing project data."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    root = base / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root
