"""
pytest conftest.py — Project-wide test fixtures and environment setup.

Sets KERAS_HOME and MPLCONFIGDIR to temp directories before any module
import to prevent PermissionError on C:\\Users\\...\\keras.json (Windows bug).
"""

import os
import sys
from pathlib import Path

# Ensure project root is on path before imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.common import configure_env
configure_env()

