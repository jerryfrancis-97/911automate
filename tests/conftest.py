"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root so 'src' package is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
