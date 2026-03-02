#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
VENV="911env"
if [ ! -d "$VENV" ]; then
  python -m venv "$VENV"
fi
if [ -f "$VENV/Scripts/activate" ]; then
  . "$VENV/Scripts/activate"   # Windows
else
  . "$VENV/bin/activate"      # Linux/macOS
fi
pip install --upgrade pip
pip install -r requirements.txt
echo "Done. Activate with: . 911env/Scripts/activate (Windows) or . 911env/bin/activate (Linux/macOS)"
