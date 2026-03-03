#!/usr/bin/env bash
# Launch the 911automate RAG API with uvicorn.
# Usage: bash scripts/run_api.sh
set -e
cd "$(dirname "$0")/.."

# Activate venv (Windows path first, then Linux/macOS)
if [ -f "911env/Scripts/activate" ]; then
  . 911env/Scripts/activate
else
  . 911env/bin/activate
fi

# AGENT_WARM_ON_START=false: instant startup, lazy agent on first /chat
export AGENT_WARM_ON_START=${AGENT_WARM_ON_START:-false}
exec uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
