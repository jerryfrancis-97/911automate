FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, API, prompts, and runtime entrypoints
COPY src/ ./src/
COPY api/ ./api/
COPY prompts/ ./prompts/
COPY config.yml ./config.yml
COPY streamlit_app.py ./streamlit_app.py
COPY streamlit_helpers.py ./streamlit_helpers.py
COPY run_ingest.py ./run_ingest.py

# Project root must be on PYTHONPATH so `from src.rag.xxx import ...` resolves
ENV PYTHONPATH=/app

# Default to CLI; docker-compose overrides command for api/streamlit services.
CMD ["python", "-m", "src.rag.cli"]
