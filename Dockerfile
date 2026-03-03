FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and prompt files
COPY src/ ./src/
COPY prompts/ ./prompts/

# Project root must be on PYTHONPATH so `from src.rag.xxx import ...` resolves
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.rag.cli"]
