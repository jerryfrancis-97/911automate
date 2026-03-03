# 911automate

Minimal production-ready Python project for 911/EMS automation with RAG support.

## Prerequisites

- Python 3.10+
- On Windows: Git Bash (from Git for Windows) to run `bootstrap.sh`

## Project Layout

```
911automate/
├── src/
│   └── rag/              # RAG module
├── tests/
├── notebooks/
├── scripts/
│   └── bootstrap.sh
├── data/
├── requirements.txt
└── README.md
```

## Setup

From the project root:

```bash
bash scripts/bootstrap.sh
```

On Linux/macOS, you can make it executable first:

```bash
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
```

## Activate Virtual Environment

After bootstrap, activate the 911env:

- **Windows:** `. 911env/Scripts/activate`
- **Linux/macOS:** `source 911env/bin/activate`

## Verify

```bash
pip list
```

You should see langchain, qdrant-client, pypdf, pytest, black, mypy.

## Development

- Run tests: `pytest`
- Format code: `black src/ tests/`
- Type check: `mypy src/`



# Backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (in a separate terminal)
streamlit run streamlit_app.py