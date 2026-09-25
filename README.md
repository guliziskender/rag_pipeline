# rag_pipeline

A RAG (Retrieval-Augmented Generation) pipeline: upload PDFs, ask questions grounded in their content.

- `app.py` — FastAPI backend: ingestion, hybrid (BM25 + dense) retrieval with reranking, and streamed, grounded answers via Claude.
- `gui.py` — Streamlit chat UI and document manager.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:
- `ANTHROPIC_API_KEY` — your Anthropic API key.
- `RAG_API_KEY` — a shared secret required on every API request. Generate one with:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
  The backend refuses to start without this set. `gui.py` reads the same variable and sends it automatically.

## Running

```bash
uvicorn app:app --reload          # backend, http://localhost:8000
streamlit run gui.py              # UI, http://localhost:8501
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite mocks all network-touching models (embeddings, LLM, reranker), so it needs no API keys and runs fully offline. CI runs it on every push and pull request.
