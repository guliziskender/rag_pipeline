import importlib.util
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_startup_fails_without_api_key(monkeypatch):
    # The RAG_API_KEY check happens before any model loading, so this needs
    # no mocking of embeddings/LLM classes -- the import should fail before
    # ever reaching those lines.
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    module_name = f"app_missing_key_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="RAG_API_KEY"):
        spec.loader.exec_module(module)


def test_documents_requires_auth(unauthenticated_client):
    response = unauthenticated_client.get("/documents")
    assert response.status_code == 401


def test_documents_rejects_wrong_key(app_module):
    wrong_client = TestClient(app_module.app, headers={"X-API-Key": "not-the-right-key"})
    response = wrong_client.get("/documents")
    assert response.status_code == 401


def test_documents_accepts_correct_key(client):
    response = client.get("/documents")
    assert response.status_code == 200


def test_ingest_requires_auth(unauthenticated_client, sample_pdf_bytes):
    response = unauthenticated_client.post(
        "/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 401


def test_delete_requires_auth(unauthenticated_client):
    response = unauthenticated_client.delete("/documents/a.pdf")
    assert response.status_code == 401


def test_query_requires_auth(unauthenticated_client):
    response = unauthenticated_client.post(
        "/query/advanced", json={"question": "hi", "history": []}
    )
    assert response.status_code == 401
