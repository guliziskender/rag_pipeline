def test_rejects_non_pdf_content_type(client, sample_pdf_bytes):
    response = client.post(
        "/ingest",
        files={"file": ("sample.txt", sample_pdf_bytes, "text/plain")},
    )
    assert response.status_code == 415


def test_rejects_empty_file(client):
    response = client.post(
        "/ingest",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_rejects_unparseable_pdf_with_detail(client):
    response = client.post(
        "/ingest",
        files={"file": ("broken.pdf", b"this is not a real pdf", "application/pdf")},
    )
    assert response.status_code == 422
    assert "broken.pdf" in response.json()["detail"]


def test_ingest_success_returns_chunk_count(client, sample_pdf_bytes):
    response = client.post(
        "/ingest",
        files={"file": ("sample_a.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample_a.pdf"
    assert body["chunks"] >= 1


def test_ingest_indexes_into_vector_store(app_module, client, sample_pdf_bytes):
    client.post(
        "/ingest",
        files={"file": ("sample_a.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert app_module.vector_store._collection.count() >= 1
