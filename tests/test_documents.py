def test_list_documents_empty(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": []}


def test_list_documents_after_ingesting_multiple(client, sample_pdf_bytes, other_pdf_bytes):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    client.post("/ingest", files={"file": ("b.pdf", other_pdf_bytes, "application/pdf")})

    response = client.get("/documents")
    assert response.status_code == 200
    filenames = {doc["filename"] for doc in response.json()["documents"]}
    assert filenames == {"a.pdf", "b.pdf"}


def test_delete_nonexistent_document_returns_404(client):
    response = client.delete("/documents/nope.pdf")
    assert response.status_code == 404


def test_delete_document_removes_it_from_list(client, sample_pdf_bytes, other_pdf_bytes):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    client.post("/ingest", files={"file": ("b.pdf", other_pdf_bytes, "application/pdf")})

    delete_response = client.delete("/documents/a.pdf")
    assert delete_response.status_code == 200
    assert delete_response.json()["chunks_removed"] >= 1

    remaining = client.get("/documents").json()["documents"]
    assert [doc["filename"] for doc in remaining] == ["b.pdf"]


def test_delete_rebuilds_bm25_index(app_module, client, sample_pdf_bytes, other_pdf_bytes):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    client.post("/ingest", files={"file": ("b.pdf", other_pdf_bytes, "application/pdf")})
    assert len(app_module.bm25_retriever.docs) == 2

    client.delete("/documents/a.pdf")
    assert len(app_module.bm25_retriever.docs) == 1
    assert app_module.bm25_retriever.docs[0].metadata["source"] == "b.pdf"


def test_deleting_last_document_clears_bm25_index(app_module, client, sample_pdf_bytes):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    client.delete("/documents/a.pdf")
    assert app_module.bm25_retriever is None
