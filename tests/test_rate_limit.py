def test_ingest_allows_up_to_limit_then_429s(client, sample_pdf_bytes):
    # /ingest is limited to 10/minute; TestClient requests all share the same
    # synthetic client host, so they share one rate-limit bucket.
    for i in range(10):
        response = client.post(
            "/ingest",
            files={"file": (f"doc_{i}.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200, f"request {i} unexpectedly rate limited"

    eleventh = client.post(
        "/ingest",
        files={"file": ("doc_10.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert eleventh.status_code == 429


def test_query_allows_up_to_limit_then_429s(client, sample_pdf_bytes):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})

    # /query/advanced is limited to 20/minute.
    for i in range(20):
        response = client.post(
            "/query/advanced", json={"question": f"question {i}", "history": []}
        )
        assert response.status_code == 200, f"request {i} unexpectedly rate limited"

    twenty_first = client.post(
        "/query/advanced", json={"question": "one too many", "history": []}
    )
    assert twenty_first.status_code == 429


def test_documents_falls_back_to_default_limit(client):
    # /documents has no route-specific @limiter.limit(...), so it's covered
    # by the Limiter's default_limits=["60/minute"] applied via middleware.
    for _ in range(60):
        response = client.get("/documents")
        assert response.status_code == 200

    sixty_first = client.get("/documents")
    assert sixty_first.status_code == 429


def test_rate_limit_is_per_client_not_global(app_module, sample_pdf_bytes, test_api_key):
    from fastapi.testclient import TestClient

    client_a = TestClient(
        app_module.app, headers={"X-API-Key": test_api_key}, client=("1.1.1.1", 123)
    )
    client_b = TestClient(
        app_module.app, headers={"X-API-Key": test_api_key}, client=("2.2.2.2", 123)
    )

    for i in range(10):
        response = client_a.post(
            "/ingest",
            files={"file": (f"doc_{i}.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200

    # client_a is now at its limit, but client_b has its own separate bucket.
    exhausted = client_a.post(
        "/ingest", files={"file": ("one_more.pdf", sample_pdf_bytes, "application/pdf")}
    )
    assert exhausted.status_code == 429

    still_fine = client_b.post(
        "/ingest", files={"file": ("first.pdf", sample_pdf_bytes, "application/pdf")}
    )
    assert still_fine.status_code == 200
