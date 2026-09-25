class FakeReranker:
    """Cross-encoder stand-in that returns a fixed score for every pair,
    so tests can force the low-confidence guardrail branch on demand."""

    def __init__(self, score):
        self.score = score

    def predict(self, pairs):
        return [self.score for _ in pairs]


def test_empty_index_returns_canned_message_without_llm_call(client, app_module):
    response = client.post(
        "/query/advanced",
        json={"question": "what is this about?", "history": []},
    )
    assert response.status_code == 200
    assert "don't have any indexed documents" in response.text
    assert app_module.call_log == []


def test_empty_index_with_history_skips_contextualization_call(client, app_module):
    # A wasted contextualization call here would mean the early bail-out
    # isn't actually short-circuiting before that LLM round-trip.
    response = client.post(
        "/query/advanced",
        json={
            "question": "what about it?",
            "history": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 200
    assert app_module.call_log == []


def test_query_after_ingest_returns_grounded_answer(client, app_module, sample_pdf_bytes):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})

    response = client.post(
        "/query/advanced",
        json={"question": "what is photosynthesis?", "history": []},
    )
    assert response.status_code == 200
    assert "a.pdf" in response.text
    assert "mocked answer" in response.text
    assert [c["method"] for c in app_module.call_log] == ["stream"]


def test_query_with_history_calls_contextualization_then_generation(
    client, app_module, sample_pdf_bytes
):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})

    response = client.post(
        "/query/advanced",
        json={
            "question": "what about it?",
            "history": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 200
    assert [c["method"] for c in app_module.call_log] == ["invoke", "stream"]


def test_confident_retrieval_has_plain_grounding_prompt(client, app_module, sample_pdf_bytes):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    app_module.reranker = FakeReranker(score=4.2)

    client.post("/query/advanced", json={"question": "what is this about?", "history": []})

    system_prompt = app_module.call_log[0]["system"]
    assert "Only answer using the Context" in system_prompt
    assert "weak match" not in system_prompt


def test_low_confidence_retrieval_adds_skepticism_caveat(client, app_module, sample_pdf_bytes):
    client.post("/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    app_module.reranker = FakeReranker(score=-2.5)

    client.post("/query/advanced", json={"question": "irrelevant question", "history": []})

    system_prompt = app_module.call_log[0]["system"]
    assert "weak match" in system_prompt


def test_hybrid_retrieval_finds_exact_keyword_match(
    client, app_module, sample_pdf_bytes, other_pdf_bytes
):
    # Fake embeddings carry no real semantic meaning, so dense-only retrieval
    # has no reliable way to prefer the right document here -- if the correct
    # one still comes back, it's because BM25's exact keyword match found it.
    client.post("/ingest", files={"file": ("photosynthesis.pdf", sample_pdf_bytes, "application/pdf")})
    client.post("/ingest", files={"file": ("zephyrion.pdf", other_pdf_bytes, "application/pdf")})

    response = client.post(
        "/query/advanced",
        json={"question": "What is the rated output of the Zephyrion-9000?", "history": []},
    )
    assert response.status_code == 200
    assert "zephyrion.pdf" in response.text
