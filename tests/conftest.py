import importlib.util
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    """Import a fresh copy of app.py with network-touching models mocked out
    and an isolated, empty Chroma persist directory.

    app.py builds its vector store, embeddings, and LLM as module-level
    globals at import time, so the only way to get one test's state from
    leaking into the next is to give each test its own module object
    entirely, imported under a unique name, rather than reusing the
    process-wide `app` module import.
    """
    import langchain_anthropic
    import langchain_huggingface
    import sentence_transformers
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    call_log = []

    def _system_message_content(prompt_value):
        try:
            messages = prompt_value.to_messages()
        except AttributeError:
            return None
        return next((m.content for m in messages if m.type == "system"), None)

    class SpyChatModel(FakeListChatModel):
        def invoke(self, prompt_value, *args, **kwargs):
            call_log.append({"method": "invoke", "system": _system_message_content(prompt_value)})
            return super().invoke(prompt_value, *args, **kwargs)

        def stream(self, prompt_value, *args, **kwargs):
            call_log.append({"method": "stream", "system": _system_message_content(prompt_value)})
            return super().stream(prompt_value, *args, **kwargs)

    # FakeListChatModel pops responses off this list in call order regardless
    # of whether the call was contextualization or generation, so keep every
    # entry identical -- tests shouldn't depend on which call consumes which.
    fake_responses = ["mocked answer"] * 50

    monkeypatch.setattr(
        langchain_huggingface,
        "HuggingFaceEmbeddings",
        lambda **kwargs: DeterministicFakeEmbedding(size=384),
    )
    monkeypatch.setattr(
        langchain_anthropic,
        "ChatAnthropic",
        lambda **kwargs: SpyChatModel(responses=fake_responses),
    )
    # Skip the real cross-encoder download; rerank_documents() degrades to a
    # plain top-n truncation when reranker is None, which is fine for tests
    # that don't specifically exercise reranking/guardrail scoring.
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", None)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))

    module_name = f"app_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.call_log = call_log

    yield module

    del sys.modules[module_name]


@pytest.fixture
def client(app_module):
    return TestClient(app_module.app)


@pytest.fixture
def sample_pdf_bytes():
    return (FIXTURES_DIR / "sample_a.pdf").read_bytes()


@pytest.fixture
def other_pdf_bytes():
    return (FIXTURES_DIR / "sample_b.pdf").read_bytes()
