from langchain_core.documents import Document


class ScoreByLength:
    """Fake cross-encoder: scores a pair by how long the doc text is, so
    the expected sort order is easy to reason about."""

    def predict(self, pairs):
        return [float(len(doc_text)) for _query, doc_text in pairs]


def test_returns_empty_list_for_no_candidates(app_module):
    assert app_module.rerank_documents("q", [], top_n=3) == []


def test_without_reranker_returns_plain_truncation(app_module):
    app_module.reranker = None
    docs = [Document(page_content=f"doc-{i}") for i in range(5)]

    result = app_module.rerank_documents("q", docs, top_n=3)

    assert result == docs[:3]


def test_with_reranker_sorts_by_score_descending(app_module):
    app_module.reranker = ScoreByLength()
    docs = [
        Document(page_content="short"),
        Document(page_content="a much longer piece of text"),
        Document(page_content="medium length"),
    ]

    result = app_module.rerank_documents("q", docs, top_n=2)

    assert [d.page_content for d in result] == [
        "a much longer piece of text",
        "medium length",
    ]


def test_annotates_documents_with_rerank_score(app_module):
    app_module.reranker = ScoreByLength()
    docs = [Document(page_content="short"), Document(page_content="longer text")]

    result = app_module.rerank_documents("q", docs, top_n=2)

    assert all("rerank_score" in d.metadata for d in result)
