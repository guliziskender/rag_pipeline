import json
import logging
import os
import secrets
import tempfile
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Request, UploadFile, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

API_KEY = os.environ.get("RAG_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "RAG_API_KEY environment variable must be set to a shared secret "
        "before starting the server. Generate one with, e.g.:\n"
        '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(provided_key: str = Security(api_key_header)) -> None:
    if not provided_key or not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app = FastAPI(dependencies=[Depends(verify_api_key)])
logger = logging.getLogger("rag_pipeline")

# Everyone authenticates with the same shared RAG_API_KEY, so there's no
# per-user identity to key rate limits on -- limit per client IP instead.
# This guards against the realistic failure mode (a script looping and
# racking up Anthropic API cost), not against an outside attacker guessing
# the key, which a shared secret can't defend against anyway.
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

# Initialize the reranker only when its optional dependency is installed.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if CrossEncoder else None

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="rag_pipeline",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)
llm = ChatAnthropic(model="claude-sonnet-5")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# BM25Retriever has no persistent index of its own -- it's an in-memory word
# frequency table built from a fixed list of Documents -- so it must be
# rebuilt from whatever's currently in Chroma whenever documents change.
bm25_retriever: Optional[BM25Retriever] = None


def rebuild_bm25_retriever() -> None:
    global bm25_retriever
    records = vector_store.get(include=["metadatas", "documents"])
    docs = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(records["documents"], records["metadatas"])
    ]
    if not docs:
        bm25_retriever = None
        return
    new_retriever = BM25Retriever.from_documents(docs)
    new_retriever.k = 10
    bm25_retriever = new_retriever


rebuild_bm25_retriever()


@app.post("/ingest", summary="Index an uploaded PDF")
@limiter.limit("10/minute")
async def ingest_document(request: Request, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        try:
            pages = await run_in_threadpool(PyPDFLoader(tmp_path).load)
        except Exception as exc:
            logger.exception("Failed to parse PDF %s", file.filename)
            raise HTTPException(
                status_code=422,
                detail=f"Could not read '{file.filename}' as a PDF: {exc}",
            ) from exc
    finally:
        os.remove(tmp_path)

    for page in pages:
        page.metadata["source"] = file.filename

    chunks = await run_in_threadpool(text_splitter.split_documents, pages)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in the PDF")

    try:
        await run_in_threadpool(vector_store.add_documents, chunks)
    except Exception as exc:
        logger.exception("Failed to index chunks for %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index '{file.filename}': {exc}",
        ) from exc

    await run_in_threadpool(rebuild_bm25_retriever)

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "detail": "PDF indexed successfully.",
    }


@app.get("/documents", summary="List indexed documents")
async def list_documents():
    records = await run_in_threadpool(vector_store.get, include=["metadatas"])

    chunk_counts: dict[str, int] = {}
    for metadata in records["metadatas"]:
        source = metadata.get("source", "Unknown")
        chunk_counts[source] = chunk_counts.get(source, 0) + 1

    return {
        "documents": [
            {"filename": source, "chunks": count}
            for source, count in sorted(chunk_counts.items())
        ]
    }


@app.delete("/documents/{filename}", summary="Remove an indexed document")
async def delete_document(filename: str):
    existing = await run_in_threadpool(vector_store.get, where={"source": filename})
    if not existing["ids"]:
        raise HTTPException(status_code=404, detail=f"No indexed chunks found for '{filename}'")

    await run_in_threadpool(vector_store.delete, ids=existing["ids"])
    await run_in_threadpool(rebuild_bm25_retriever)

    return {
        "filename": filename,
        "chunks_removed": len(existing["ids"]),
        "detail": "Document removed from the index.",
    }

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class AdvancedQueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []

def rerank_documents(query: str, docs: list, top_n: int = 3):
    """Re-scores retrieved candidate chunks using a joint Cross-Encoder model."""
    if not docs:
        return []
    if reranker is None:
        return docs[:top_n]
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
        
    sorted_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    return sorted_docs[:top_n]

def no_documents_response(search_query: str) -> StreamingResponse:
    def no_context_stream():
        yield f"METADATA:{json.dumps({'sources': [], 'standalone_query': search_query})}\n"
        yield "I don't have any indexed documents to answer that from. Upload a PDF first."
    return StreamingResponse(no_context_stream(), media_type="text/plain")


@app.post("/query/advanced", summary="Advanced RAG with Memory & Reranking")
@limiter.limit("20/minute")
async def query_rag_advanced(request: Request, query_request: AdvancedQueryRequest):
    # 0. Bail out before paying for a contextualization LLM call that
    # retrieval could never use anyway if nothing has ever been indexed.
    indexed_count = await run_in_threadpool(vector_store._collection.count)
    if indexed_count == 0:
        return no_documents_response(query_request.question)

    # 1. Reformat chat history for LangChain
    chat_history = []
    for msg in query_request.history:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    search_query = query_request.question

    # 2. History Contextualization Step
    if chat_history:
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        context_chain = contextualize_q_prompt | llm
        response = await run_in_threadpool(
            context_chain.invoke, {"chat_history": chat_history, "input": query_request.question}
        )
        search_query = response.content

    # 3. Retrieve Candidate Pool (k=10 for broad recall), blending dense
    # (semantic) search with BM25 (exact keyword/term) search so queries
    # that hinge on specific names or jargon aren't missed by embeddings
    # alone. Falls back to dense-only if BM25 has no documents yet.
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    if bm25_retriever is not None:
        retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5]
        )
    else:
        retriever = dense_retriever
    candidate_docs = await run_in_threadpool(retriever.invoke, search_query)

    # 4. Rerank down to top k=3 high-precision chunks
    top_docs = await run_in_threadpool(rerank_documents, search_query, candidate_docs, top_n=3)

    if not top_docs:
        # Defensive fallback: a concurrent DELETE /documents/{filename} could
        # remove the last indexed chunk between the count check above and here.
        return no_documents_response(search_query)

    sources = list(set([doc.metadata.get("source", "Unknown") for doc in top_docs]))
    context_text = "\n\n".join([doc.page_content for doc in top_docs])

    # Cross-encoder logits for this model are roughly centered on 0: positive
    # scores tend to be genuine matches, negative scores are the retriever's
    # "least bad" guess rather than an actual answer to the question.
    best_score = top_docs[0].metadata.get("rerank_score")
    low_confidence = best_score is not None and best_score < 0.0

    # 5. Generation Step
    grounding_instruction = (
        "You are an expert technical assistant. Only answer using the Context "
        "below. If the Context does not contain the answer, say the indexed "
        "documents don't cover this — do not use your own general knowledge."
    )
    if low_confidence:
        grounding_instruction += (
            " The retrieval system flagged this Context as a weak match for the "
            "question, so treat it with extra skepticism before relying on it."
        )

    qa_system_prompt = f"{grounding_instruction}\n\nContext:\n{context_text}"
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    gen_chain = qa_prompt | llm

    def generate_tokens():
        yield f"METADATA:{json.dumps({'sources': sources, 'standalone_query': search_query})}\n"
        for chunk in gen_chain.stream({"chat_history": chat_history, "input": query_request.question}):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate_tokens(), media_type="text/plain")