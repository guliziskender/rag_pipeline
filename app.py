import json
import logging
import os
import tempfile
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

app = FastAPI()
logger = logging.getLogger("rag_pipeline")

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


@app.post("/ingest", summary="Index an uploaded PDF")
async def ingest_document(file: UploadFile = File(...)):
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
            pages = PyPDFLoader(tmp_path).load()
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

    chunks = text_splitter.split_documents(pages)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in the PDF")

    try:
        vector_store.add_documents(chunks)
    except Exception as exc:
        logger.exception("Failed to index chunks for %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index '{file.filename}': {exc}",
        ) from exc

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "detail": "PDF indexed successfully.",
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

@app.post("/query/advanced", summary="Advanced RAG with Memory & Reranking")
async def query_rag_advanced(request: AdvancedQueryRequest):
    # 1. Reformat chat history for LangChain
    chat_history = []
    for msg in request.history:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    search_query = request.question

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
        search_query = context_chain.invoke({"chat_history": chat_history, "input": request.question}).content

    # 3. Retrieve Candidate Pool (k=10 for broad recall)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    candidate_docs = retriever.invoke(search_query)

    # 4. Rerank down to top k=3 high-precision chunks
    top_docs = rerank_documents(search_query, candidate_docs, top_n=3)
    
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in top_docs]))
    context_text = "\n\n".join([doc.page_content for doc in top_docs])

    # 5. Generation Step
    qa_system_prompt = (
        "You are an expert technical assistant. Use the retrieved context to answer "
        "the user's question. If you don't know the answer, state that you don't know.\n\n"
        f"Context:\n{context_text}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    gen_chain = qa_prompt | llm

    def generate_tokens():
        yield f"METADATA:{json.dumps({'sources': sources, 'standalone_query': search_query})}\n"
        for chunk in gen_chain.stream({"chat_history": chat_history, "input": request.question}):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate_tokens(), media_type="text/plain")