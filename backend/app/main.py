import os
import time
import uuid
import tempfile
import requests
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

app = FastAPI(title="DocuChat RAG API", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tracks the last time any real (non-self-ping) request hit the app, so the
# keep-alive loop can skip pinging when genuine traffic has already reset
# Render's spin-down timer on its own.
_last_request_time = time.time()


@app.middleware("http")
async def _track_last_request(request, call_next):
    global _last_request_time
    if request.url.path != "/api/health":
        _last_request_time = time.time()
    return await call_next(request)

# Models and Client Initialization
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-3.5-flash-lite"

print("Initializing Embedding Model...")
embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
print("Embedding Model initialized.")


# Global in-memory Ephemeral Chroma client
chroma_client = chromadb.EphemeralClient()

# In-memory session metadata registry: session_id -> metadata
session_metadata = {}

# --- Memory management ---
# EphemeralClient keeps every uploaded document's chunks and embeddings
# resident in RAM for as long as the process runs, with no automatic expiry.
# Abandoned sessions (user closes the tab without deleting) otherwise
# accumulate forever and eventually exceed the instance's memory limit.
# These settings auto-evict inactive sessions so memory doesn't grow
# unbounded on a free-tier instance shared across multiple users.
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 20 * 60))  # 20 min of inactivity
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", 10 * 60))  # sweep every 10 min


def _touch_session(session_id: str):
    """Record activity so the cleanup sweep doesn't evict a session still in use."""
    if session_id in session_metadata:
        session_metadata[session_id]["last_active"] = time.time()


def _evict_session(session_id: str):
    collection_name = f"docuchat_{session_id}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception as e:
        print(f"Cleanup: error deleting collection {collection_name}: {e}")
    session_metadata.pop(session_id, None)


# --- Conversation memory ---
# Stored server-side per session_id (the same key that already isolates each
# user's uploaded documents), so multiple users never see each other's
# history. Capped to the last MAX_HISTORY_TURNS exchanges, with each answer
# truncated, so it stays a small, bounded amount of memory per session.
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 5))
HISTORY_ANSWER_CHAR_LIMIT = 500


def _get_history(session_id: str):
    meta = session_metadata.get(session_id)
    return meta.get("history", []) if meta else []


def _append_history(session_id: str, question: str, answer: str):
    meta = session_metadata.get(session_id)
    if meta is None:
        return
    history = meta.setdefault("history", [])
    history.append({
        "question": question,
        "answer": answer[:HISTORY_ANSWER_CHAR_LIMIT]
    })
    # Keep only the most recent turns so this can't grow unbounded.
    if len(history) > MAX_HISTORY_TURNS:
        meta["history"] = history[-MAX_HISTORY_TURNS:]


def _format_history(history):
    if not history:
        return ""
    lines = ["Recent conversation:"]
    for turn in history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines) + "\n"


async def _cleanup_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        now = time.time()
        expired = [
            sid for sid, meta in list(session_metadata.items())
            if now - meta.get("last_active", now) > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            print(f"Cleanup: evicting inactive session {sid}")
            _evict_session(sid)


# --- Keep-alive (self-ping) ---
# Render's free tier spins the service down after ~15 min with no inbound
# HTTP traffic. This periodically pings the service's own public health
# endpoint to keep it continuously awake. RENDER_EXTERNAL_URL is set
# automatically by Render; set KEEP_ALIVE=0 to disable (e.g. for local dev).
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "1") == "1"
KEEP_ALIVE_INTERVAL_SECONDS = int(os.getenv("KEEP_ALIVE_INTERVAL_SECONDS", 10 * 60))  # every 10 min


async def _keep_alive_loop():
    external_url = os.getenv("RENDER_EXTERNAL_URL")
    if not external_url:
        print("Keep-alive: RENDER_EXTERNAL_URL not set, skipping self-ping loop.")
        return
    health_url = external_url.rstrip("/") + "/api/health"
    while True:
        await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)
        # Real user traffic already resets Render's spin-down timer on its
        # own, so only self-ping if the site has genuinely been idle for
        # the full interval — avoids pinging (and any blocking-call delay)
        # while someone is actively using the app.
        idle_seconds = time.time() - _last_request_time
        if idle_seconds < KEEP_ALIVE_INTERVAL_SECONDS:
            continue
        try:
            await asyncio.to_thread(requests.get, health_url, timeout=10)
        except Exception as e:
            print(f"Keep-alive: ping failed: {e}")


@app.on_event("startup")
async def _start_cleanup_task():
    asyncio.create_task(_cleanup_loop())
    if KEEP_ALIVE:
        asyncio.create_task(_keep_alive_loop())


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant that answers questions using ONLY the provided context (excerpts from the user's uploaded document(s)).

- Each excerpt in the context is labeled with its source filename and page number, like "[Source: file.pdf, Page 2]".
- If multiple different files appear in the context, mention which file(s) your answer is based on (e.g. "According to report.pdf, ..."). If there is only one file, you don't need to repeat its name every time.
- If the user asks a specific factual question and the answer is not present in the context, say: "I could not find the answer in the document."
- If the user gives an open-ended or general request (e.g. "explain", "explain this", "summarize", "what is this about", "give me an overview"), do NOT treat it as a missing-answer case. Instead, use the provided context to explain or summarize what it covers, in your own words, as clearly and helpfully as possible. If multiple files are present, organize the summary by file where it makes sense.
- You may be given recent conversation history below the context. Use it to understand follow-up questions (e.g. "what about that", "explain more", "and page 2?") and keep your answers consistent with what you said before. The conversation history is for context only — it is not itself a source of facts about the document; still ground factual claims in the provided context.
- Never invent facts that aren't in the context, but do your best to be helpful with whatever context is provided.
""",
        ),
        (
            "human",
            """Context:
{context}
{history}
Question:
{question}
""",
        ),
    ]
)

class QueryRequest(BaseModel):
    session_id: str
    query: str

class DeleteRequest(BaseModel):
    session_id: str

class UrlUploadRequest(BaseModel):
    url: str
    session_id: Optional[str] = None


def estimate_relevance(vectorstore, query: str):
    """Real relevance readout from similarity scores — purely additive,
    does not change what context is fed to the LLM."""
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=4)
        scores = [score for _, score in results if score is not None]
        return round(sum(scores) / len(scores) * 100, 1) if scores else None
    except Exception as e:
        print(f"Error estimating relevance: {e}")
        return None


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/status/{session_id}")
def get_status(session_id: str):
    if session_id in session_metadata:
        return {
            "active": True,
            **session_metadata[session_id]
        }
    return {"active": False}


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None)
):
    if not session_id or session_id == "null" or session_id == "undefined":
        session_id = uuid.uuid4().hex

    if session_id not in session_metadata:
        session_metadata[session_id] = {
            "doc_names": [],
            "num_pages": 0,
            "num_chunks": 0,
            "last_active": time.time()
        }

    all_docs = []
    new_doc_names = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        
        # Save upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            try:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to write file {file.filename}: {e}")

        # Parse PDF
        try:
            loader = PyPDFLoader(tmp_file_path)
            loaded_pages = loader.load()
            # PyPDFLoader stamps metadata["source"] with the temp file path;
            # overwrite it with the original filename so downstream sources
            # (chat citations) show the real PDF name instead of a temp path.
            for page in loaded_pages:
                page.metadata["source"] = file.filename
            all_docs.extend(loaded_pages)
            new_doc_names.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF {file.filename}: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No valid PDF documents parsed.")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Index chunks in Chroma
    try:
        collection_name = f"docuchat_{session_id}"
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        vectorstore.add_documents(chunks)
        
        # Update session metadata
        meta = session_metadata[session_id]
        meta["doc_names"] = list(set(meta["doc_names"] + new_doc_names))
        meta["num_pages"] += len(all_docs)
        meta["num_chunks"] += len(chunks)

        return {
            "session_id": session_id,
            "doc_names": meta["doc_names"],
            "num_pages": meta["num_pages"],
            "num_chunks": meta["num_chunks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {e}")


@app.post("/api/upload-url")
async def upload_url(req: UrlUploadRequest):
    session_id = req.session_id
    if not session_id or session_id == "null" or session_id == "undefined":
        session_id = uuid.uuid4().hex

    if session_id not in session_metadata:
        session_metadata[session_id] = {
            "doc_names": [],
            "num_pages": 0,
            "num_chunks": 0,
            "last_active": time.time()
        }

    # Fetch document from URL
    try:
        resp = requests.get(req.url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {e}")

    # Deduce name
    name = req.url.rstrip("/").split("/")[-1] or "document.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"

    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        try:
            tmp_file.write(resp.content)
            tmp_file_path = tmp_file.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write temp file: {e}")

    # Parse, split, index
    try:
        loader = PyPDFLoader(tmp_file_path)
        all_docs = loader.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No PDF pages found in document.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    try:
        collection_name = f"docuchat_{session_id}"
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        vectorstore.add_documents(chunks)

        meta = session_metadata[session_id]
        meta["doc_names"] = list(set(meta["doc_names"] + [name]))
        meta["num_pages"] += len(all_docs)
        meta["num_chunks"] += len(chunks)

        return {
            "session_id": session_id,
            "doc_names": meta["doc_names"],
            "num_pages": meta["num_pages"],
            "num_chunks": meta["num_chunks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")


# Generic/open-ended requests where the user isn't asking about a specific
# fact, but wants an explanation or summary of the document as a whole.
GENERIC_QUERY_PATTERNS = [
    "explain", "summarize", "summarise", "summary", "overview",
    "what is this", "what's this", "whats this", "about this document",
    "about this file", "about this pdf", "tell me about this",
    "describe this", "elaborate", "what does this document say",
    "give me a summary", "give me an overview",
]


def is_generic_query(query: str) -> bool:
    """True for vague, open-ended requests ('explain', 'summarize', etc.)
    that aren't asking about a specific fact in the document."""
    normalized = query.strip().lower().rstrip("?!. ")
    if not normalized:
        return False
    if len(normalized.split()) <= 4:
        for pattern in GENERIC_QUERY_PATTERNS:
            if pattern in normalized:
                return True
    return normalized in {"explain", "summarize", "summarise", "summary", "overview"}


@app.post("/api/query")
async def query_index(req: QueryRequest):
    session_id = req.session_id
    query = req.query

    if session_id not in session_metadata:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload documents again.")

    _touch_session(session_id)

    try:
        collection_name = f"docuchat_{session_id}"
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

        generic = is_generic_query(query)
        history = _get_history(session_id)

        if generic:
            # Vague requests like "explain" or "summarize" carry little
            # semantic signal on their own, so similarity search against the
            # literal query text tends to pull back a handful of unrelated
            # chunks. Instead, pull a broader, more representative sample of
            # the document using a retrieval query aimed at general content.
            retrieval_query = "main topics, key points, and overall summary of the document"
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.3},
            )
        else:
            # Fold the previous question into retrieval too, so short
            # follow-ups ("what about that?", "and page 2?") that carry
            # little meaning on their own still retrieve relevant chunks
            # based on what was being discussed.
            if history:
                retrieval_query = f"{history[-1]['question']} {query}"
            else:
                retrieval_query = query
            # Retrieve documents using MMR
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
            )

        start_time = time.time()
        docs = retriever.invoke(retrieval_query)
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}, Page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

        final_prompt = PROMPT.invoke({
            "context": context,
            "history": _format_history(history),
            "question": query
        })

        # Initialize LLM
        # Set api_key explicitly or rely on env
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured on the backend server.")

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=google_api_key)
        response = llm.invoke(final_prompt)


        # Normalize response content to a plain string.
        # Gemini (langchain-google-genai) can return `content` as a list of
        # content blocks (e.g. [{"type": "text", "text": "...", "extras": {...}}])
        # instead of a plain string, unlike Mistral. Extract and join just the text.
        raw_content = response.content
        if isinstance(raw_content, str):
            answer_text = raw_content
        elif isinstance(raw_content, list):
            parts = []
            for block in raw_content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        parts.append(block["text"])
            answer_text = "".join(parts)
        else:
            answer_text = str(raw_content)

        elapsed = round(time.time() - start_time, 2)
        relevance = estimate_relevance(vectorstore, query)

        _append_history(session_id, query, answer_text)

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", "?"),
                "text": d.page_content[:300] + "…"
            }
            for d in docs
        ]

        return {
            "content": answer_text,
            "elapsed": elapsed,
            "relevance": relevance,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query: {e}")


@app.post("/api/delete")
async def delete_index(req: DeleteRequest):
    session_id = req.session_id
    _evict_session(session_id)
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    # Bind to port from env or default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
