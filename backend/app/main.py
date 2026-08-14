import os
import time
import uuid
import tempfile
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
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

# Models and Client Initialization
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral-small-2506"

print("Initializing Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print("Embedding Model initialized.")

# Global in-memory Ephemeral Chroma client
chroma_client = chromadb.EphemeralClient()

# In-memory session metadata registry: session_id -> metadata
session_metadata = {}

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}
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
            "num_chunks": 0
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
            "num_chunks": 0
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


@app.post("/api/query")
async def query_index(req: QueryRequest):
    session_id = req.session_id
    query = req.query

    if session_id not in session_metadata:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload documents again.")

    try:
        collection_name = f"docuchat_{session_id}"
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

        # Retrieve documents using MMR
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )

        start_time = time.time()
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        final_prompt = PROMPT.invoke({"context": context, "question": query})

        # Initialize LLM
        # Set api_key explicitly or rely on env
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise HTTPException(status_code=500, detail="MISTRAL_API_KEY is not configured on the backend server.")

        llm = ChatMistralAI(model=LLM_MODEL_NAME, api_key=mistral_api_key)
        response = llm.invoke(final_prompt)
        
        elapsed = round(time.time() - start_time, 2)
        relevance = estimate_relevance(vectorstore, query)

        sources = [
            {"page": d.metadata.get("page", "?"), "text": d.page_content[:300] + "…"}
            for d in docs
        ]

        return {
            "content": response.content,
            "elapsed": elapsed,
            "relevance": relevance,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query: {e}")


@app.post("/api/delete")
async def delete_index(req: DeleteRequest):
    session_id = req.session_id

    # Remove collection if exists in Chroma
    collection_name = f"docuchat_{session_id}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception as e:
        # Might fail if collection doesn't exist, we can ignore
        print(f"Error deleting collection: {e}")

    # Remove from session metadata
    if session_id in session_metadata:
        del session_metadata[session_id]

    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    # Bind to port from env or default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
