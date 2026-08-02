"""
DocuChat — RAG document intelligence
Warm editorial dark theme (Syne / DM Mono / DM Sans), amber-orange accents,
live processing pipeline. Presentation-only — retrieval, prompt, embedding
model, and LLM are unchanged from the original backend.

Run with: streamlit run app.py
"""

import os
import tempfile
import time
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Streamlit Cloud provides secrets via st.secrets rather than a .env file.
for _key in ["MISTRAL_API_KEY", "HUGGINGFACEHUB_API_TOKEN"]:
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

st.set_page_config(
    page_title="DocuChat · RAG Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral-small-2506"

# =============================================================================
# CUSTOM CSS — warm editorial dark theme
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Hero ── */
.hero { text-align: center; padding: 3.5rem 0 2.5rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.25em; text-transform: uppercase;
    color: #ff8c32; margin-bottom: 1rem; opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800; line-height: 1.0; letter-spacing: -0.03em;
    color: #f0ebe0; margin: 0 0 1rem;
}
.hero h1 span { color: #ff8c32; }
.hero-sub {
    font-size: 1.05rem; font-weight: 300; color: #a09890;
    max-width: 520px; width: 100%;
    margin: 0 auto !important; line-height: 1.65;
    text-align: center !important;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent);
    margin: 2rem 0;
}

/* ── Input card ── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,140,50,0.15);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(255,140,50,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #ff8c32 !important; }
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: #a09890 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.2rem !important;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s !important;
    box-shadow: 0 4px 20px rgba(255,140,50,0.3) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,140,50,0.4) !important;
    opacity: 0.95 !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
    background: rgba(255,255,255,0.06) !important;
    color: #605850 !important;
    box-shadow: none !important;
}

/* ── Pipeline step cards ── */
.step-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.7rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.step-card.active { border-color: rgba(255,140,50,0.4); background: rgba(255,140,50,0.04); }
.step-card.done { border-color: rgba(80,200,120,0.3); background: rgba(80,200,120,0.03); }
.step-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    border-radius: 14px 0 0 14px; background: rgba(255,255,255,0.05); transition: background 0.3s;
}
.step-card.active::before { background: #ff8c32; }
.step-card.done::before { background: #50c878; }

.step-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.2rem; }
.step-num { font-family: 'DM Mono', monospace; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.15em; color: #ff8c32; opacity: 0.7; }
.step-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #f0ebe0; }
.step-status { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; }
.status-waiting { color: #555; }
.status-running { color: #ff8c32; }
.status-done { color: #50c878; }
.step-desc { font-size: 0.82rem; color: #706860; margin-top: 0.3rem; }

/* ── Result / report / feedback panels ── */
.report-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,140,50,0.2);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-top: 0.6rem;
    margin-bottom: 1rem;
}
.feedback-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(80,200,120,0.2);
    border-radius: 16px;
    padding: 1.4rem 2.2rem;
    margin-bottom: 1.5rem;
}
.query-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 2.2rem;
    margin-top: 1.4rem;
    margin-bottom: 0.6rem;
}
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 1rem; padding-bottom: 0.6rem;
}
.panel-label.orange { color: #ff8c32; border-bottom: 1px solid rgba(255,140,50,0.15); }
.panel-label.green { color: #50c878; border-bottom: 1px solid rgba(80,200,120,0.15); }
.panel-label.muted { color: #706860; border-bottom: 1px solid rgba(255,255,255,0.06); }

.source-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 2px solid #50c878;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #a09890;
    line-height: 1.55;
}
.source-card b {
    color: #50c878; font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; font-family: 'DM Mono', monospace;
}

.section-heading {
    font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700;
    color: #f0ebe0; margin: 1.5rem 0 1rem;
}

/* ── Chat input ── */
[data-testid="stChatInput"] { background: transparent !important; }
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.04) !important;
    color: #f0ebe0 !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 20px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #ff8c32 !important;
    box-shadow: 0 0 0 3px rgba(255,140,50,0.12) !important;
}
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%) !important;
    border-radius: 50% !important;
}

/* ── Alerts / expanders ── */
[data-testid="stAlertContentSuccess"] {
    background: rgba(80,200,120,0.06) !important;
    border: 1px solid rgba(80,200,120,0.25) !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentWarning"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #a09890 !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #a09890 !important;
    letter-spacing: 0.1em !important;
}

.stSpinner > div { color: #ff8c32 !important; }
.stProgress > div > div { background: linear-gradient(90deg, #ff8c32, #ff5a1a) !important; }

.chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.doc-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-size: 0.75rem;
    color: #a09890;
    font-family: 'DM Sans', sans-serif;
}
.chip-label {
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    color: #605850; letter-spacing: 0.1em;
    display: flex; align-items: center;
}

.notice {
    font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #605850;
    text-align: center; margin-top: 3rem; letter-spacing: 0.08em;
}

/* ── Mobile: force columns to stack full-width ── */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .block-container {
        padding: 1.2rem 1rem 3rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================
def step_card_html(num: str, title: str, state: str, desc: str = "") -> str:
    status_map = {
        "waiting": ("WAITING", "status-waiting"),
        "running": ("● RUNNING", "status-running"),
        "done": ("✓ DONE", "status-done"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    desc_html = f'<div class="step-desc">{desc}</div>' if desc else ""
    return f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
        {desc_html}
    </div>
    """


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    defaults = {
        "retriever": None,
        "vectorstore": None,
        "chat_history": [],
        "doc_names": [],
        "num_chunks": 0,
        "num_pages": 0,
        "queries_asked": 0,
        "last_query": None,
        "view": "home",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# =============================================================================
# CACHED RESOURCES / BACKEND (unchanged retrieval, prompt, embeddings, LLM)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatMistralAI(model=LLM_MODEL_NAME)


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


def estimate_relevance(vectorstore, query: str):
    """Real relevance readout from similarity scores — purely additive,
    does not change what context is fed to the LLM."""
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=4)
        scores = [score for _, score in results if score is not None]
        return round(sum(scores) / len(scores) * 100, 1) if scores else None
    except Exception:
        return None


# =============================================================================
# HOME PAGE — hero, upload, pipeline
# =============================================================================
if st.session_state.view == "home":

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Retrieval Augmented Generation</div>
        <h1>Docu<span>Chat</span></h1>
        <p class="hero-sub">
            Upload your documents and get answers grounded strictly in their
            content.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Upload ----
    _sp1, upload_col, _sp2 = st.columns([1, 2, 1])
    with upload_col:
        tab_file, tab_link = st.tabs(["📁 Upload File", "🔗 Paste Link"])

        build_clicked = False
        pending_files = []  # list of (display_name, local_temp_path)

        with tab_file:
            uploaded_files = st.file_uploader(
                "Upload PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
            )
            if uploaded_files:
                st.caption(f"✅ {len(uploaded_files)} file(s) received: " + ", ".join(f.name for f in uploaded_files))
            else:
                st.caption("No file selected yet.")
            if st.button("⚡  Build from Uploaded File(s)", use_container_width=True, disabled=not uploaded_files):
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(f.getvalue())
                        pending_files.append((f.name, tmp_file.name))
                build_clicked = True

        with tab_link:
            st.caption("Trouble uploading on mobile? Paste a direct link to a PDF instead (Google Drive/Dropbox direct-download link, or any public PDF URL).")
            pdf_url = st.text_input(
                "PDF URL", placeholder="https://example.com/document.pdf", label_visibility="collapsed"
            )
            if st.button("⚡  Fetch & Build from Link", use_container_width=True, disabled=not pdf_url):
                try:
                    with st.spinner("Downloading PDF…"):
                        resp = requests.get(pdf_url, timeout=30)
                        resp.raise_for_status()
                        content_type = resp.headers.get("Content-Type", "").lower()
                        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
                            st.warning("This link doesn't look like a direct PDF file — trying anyway.")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(resp.content)
                            name = pdf_url.rstrip("/").split("/")[-1] or "document.pdf"
                            if not name.lower().endswith(".pdf"):
                                name += ".pdf"
                            pending_files.append((name, tmp_file.name))
                    build_clicked = True
                except Exception as e:
                    st.error(f"Couldn't fetch that link: {e}")

        if st.session_state.doc_names:
            chips = "".join(f'<span class="doc-chip">{name}</span>' for name in st.session_state.doc_names)
            st.markdown(
                f'<div class="chip-row"><span class="chip-label">INDEXED →&nbsp;</span>{chips}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Go to Chat →", use_container_width=True):
                st.session_state.view = "chat"
                st.rerun()

    # ---- Pipeline ----
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    p1 = pc1.empty()
    p2 = pc2.empty()
    p3 = pc3.empty()
    p4 = pc4.empty()

    ready = st.session_state.retriever is not None
    initial_state = "done" if ready else "waiting"
    p1.markdown(step_card_html("01", "Load Documents", initial_state), unsafe_allow_html=True)
    p2.markdown(step_card_html("02", "Split into Chunks", initial_state), unsafe_allow_html=True)
    p3.markdown(step_card_html("03", "Generate Embeddings", initial_state), unsafe_allow_html=True)
    p4.markdown(step_card_html("04", "Build Vector Index", initial_state), unsafe_allow_html=True)

    if build_clicked and pending_files:
        # ── Step 1: load ──
        p1.markdown(step_card_html("01", "Load Documents", "running"), unsafe_allow_html=True)
        all_docs = []
        for _name, file_path in pending_files:
            try:
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())
            finally:
                os.unlink(file_path)
        p1.markdown(step_card_html("01", "Load Documents", "done", f"{len(all_docs)} pages parsed"), unsafe_allow_html=True)

        # ── Step 2: split ──
        p2.markdown(step_card_html("02", "Split into Chunks", "running"), unsafe_allow_html=True)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        p2.markdown(step_card_html("02", "Split into Chunks", "done", f"{len(chunks)} chunks created"), unsafe_allow_html=True)

        # ── Step 3: embed ──
        p3.markdown(step_card_html("03", "Generate Embeddings", "running"), unsafe_allow_html=True)
        embeddings = load_embedding_model()
        p3.markdown(step_card_html("03", "Generate Embeddings", "done", "Embedding model ready"), unsafe_allow_html=True)

        # ── Step 4: index ──
        p4.markdown(step_card_html("04", "Build Vector Index", "running"), unsafe_allow_html=True)
        try:
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.num_chunks = len(chunks)
            st.session_state.num_pages = len(all_docs)
            st.session_state.doc_names = [name for name, _ in pending_files]
            st.session_state.chat_history = []
            st.session_state.queries_asked = 0
            p4.markdown(step_card_html("04", "Build Vector Index", "done", "Index ready"), unsafe_allow_html=True)
            st.session_state.view = "chat"
            st.rerun()
        except Exception as e:
            p4.markdown(step_card_html("04", "Build Vector Index", "waiting", "Failed — see error below"), unsafe_allow_html=True)
            st.error(f"Couldn't build the index: {e}")


# =============================================================================
# CHAT PAGE
# =============================================================================
elif st.session_state.view == "chat" and st.session_state.retriever is not None:
    top_a, top_b = st.columns([1, 5])
    with top_a:
        if st.button("← Back", use_container_width=True):
            st.session_state.view = "home"
            st.rerun()
    with top_b:
        chips = "".join(f'<span class="doc-chip">{name}</span>' for name in st.session_state.doc_names)
        st.markdown(
            f'<div class="chip-row"><span class="chip-label">INDEXED →&nbsp;</span>{chips}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-heading">Ask a Question</div>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="query-panel"><div class="panel-label muted">You asked</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="report-panel"><div class="panel-label orange">📝 Response</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            meta_bits = [f'⏱ {msg.get("elapsed", "—")}s']
            if msg.get("relevance") is not None:
                meta_bits.append(f'Relevance ~{msg["relevance"]}%')
            feedback_body = " &nbsp;·&nbsp; ".join(meta_bits)
            st.markdown(
                f'<div class="feedback-panel"><div class="panel-label green">🧐 Retrieval Detail</div>{feedback_body}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                with st.expander(f"View {len(msg['sources'])} source excerpts"):
                    for i, s in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f'<div class="source-card"><b>Source {i} · Page {s["page"]}</b><br>{s["text"]}</div>',
                            unsafe_allow_html=True,
                        )

    query = st.chat_input("Ask a question about your document…")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.last_query = query
        st.session_state.queries_asked += 1

        with st.spinner("Retrieving context and generating a response…"):
            start = time.time()
            docs = st.session_state.retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)
            final_prompt = PROMPT.invoke({"context": context, "question": query})
            llm = load_llm()
            response = llm.invoke(final_prompt)
            elapsed = round(time.time() - start, 2)
            relevance = estimate_relevance(st.session_state.vectorstore, query)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response.content,
                "elapsed": elapsed,
                "relevance": relevance,
                "sources": [
                    {"page": d.metadata.get("page", "?"), "text": d.page_content[:300] + "…"}
                    for d in docs
                ],
            }
        )
        st.rerun()

    col_a, col_b, _ = st.columns([1, 1, 4])
    with col_a:
        if st.button("New Chat", use_container_width=True, disabled=not st.session_state.chat_history):
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        if st.button("Delete Index", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.vectorstore = None
            st.session_state.doc_names = []
            st.session_state.num_chunks = 0
            st.session_state.num_pages = 0
            st.session_state.chat_history = []
            st.session_state.queries_asked = 0
            st.session_state.view = "home"
            st.rerun()

else:
    st.session_state.view = "home"
    st.rerun()


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="notice">
    DocuChat · Powered by LangChain · RAG
</div>
""", unsafe_allow_html=True)
