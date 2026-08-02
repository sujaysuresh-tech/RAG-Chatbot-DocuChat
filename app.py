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
import uuid
from datetime import datetime

import chromadb
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
    initial_sidebar_state="expanded" if st.session_state.get("retriever") is not None else "collapsed",
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

html, body {
    overflow-x: hidden;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
    overflow-x: hidden;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Keep the header container itself (it holds the native sidebar toggle) but
   make it transparent and hide just the Deploy/menu toolbar inside it. */
[data-testid="stHeader"] {
    background: transparent !important;
    visibility: visible !important;
    height: 3.2rem !important;
}
[data-testid="stToolbar"] {
    visibility: hidden !important;
}
[data-testid="stHeader"] button svg {
    fill: #f0ebe0 !important;
}
[data-testid="stHeader"] button {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
}

/* ── Sidebar (doc management, appears after a successful upload) ── */
[data-testid="stSidebar"] {
    background: #0d0d12 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #e8e4dc;
}
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Hero ── */
.hero { text-align: center; padding: 3.5rem 0 2.5rem; max-width: 100%; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.25em; text-transform: uppercase;
    color: #ff8c32; margin-bottom: 1rem; opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.1rem, 9vw, 5rem);
    font-weight: 800; line-height: 1.05; letter-spacing: -0.03em;
    color: #f0ebe0; margin: 0 0 1rem;
    word-break: break-word;
    max-width: 100%;
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

/* ── Small back button ── */
.back-btn-wrap .stButton > button {
    width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.04) !important;
    color: #e8e4dc !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}
.back-btn-wrap .stButton > button:hover {
    background: rgba(255,140,50,0.12) !important;
    color: #ff8c32 !important;
    transform: none !important;
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
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}
[data-testid="stBottomBlockContainer"] {
    padding: 0.5rem 0 !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
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

/* ── Mobile: force columns to stack full-width, tighten the hero ── */
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
    .hero {
        padding: 1.6rem 0 1.2rem !important;
    }
    .hero-eyebrow {
        font-size: 0.58rem !important;
        letter-spacing: 0.16em !important;
    }
    .hero h1 {
        font-size: clamp(1.9rem, 11vw, 2.8rem) !important;
        letter-spacing: -0.02em !important;
    }
    .hero-sub {
        font-size: 0.9rem !important;
        padding: 0 0.4rem;
    }
    .input-card {
        padding: 1.3rem 1.2rem !important;
    }
    .step-card {
        padding: 1rem 1.1rem !important;
    }
    .report-panel, .feedback-panel, .query-panel {
        padding: 1.2rem 1.3rem !important;
    }
}

@media (max-width: 400px) {
    .hero h1 {
        font-size: clamp(1.6rem, 12vw, 2.2rem) !important;
    }
}
/* ── Chat window (bounded, scrollable, bubble-style) ── */
.chat-window {
    max-height: 58vh;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    padding: 0.4rem 0.2rem 0.8rem;
    margin-bottom: 1rem;
}
.chat-window::-webkit-scrollbar { width: 6px; }
.chat-window::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.15);
    border-radius: 9999px;
}

.msg-bubble {
    padding: 0.7rem 1.1rem;
    border-radius: 14px;
    max-width: 78%;
    font-size: 14.5px;
    line-height: 1.55;
    word-wrap: break-word;
    animation: messageIn 0.3s cubic-bezier(0.22, 1, 0.36, 1);
}
@keyframes messageIn {
    from { opacity: 0; transform: translateY(8px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.user-bubble {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%);
    color: #0a0a0f;
    margin-left: auto;
    border-bottom-right-radius: 4px;
    box-shadow: 0 4px 16px rgba(255,140,50,0.22);
    font-weight: 500;
}

.bot-bubble {
    background: rgba(255,255,255,0.03);
    color: #e8e4dc;
    border: 1px solid rgba(255,255,255,0.07);
    align-self: flex-start;
    border-bottom-left-radius: 4px;
}
.bot-bubble .bubble-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #706860;
    margin-top: 0.6rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.bot-bubble details {
    margin-top: 0.5rem;
}
.bot-bubble details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #a09890 !important;
    letter-spacing: 0.08em !important;
    cursor: pointer;
}
.bot-bubble .source-card { margin-top: 0.5rem; }

.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0.6rem 1.1rem;
}
.typing-indicator span {
    width: 6px; height: 6px; border-radius: 50%;
    background: #706860;
    animation: typingBounce 1.2s ease-in-out infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.3s; }
@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-5px); opacity: 1; }
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


def get_ephemeral_chroma_client():
    """Fresh, in-memory-only Chroma client. Falls back for older chromadb
    versions that predate EphemeralClient()."""
    try:
        return chromadb.EphemeralClient()
    except AttributeError:
        from chromadb.config import Settings
        return chromadb.Client(Settings(is_persistent=False))


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
# SHARED UPLOAD CONTROLS (used on the landing page, and later in the sidebar)
# =============================================================================
def render_upload_controls(location: str):
    """Renders the Upload File / Paste Link tabs. `location` is just a key
    suffix ('main' or 'sidebar') so widget keys stay unique."""
    tab_file, tab_link = st.tabs(["Upload File", "Paste Link"])
    build_clicked = False
    pending_files = []  # list of (display_name, local_temp_path)

    with tab_file:
        uploaded_files = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True,
            label_visibility="collapsed", key=f"uploader_{location}",
        )
        if uploaded_files:
            st.caption(f"✅ {len(uploaded_files)} file(s) received: " + ", ".join(f.name for f in uploaded_files))
        else:
            st.caption("No file selected yet.")
        if st.button("Build Knowledge Base", use_container_width=True,
                     disabled=not uploaded_files, key=f"build_btn_{location}"):
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(f.getvalue())
                    pending_files.append((f.name, tmp_file.name))
            build_clicked = True

    with tab_link:
        st.caption("Trouble uploading on mobile? Paste a direct link to a PDF instead (Google Drive/Dropbox direct-download link, or any public PDF URL).")
        pdf_url = st.text_input(
            "PDF URL", placeholder="https://example.com/document.pdf",
            label_visibility="collapsed", key=f"url_input_{location}",
        )
        if st.button("Fetch & Build from Link", use_container_width=True,
                     disabled=not pdf_url, key=f"url_btn_{location}"):
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

    return build_clicked, pending_files


def process_upload(pending_files, placeholders):
    """Runs the load → split → embed → index pipeline, updating the four
    step-card placeholders as it goes, then reruns the app."""
    p1, p2, p3, p4 = placeholders

    p1.markdown(step_card_html("01", "Load Documents", "running"), unsafe_allow_html=True)
    all_docs = []
    for _name, file_path in pending_files:
        try:
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
        finally:
            os.unlink(file_path)
    p1.markdown(step_card_html("01", "Load Documents", "done", f"{len(all_docs)} pages parsed"), unsafe_allow_html=True)

    p2.markdown(step_card_html("02", "Split into Chunks", "running"), unsafe_allow_html=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    p2.markdown(step_card_html("02", "Split into Chunks", "done", f"{len(chunks)} chunks created"), unsafe_allow_html=True)

    p3.markdown(step_card_html("03", "Generate Embeddings", "running"), unsafe_allow_html=True)
    embeddings = load_embedding_model()
    p3.markdown(step_card_html("03", "Generate Embeddings", "done", "Embedding model ready"), unsafe_allow_html=True)

    p4.markdown(step_card_html("04", "Build Vector Index", "running"), unsafe_allow_html=True)
    try:
        # Fresh in-memory client + uniquely-named collection every time, so
        # old documents never leak into a new session's answers.
        chroma_client = get_ephemeral_chroma_client()
        collection_name = f"docuchat_{uuid.uuid4().hex}"
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=chroma_client,
            collection_name=collection_name,
        )
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
        st.rerun()
    except Exception as e:
        p4.markdown(step_card_html("04", "Build Vector Index", "waiting", "Failed — see error below"), unsafe_allow_html=True)
        st.error(f"Couldn't build the index: {e}")


# =============================================================================
# ROUTING — landing page until a document is indexed, then sidebar + chat
# =============================================================================
ready = st.session_state.retriever is not None

if not ready:
    # ---------------- LANDING PAGE (no sidebar) ----------------
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

    _sp1, upload_col, _sp2 = st.columns([1, 2, 1])
    with upload_col:
        build_clicked, pending_files = render_upload_controls("main")

    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    p1, p2, p3, p4 = pc1.empty(), pc2.empty(), pc3.empty(), pc4.empty()
    p1.markdown(step_card_html("01", "Load Documents", "waiting"), unsafe_allow_html=True)
    p2.markdown(step_card_html("02", "Split into Chunks", "waiting"), unsafe_allow_html=True)
    p3.markdown(step_card_html("03", "Generate Embeddings", "waiting"), unsafe_allow_html=True)
    p4.markdown(step_card_html("04", "Build Vector Index", "waiting"), unsafe_allow_html=True)

    if build_clicked and pending_files:
        process_upload(pending_files, (p1, p2, p3, p4))

    st.markdown("""
    <div class="notice">
        DocuChat · Powered by LangChain · RAG
    </div>
    """, unsafe_allow_html=True)

else:
    # ---------------- SIDEBAR: appears only once a document is indexed ----------------
    with st.sidebar:
        st.markdown('<div class="hero-eyebrow" style="text-align:left;">DocuChat</div>', unsafe_allow_html=True)

        chips = "".join(f'<span class="doc-chip">{name}</span>' for name in st.session_state.doc_names)
        st.markdown(
            f'<div class="chip-row"><span class="chip-label">INDEXED →&nbsp;</span>{chips}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{st.session_state.num_pages} pages · {st.session_state.num_chunks} chunks")

        if st.button("Delete Index", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.vectorstore = None
            st.session_state.doc_names = []
            st.session_state.num_chunks = 0
            st.session_state.num_pages = 0
            st.session_state.chat_history = []
            st.session_state.queries_asked = 0
            st.rerun()

        st.markdown('<div class="section-heading" style="font-size:1rem; margin:1.4rem 0 0.6rem;">Add More Documents</div>', unsafe_allow_html=True)
        build_clicked, pending_files = render_upload_controls("sidebar")

        sp1, sp2, sp3, sp4 = st.empty(), st.empty(), st.empty(), st.empty()
        if build_clicked and pending_files:
            process_upload(pending_files, (sp1, sp2, sp3, sp4))

    # ---------------- MAIN: chat ----------------
    def render_source_details(sources) -> str:
        if not sources:
            return ""
        cards = "".join(
            f'<div class="source-card"><b>Source {i} · Page {s["page"]}</b><br>{s["text"]}</div>'
            for i, s in enumerate(sources, start=1)
        )
        return f'<details><summary>View {len(sources)} source excerpts</summary>{cards}</details>'

    bubbles_html = '<div class="chat-window">'
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            bubbles_html += f'<div class="msg-bubble user-bubble">{msg["content"]}</div>'
        else:
            meta_bits = [f'⏱ {msg.get("elapsed", "—")}s']
            if msg.get("relevance") is not None:
                meta_bits.append(f'Relevance ~{msg["relevance"]}%')
            meta_line = " &nbsp;·&nbsp; ".join(meta_bits)
            sources_html = render_source_details(msg.get("sources"))
            bubbles_html += (
                f'<div class="msg-bubble bot-bubble">{msg["content"]}'
                f'<div class="bubble-meta">{meta_line}</div>{sources_html}</div>'
            )
    bubbles_html += "</div>"
    chat_window = st.empty()
    chat_window.markdown(bubbles_html, unsafe_allow_html=True)

    query = st.chat_input("Ask a question about your document…")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.last_query = query
        st.session_state.queries_asked += 1

        thinking_html = (
            bubbles_html[:-6]
            + f'<div class="msg-bubble user-bubble">{query}</div>'
            + '<div class="typing-indicator"><span></span><span></span><span></span></div></div>'
        )
        chat_window.markdown(thinking_html, unsafe_allow_html=True)

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
