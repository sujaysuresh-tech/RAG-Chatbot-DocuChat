"""
DocuChat — Premium RAG frontend (v2)
=====================================
Presentation-only layer for a Retrieval-Augmented Generation (RAG)
application. Retrieval logic, prompt template, embedding model, and LLM
are unchanged from the original backend.

Run with: streamlit run app.py
"""

import os
import tempfile
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="DocuChat — AI Document Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral-small-2506"

# =============================================================================
# ICONS — minimal inline SVGs (stroke-based, inherits color) instead of emoji
# =============================================================================
ICONS = {
    "logo": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2 L22 12 L12 22 L2 12 Z"/></svg>',
    "upload": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M12 4l-5 5M12 4l5 5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>',
    "document": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>',
    "database": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/></svg>',
    "chunks": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    "chat": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "clock": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>',
    "spark": '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l1.9 5.8L20 10l-6.1 2.2L12 18l-1.9-5.8L4 10l6.1-2.2z"/></svg>',
    "user": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg>',
}


def icon(name: str) -> str:
    return ICONS.get(name, "")


# =============================================================================
# CUSTOM CSS — layered depth, noise texture, refined type scale
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --bg-primary: #0B0F19;
    --bg-secondary: #121826;
    --bg-card: #181F2F;
    --bg-card-hover: #1D2538;
    --accent: #6366F1;
    --accent-hover: #7C83FF;
    --accent-dim: rgba(99,102,241,0.14);
    --border: rgba(255,255,255,0.07);
    --border-strong: rgba(255,255,255,0.13);
    --text-primary: #FFFFFF;
    --text-secondary: #AEB8CE;
    --text-muted: #626C87;
    --success: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-pill: 999px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.24);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.28), 0 1px 3px rgba(0,0,0,0.3);
    --shadow-lg: 0 12px 40px rgba(0,0,0,0.38), 0 2px 8px rgba(0,0,0,0.24);
    --shadow-glow: 0 8px 30px rgba(99,102,241,0.22);
}

html, body, [class*="css"] {
    font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;
}

.stApp {
    background:
        radial-gradient(ellipse 65% 45% at 15% -8%, rgba(99,102,241,0.13), transparent 55%),
        radial-gradient(ellipse 55% 40% at 105% 5%, rgba(124,131,255,0.08), transparent 55%),
        radial-gradient(ellipse 50% 35% at 50% 105%, rgba(99,102,241,0.05), transparent 55%),
        var(--bg-primary);
    color: var(--text-primary);
}

/* subtle film-grain overlay for texture, premium-app signature */
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

.block-container { animation: pageIn 0.4s ease; }
@keyframes pageIn { from { opacity: 0; } to { opacity: 1; } }

/* ---------------------------------------------------------------------- */
/* Top bar                                                                 */
/* ---------------------------------------------------------------------- */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 22px;
    background: rgba(18,24,38,0.7);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 26px;
    box-shadow: var(--shadow-md);
}
.topbar-left { display: flex; align-items: center; gap: 11px; }
.brand-mark {
    width: 30px; height: 30px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--accent), var(--accent-hover));
    color: #fff;
    box-shadow: var(--shadow-glow);
}
.topbar-name {
    font-family: "Manrope", sans-serif;
    font-weight: 700;
    font-size: 15.5px;
    letter-spacing: -0.2px;
}
.topbar-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.meta-chip {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 5px 12px 5px 10px;
    border-radius: var(--radius-pill);
    letter-spacing: 0.1px;
}
.meta-chip.status-live { color: var(--success); border-color: rgba(16,185,129,0.25); }
.meta-chip.status-idle { color: var(--text-muted); }
.pulse-dot { position: relative; width: 6px; height: 6px; border-radius: 50%; }
.pulse-dot.live { background: var(--success); }
.pulse-dot.live::after {
    content: ""; position: absolute; inset: -4px; border-radius: 50%;
    border: 1px solid var(--success); animation: ping 1.8s ease-out infinite;
}
.pulse-dot.idle { background: var(--text-muted); }
@keyframes ping { 0% { transform: scale(0.8); opacity: 0.8; } 100% { transform: scale(2.2); opacity: 0; } }

/* ---------------------------------------------------------------------- */
/* Hero                                                                    */
/* ---------------------------------------------------------------------- */
.hero { text-align: center; padding: 56px 20px 40px; }
.hero .kicker {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 700; letter-spacing: 1.6px;
    text-transform: uppercase; color: var(--accent-hover);
    background: var(--accent-dim);
    border: 1px solid rgba(99,102,241,0.25);
    padding: 6px 14px; border-radius: var(--radius-pill);
    margin-bottom: 22px;
}
.hero h1 {
    font-family: "Manrope", sans-serif;
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1.4px;
    line-height: 1.1;
    margin: 0 0 14px;
    background: linear-gradient(135deg, #ffffff 25%, #A7ACFF 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}
.hero p {
    font-size: 15.5px;
    color: var(--text-secondary);
    max-width: 460px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ---------------------------------------------------------------------- */
/* Sidebar                                                                 */
/* ---------------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-secondary), #0E1220) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

.sb-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 26px; }
.sb-brand .brand-mark { width: 28px; height: 28px; }
.sb-brand span { font-family: "Manrope", sans-serif; font-weight: 700; font-size: 15px; }

.sb-section-label {
    display: flex; align-items: center; gap: 6px;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 24px 0 11px;
}

.upload-hint {
    display: flex; align-items: flex-start; gap: 9px;
    background: var(--accent-dim);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    margin-bottom: 12px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
}
.upload-hint svg { flex-shrink: 0; margin-top: 2px; color: var(--accent-hover); }

.doc-item {
    display: flex;
    align-items: center;
    gap: 9px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 9px 12px;
    margin-bottom: 6px;
    font-size: 12.5px;
    color: var(--text-secondary);
    transition: border-color 0.15s, background 0.15s;
}
.doc-item:hover { border-color: var(--border-strong); background: var(--bg-card-hover); }
.doc-item svg { color: var(--accent-hover); flex-shrink: 0; }

.db-status-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    box-shadow: var(--shadow-sm);
}
.db-status-card .label {
    display: flex; align-items: center; gap: 6px;
    font-size: 10.5px; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.8px;
}
.db-status-card .value { font-size: 14.5px; font-weight: 600; margin-top: 6px; }
.value-ready { color: var(--success); }
.value-empty { color: var(--text-muted); }

.sb-footer {
    margin-top: 34px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 10.5px;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.2px;
}

/* ---------------------------------------------------------------------- */
/* Buttons                                                                 */
/* ---------------------------------------------------------------------- */
.stButton > button {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 9px 16px;
    font-weight: 600;
    font-size: 13px;
    width: 100%;
    transition: all 0.18s cubic-bezier(0.2, 0, 0.1, 1);
    box-shadow: var(--shadow-sm);
}
.stButton > button:hover {
    border-color: var(--accent);
    background: var(--bg-card-hover);
    transform: translateY(-1.5px);
    box-shadow: var(--shadow-md);
}
.stButton > button:active { transform: translateY(0); }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-hover));
    border: none;
    color: #fff;
    box-shadow: var(--shadow-glow);
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 10px 34px rgba(99,102,241,0.32);
}

/* ---------------------------------------------------------------------- */
/* File uploader                                                          */
/* ---------------------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--bg-card-hover) !important;
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: var(--text-secondary) !important; }

/* ---------------------------------------------------------------------- */
/* Stat cards                                                             */
/* ---------------------------------------------------------------------- */
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 17px 19px;
    box-shadow: var(--shadow-sm);
    transition: transform 0.18s, border-color 0.18s, box-shadow 0.18s;
}
.stat-card:hover { transform: translateY(-2px); border-color: var(--border-strong); box-shadow: var(--shadow-md); }
.stat-card .stat-icon {
    width: 30px; height: 30px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    background: var(--accent-dim); color: var(--accent-hover);
    margin-bottom: 12px;
}
.stat-card .stat-label { font-size: 10.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.7px; font-weight: 600; }
.stat-card .stat-value { font-family: "Manrope", sans-serif; font-size: 25px; font-weight: 700; margin-top: 5px; letter-spacing: -0.5px; }

/* ---------------------------------------------------------------------- */
/* Chat — custom avatar row + bubbles                                     */
/* ---------------------------------------------------------------------- */
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] { display: none; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display: none; }

[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 18px !important;
    margin-bottom: 14px !important;
    box-shadow: var(--shadow-sm);
    animation: fadeInUp 0.32s cubic-bezier(0.2, 0, 0.1, 1);
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] p { font-size: 14.5px; line-height: 1.65; color: var(--text-primary); margin: 0; }

[data-testid="stChatMessage"]:has(.role-user) {
    background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(124,131,255,0.04)) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
}

.msg-avatar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.msg-avatar {
    width: 22px; height: 22px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.msg-avatar.assistant { background: linear-gradient(135deg, var(--accent), var(--accent-hover)); color: #fff; }
.msg-avatar.user { background: var(--bg-card-hover); border: 1px solid var(--border-strong); color: var(--text-secondary); }
.msg-name { font-size: 12.5px; font-weight: 600; color: var(--text-primary); }
.msg-time { font-size: 10.5px; color: var(--text-muted); margin-left: 2px; }

.msg-meta {
    display: flex; align-items: center; gap: 8px;
    font-size: 10.5px; color: var(--text-muted);
    margin-top: 10px; padding-top: 10px;
    border-top: 1px solid var(--border);
}
.msg-meta svg { vertical-align: -1px; margin-right: 3px; }
.confidence-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10px; font-weight: 600;
    padding: 3px 9px;
    border-radius: var(--radius-pill);
    border: 1px solid rgba(16,185,129,0.25);
    color: var(--success);
    background: rgba(16,185,129,0.08);
}

/* ---------------------------------------------------------------------- */
/* Source cards                                                           */
/* ---------------------------------------------------------------------- */
.source-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 15px;
    margin-bottom: 8px;
    font-size: 12.5px;
    color: var(--text-secondary);
    line-height: 1.55;
    position: relative;
    padding-left: 34px;
}
.source-card .src-num {
    position: absolute; left: 12px; top: 12px;
    width: 16px; height: 16px; border-radius: 5px;
    background: var(--accent-dim); color: var(--accent-hover);
    font-size: 10px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.source-card b {
    color: var(--accent-hover);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
}

[data-testid="stExpander"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary { font-size: 12.5px !important; color: var(--text-secondary) !important; }

/* ---------------------------------------------------------------------- */
/* Thinking indicator                                                      */
/* ---------------------------------------------------------------------- */
.thinking-row { display: flex; align-items: center; gap: 6px; padding: 4px 2px; }
.thinking-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent-hover);
    animation: bounce 1.2s ease-in-out infinite;
}
.thinking-dot:nth-child(2) { animation-delay: 0.15s; }
.thinking-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-4px); opacity: 1; }
}

/* ---------------------------------------------------------------------- */
/* Chat input                                                             */
/* ---------------------------------------------------------------------- */
[data-testid="stChatInput"] { background: transparent !important; }
[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 22px !important;
    box-shadow: var(--shadow-sm);
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.16) !important;
}
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-hover)) !important;
    border-radius: 50% !important;
    box-shadow: var(--shadow-glow);
}

/* ---------------------------------------------------------------------- */
/* Alerts / progress                                                      */
/* ---------------------------------------------------------------------- */
[data-testid="stAlertContentSuccess"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(16,185,129,0.28) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
[data-testid="stAlertContentInfo"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
}
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent-hover)) !important; }

hr { border-color: var(--border) !important; }

/* ---------------------------------------------------------------------- */
/* Scrollbar                                                              */
/* ---------------------------------------------------------------------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }

@media (max-width: 900px) {
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
        "show_sources": True,
        "show_relevance": True,
        "last_query": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# =============================================================================
# CACHED RESOURCES (embeddings / LLM — unchanged from original backend)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatMistralAI(model=LLM_MODEL_NAME)


def build_prompt() -> ChatPromptTemplate:
    """Unchanged prompt template from the original backend."""
    return ChatPromptTemplate.from_messages(
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


PROMPT = build_prompt()


# =============================================================================
# BACKEND HELPERS
# =============================================================================
def process_pdfs(uploaded_files):
    """Load, chunk, and embed one or more PDFs into a fresh in-memory
    Chroma vector store. Retrieval parameters unchanged from the backend."""
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        try:
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
        finally:
            os.unlink(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    embeddings = load_embedding_model()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    return vectorstore, retriever, len(chunks), len(all_docs)


def estimate_relevance(vectorstore, query: str):
    """Best-effort relevance readout from real similarity scores. Purely
    additive — does not change what context is fed to the LLM."""
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=4)
        scores = [score for _, score in results if score is not None]
        if not scores:
            return None
        return round(sum(scores) / len(scores) * 100, 1)
    except Exception:
        return None


def avatar_row(role: str, ts: str) -> str:
    if role == "user":
        return (
            f'<div class="msg-avatar-row role-user"><div class="msg-avatar user">{icon("user")}</div>'
            f'<span class="msg-name">You</span><span class="msg-time">{ts}</span></div>'
        )
    return (
        f'<div class="msg-avatar-row"><div class="msg-avatar assistant">{icon("logo")}</div>'
        f'<span class="msg-name">DocuChat</span><span class="msg-time">{ts}</span></div>'
    )


def answer_query(query: str, chat_box):
    """Runs retrieval + LLM generation, rendering a thinking indicator,
    a streamed answer, and a full answer card with sources."""
    ts_user = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({"role": "user", "content": query, "ts": ts_user})
    st.session_state.last_query = query
    st.session_state.queries_asked += 1

    with chat_box:
        with st.chat_message("user"):
            st.markdown(avatar_row("user", ts_user), unsafe_allow_html=True)
            st.markdown(query)

        with st.chat_message("assistant"):
            ts_bot = datetime.now().strftime("%H:%M")
            st.markdown(avatar_row("assistant", ts_bot), unsafe_allow_html=True)
            thinking = st.empty()
            thinking.markdown(
                '<div class="thinking-row"><div class="thinking-dot"></div>'
                '<div class="thinking-dot"></div><div class="thinking-dot"></div></div>',
                unsafe_allow_html=True,
            )

            start = time.time()
            docs = st.session_state.retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)
            final_prompt = PROMPT.invoke({"context": context, "question": query})

            llm = load_llm()
            answer_slot = st.empty()
            full_text = ""
            try:
                for chunk in llm.stream(final_prompt):
                    if not full_text:
                        thinking.empty()
                    full_text += chunk.content or ""
                    answer_slot.markdown(full_text + "▌")
                answer_slot.markdown(full_text)
            except Exception:
                thinking.empty()
                response = llm.invoke(final_prompt)
                full_text = response.content
                answer_slot.markdown(full_text)

            elapsed = round(time.time() - start, 2)
            relevance = estimate_relevance(st.session_state.vectorstore, query) if st.session_state.show_relevance else None

            meta_bits = [f'{icon("clock")} {elapsed}s']
            if relevance is not None:
                meta_bits.append(f'<span class="confidence-badge">{icon("spark")} {relevance}% relevance</span>')
            st.markdown(f'<div class="msg-meta">{" &nbsp;·&nbsp; ".join(meta_bits)}</div>', unsafe_allow_html=True)

            if st.session_state.show_sources and docs:
                with st.expander(f"View {len(docs)} source excerpts"):
                    for i, doc in enumerate(docs, start=1):
                        page = doc.metadata.get("page", "?")
                        excerpt = doc.page_content[:300] + "…"
                        st.markdown(
                            f'<div class="source-card"><span class="src-num">{i}</span>'
                            f'<b>Page {page}</b><br>{excerpt}</div>',
                            unsafe_allow_html=True,
                        )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": full_text,
            "ts": ts_bot,
            "sources": [
                {"page": d.metadata.get("page", "?"), "text": d.page_content[:300] + "…"}
                for d in docs
            ],
            "elapsed": elapsed,
            "relevance": relevance,
        }
    )


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f'<div class="sb-brand"><div class="brand-mark">{icon("logo")}</div>'
            "<span>DocuChat</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="sb-section-label">{icon("document")} Documents</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="upload-hint">{icon("upload")}'
            "<span>Drop one or more PDFs — they'll be chunked, embedded, "
            "and indexed into a fresh knowledge base.</span></div>",
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
        )

        if st.session_state.doc_names:
            for name in st.session_state.doc_names:
                st.markdown(
                    f'<div class="doc-item">{icon("document")}{name}</div>', unsafe_allow_html=True
                )

        build_clicked = st.button(
            "Create Vector Database", type="primary", disabled=not uploaded_files, use_container_width=True
        )

        if build_clicked and uploaded_files:
            progress = st.progress(0, text="Reading documents…")
            try:
                progress.progress(25, text="Splitting into chunks…")
                vectorstore, retriever, n_chunks, n_pages = process_pdfs(uploaded_files)
                progress.progress(75, text="Generating embeddings…")

                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.num_chunks = n_chunks
                st.session_state.num_pages = n_pages
                st.session_state.doc_names = [f.name for f in uploaded_files]
                st.session_state.chat_history = []
                st.session_state.queries_asked = 0

                progress.progress(100, text="Database ready.")
                time.sleep(0.4)
                progress.empty()
                st.success(f"Indexed {len(uploaded_files)} document(s) · {n_chunks} chunks")
            except Exception as e:
                progress.empty()
                st.error(f"Couldn't build the database: {e}")

        st.markdown(
            f'<div class="sb-section-label">{icon("database")} Knowledge Base</div>',
            unsafe_allow_html=True,
        )
        status_ready = st.session_state.retriever is not None
        st.markdown(
            f"""
            <div class="db-status-card">
                <div class="label">{icon("database")} Status</div>
                <div class="value {'value-ready' if status_ready else 'value-empty'}">
                    {'● Ready' if status_ready else '○ Empty'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Chat", use_container_width=True, disabled=not status_ready):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("Delete DB", use_container_width=True, disabled=not status_ready):
                st.session_state.retriever = None
                st.session_state.vectorstore = None
                st.session_state.doc_names = []
                st.session_state.num_chunks = 0
                st.session_state.num_pages = 0
                st.session_state.chat_history = []
                st.session_state.queries_asked = 0
                st.rerun()

        st.markdown('<div class="sb-section-label">Settings</div>', unsafe_allow_html=True)
        st.session_state.show_sources = st.toggle("Show source excerpts", value=st.session_state.show_sources)
        st.session_state.show_relevance = st.toggle("Show relevance score", value=st.session_state.show_relevance)

        with st.expander("About"):
            st.caption(
                "DocuChat answers questions strictly from the documents you upload, "
                "using retrieval-augmented generation. Nothing is sent to the model "
                "except the retrieved passages relevant to your question."
            )

        st.markdown(
            '<div class="sb-footer">DocuChat · Built with Streamlit &amp; LangChain</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# TOP BAR
# =============================================================================
def render_topbar():
    ready = st.session_state.retriever is not None
    status_class = "status-live" if ready else "status-idle"
    dot_class = "live" if ready else "idle"
    status_text = "Database Ready" if ready else "No Database"

    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-left">
                <div class="brand-mark">{icon("logo")}</div>
                <div class="topbar-name">DocuChat</div>
            </div>
            <div class="topbar-right">
                <span class="meta-chip">Model · {LLM_MODEL_NAME}</span>
                <span class="meta-chip">Embeddings · MiniLM-L6-v2</span>
                <span class="meta-chip {status_class}"><span class="pulse-dot {dot_class}"></span>{status_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STATS DASHBOARD
# =============================================================================
def render_stats():
    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">{icon("document")}</div>
                <div class="stat-label">Documents</div>
                <div class="stat-value">{len(st.session_state.doc_names)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">{icon("chunks")}</div>
                <div class="stat-label">Chunks</div>
                <div class="stat-value">{st.session_state.num_chunks}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">{icon("chat")}</div>
                <div class="stat-label">Queries Asked</div>
                <div class="stat-value">{st.session_state.queries_asked}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">{icon("database")}</div>
                <div class="stat-label">Database</div>
                <div class="stat-value" style="font-size:15px; color: var(--success);">Ready</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN
# =============================================================================
render_sidebar()
render_topbar()

if st.session_state.retriever is None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="kicker">{icon("spark")} AI DOCUMENT INTELLIGENCE</div>
            <h1>Talk to your documents</h1>
            <p>Upload one or more PDFs from the sidebar and build a knowledge base —
            then ask anything and get answers grounded strictly in your content.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    render_stats()

    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(avatar_row(msg["role"], msg.get("ts", "")), unsafe_allow_html=True)
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    meta_bits = [f'{icon("clock")} {msg.get("elapsed", "—")}s']
                    if msg.get("relevance") is not None:
                        meta_bits.append(
                            f'<span class="confidence-badge">{icon("spark")} {msg["relevance"]}% relevance</span>'
                        )
                    st.markdown(
                        f'<div class="msg-meta">{" &nbsp;·&nbsp; ".join(meta_bits)}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.session_state.show_sources and msg.get("sources"):
                        with st.expander(f"View {len(msg['sources'])} source excerpts"):
                            for i, s in enumerate(msg["sources"], start=1):
                                st.markdown(
                                    f'<div class="source-card"><span class="src-num">{i}</span>'
                                    f'<b>Page {s["page"]}</b><br>{s["text"]}</div>',
                                    unsafe_allow_html=True,
                                )

    action_cols = st.columns([1, 1, 6])
    with action_cols[0]:
        if st.session_state.last_query and st.button("↻ Regenerate", use_container_width=True):
            if len(st.session_state.chat_history) >= 2:
                st.session_state.chat_history.pop()
                st.session_state.chat_history.pop()
            answer_query(st.session_state.last_query, chat_box)
            st.rerun()
    with action_cols[1]:
        if st.button("Clear Chat", use_container_width=True, disabled=not st.session_state.chat_history):
            st.session_state.chat_history = []
            st.rerun()

    query = st.chat_input("Ask a question about your document…")
    if query:
        answer_query(query, chat_box)
        st.rerun()