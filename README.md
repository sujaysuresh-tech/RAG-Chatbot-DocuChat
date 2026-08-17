<div align="center">

# 📚 DocuChat 🤖

### 💬 Chat With Your PDFs, Powered by RAG

<img src="https://skillicons.dev/icons?i=html,css,js,react,vite,python,fastapi,vercel,git,github&theme=dark" />

<br/>

[Overview](#-overview) • [Features](#-features) • [Tech Stack](#️-tech-stack) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Roadmap](#-roadmap)

</div>

---

## 🌟 Overview

**DocuChat** is a full-stack Retrieval-Augmented Generation (RAG) chatbot that lets you upload one or more PDFs and have a real conversation about their contents. 📄💬 Instead of relying on an LLM's raw memory, DocuChat chunks and embeds your documents, retrieves the most relevant passages for every question, and grounds Gemini's answers directly in that retrieved context — with page-level source citations so you always know where an answer came from. 🔍

The project explores a production-style RAG pipeline end-to-end: a React frontend for upload and chat, a FastAPI backend for chunking/embedding/retrieval, ChromaDB as the vector store, and Google Gemini for generation — all wired together with session isolation so multiple people can use it at once without their documents or conversations ever mixing. 🔐

🔗 **Live demo:** [Live Demo](https://rag-chatbot-docuchat.vercel.app/)

---

## 🚀 Features

- 📤 **Multi-PDF upload** — drag-and-drop or upload-by-URL, with several documents indexed into one session
- 🧠 **Retrieval-Augmented Generation** — answers are grounded in the actual document content, not just the model's memory
- 📎 **Source citations** — every answer links back to the exact filename and page it came from
- 🗣️ **Conversational memory** — follow-up questions ("what about that?", "explain more") are understood in context, per session
- 📝 **Smart handling of open-ended asks** — "explain", "summarize", and "give me an overview" pull a broad, representative sample of the document instead of failing to find a specific "answer"
- 🌗 **Dark mode UI** with markdown, math (KaTeX), and GFM rendering in chat responses
- 👥 **Multi-user session isolation** — every user gets their own vector collection and chat history, with automatic cleanup of inactive sessions
- ☁️ **Always-on backend** on Render's free tier via a self-pinging keep-alive loop

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| 🎨 Frontend | React 19, Vite |
| 📝 Rendering | react-markdown, remark-gfm, remark-math, KaTeX |
| ⚙️ Backend | FastAPI (Python) |
| 🔗 Orchestration | LangChain |
| 🧩 Vector Store | ChromaDB |
| 🧬 Embeddings | HuggingFace Inference Endpoint (`all-MiniLM-L6-v2`) |
| 🤖 AI Model | Google Gemini |
| 📄 PDF Parsing | pypdf / PyPDFLoader |
| ☁️ Deployment | Frontend on Vercel · Backend on Render |

---

## 🏗️ Architecture

```
RAG-Chatbot-DocuChat/
├── 📁 backend/
│   ├── 📁 app/
│   │   └── 🐍 main.py         # FastAPI app — upload, query, session & memory management
│   └── 📄 requirements.txt
├── 📁 frontend/
│   ├── 📁 src/
│   │   ├── ⚛️ App.jsx          # Upload screen, chat UI, source citations
│   │   ├── 🎨 index.css        # Dark mode styling
│   │   └── 🚀 main.jsx
│   ├── 📄 index.html
│   └── 📦 package.json
└── 📘 README.md
```

**Flow:** 📄 user uploads PDF(s) → backend parses pages with `PyPDFLoader`, splits them into overlapping chunks, and embeds them into a **per-session** ChromaDB collection → 🧑 user asks a question → the query (plus recent chat history, for follow-ups) is used to retrieve the most relevant chunks via MMR search → those chunks, labeled with filename and page, are passed to **Gemini** along with the conversation history → the grounded answer streams back to the chat window with clickable source excerpts. 🔍

Every session is keyed by a unique `session_id`, so each user's documents, embeddings, and chat history live in complete isolation from everyone else's — and inactive sessions are automatically evicted to keep the backend's memory footprint bounded. 🧹

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sujaysuresh-tech/RAG-Chatbot-DocuChat.git
cd RAG-Chatbot-DocuChat
```

**Backend:**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

---

## 🔑 Configuration

DocuChat's backend needs API keys for embeddings and generation.

Add these as environment variables (locally in a `.env` file inside `backend/`, or in your Render project settings for deployment):

```bash
GOOGLE_API_KEY=your-gemini-api-key
HUGGINGFACEHUB_API_TOKEN=your-huggingface-token
```

> ⚠️ Never commit your API keys to a public repository — that's why they live in environment variables, kept off the client and out of version control. 🙅‍♂️

---

## 💡 Usage

1. Open the app locally, or try the **[live demo](https://rag-chatbot-docuchat.vercel.app/)** 🔴
2. Upload one or more PDFs via drag-and-drop 📤
3. Ask a question, or try something open-ended like "summarize this document" 💬
4. Expand **"View source excerpts"** under any answer to see exactly which file and page it came from 📎

---

## 🗺️ Roadmap

- [ ] 🖇️ Support for more file types (DOCX, TXT, URLs beyond PDF)
- [ ] 📡 Streaming responses (token-by-token rendering)
- [ ] 🗂️ Persistent, resumable sessions across visits
- [ ] 📊 Per-document summaries alongside the chat view
- [ ] ☀️ Light mode toggle

---


<div align="center">

Built by [Sujay Suresh](https://github.com/sujaysuresh-tech) · 🔗 [Live Demo](https://rag-chatbot-docuchat.vercel.app/) 

</div>
