import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [session, setSession] = useState(null);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [queryInput, setQueryInput] = useState('');
  const [error, setError] = useState('');
  
  // Pipeline Step States: 'waiting' | 'running' | 'done'
  const [stepStates, setStepStates] = useState({
    step1: 'waiting',
    step2: 'waiting',
    step3: 'waiting',
    step4: 'waiting',
  });
  const [stepDetails, setStepDetails] = useState({
    step1: '',
    step2: '',
    step3: '',
    step4: '',
  });

  const chatWindowRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll chat window
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [chatHistory, chatLoading]);

  // Check session on mount
  useEffect(() => {
    const existingSessionId = localStorage.getItem('docuchat_session_id');
    if (existingSessionId) {
      fetch(`${API_URL}/api/status/${existingSessionId}`)
        .then((res) => {
          if (!res.ok) throw new Error('Failed to verify session');
          return res.json();
        })
        .then((data) => {
          if (data.active) {
            setSession({
              session_id: existingSessionId,
              doc_names: data.doc_names,
              num_pages: data.num_pages,
              num_chunks: data.num_chunks
            });
          } else {
            localStorage.removeItem('docuchat_session_id');
          }
        })
        .catch((err) => {
          console.warn('Session inactive or offline:', err.message);
          localStorage.removeItem('docuchat_session_id');
        });
    }
  }, []);

  // Run pipeline simulation during loading
  const runPipelineSimulation = (promise) => {
    setLoading(true);
    setError('');
    setStepStates({
      step1: 'running',
      step2: 'waiting',
      step3: 'waiting',
      step4: 'waiting',
    });
    setStepDetails({
      step1: 'Reading document data...',
      step2: '',
      step3: '',
      step4: '',
    });

    const timers = [];

    // Simulate Step 1 -> 2
    timers.push(setTimeout(() => {
      setStepStates(prev => ({ ...prev, step1: 'done', step2: 'running' }));
      setStepDetails(prev => ({ 
        ...prev, 
        step1: 'Documents successfully parsed', 
        step2: 'Splitting into 1000-character segments...' 
      }));
    }, 1500));

    // Simulate Step 2 -> 3
    timers.push(setTimeout(() => {
      setStepStates(prev => ({ ...prev, step2: 'done', step3: 'running' }));
      setStepDetails(prev => ({ 
        ...prev, 
        step2: 'Text chunks prepared', 
        step3: 'Generating dense vector representations...' 
      }));
    }, 3000));

    // Simulate Step 3 -> 4
    timers.push(setTimeout(() => {
      setStepStates(prev => ({ ...prev, step3: 'done', step4: 'running' }));
      setStepDetails(prev => ({ 
        ...prev, 
        step3: 'Embeddings loaded into memory', 
        step4: 'Building index structure...' 
      }));
    }, 4500));

    promise
      .then((data) => {
        // Clear simulated timeouts
        timers.forEach(t => clearTimeout(t));

        // Set all to done with actual values
        setStepStates({
          step1: 'done',
          step2: 'done',
          step3: 'done',
          step4: 'done',
        });
        setStepDetails({
          step1: `${data.num_pages} pages parsed`,
          step2: `${data.num_chunks} chunks created`,
          step3: 'Embedding model ready',
          step4: 'Vector database ready',
        });

        // Store session
        localStorage.setItem('docuchat_session_id', data.session_id);
        
        // Wait a second for user to view the complete pipeline before showing chat
        setTimeout(() => {
          setSession(data);
          setLoading(false);
          setFiles([]);
          setUrlInput('');
        }, 1200);
      })
      .catch((err) => {
        timers.forEach(t => clearTimeout(t));
        setError(err.message || 'Build failed.');
        setLoading(false);
        setStepStates({
          step1: 'waiting',
          step2: 'waiting',
          step3: 'waiting',
          step4: 'waiting',
        });
        setStepDetails({
          step1: '',
          step2: '',
          step3: '',
          step4: '',
        });
      });
  };

  const handleFileUpload = (e) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleBuildFromFiles = (e) => {
    e.preventDefault();
    if (files.length === 0) return;

    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    
    const existingSessionId = session?.session_id || '';
    if (existingSessionId) {
      formData.append('session_id', existingSessionId);
    }

    const uploadPromise = fetch(`${API_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    }).then(async (res) => {
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Upload failed');
      }
      return res.json();
    });

    runPipelineSimulation(uploadPromise);
  };

  const handleQuery = (e) => {
    e.preventDefault();
    if (!queryInput.trim() || !session || chatLoading) return;

    const query = queryInput.trim();
    setQueryInput('');
    
    // Add user message to history
    setChatHistory(prev => [...prev, { role: 'user', content: query }]);
    setChatLoading(true);

    fetch(`${API_URL}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: session.session_id,
        query: query
      }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Query failed. Session might have expired.');
        }
        return res.json();
      })
      .then((data) => {
        setChatHistory(prev => [
          ...prev,
          {
            role: 'assistant',
            content: data.content,
            elapsed: data.elapsed,
            relevance: data.relevance,
            sources: data.sources
          }
        ]);
        setChatLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setChatLoading(false);
        // Add error notification directly into chat window
        setChatHistory(prev => [
          ...prev,
          {
            role: 'assistant',
            content: `Error: ${err.message}. Please upload documents again to reinitialize the session.`,
            elapsed: 0,
            relevance: null,
            sources: []
          }
        ]);
      });
  };

  const handleDeleteIndex = () => {
    if (!session) return;
    
    fetch(`${API_URL}/api/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: session.session_id }),
    })
      .then(() => {
        localStorage.removeItem('docuchat_session_id');
        setSession(null);
        setChatHistory([]);
        setError('');
      })
      .catch((err) => {
        console.error('Delete failed:', err);
        // Force state reset anyway
        localStorage.removeItem('docuchat_session_id');
        setSession(null);
        setChatHistory([]);
      });
  };

  const getStepCardHtml = (num, title, state, desc) => {
    const statusLabel = {
      waiting: 'WAITING',
      running: '● RUNNING',
      done: '✓ DONE',
    }[state];

    return (
      <div className={`step-card ${state === 'running' ? 'active' : state === 'done' ? 'done' : ''}`}>
        <div className="step-header">
          <span className="step-num">{num}</span>
          <span className="step-title">{title}</span>
          <span className={`step-status status-${state}`}>{statusLabel}</span>
        </div>
        {desc && <div className="step-desc">{desc}</div>}
      </div>
    );
  };

  const dragOver = (e) => {
    e.preventDefault();
  };

  const fileDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      setFiles(Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf'));
    }
  };

  return (
    <div className={`app-container ${!session ? 'no-glow' : ''}`}>
      {/* Header */}
      {session && (
        <header className="app-header">
          <a href="/" className="logo">
            Docu<span>Chat</span>
          </a>
          <button onClick={handleDeleteIndex} className="btn-secondary" style={{ width: 'auto' }}>
            Delete Index
          </button>
        </header>
      )}

      {/* Main Layout */}
      {!session ? (
        // ---------------- LANDING VIEW ----------------
        <div className="landing-container">
          <div className="hero">
            <div className="hero-eyebrow">Retrieval Augmented Generation</div>
            <h1>Docu<span>Chat</span></h1>
            <p className="hero-sub">
              Upload your documents and get answers grounded strictly in their content.
            </p>
          </div>

          <div className="input-card">
            {error && <div className="alert alert-danger">{error}</div>}

            <div className="tab-content">
              <form onSubmit={handleBuildFromFiles}>
                  <div 
                    className={`dropzone ${loading ? 'dropzone-loading' : ''}`} 
                    onClick={() => !loading && fileInputRef.current?.click()}
                    onDragOver={dragOver}
                    onDrop={fileDrop}
                  >
                    <input 
                      ref={fileInputRef}
                      type="file" 
                      multiple 
                      accept=".pdf" 
                      onChange={handleFileUpload} 
                      style={{ display: 'none' }}
                      disabled={loading}
                    />
                    {loading ? (
                      <>
                        <div className="chip-label" style={{ fontSize: '0.6rem' }}>Indexing status:</div>
                        <span style={{ color: 'var(--orange)' }}>
                          {stepStates.step4 === 'running' ? 'Rebuilding vector index...' :
                           stepStates.step3 === 'running' ? 'Generating embeddings...' :
                           stepStates.step2 === 'running' ? 'Splitting texts...' : 'Parsing documents...'}
                        </span>
                      </>
                    ) : (
                      <>
                        <div className="dropzone-icon">📄</div>
                        <span>Drag and drop your PDFs here, or click to browse</span>
                        <small>Only PDF files are supported</small>
                      </>
                    )}
                  </div>
                  {files.length > 0 && !loading && (
                    <div className="file-list-preview">
                      {files.map((file, idx) => (
                        <div key={idx} className="file-item">
                          📄 {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                        </div>
                      ))}
                    </div>
                  )}
                  <button 
                    type="submit" 
                    className="btn-primary" 
                    disabled={files.length === 0 || loading}
                  >
                    {loading ? <div className="spinner"></div> : 'Build'}
                  </button>
              </form>
            </div>
          </div>
        </div>
      ) : (
        // ---------------- SPLIT CHAT/SIDEBAR VIEW ----------------
        <div className="layout-split">
          {/* Sidebar */}
          <aside className="sidebar">
            <div className="indexed-row">
              <div className="chip-label">Indexed Documents</div>
              <div className="chip-row">
                {session.doc_names && session.doc_names.map((name, i) => (
                  <span key={i} className="doc-chip" title={name}>{name}</span>
                ))}
              </div>
              <div className="stats-caption">
                {session.num_pages} pages · {session.num_chunks} chunks
              </div>
            </div>

            <div className="sidebar-divider"></div>

            <div>
              <form onSubmit={handleBuildFromFiles}>
                  <div 
                    className={`dropzone ${loading ? 'dropzone-loading' : ''}`} 
                    onClick={() => !loading && fileInputRef.current?.click()}
                    style={{ padding: '1rem 0.5rem', gap: '0.2rem' }}
                  >
                    <input 
                      ref={fileInputRef}
                      type="file" 
                      multiple 
                      accept=".pdf" 
                      onChange={handleFileUpload} 
                      style={{ display: 'none' }}
                      disabled={loading}
                    />
                    {loading ? (
                      <>
                        <div className="chip-label" style={{ fontSize: '0.6rem' }}>Indexing status:</div>
                        <span style={{ fontSize: '0.75rem', color: 'var(--orange)' }}>
                          {stepStates.step4 === 'running' ? 'Rebuilding vector index...' :
                           stepStates.step3 === 'running' ? 'Generating embeddings...' :
                           stepStates.step2 === 'running' ? 'Splitting texts...' : 'Parsing documents...'}
                        </span>
                      </>
                    ) : (
                      <>
                        <div style={{ fontSize: '1.2rem' }}>📄</div>
                        <span style={{ fontSize: '0.8rem' }}>Upload more PDFs</span>
                      </>
                    )}
                  </div>
                  {files.length > 0 && !loading && (
                    <div className="file-list-preview" style={{ maxHeight: '60px' }}>
                      {files.map((file, idx) => (
                        <div key={idx} className="file-item">
                          {file.name}
                        </div>
                      ))}
                    </div>
                  )}
                  <button 
                    type="submit" 
                    className="btn-primary" 
                    disabled={files.length === 0 || loading}
                    style={{ padding: '0.5rem 1rem', fontSize: '0.85rem', marginTop: '0.5rem' }}
                  >
                    {loading ? <div className="spinner" style={{ width: '15px', height: '15px' }}></div> : 'Build'}
                  </button>
              </form>
            </div>
          </aside>

          {/* Chat Interface */}
          <main className="chat-panel">
            <div className="chat-window" ref={chatWindowRef}>
              {chatHistory.length === 0 && (
                <div style={{ margin: 'auto', textAlign: 'center', opacity: 0.5, maxWidth: '400px' }}>
                  <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>💬</div>
                  <h3>Knowledge Base Ready</h3>
                  <p style={{ fontSize: '0.85rem', marginTop: '0.5rem', lineHeight: 1.5 }}>
                    Ask any question about your indexed documents. The system will retrieve matching sections and synthesize an answer.
                  </p>
                </div>
              )}

              {chatHistory.map((msg, index) => (
                <div key={index} className={`msg-bubble ${msg.role === 'user' ? 'user-bubble' : 'bot-bubble'}`}>
                  {msg.role === 'assistant' ? (
                    <div className="markdown-body">
                      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <span>{msg.content}</span>
                  )}
                  {msg.role === 'assistant' && (
                    <>
                      <div className="bubble-meta">
                        <span>⏱ {msg.elapsed}s</span>
                        {msg.relevance !== null && <span>Relevance ~{msg.relevance}%</span>}
                      </div>
                      {msg.sources && msg.sources.length > 0 && (
                        <details>
                          <summary>View {msg.sources.length} source excerpts</summary>
                          {msg.sources.map((src, i) => (
                            <div key={i} className="source-card">
                              <b>Source {i + 1} · Page {src.page}</b>
                              {src.text}
                            </div>
                          ))}
                        </details>
                      )}
                    </>
                  )}
                </div>
              ))}

              {chatLoading && (
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              )}
            </div>

            <div className="chat-input-container">
              <form onSubmit={handleQuery} className="chat-form">
                <input 
                  type="text" 
                  className="chat-input" 
                  placeholder="Ask a question about your document…" 
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  disabled={chatLoading}
                  required
                />
                <button type="submit" className="chat-submit-btn" disabled={!queryInput.trim() || chatLoading}>
                  <svg viewBox="0 0 24 24">
                    <path d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
                  </svg>
                </button>
              </form>
            </div>
          </main>
        </div>
      )}
    </div>
  );
}

export default App;
