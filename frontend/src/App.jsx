import React, { useState, useEffect, useRef } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SUGGESTIONS = [
  "Identify the primary objectives and key results",
  "Summarize the key takeaways from this document",
  "What are the major performance highlights mentioned?",
  "Are there any significant risk assessment notes?"
];

function App() {
  const [session, setSession] = useState(null);
  const [activeTab, setActiveTab] = useState('file'); // 'file' | 'url'
  const [files, setFiles] = useState([]);
  const [urlInput, setUrlInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [queryInput, setQueryInput] = useState('');
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [activeCitationIndex, setActiveCitationIndex] = useState(null); // tracking clicked citation

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
      step1: 'Accessing PDF stream...',
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
        step2: 'Tokenizing into 1000-character segments...' 
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
        step3: 'Embeddings computed remotely', 
        step4: 'Assembling Chroma index structures...' 
      }));
    }, 4500));

    promise
      .then((data) => {
        timers.forEach(t => clearTimeout(t));

        setStepStates({
          step1: 'done',
          step2: 'done',
          step3: 'done',
          step4: 'done',
        });
        setStepDetails({
          step1: `${data.num_pages} pages parsed`,
          step2: `${data.num_chunks} chunks indexed`,
          step3: 'Embeddings generated in cloud',
          step4: 'Index mapping verified',
        });

        localStorage.setItem('docuchat_session_id', data.session_id);
        
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

  const submitBuildFromFiles = (e) => {
    if (e) e.preventDefault();
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

  const submitBuildFromUrl = (e) => {
    if (e) e.preventDefault();
    if (!urlInput) return;

    const existingSessionId = session?.session_id || '';
    const payload = {
      url: urlInput,
      session_id: existingSessionId || null,
    };

    const uploadPromise = fetch(`${API_URL}/api/upload-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(async (res) => {
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to index URL');
      }
      return res.json();
    });

    runPipelineSimulation(uploadPromise);
  };

  const executeQuery = (queryText) => {
    if (!queryText.trim() || !session || chatLoading) return;

    setChatHistory(prev => [...prev, { role: 'user', content: queryText }]);
    setChatLoading(true);
    setActiveCitationIndex(null);

    fetch(`${API_URL}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: session.session_id,
        query: queryText
      }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Session expired or request failed.');
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
        setChatHistory(prev => [
          ...prev,
          {
            role: 'assistant',
            content: `Error: ${err.message}. Re-uploading documents may be required to re-establish the connection.`,
            elapsed: 0,
            relevance: null,
            sources: []
          }
        ]);
      });
  };

  const handleQuerySubmit = (e) => {
    e.preventDefault();
    if (!queryInput.trim()) return;
    executeQuery(queryInput);
    setQueryInput('');
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
        setSearchTerm('');
      })
      .catch((err) => {
        console.error('Index deletion error:', err);
        localStorage.removeItem('docuchat_session_id');
        setSession(null);
        setChatHistory([]);
      });
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

  // Filter doc names based on search term
  const filteredDocs = session?.doc_names?.filter(name => 
    name.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <a href="/" className="logo">
          <div className="logo-icon"></div>
          <span>INSIGHT</span>AI
        </a>
        <div className="header-actions">
          {session && (
            <button onClick={handleDeleteIndex} className="btn-secondary" style={{ width: 'auto', padding: '0.4rem 0.8rem' }}>
              Reset Workspace
            </button>
          )}
        </div>
      </header>

      {/* Main Grid */}
      <div className="layout-split">
        {/* Left Sidebar (Only visible when document session is active) */}
        {session && (
          <aside className="sidebar">
            <div className="sidebar-header">
              <div className="sidebar-title">
                Documents
                <span className="doc-count-badge">{session.doc_names.length} files</span>
              </div>
              
              {/* Search input */}
              <div className="search-container">
                <span className="search-icon">🔍</span>
                <input 
                  type="text" 
                  className="search-input" 
                  placeholder="Search index..." 
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>

            {/* Document Cards Feed */}
            <div className="doc-list">
              {filteredDocs.length === 0 ? (
                <div style={{ padding: '2rem 1rem', textAlignment: 'center', opacity: 0.4, fontSize: '0.8rem' }}>
                  No matching files found.
                </div>
              ) : (
                filteredDocs.map((name, i) => (
                  <div key={i} className="doc-card active">
                    <span className="doc-card-icon">📄</span>
                    <div className="doc-card-details">
                      <div className="doc-card-title" title={name}>{name}</div>
                      <div className="doc-card-meta">
                        <span className="doc-card-badge">INDEXED</span>
                        <span>PDF</span>
                      </div>
                    </div>
                  </div>
                ))
              )}

              {/* Collapsible Incremental Uploader */}
              <div className="sidebar-divider" style={{ margin: '1rem 0' }}></div>
              <div style={{ padding: '0 0.25rem' }}>
                <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '0.5rem', fontWeight: 600 }}>
                  Add Documents
                </div>
                <div className="tab-headers" style={{ marginBottom: '0.75rem' }}>
                  <button 
                    onClick={() => { setActiveTab('file'); setError(''); }} 
                    className={`tab-button ${activeTab === 'file' ? 'active' : ''}`}
                    disabled={loading}
                  >
                    File
                  </button>
                  <button 
                    onClick={() => { setActiveTab('url'); setError(''); }} 
                    className={`tab-button ${activeTab === 'url' ? 'active' : ''}`}
                    disabled={loading}
                  >
                    Link
                  </button>
                </div>

                {activeTab === 'file' ? (
                  <form onSubmit={submitBuildFromFiles}>
                    <div 
                      className="uploader-panel" 
                      onClick={() => !loading && fileInputRef.current?.click()}
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
                      <div className="uploader-icon">📂</div>
                      <span>Select PDFs</span>
                    </div>
                    {files.length > 0 && (
                      <div className="file-list-preview" style={{ maxHeight: '60px', marginTop: '0.5rem' }}>
                        {files.map((file, idx) => (
                          <div key={idx} className="file-item" style={{ fontSize: '0.75rem' }}>
                            {file.name}
                          </div>
                        ))}
                      </div>
                    )}
                    <button 
                      type="submit" 
                      className="btn-primary" 
                      disabled={files.length === 0 || loading}
                      style={{ padding: '0.5rem 1rem', fontSize: '0.8rem', marginTop: '0.5rem' }}
                    >
                      {loading ? <div className="spinner"></div> : 'Append Files'}
                    </button>
                  </form>
                ) : (
                  <form onSubmit={submitBuildFromUrl} className="input-text-group">
                    <input 
                      type="url" 
                      placeholder="https://example.com/document.pdf" 
                      value={urlInput}
                      onChange={(e) => setUrlInput(e.target.value)}
                      disabled={loading}
                      style={{ padding: '0.45rem 0.75rem', fontSize: '0.8rem' }}
                      required
                    />
                    <button 
                      type="submit" 
                      className="btn-primary" 
                      disabled={!urlInput || loading}
                      style={{ padding: '0.5rem 1rem', fontSize: '0.8rem', marginTop: '0.25rem' }}
                    >
                      {loading ? <div className="spinner"></div> : 'Append URL'}
                    </button>
                  </form>
                )}
              </div>
            </div>

            {/* Sidebar Stats Panel */}
            <div className="sidebar-stats">
              <div className="stats-grid">
                <div className="stat-box">
                  <div className="stat-val">{session.doc_names.length}</div>
                  <div className="stat-label">Documents</div>
                </div>
                <div className="stat-box">
                  <div className="stat-val">{session.num_pages}</div>
                  <div className="stat-label">Pages</div>
                </div>
                <div className="stat-box">
                  <div className="stat-val">{(session.num_chunks * 0.15).toFixed(1)} KB</div>
                  <div className="stat-label">Storage</div>
                </div>
              </div>
              <div className="stats-chart">
                <div className="chart-bar"><div className="chart-bar-fill" style={{ height: '70%' }}></div></div>
                <div className="chart-bar"><div className="chart-bar-fill" style={{ height: '45%' }}></div></div>
                <div className="chart-bar"><div className="chart-bar-fill" style={{ height: '90%' }}></div></div>
                <div className="chart-bar"><div className="chart-bar-fill" style={{ height: '60%' }}></div></div>
                <div className="chart-bar"><div className="chart-bar-fill" style={{ height: '30%' }}></div></div>
                <div className="chart-bar"><div className="chart-bar-fill" style={{ height: '80%' }}></div></div>
                <div className="chart-bar"><div className="chart-bar-fill" style={{ height: '50%' }}></div></div>
              </div>
            </div>
          </aside>
        )}

        {/* Chat / Workspace Panel */}
        <main className="chat-panel">
          {/* Header containing metadata */}
          <div className="chat-header">
            <div className="chat-header-title">
              {session ? (
                <>Discussion: <span>Active Workspace</span></>
              ) : (
                <>System Ingestion Pipeline</>
              )}
            </div>
            {session && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                <span>DB: CHROMA</span>
                <span>•</span>
                <span>MODEL: MISTRAL</span>
              </div>
            )}
          </div>

          <div className="chat-window" ref={chatWindowRef}>
            {!session ? (
              // ---------------- EMPTY LANDING SCREEN (UPLOADER) ----------------
              <div className="chat-empty-state" style={{ maxWidth: '480px' }}>
                <div className="empty-icon-wrap">🚀</div>
                <h3>Unified Knowledge Workspace</h3>
                <p>
                  Deploy documents to create your custom vector index. Ask complex questions and generate grounded insights.
                </p>

                <div className="input-card" style={{ width: '100%', maxWidth: '100%', background: 'var(--panel-bg)', padding: '1.5rem', marginTop: '0.5rem' }}>
                  {error && <div className="alert alert-danger" style={{ marginBottom: '1rem' }}>{error}</div>}

                  <div className="tab-headers" style={{ marginBottom: '1.25rem' }}>
                    <button 
                      onClick={() => { setActiveTab('file'); setError(''); }} 
                      className={`tab-button ${activeTab === 'file' ? 'active' : ''}`}
                      disabled={loading}
                    >
                      Upload File
                    </button>
                    <button 
                      onClick={() => { setActiveTab('url'); setError(''); }} 
                      className={`tab-button ${activeTab === 'url' ? 'active' : ''}`}
                      disabled={loading}
                    >
                      Paste Link
                    </button>
                  </div>

                  {activeTab === 'file' ? (
                    <form onSubmit={submitBuildFromFiles}>
                      <div 
                        className="dropzone" 
                        onClick={() => !loading && fileInputRef.current?.click()}
                        onDragOver={dragOver}
                        onDrop={fileDrop}
                        style={{ borderStyle: 'dashed' }}
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
                        <div className="dropzone-icon">📄</div>
                        <span>Drag and drop PDFs here, or click to browse</span>
                        <small>Supports direct parsing of multiple page documents</small>
                      </div>
                      {files.length > 0 && (
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
                        {loading ? <div className="spinner"></div> : 'Build Knowledge Base'}
                      </button>
                    </form>
                  ) : (
                    <form onSubmit={submitBuildFromUrl} className="input-text-group">
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        Input direct public PDF endpoint:
                      </span>
                      <input 
                        type="url" 
                        placeholder="https://example.com/document.pdf" 
                        value={urlInput}
                        onChange={(e) => setUrlInput(e.target.value)}
                        disabled={loading}
                        required
                        className="text-input"
                      />
                      <button 
                        type="submit" 
                        className="btn-primary" 
                        disabled={!urlInput || loading}
                      >
                        {loading ? <div className="spinner"></div> : 'Fetch & Build Workspace'}
                      </button>
                    </form>
                  )}
                </div>

                {loading && (
                  <div className="pipeline-container">
                    <div className={`pipeline-step-item ${stepStates.step1 === 'running' ? 'active' : stepStates.step1 === 'done' ? 'done' : ''}`}>
                      <div className="step-indicator"></div>
                      <div className="step-info">
                        <span className="step-label">01 Load Documents</span>
                        <span className="step-detail">{stepDetails.step1}</span>
                      </div>
                    </div>
                    <div className={`pipeline-step-item ${stepStates.step2 === 'running' ? 'active' : stepStates.step2 === 'done' ? 'done' : ''}`}>
                      <div className="step-indicator"></div>
                      <div className="step-info">
                        <span className="step-label">02 Split into Chunks</span>
                        <span className="step-detail">{stepDetails.step2}</span>
                      </div>
                    </div>
                    <div className={`pipeline-step-item ${stepStates.step3 === 'running' ? 'active' : stepStates.step3 === 'done' ? 'done' : ''}`}>
                      <div className="step-indicator"></div>
                      <div className="step-info">
                        <span className="step-label">03 Generate Embeddings</span>
                        <span className="step-detail">{stepDetails.step3}</span>
                      </div>
                    </div>
                    <div className={`pipeline-step-item ${stepStates.step4 === 'running' ? 'active' : stepStates.step4 === 'done' ? 'done' : ''}`}>
                      <div className="step-indicator"></div>
                      <div className="step-info">
                        <span className="step-label">04 Build Vector Index</span>
                        <span className="step-detail">{stepDetails.step4}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              // ---------------- ACTIVE CONVERSATION SCREEN ----------------
              <>
                {chatHistory.length === 0 && (
                  <div className="chat-empty-state">
                    <div className="empty-icon-wrap" style={{ backgroundColor: 'var(--emerald-alpha)', color: 'var(--emerald)' }}>💬</div>
                    <h3>Insights Engine Active</h3>
                    <p style={{ marginBottom: '1.5rem' }}>
                      Ask questions, verify sources, and interact directly with your uploaded documents.
                    </p>
                    <div className="suggestions-grid">
                      {SUGGESTIONS.map((sug, i) => (
                        <button 
                          key={i} 
                          className="suggestion-chip" 
                          onClick={() => executeQuery(sug)}
                        >
                          {sug}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {chatHistory.map((msg, index) => (
                  <div key={index} className={`msg-bubble ${msg.role === 'user' ? 'user-bubble' : 'bot-bubble'}`}>
                    <div className="msg-content">
                      {msg.content}

                      {/* Render citations directly in bot message cards */}
                      {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                        <div className="citations-container">
                          {msg.sources.map((src, i) => {
                            const isSelected = activeCitationIndex === `${index}-${i}`;
                            return (
                              <div key={i} style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                                <button 
                                  className="citation-card" 
                                  onClick={() => setActiveCitationIndex(isSelected ? null : `${index}-${i}`)}
                                >
                                  <span>🔍</span>
                                  <span>Source {i + 1} • Page {src.page}</span>
                                </button>
                                {isSelected && (
                                  <div className="citation-text-panel">
                                    {src.text}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>

                    {msg.role === 'assistant' && (
                      <div className="bubble-meta">
                        <span>⏱ {msg.elapsed}s</span>
                        {msg.relevance !== null && <span>Relevance: {msg.relevance}%</span>}
                      </div>
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
              </>
            )}
          </div>

          {/* Bottom Chat Form Input */}
          {session && (
            <div className="chat-input-container">
              <form onSubmit={handleQuerySubmit} className="chat-form">
                <input 
                  type="text" 
                  className="chat-input" 
                  placeholder="Ask Insight AI about your documents..." 
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
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
