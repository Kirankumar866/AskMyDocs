"use client";

import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────

interface Citation {
  chunk_id: number;
  source: string;
  relevance: string;
}

interface Source {
  chunk_number: number;
  source: string;
  filename: string;
  page: number | string;
  content_preview: string;
  rerank_score: number | null;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  sources?: Source[];
  confidence?: string;
  timestamp: Date;
}

interface HealthStatus {
  status: string;
  collection_size: number;
  embedding_model: string;
  llm_model: string;
}

// ── Icons ────────────────────────────────────────────────────────────

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function DocumentIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

function LoaderIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
      <line x1="12" y1="2" x2="12" y2="6" />
      <line x1="12" y1="18" x2="12" y2="22" />
      <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
      <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
      <line x1="2" y1="12" x2="6" y2="12" />
      <line x1="18" y1="12" x2="22" y2="12" />
      <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
      <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2A5.5 5.5 0 0 0 4 7.5c0 1.33.47 2.55 1.26 3.5H5a4 4 0 0 0-1 7.9V19a2 2 0 0 0 2 2h2" />
      <path d="M14.5 2A5.5 5.5 0 0 1 20 7.5c0 1.33-.47 2.55-1.26 3.5H19a4 4 0 0 1 1 7.9V19a2 2 0 0 1-2 2h-2" />
      <path d="M12 2v20" />
    </svg>
  );
}

// ── Main Component ───────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [showSources, setShowSources] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check health on mount
  useEffect(() => {
    fetchHealth();
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 150) + "px";
    }
  }, [input]);

  async function fetchHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage.content }),
      });

      const data = await res.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer || "Sorry, I could not generate an answer.",
        citations: data.citations || [],
        sources: data.sources || [],
        confidence: data.confidence || "low",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "⚠️ Could not connect to the backend. Make sure the Python server is running on port 8000.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadSuccess(null);

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await fetch(`${API_BASE}/api/ingest`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setUploadSuccess(
        `✅ Ingested ${data.chunks_created} chunks from ${files.length} file(s). Total: ${data.collection_total} chunks.`
      );
      fetchHealth();
    } catch {
      setUploadSuccess("❌ Upload failed. Is the backend running?");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function getConfidenceBadge(confidence: string | undefined) {
    if (!confidence) return null;
    const colors: Record<string, string> = {
      high: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
      medium: "bg-amber-500/10 text-amber-400 border-amber-500/20",
      low: "bg-red-500/10 text-red-400 border-red-500/20",
    };
    return (
      <span
        className={`inline-flex items-center gap-1 px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wider rounded-full border ${colors[confidence] || colors.low}`}
      >
        {confidence} confidence
      </span>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-grid" id="app-root">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header
        className="flex items-center justify-between px-6 py-4 border-b border-border/50 backdrop-blur-xl bg-background/80"
        id="header"
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-accent/10 text-accent glow-accent">
            <BrainIcon />
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-tight text-foreground">
              Ask My Docs
            </h1>
            <p className="text-[0.7rem] text-muted-foreground">
              Hybrid RAG · Cross-Encoder Reranking · Citations
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Health indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-border text-xs text-muted-foreground">
            <span
              className={`w-2 h-2 rounded-full ${health ? "bg-success" : "bg-error"}`}
            />
            {health
              ? `${health.collection_size} chunks`
              : "Offline"}
          </div>

          {/* Upload button */}
          <label
            htmlFor="file-upload"
            className="flex items-center gap-2 px-4 py-2 text-xs font-medium text-foreground bg-card hover:bg-card-hover border border-border hover:border-border-hover rounded-lg cursor-pointer transition-all duration-200"
            id="upload-button"
          >
            {isUploading ? <LoaderIcon /> : <UploadIcon />}
            {isUploading ? "Uploading..." : "Upload Docs"}
          </label>
          <input
            id="file-upload"
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.mdx"
            className="hidden"
            onChange={handleFileUpload}
          />
        </div>
      </header>

      {/* ── Upload feedback ────────────────────────────────────── */}
      {uploadSuccess && (
        <div
          className="mx-6 mt-3 px-4 py-2.5 rounded-lg text-xs font-medium border animate-fade-in-up
          border-accent/20 bg-accent/5 text-accent"
          id="upload-feedback"
        >
          {uploadSuccess}
        </div>
      )}

      {/* ── Messages ───────────────────────────────────────────── */}
      <main
        className="flex-1 overflow-y-auto px-4 py-6"
        id="messages-container"
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center gap-6 animate-fade-in-up">
            <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 text-accent pulse-glow">
              <BrainIcon />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-foreground mb-2">
                Ask anything about your docs
              </h2>
              <p className="text-sm text-muted-foreground max-w-md">
                Upload PDFs, Markdown, or text files, then ask questions.
                Answers include citations to the exact source chunks.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 justify-center mt-2">
              {[
                "What is this document about?",
                "Summarize the key features",
                "What architecture is used?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => setInput(q)}
                  className="px-3 py-1.5 text-xs text-muted-foreground bg-card hover:bg-card-hover border border-border hover:border-accent/30 rounded-lg transition-all duration-200"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-4">
            {messages.map((msg, i) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fade-in-up`}
                style={{ animationDelay: `${i * 0.05}s` }}
              >
                <div
                  className={`max-w-[85%] ${
                    msg.role === "user"
                      ? "bg-accent/15 border-accent/20 text-foreground"
                      : "bg-card border-border"
                  } border rounded-2xl px-4 py-3`}
                >
                  {/* Role label */}
                  <div className="flex items-center gap-2 mb-1.5">
                    {msg.role === "assistant" && (
                      <SparkleIcon />
                    )}
                    <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground">
                      {msg.role === "user" ? "You" : "Ask My Docs"}
                    </span>
                    {msg.role === "assistant" &&
                      getConfidenceBadge(msg.confidence)}
                  </div>

                  {/* Content */}
                  <div className="text-sm leading-relaxed whitespace-pre-wrap">
                    {msg.content}
                  </div>

                  {/* Sources toggle */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-2 border-t border-border/50">
                      <button
                        onClick={() =>
                          setShowSources(
                            showSources === msg.id ? null : msg.id
                          )
                        }
                        className="flex items-center gap-1.5 text-[0.7rem] font-medium text-accent hover:text-accent-hover transition-colors"
                        id={`sources-toggle-${msg.id}`}
                      >
                        <DocumentIcon />
                        {showSources === msg.id
                          ? "Hide sources"
                          : `View ${msg.sources.length} source(s)`}
                      </button>

                      {showSources === msg.id && (
                        <div className="mt-2 space-y-2 animate-fade-in-up">
                          {msg.sources.map((src, j) => (
                            <div
                              key={j}
                              className="px-3 py-2 rounded-lg bg-background/50 border border-border/50 text-xs"
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-semibold text-foreground">
                                  [{src.chunk_number}] {src.filename}
                                </span>
                                <span className="text-muted-foreground">
                                  Page {src.page}
                                  {src.rerank_score !== null &&
                                    ` · Score: ${src.rerank_score.toFixed(3)}`}
                                </span>
                              </div>
                              <p className="text-muted-foreground leading-relaxed">
                                {src.content_preview}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Loading indicator */}
            {isLoading && (
              <div className="flex justify-start animate-fade-in-up">
                <div className="bg-card border border-border rounded-2xl px-4 py-3">
                  <div className="flex items-center gap-2 mb-1.5">
                    <SparkleIcon />
                    <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground">
                      Ask My Docs
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <LoaderIcon />
                    <span>Searching documents & generating answer...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* ── Input Area ─────────────────────────────────────────── */}
      <footer
        className="border-t border-border/50 bg-background/80 backdrop-blur-xl px-4 py-4"
        id="input-area"
      >
        <form
          onSubmit={handleSubmit}
          className="max-w-3xl mx-auto flex items-end gap-3"
        >
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask a question about your documents..."
              rows={1}
              className="w-full px-4 py-3 text-sm text-foreground bg-card border border-border hover:border-border-hover focus:border-accent/50 focus:ring-1 focus:ring-accent/20 rounded-xl resize-none outline-none transition-all duration-200 placeholder:text-muted"
              id="query-input"
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="flex items-center justify-center w-11 h-11 rounded-xl bg-accent hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed text-white transition-all duration-200 glow-accent"
            id="send-button"
          >
            {isLoading ? <LoaderIcon /> : <SendIcon />}
          </button>
        </form>

        <p className="text-center text-[0.6rem] text-muted mt-2">
          Hybrid retrieval (BM25 + Vector) → Cross-encoder Reranking → GPT-4o
          with citation enforcement
        </p>
      </footer>
    </div>
  );
}
