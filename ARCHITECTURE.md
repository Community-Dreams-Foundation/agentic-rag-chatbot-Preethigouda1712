# Architecture Overview

## Goal
The Agentic RAG Chatbot is a local search system that enables users to ask questions about their documents and receive grounded answers with citations.

---

## High-Level Flow

### 1) Ingestion (Upload → Parse → Chunk)
**Supported inputs:** Plain text (.txt), Markdown (.md)
**Parsing/Chunking:** Files split by paragraph boundaries into ~500-char chunks
**Metadata:** source filename, chunk_id, timestamp

### 2) Indexing / Storage
**Vector store:** FAISS with L2 distance
**Embeddings:** Google Gemini text-embedding-004 (768-dim)
**Persistence:** Saved to ./rag_index/ with FAISS index + metadata JSON

### 3) Retrieval + Grounded Answering
**Retrieval:** Top-k similarity search (k=2 default)
**Citations:** Include source, chunk_id, snippet, relevance score
**Failure:** Gracefully reports when insufficient information available

### 4) Memory System (Selective)
**Stores:** High-signal user preferences, workflow insights, org learnings
**Excludes:** PII, secrets, raw transcripts, redundant info
**Confidence:** Only writes insights with >70% confidence score
**Format:** Timestamped bullets in USER_MEMORY.md and COMPANY_MEMORY.md

### 5) Answer Generation
**Model:** Google Gemini Pro (temp 0.7, max 1000 tokens)
**Grounding:** Context injected with labeled chunks
**Safety:** Explicit instruction to decline answering without sufficient context

---

## System Components

### rag_system.py
Core RAG implementation with four main classes:
- **Document**: Text chunk with metadata (content, source, chunk_id, page, created_at)
- **DocumentIngestor**: File loading and chunking; `ingest_file()` / `ingest_text()` methods
- **RAGSystem**: Main orchestrator; `add_document()`, `search()`, `answer_question()`, persistence methods
- **MemoryManager**: Insight extraction; `extract_insights()`, `add_memory()` methods

### chatbot.py
User-facing CLI interface:
- **RAGChatbot**: Interactive mode with commands (add, ask, status, help, exit)
- **run_sanity_check()**: Generates evaluation artifacts in required format

### Workflow
1. `python chatbot.py` → Interactive prompt
2. `add <file>` → DocumentIngestor chunks file, RAGSystem embeds and indexes
3. `ask <question>` → Query embedded, FAISS searches, LLM generates answer with citations
4. MemoryManager extracts insights automatically
5. Index auto-saved after each addition

## Deployment & Reproducibility

**Requirements:**
- Python 3.8+
- Dependencies: google-genai, faiss-cpu, numpy, python-dotenv, pytest

**Setup:**
- `pip install -r requirements.txt && export GOOGLE_API_KEY=...`
- Run: `python chatbot.py` (interactive) or `python chatbot.py sanity` (evaluation)
- Sanity check: Generates `artifacts/sanity_output.json`

**Fallback Behavior:**
- Works without API key using mock embeddings
- Graceful degradation for testing and demos
- Deterministic embeddings for reproducibility

## Tradeoffs & Future Improvements

**Why FAISS?**
- Fast in-memory search, scales to millions of documents
- Easy serialization for offline use
- No external services required (local-first)

**Why confidence-based memory?**
- Avoids storing noise and PII
- High signal-to-noise ratio for knowledge base
- Explicit security properties

**Future enhancements:**
- Hybrid search (BM25 + semantic similarity)
- Multi-format support (PDF, HTML, JSON)
- LLM-based reranking for precision
- Conversation memory with multi-turn context
- Multi-user support with separate indices
- GPU acceleration via FAISS-GPU
