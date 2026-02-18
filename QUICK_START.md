# Quick Reference Guide

## Installation

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Set Google API key (required for real embeddings)
export GOOGLE_API_KEY="your-key-here"  # On Unix/Mac or setx GOOGLE_API_KEY "your-key-here" on Windows
```

## Running the Chatbot

### Interactive Mode
```bash
python chatbot.py
```

Commands:
- `add <file_path>` - Add a document to the index
- `ask <question>` - Ask a question about indexed documents
- `status` - Show indexing status
- `help` - Show help message
- `exit` - Exit the chatbot

### Command Line Mode
```bash
python chatbot.py add sample_docs/sample_document.txt
python chatbot.py ask "What is RAG?"
python chatbot.py sanity
```

## Project Files

### Core Implementation
- **rag_system.py** - RAG system implementation
  - `Document` - Represents document chunks
  - `DocumentIngestor` - Loads and chunks files
  - `RAGSystem` - Main RAG orchestrator
  - `MemoryManager` - Extracts and stores insights

- **chatbot.py** - Interactive CLI and evaluation
  - `RAGChatbot` - Interactive command interface
  - `run_sanity_check()` - Generates evaluation artifacts

### Data & Configuration
- **requirements.txt** - Python dependencies
- **sample_docs/** - Sample documents for testing
  - `sample_document.txt` - RAG system documentation

### Output Files
- **rag_index/** - Persisted FAISS index
  - `faiss.index` - Vector index
  - `embeddings.npy` - Document embeddings
  - `documents.json` - Document metadata

- **USER_MEMORY.md** - User-specific memories
- **COMPANY_MEMORY.md** - Organization-wide memories
- **artifacts/sanity_output.json** - Evaluation output

### Documentation
- **README.md** - Main project documentation
- **ARCHITECTURE.md** - System architecture details
- **EVAL_QUESTIONS.md** - Example evaluation questions

## How RAG Works

1. **Ingestion**: Document → Chunks → Embeddings
2. **Indexing**: Embeddings stored in FAISS for fast search
3. **Retrieval**: Query → Embedding → Top-k similar chunks
4. **Generation**: Context + Query → LLM → Answer with citations

## Example Workflow

```bash
# Start interactive mode
$ python chatbot.py

chatbot> add sample_docs/sample_document.txt
✓ Successfully added 8 chunks from sample_document.txt

chatbot> ask What is RAG?
Searching...

Answer:
RAG (Retrieval-Augmented Generation) combines document retrieval 
with language models to provide grounded, cited answers...

Citations:
  [1] sample_document.txt (relevance: 0.95)
      RAG is a technique that enhances large language models...

chatbot> status
📊 Indexing Status:
   Total documents indexed: 8
   Unique sources: 1
     - sample_document.txt: 8 chunks

chatbot> exit
Goodbye!
```

## Testing & Evaluation

```bash
# Run sanity check (generates artifacts/sanity_output.json)
python chatbot.py sanity

# Verify output format
python scripts/verify_output.py artifacts/sanity_output.json
```

## Key Features

✅ **Feature A - RAG + Citations**
- Document ingestion and chunking
- Google Gemini-powered embeddings
- FAISS-based retrieval
- Grounded answers with citations

✅ **Feature B - Persistent Memory**
- Automatic insight extraction
- Selective memory writing
- USER_MEMORY.md and COMPANY_MEMORY.md
- High-confidence filtering (>70%)

## Customization

### Change chunk size
```python
rag = RAGSystem()
rag.ingestor.chunk_size = 1000  # Default is 500
```

### Change number of retrieved results
```python
answer, citations = rag.answer_question(question, top_k=5)  # Default is 2
```

### Use with Google API key
```bash
export GOOGLE_API_KEY="your-key-here"
python chatbot.py
# Now uses real Google Gemini embeddings
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "GOOGLE_API_KEY not set" | Set API key: `export GOOGLE_API_KEY="your-key-here"` |
| "No documents indexed" | Add documents first using `add` command |
| Slow first query | First embedding takes time; subsequent queries are faster |
| ImportError | Run `pip install -r requirements.txt` |

## Performance Notes

- **First document**: Takes ~5-10 seconds (API call)
- **Subsequent documents**: ~2-5 seconds each
- **Query**: ~1-2 seconds for retrieval + generation
- **Index size**: ~6KB per document chunk

## Next Steps

1. Update `README.md` with your participant info
2. Record a 5-10 min video walkthrough
3. Run `python chatbot.py sanity` to generate evaluation artifacts
4. Push to GitHub Classroom repository
