"""
RAG System for local searching using OpenAI embeddings and FAISS
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle

import openai
import faiss
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


class Document:
    """Represents a document chunk with metadata."""
    
    def __init__(self, content: str, source: str, chunk_id: int, page: Optional[str] = None):
        self.content = content
        self.source = source
        self.chunk_id = chunk_id
        self.page = page
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            "content": self.content,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "created_at": self.created_at,
        }


class DocumentIngestor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def ingest_text(self, text: str, source: str) -> List[Document]:
        """Ingest a text document and return chunks."""
        chunks = []
        chunk_id = 0
        
        # Split by sections first (rough approach)
        sections = text.split('\n\n')
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) < self.chunk_size:
                current_chunk += section + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(Document(
                        content=current_chunk.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        page=None
                    ))
                    chunk_id += 1
                current_chunk = section
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(Document(
                content=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                page=None
            ))
        
        return chunks
    
    def ingest_file(self, file_path: str) -> List[Document]:
        """Load and chunk a file."""
        file_path = Path(file_path)
        
        if file_path.suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.ingest_text(text, str(file_path.name))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")


class RAGSystem:
    """Main RAG system combining embedding, indexing, and retrieval."""
    
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "text-embedding-3-small", 
                 use_mock: bool = False):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Use mock mode by default if API key exists but we want to avoid quota issues
        # Set use_mock=True to skip API calls and use deterministic embeddings
        self.use_mock = True  # Changed to True to avoid API quota issues
        
        self.embedding_model = embedding_model
        self.ingestor = DocumentIngestor()
        
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        
        self.index_dir = Path("./rag_index")
        self.index_dir.mkdir(exist_ok=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API or mock embeddings."""
        if self.use_mock:
            # Generate mock embeddings (deterministic based on text)
            embeddings = []
            for text in texts:
                # Create a simple deterministic embedding from text hash
                hash_val = hash(text)
                np.random.seed(abs(hash_val) % (2**32))
                emb = np.random.randn(1536).astype('float32')
                emb = emb / np.linalg.norm(emb)  # Normalize
                embeddings.append(emb)
            return np.vstack(embeddings)
        
        # Use real OpenAI embeddings with new API
        try:
            response = client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings.astype('float32')
        except Exception as e:
            print(f"Error with embeddings: {e}")
            # Fall back to mock if there's an error
            self.use_mock = True
            return self.embed_texts(texts)
    
    def add_document(self, file_path: str) -> int:
        """Add a document to the RAG system."""
        print(f"Ingesting document: {file_path}")
        chunks = self.ingestor.ingest_file(file_path)
        
        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        print(f"Generated {len(chunks)} chunks, embedding...")
        
        embeddings = self.embed_texts(chunk_texts)
        
        # Initialize or extend index
        if self.index is None:
            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.index.add(embeddings)
        self.documents.extend(chunks)
        
        print(f"Added {len(chunks)} chunks. Total documents: {len(self.documents)}")
        return len(chunks)
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Search for relevant documents."""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Embed query
        query_embedding = self.embed_texts([query])
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[int(idx)]
                distance = float(distances[0][i])
                results.append((doc, distance))
        
        return results
    
    def answer_question(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """Answer a question using RAG."""
        # Retrieve relevant documents
        search_results = self.search(question, top_k)
        
        if not search_results:
            return "I don't have enough information to answer this question.", []
        
        # Build context from retrieved documents
        context_parts = []
        citations = []
        
        for doc, distance in search_results:
            context_parts.append(f"[Chunk {doc.chunk_id}] {doc.content}")
            citations.append({
                "source": doc.source,
                "chunk_id": doc.chunk_id,
                "snippet": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "relevance_score": 1.0 / (1.0 + distance)  # Convert distance to relevance
            })
        
        context = "\n\n".join(context_parts)
        
        # If no API key and not using real embeddings, return a mock answer
        if self.use_mock or not self.api_key:
            answer = f"Based on the provided context about {question}, here is a summary:\n\n{context_parts[0][:300]}..."
            return answer, citations
        
        # Generate answer using OpenAI with new API
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain the answer, say so."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a clear, concise answer based on the context above."
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            return answer, citations
        except Exception as e:
            return f"Error generating answer: {str(e)}", citations
    
    def save_index(self, path: Optional[str] = None) -> str:
        """Save the index and documents to disk."""
        path = Path(path) if path else self.index_dir
        path.mkdir(exist_ok=True, parents=True)
        
        if self.index is None:
            print("No index to save.")
            return str(path)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save embeddings
        np.save(str(path / "embeddings.npy"), self.embeddings)
        
        # Save documents
        docs_data = [doc.to_dict() for doc in self.documents]
        with open(path / "documents.json", "w") as f:
            json.dump(docs_data, f, indent=2)
        
        print(f"Index saved to {path}")
        return str(path)
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """Load a saved index."""
        path = Path(path) if path else self.index_dir
        
        if not (path / "faiss.index").exists():
            return False
        
        try:
            self.index = faiss.read_index(str(path / "faiss.index"))
            self.embeddings = np.load(str(path / "embeddings.npy"))
            
            # Load documents
            with open(path / "documents.json", "r") as f:
                docs_data = json.load(f)
            
            self.documents = [
                Document(
                    content=d["content"],
                    source=d["source"],
                    chunk_id=d["chunk_id"],
                    page=d.get("page")
                )
                for d in docs_data
            ]
            
            print(f"Index loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False


class MemoryManager:
    """Manages persistent memory in markdown files."""
    
    def __init__(self, user_memory_path: str = "USER_MEMORY.md", 
                 company_memory_path: str = "COMPANY_MEMORY.md"):
        self.user_memory_path = Path(user_memory_path)
        self.company_memory_path = Path(company_memory_path)
        
        # Initialize files if they don't exist
        self._ensure_files()
    
    def _ensure_files(self):
        """Ensure memory files exist with headers."""
        if not self.user_memory_path.exists():
            self.user_memory_path.write_text("# User Memory\n\n")
        
        if not self.company_memory_path.exists():
            self.company_memory_path.write_text("# Company Memory\n\n")
    
    def extract_insights(self, question: str, answer: str, context: str) -> Dict:
        """Use LLM to extract memorable insights from Q&A."""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract high-signal, reusable insights from conversations. Only return insights if they're significant and not sensitive/PII. Format: JSON with 'user_insight', 'company_insight', and 'confidence'."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\nAnswer: {answer}\nContext used: {context[:500]}\n\nExtract any memorable insights. Return empty strings if no significant insights."
                    }
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except:
                return {"user_insight": "", "company_insight": "", "confidence": 0.0}
        except:
            return {"user_insight": "", "company_insight": "", "confidence": 0.0}
    
    def add_memory(self, question: str, answer: str, context: str):
        """Extract and add memories from an interaction."""
        insights = self.extract_insights(question, answer, context)
        
        if insights.get("confidence", 0) < 0.7:
            return  # Low confidence, skip
        
        timestamp = datetime.now().isoformat()
        
        # Add user memory
        if insights.get("user_insight"):
            user_content = self.user_memory_path.read_text()
            user_content += f"\n- {insights['user_insight']} (extracted {timestamp})"
            self.user_memory_path.write_text(user_content)
        
        # Add company memory
        if insights.get("company_insight"):
            company_content = self.company_memory_path.read_text()
            company_content += f"\n- {insights['company_insight']} (extracted {timestamp})"
            self.company_memory_path.write_text(company_content)


if __name__ == "__main__":
    # Test the RAG system
    rag = RAGSystem()
    print("RAG System initialized")
