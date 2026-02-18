"""
RAG System for local searching using Google Gemini embeddings and FAISS
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from pypdf import PdfReader

import google.genai as genai
import faiss
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Google Gemini client
api_key = os.getenv("GOOGLE_API_KEY")
client = None
if api_key:
    client = genai.Client(api_key=api_key)


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
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Improved chunking with overlap using sliding window.
        Works better than simple \n\n splitting.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())

            start += self.chunk_size - self.overlap

        return [c for c in chunks if c]

    def ingest_text(self, text: str, source: str, page: Optional[str] = None,
                    starting_chunk_id: int = 0) -> List[Document]:
        """Ingest raw text and return chunked Documents."""
        chunks = []
        text_chunks = self._chunk_text(text)

        for i, chunk in enumerate(text_chunks):
            chunks.append(Document(
                content=chunk,
                source=source,
                chunk_id=starting_chunk_id + i,
                page=page
            ))

        return chunks

    def ingest_file(self, file_path: str) -> List[Document]:
        """Load and chunk a file with page-aware PDF support."""
        file_path = Path(file_path)

        # ---------- TEXT / MARKDOWN ----------
        if file_path.suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            return self.ingest_text(text, str(file_path.name))

        # ---------- PDF SUPPORT ----------
        elif file_path.suffix == '.pdf':
            reader = PdfReader(str(file_path))
            documents = []
            chunk_id = 0

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()

                # Skip empty or scanned pages
                if not page_text or not page_text.strip():
                    continue

                page_chunks = self.ingest_text(
                    text=page_text,
                    source=str(file_path.name),
                    page=str(page_number),
                    starting_chunk_id=chunk_id
                )

                documents.extend(page_chunks)
                chunk_id += len(page_chunks)

            return documents

        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")


class RAGSystem:
    """Main RAG system combining embedding, indexing, and retrieval."""
    
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "text-embedding-004", 
                 use_mock: bool = False):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        # Use real Google Gemini embeddings for accurate semantic search
        # Falls back to mock embeddings if API key is missing or quota exceeded
        self.use_mock = use_mock or not self.api_key  # Use mock if no API key
        self.vocabulary = {}  # For semantic embeddings
        
        self.embedding_model = embedding_model
        self.ingestor = DocumentIngestor()
        
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        
        self.index_dir = Path("./rag_index")
        self.index_dir.mkdir(exist_ok=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Google API or semantic mock embeddings."""
        if self.use_mock:
            # Generate semantic embeddings based on TF-IDF
            import re
            from collections import Counter
            
            embeddings = []
            
            # Build vocabulary from all texts if not already done
            if not self.vocabulary:
                all_words = Counter()
                for text in texts:
                    words = re.findall(r'\b\w+\b', text.lower())
                    all_words.update(words)
                # Keep top 1536 words as vocabulary (size of embeddings)
                self.vocabulary = {word: idx for idx, (word, _) in enumerate(
                    all_words.most_common(1536)
                )}
            
            # Create TF-IDF style embeddings
            for text in texts:
                emb = np.zeros(1536, dtype='float32')
                words = re.findall(r'\b\w+\b', text.lower())
                word_counts = Counter(words)
                
                # Fill embedding with TF scores for known words
                for word, count in word_counts.items():
                    if word in self.vocabulary:
                        idx = self.vocabulary[word]
                        # TF score: log(1 + count) normalized
                        emb[idx] = np.log(1 + count)
                
                # Normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)
            
            return np.vstack(embeddings)
        
        # Use real Google Gemini embeddings
        try:
            if not client:
                raise ValueError("Google API client not initialized. Check GOOGLE_API_KEY in .env")
            
            embeddings = []
            for text in texts:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text
                )
                # Extract embedding from the response
                embedding_values = response.embeddings[0].values
                embeddings.append(embedding_values)
            return np.array(embeddings, dtype='float32')
        except Exception as e:
            print(f"Error with embeddings: {e}")
            # Fall back to semantic mock if there's an error
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
            answer = f"Based on the provided context, here is what I found:\n\n{context_parts[0][:800]}"
            if not answer.endswith(('.', '!', '?')):
                answer += "..."
            return answer, citations
        
        # Generate answer using Google Gemini
        try:
            if not client:
                raise ValueError("Google API client not initialized. Check GOOGLE_API_KEY in .env")
            
            prompt = f"You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain the answer, say so.\n\nContext:\n{context}\n\nQuestion: {question}\n\nProvide a clear, concise answer based on the context above."
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 1000,
                }
            )
            
            answer = response.text
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
        
        # Save vocabulary for semantic embeddings
        with open(path / "vocabulary.json", "w") as f:
            json.dump(self.vocabulary, f, indent=2)
        
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
            
            # Load vocabulary for semantic embeddings
            vocab_path = path / "vocabulary.json"
            if vocab_path.exists():
                with open(vocab_path, "r") as f:
                    self.vocabulary = json.load(f)
            
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
        """Extract HIGH-SIGNAL insights dynamically based on actual content."""
        import re
        from collections import Counter
        
        combined_text = (question + " " + answer).lower()
        
        # ===== Extract key concepts from the answer (NOT hardcoded) =====
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can',
                      'that', 'this', 'these', 'those', 'it', 'its', 'in', 'on', 'at', 'to', 'for', 'of',
                      'as', 'by', 'with', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                      'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
                      'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                      'what', 'which', 'who', 'whom', 'whose', 'would', 'could', 'should', 'may', 'might'}
        
        # Extract single important words (4+ chars, which are likely nouns/verbs)
        words = re.findall(r'\b[a-z]{4,}\b', answer.lower())
        important_words = [w for w in words if w not in stop_words]
        word_freq = Counter(important_words)
        
        # Get top concepts by frequency AND length (words that repeat OR are 6+ chars important)
        concepts = [word for word, count in word_freq.most_common(10) if count >= 2 or len(word) >= 6]
        
        # ===== COMPANY INSIGHTS: Identify patterns from ACTUAL answer content =====
        company_insight = ""
        
        # Only generate if we have actual important concepts
        if concepts:
            # Pattern 1: Multiple related concepts = architectural insight
            if len(concepts) >= 2:
                company_insight = f"{concepts[0].title()} and {concepts[1]} are interconnected components"
            # Pattern 2: Single strong concept with depth = domain knowledge
            elif len(concepts) >= 1:
                concept_word = concepts[0]
                concept_sentences = [s for s in answer.split('.') if concept_word in s.lower()]
                if len(concept_sentences) >= 2 or len(answer) > 100:
                    company_insight = f"{concept_word.title()} has multiple important considerations"
                else:
                    company_insight = f"{concept_word.title()} is a key component"
        
        # Fallback: Extract from answer if still empty and answer is substantial
        if not company_insight and len(answer) > 80:
            nouns = re.findall(r'\b[a-z]{6,}\b', answer.lower())
            if nouns:
                company_insight = f"{nouns[0].title()} is a fundamental component"
        
        # ===== USER INSIGHTS: Infer user needs from question patterns =====
        user_insight = ""
        question_lower = question.lower()
        
        # Pattern-based user intent detection (generalized)
        if question_lower.startswith("how"):
            user_insight = "User seeks practical implementation and hands-on guidance"
        elif question_lower.startswith("what is") or "explain" in question_lower:
            user_insight = "User is building foundational knowledge and understanding"
        elif "why" in question_lower:
            user_insight = "User focuses on understanding rationale and causation"
        elif "best" in question_lower or "should" in question_lower:
            user_insight = "User is evaluating options and making design decisions"
        elif "problem" in question_lower or "issue" in question_lower or "fix" in question_lower:
            user_insight = "User is troubleshooting and solving specific challenges"
        elif "compare" in question_lower or "difference" in question_lower:
            user_insight = "User evaluates trade-offs and alternative approaches"
        
        # Only extract if we have meaningful insights
        confidence = 0.85 if (user_insight or company_insight) else 0.0
        
        return {
            "user_insight": user_insight,
            "company_insight": company_insight,
            "confidence": confidence
        }
    
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
