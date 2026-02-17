import sys
sys.path.insert(0, '.')

from rag_system import RAGSystem

# Test what a fresh instance loads
rag = RAGSystem()
print(f"Before load_index: {len(rag.documents)} documents")

rag.load_index()
print(f"After load_index: {len(rag.documents)} documents")
print(f"Sources loaded: {set(d.source for d in rag.documents)}")
