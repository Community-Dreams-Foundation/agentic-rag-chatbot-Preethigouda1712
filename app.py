from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path
from rag_system import RAGSystem, MemoryManager

app = Flask(__name__)
CORS(app)

# Initialize RAG system
rag = RAGSystem()
memory = MemoryManager()

# Try to load existing index
rag.load_index()

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current indexing status."""
    document_sources = set(d.source for d in rag.documents)
    return jsonify({
        'total_documents': len(rag.documents),
        'unique_sources': len(document_sources),
        'sources': list(document_sources)
    })

@app.route('/api/add-document', methods=['POST'])
def add_document():
    """Add a document to the RAG system."""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        # Save file temporarily
        temp_path = Path('temp_uploads') / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        file.save(temp_path)
        
        # Add to RAG system
        chunks_added = rag.add_document(str(temp_path))
        rag.save_index()
        
        # Clean up
        temp_path.unlink()
        
        return jsonify({
            'success': True,
            'message': f'Added {chunks_added} chunks from {file.filename}',
            'chunks_added': chunks_added
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question to the RAG system."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if len(rag.documents) == 0:
            return jsonify({'error': 'No documents indexed. Please add documents first.'}), 400
        
        # Get answer with citations
        answer, citations = rag.answer_question(question, top_k=2)
        
        # Try to extract memory
        try:
            context = "\n".join([c["snippet"] for c in citations])
            memory.add_memory(question, answer, context)
        except:
            pass  # Memory extraction is optional
        
        return jsonify({
            'success': True,
            'answer': answer,
            'citations': [
                {
                    'source': c['source'],
                    'chunk_id': c['chunk_id'],
                    'snippet': c['snippet'],
                    'relevance_score': c['relevance_score']
                }
                for c in citations
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory', methods=['GET'])
def get_memory():
    """Get memory contents."""
    try:
        user_mem = Path('USER_MEMORY.md').read_text() if Path('USER_MEMORY.md').exists() else ''
        company_mem = Path('COMPANY_MEMORY.md').read_text() if Path('COMPANY_MEMORY.md').exists() else ''
        
        return jsonify({
            'user_memory': user_mem,
            'company_memory': company_mem
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_index():
    """Clear the index."""
    try:
        global rag
        rag = RAGSystem()
        
        # Remove index files
        import shutil
        if Path('rag_index').exists():
            shutil.rmtree('rag_index')
        
        return jsonify({'success': True, 'message': 'Index cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
