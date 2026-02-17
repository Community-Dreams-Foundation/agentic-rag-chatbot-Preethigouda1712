import json
from pathlib import Path

docs = json.load(open('rag_index/documents.json'))
print(f'Total docs: {len(docs)}')
print(f'Sources: {set(d["source"] for d in docs)}')

# Count by source
from collections import Counter
counts = Counter(d['source'] for d in docs)
for source, count in counts.items():
    print(f'  {source}: {count} chunks')
