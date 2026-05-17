🔍 Multi-Format Document Retrieval System
A local document retrieval system that finds the most relevant passages across mixed file collections using hybrid search — combining semantic embeddings with keyword matching.
What it does
Given a natural language query, the system searches through a folder of mixed documents and returns the top 5 most relevant passages with source filename and relevance score.
Handles two types of queries:

Broad: "What was the main finding of the brand study?"
Narrow: "What does the variable frm_brand_awareness measure?"

Supported File Types
FormatLibraryPDFpdfplumberPowerPoint (.pptx)python-pptxWord (.docx)python-docxEmail (.eml)Python standard libraryJSON glossaryjson
How it works
Documents (PDF, PPTX, DOCX, EML, JSON)
        ↓
Text extraction per file type
        ↓
Split into chunks
        ↓
Sentence Transformers embeddings (all-MiniLM-L6-v2)
        ↓
Query → embedding + keyword matching
        ↓
Hybrid Score = 70% semantic + 30% keyword
        ↓
Top 5 results with source and score
Key Design Decisions
Hybrid Search — pure semantic embeddings struggled with narrow queries like exact variable names. Adding keyword matching (30% weight) significantly improved retrieval for specific terms.
Different chunk sizes per file type — JSON glossaries use smaller chunks (30 words) so each variable entry stays together. Other documents use larger chunks (500 words) for better context.
Local only — no external LLM APIs. Uses sentence-transformers locally for embeddings.
Global caching — model and embeddings are loaded once and cached in memory for faster subsequent queries.
Installation
bashpip install -r requirements.txt
Usage
pythonfrom retrieve import retrieve

results = retrieve("What was the main research methodology?")

for r in results:
    print(f"Score: {r['score']:.2f} | Source: {r['source']}")
    print(r['text'][:200])
    print()
Output format
Each result returns:
python{
    "text": "...",      # passage content
    "source": "...",    # source filename
    "score": 0.82       # relevance score (higher = better)
}
Run local tests
bashpython test_local.py
Evaluation
The system handles two distinct retrieval challenges differently:
Broad queries are answered through semantic similarity — the embedding model finds conceptually related passages even when exact keywords differ.
Narrow queries (specific variable names, exact terms) benefit from the keyword matching component which ensures exact term matches are prioritized.
Note: keyword-based local scoring underestimates real performance since it measures exact word overlap rather than semantic relevance. The system often retrieves semantically correct passages that don't share exact keywords with the query.
Tech Stack

Python 3.11
sentence-transformers (all-MiniLM-L6-v2)
pdfplumber, python-pptx, python-docx
numpy
