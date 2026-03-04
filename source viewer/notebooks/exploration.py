"""
notebooks/exploration.py
─────────────────────────
Interactive exploration script — run cell by cell in VS Code,
Jupyter (as a .py with Jupytext), or plain Python.

Usage:
    python notebooks/exploration.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 1. Ingest a document ──────────────────────────────────────────────────────
# from data_wrangling.loader import load_and_split
# from data_wrangling.vectorstore import ingest_documents
#
# chunks = load_and_split("data/raw/my_document.pdf", source_type="pdf")
# print(f"Loaded {len(chunks)} chunks")
# n = ingest_documents(chunks)
# print(f"Ingested {n} chunks")

# ── 2. Run a query ────────────────────────────────────────────────────────────
# from models.rag_pipeline import run_rag_query
# from models.schemas import QueryRequest
#
# req = QueryRequest(query="What is the main topic of the document?")
# resp = run_rag_query(req)
# print(resp.answer)
# for src in resp.sources:
#     print(f"  [{src.score:.3f}] {src.source} — {src.text[:120]}")

# ── 3. Direct similarity search ───────────────────────────────────────────────
# from data_wrangling.vectorstore import similarity_search
#
# results = similarity_search("budget allocation", top_k=5)
# for doc, score in results:
#     print(f"Score: {score:.3f} | {doc.page_content[:200]}\n")

print("Exploration script loaded. Uncomment sections above to run.")
