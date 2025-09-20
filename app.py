import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from methods.baseline import baseline_search
from methods.reranker import DocSearch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_DIR, "sql_store", "chunks.db")
CHROMA_PATH = os.path.join(ROOT_DIR, "chromadb_store")
MODEL_PATH = os.path.join(ROOT_DIR, "model", "learned_reranker.pkl")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    mode: Optional[str] = "learned"

app = FastAPI(title="RAG Document Search API")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

searcher = DocSearch(
    model=embedding_model, 
    db_path=DB_PATH, 
    chroma_path=CHROMA_PATH, 
    model_file=MODEL_PATH
)

@app.post("/ask")
def ask(request: AskRequest):
    mode = request.mode.lower()
    query = request.query
    top_k = request.top_k

    if mode == "baseline":
        results = baseline_search(
            model=embedding_model, 
            query=query, 
            top_k=top_k, 
            chroma_path=CHROMA_PATH
        )
    elif mode in ["hybrid", "learned"]:
        use_learned = mode == "learned"
        results = searcher.query_docs(query, top_k=top_k, use_learned=use_learned)
    else:
        return {"error": "Invalid mode. Choose 'baseline', 'hybrid', or 'learned'."}

    simplified = [
        {
            "doc_name": r.get("doc_name"),
            "doc_title": r.get("doc_title"),
            "doc_url": r.get("doc_url"),
            "page_num": r.get("page_num"),
            "chunk_index": r.get("chunk_index"),
            "score": r.get("score"),
            "content": r.get("content")[:1000]
        }
        for r in results
    ]

    return {"query": query, "mode": mode, "results": simplified}
