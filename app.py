import logging, os, re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import *
from sentence_transformers import SentenceTransformer, util
from methods.baseline import baseline_search
from methods.reranker import DocSearch

logging.basicConfig(level=logging.ERROR)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = ROOT_DIR + "\\sql_store\\chunks.db"
CHROMA_PATH = ROOT_DIR + "\\chromadb_store"
MODEL_PATH = ROOT_DIR + "\\model\\learned_reranker.pkl"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "learned"

app = FastAPI(title="Document Search API")

emb_mod = SentenceTransformer("all-MiniLM-L6-v2")

srch = DocSearch(
    model=emb_mod, db_path=DB_PATH, chroma_path=CHROMA_PATH, model_file=MODEL_PATH
)

@app.post("/ask")
def ask(req: AskRequest):
    m = req.mode.lower()
    q = req.query
    k = req.top_k

    if m == "baseline":
        res = baseline_search(model=emb_mod, q=q, top_k=k, chroma_path=CHROMA_PATH)
    elif m in ["hybrid", "learned"]:
        ul = m == "learned"
        res = srch.query_docs(q, top_k=k, ul=ul)
    else:
        return {"error": "Invalid mode. Choose 'baseline', 'hybrid', or 'learned'."}

    simp = []
    for r in res:
        simp.append({
            "doc_name": r.get("doc_name"),
            "doc_title": r.get("doc_title"),
            "doc_url": r.get("doc_url"),
            "page_num": r.get("page_num"),
            "chunk_index": r.get("chunk_index"),
            "score": r.get("score"),
            "content": r.get("content")[:1000]
        })

    if not simp or simp[0]['score'] < 0.5: 
        return {"answer": None, "reranker_used": m, "contexts": simp, "details": "Could not find a sufficiently relevant document chunk to form an answer."}
    
    tct = simp[0]["content"]
    
    nt = re.sub(r'\s*\n\s*', ' ', tct).strip()
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', nt)
    sents = [s.strip() for s in sents if s.strip()]

    ans = None
    if sents:
        qe = emb_mod.encode(q)
        se = emb_mod.encode(sents)
        sims = util.cos_sim(qe, se)
        bsi = sims.argmax()
        
        ap = [sents[bsi]]

        if bsi > 0:
            ap.insert(0, sents[bsi - 1])

        if bsi < len(sents) - 1:
            ap.append(sents[bsi + 1])

        ans = " ".join(ap)
    else:
        ans = tct 

    return {"answer": ans, "reranker_used": m, "contexts": simp}