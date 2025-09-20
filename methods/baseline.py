import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import *

def baseline_search(model: SentenceTransformer, q: str, chroma_path: str, top_k: int ) -> List[Dict]:
    
    cli = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    coll = cli.get_collection("safety_docs")

    q_emb = model.encode([q]).tolist()[0]

    res = coll.query(query_embeddings=[q_emb], n_results=top_k)
    docs, metas, ids, dists = res["documents"][0], res["metadatas"][0], res["ids"][0], res["distances"][0]

    v_scores = [1 - d for d in dists]

    return [
        {"doc_id": did, "doc_name": m.get("doc_name", ""), "doc_title": m.get("doc_title", ""), "doc_url": m.get("doc_url", ""), "page_num": m.get("page_num"), "chunk_index": m.get("chunk_index"), "score": s, "content": d}
        for d, m, did, s in zip(docs, metas, ids, v_scores)
    ]
