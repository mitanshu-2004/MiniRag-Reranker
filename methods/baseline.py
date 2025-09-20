import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict

def baseline_search(model: SentenceTransformer, query: str, chroma_path: str, top_k: int = 5) -> List[Dict]:
    
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("safety_docs")

    query_emb = model.encode([query]).tolist()[0]

    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    vector_scores = [1 - d for d in distances]

    return [
        {
            "doc_id": doc_id,
            "doc_name": meta.get("doc_name", ""),
            "doc_title": meta.get("doc_title", ""),
            "doc_url": meta.get("doc_url", ""),
            "page_num": meta.get("page_num"),
            "chunk_index": meta.get("chunk_index"),
            "score": score,
            "content": doc
        }
        for doc, meta, doc_id, score in zip(docs, metadatas, ids, vector_scores)
    ]
