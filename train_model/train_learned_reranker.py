import sqlite3
import chromadb
from chromadb.config import Settings
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from methods.features import extract_features
from .training_queries import training_data

# --------------------------
# Candidate retrieval
# --------------------------
def get_candidates(query, query_emb, model, collection, db_path="chunks.db", top_k=20):
    vector_results = collection.query(query_embeddings=[query_emb], n_results=top_k)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    query_clean = re.sub(r"[^\w\s]", "", query)
    query_escaped = query_clean.replace('"', '""')
    fts_query = f'"{query_escaped}"'
    cur.execute(f"""
        SELECT c.id, c.doc_name, c.doc_title, c.chunk_index, c.page_num, c.content,
               bm25(chunks_fts) AS score
        FROM chunks c
        JOIN chunks_fts fts ON c.id = fts.rowid
        WHERE chunks_fts MATCH {fts_query}
        ORDER BY score LIMIT 30
    """)
    fts_rows = cur.fetchall()
    conn.close()

    candidates = []
    for i, cid in enumerate(vector_results["ids"][0]):
        meta = vector_results["metadatas"][0][i]
        doc = vector_results["documents"][0][i]
        vscore = 1 - vector_results["distances"][0][i]
        fts_score = next((r["score"] for r in fts_rows if r["id"] == cid), 0)
        meta = dict(meta)
        meta["content"] = doc
        candidates.append((meta, vscore, fts_score))

    return candidates

# --------------------------
# Labeling
# --------------------------
def label_candidate(chunk, keywords):
    text = chunk["content"].lower()
    return int(any(k.lower() in text for k in keywords))

# --------------------------
# Training
# --------------------------
def train_model(model, chroma_path, db_path, model_save_path):
    
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("safety_docs")
    

    X, y = [], []

    for item in training_data:
        query = item["query"]
        keywords = item["keywords"]
        print(f"Processing query: {query}")

        query_emb = model.encode([query]).tolist()[0]
        candidates = get_candidates(query, query_emb, model, collection, db_path=db_path)

        for meta, vscore, ftscore in candidates:
            feats = extract_features(query, meta, vscore, ftscore)
            label = label_candidate(meta, keywords)
            X.append(feats)
            y.append(label)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, y)

    with open(model_save_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"✅ Reranker model trained and saved as {model_save_path}")
