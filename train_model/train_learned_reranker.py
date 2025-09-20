import sqlite3, re, pickle, numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from .questions import training_data

def extract_features(q, m, vs, fs):
    th = int(any(w.lower() in m.get("doc_title", "").lower() for w in q.split()))
    ql = len(q.split())
    cl = len(m.get("content", "").split())
    itc = int(m.get("chunk_index", 0) == 0)
    return [vs, fs, th, ql, cl, itc]


def get_candidates(q, qe, mod, coll, dp="chunks.db", k=20):
    v_res = coll.query(query_embeddings=[qe], n_results=k)

    con = sqlite3.connect(dp); con.row_factory = sqlite3.Row; cur = con.cursor()
    qc = re.sub(r"[^\w\s]", "", q); qe_ = qc.replace('"', '""'); fts_q = f'"{qe_}"'
    cur.execute(f"""SELECT c.id, c.doc_name, c.doc_title, c.chunk_index, c.page_num, c.content, bm25(chunks_fts) AS score FROM chunks c JOIN chunks_fts fts ON c.id = fts.rowid WHERE chunks_fts MATCH {fts_q} ORDER BY score LIMIT 30""")
    fts_rows = cur.fetchall(); con.close()

    cands = []
    for i, cid in enumerate(v_res["ids"][0]):
        meta = v_res["metadatas"][0][i]; doc = v_res["documents"][0][i]; vs = 1 - v_res["distances"][0][i]
        fs = next((r["score"] for r in fts_rows if r["id"] == cid), 0)
        meta = dict(meta); meta["content"] = doc
        cands.append((meta, vs, fs))

    return cands

def label_candidate(c, kw):
    txt = c["content"].lower()
    return int(any(k.lower() in txt for k in kw))

def train_model(model, chroma_path, db_path, model_save_path):
    
    cli = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    coll = cli.get_collection("safety_docs")
    

    X, y = [], []

    for item in training_data:
        q = item["query"]; kw = item["keywords"]
        print(f"Processing query: {q}")

        qe = model.encode([q]).tolist()[0]
        cands = get_candidates(q, qe, model, coll, dp=db_path)

        for meta, vs, fs in cands:
            feats = extract_features(q, meta, vs, fs)
            lbl = label_candidate(meta, kw)
            X.append(feats); y.append(lbl)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, y)

    with open(model_save_path, "wb") as f: pickle.dump(clf, f)

    print(f"✅ Reranker model trained and saved as {model_save_path}\n")
