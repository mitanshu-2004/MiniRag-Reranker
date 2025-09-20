import sqlite3, re, pickle, numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from methods.features import extract_features
from ..questions import training_data

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

def train_model(mod, cp, dp, msp):
    
    cli = chromadb.PersistentClient(path=cp, settings=Settings(anonymized_telemetry=False))
    coll = cli.get_collection("safety_docs")
    

    X, y = [], []

    for item in training_data:
        q = item["query"]; kw = item["keywords"]
        print(f"Processing query: {q}")

        qe = mod.encode([q]).tolist()[0]
        cands = get_candidates(q, qe, mod, coll, dp=dp)

        for meta, vs, fs in cands:
            feats = extract_features(q, meta, vs, fs)
            lbl = label_candidate(meta, kw)
            X.append(feats); y.append(lbl)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, y)

    with open(msp, "wb") as f: pickle.dump(clf, f)

    print(f"✅ Reranker model trained and saved as {msp}")
