import sqlite3, re, pickle, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import chromadb
from chromadb.config import Settings
from typing import *

def extract_features(q, m, vs, fs):
    th = int(any(w.lower() in m.get("doc_title", "").lower() for w in q.split()))
    ql = len(q.split())
    cl = len(m.get("content", "").split())
    itc = int(m.get("chunk_index", 0) == 0)
    return [vs, fs, th, ql, cl, itc]

class DocSearch:
    def __init__(self, model:SentenceTransformer, db_path: str, chroma_path: str, model_file: str, a=0.6):
        self.model = model
        self.db_path = db_path
        self.a = a
        self.model_file = model_file
        self.scl = MinMaxScaler()

        cli = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
        self.coll = cli.get_collection("safety_docs")

        try:
            with open(model_file, "rb") as f: self.clf: Optional[LogisticRegression] = pickle.load(f)
        except FileNotFoundError: self.clf = None

    def get_vector_candidates(self, qe, k=5):
        res = self.coll.query(query_embeddings=[qe], n_results=k)
        docs, metas, ids, dists = res["documents"][0], res["metadatas"][0], res["ids"][0], res["distances"][0]
        vs = [1 - d for d in dists]
        return list(zip(ids, docs, metas, vs))

    def get_fts_candidates(self, q, k=30):
        con = sqlite3.connect(self.db_path); cur = con.cursor(); con.row_factory = sqlite3.Row
        qc = re.sub(r"[^\w\s]", "", q); fts_q = f'"{qc}"'
        sql = f"""SELECT c.id, c.doc_name, c.doc_title, c.doc_url, c.chunk_index, c.page_num, c.content, bm25(chunks_fts) AS score FROM chunks c JOIN chunks_fts fts ON c.id = fts.rowid WHERE chunks_fts MATCH ? ORDER BY score LIMIT {k}"""
        cur.execute(sql, (fts_q,)); rows = cur.fetchall(); con.close()
        return [(r["id"], r["content"], dict(r), r["score"]) for r in rows]

    def hybrid_rerank(self, vc, fc, k=5) -> List[Dict[str, Any]]:
        cd = {}
        for cid, doc, meta, score in vc + fc:
            if cid not in cd: cd[cid] = {"doc": doc, "meta": meta, "vector_score":0, "fts_score":0, "hybrid_score":0}
            if (cid, doc, meta, score) in vc: cd[cid]["vector_score"] = score
            if (cid, doc, meta, score) in fc: cd[cid]["fts_score"] = score

        vs = [v["vector_score"] for v in cd.values()]
        fs = [v["fts_score"] for v in cd.values()]
        nv = self.scl.fit_transform(np.array(vs).reshape(-1,1)).flatten() if vs else []
        nf = self.scl.fit_transform(np.array(fs).reshape(-1,1)).flatten() if fs else []

        for idx, key in enumerate(cd.keys()):
            cd[key]["vector_score"] = nv[idx] if nv.size else 0
            cd[key]["fts_score"] = nf[idx] if nf.size else 0
            cd[key]["hybrid_score"] = self.a*cd[key]["vector_score"] + (1-self.a)*cd[key]["fts_score"]

        return sorted(cd.values(), key=lambda x:x["hybrid_score"], reverse=True)[:k]

    def learned_rerank(self, cands, q):
        feats, ids = [], []
        for cid, doc, meta, score in cands:
            vs = meta.get("vector_score", 0)
            fs = meta.get("fts_score", 0)
            meta['content'] = doc 
            f = extract_features(q, meta, vs, fs)
            feats.append(f)
            ids.append(cid)

        X = np.array(feats)
        if self.clf is None:
            y = np.array([1 if i<len(X)//2 else 0 for i in range(len(X))])
            self.clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            self.clf.fit(X,y)
            with open(self.model_file,"wb") as f: pickle.dump(self.clf,f)

        probs = self.clf.predict_proba(X)[:,1]
        reranked = sorted(zip(cands, probs), key=lambda x:x[1], reverse=True)
        return [(c[0], c[1], c[2], p) for c,p in reranked]

    def query_docs(self, q, top_k, ul=True):
        qe = self.model.encode([q]).tolist()[0]
        vc = self.get_vector_candidates(qe, k=top_k)
        fc = self.get_fts_candidates(q, k=30)
        hc = self.hybrid_rerank(vc, fc, k=top_k)

        if ul:
            final = self.learned_rerank([(idx,c["doc"],c["meta"],c["hybrid_score"]) for idx,c in enumerate(hc)], q)
        else: final = [(idx,c["doc"],c["meta"],c["hybrid_score"]) for idx,c in enumerate(hc)]

        return [
            {"doc_name": c[2].get("doc_name",""), "doc_title": c[2].get("doc_title",""), "doc_url": c[2].get("doc_url",""), "page_num": c[2].get("page_num"), "chunk_index": c[2].get("chunk_index"), "score": c[3], "content": c[1]}
            for c in final
        ]
