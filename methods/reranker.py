import sqlite3, re, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional


def extract_features(query, meta, vector_score, fts_score):
    title_hit = int(any(word.lower() in meta.get("doc_title", "").lower() for word in query.split()))
    query_len = len(query.split())
    chunk_len = len(meta.get("content", "").split())
    is_title_chunk = int(meta.get("chunk_index", 0) == 0)
    return [vector_score, fts_score, title_hit, query_len, chunk_len, is_title_chunk]


class DocSearch:
    def __init__(self, model: SentenceTransformer, db_path: str,
                 chroma_path: str, model_file: str, alpha=0.6):
        self.model = model
        self.db_path = db_path
        self.alpha = alpha
        self.model_file = model_file
        self.scaler = MinMaxScaler()

        client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_collection("safety_docs")

        try:
            with open(model_file, "rb") as f:
                self.clf: Optional[LogisticRegression] = pickle.load(f)
        except FileNotFoundError:
            self.clf = None

    def get_vector_candidates(self, query_emb, top_k=5):
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        docs, metadatas, ids, distances = results["documents"][0], results["metadatas"][0], results["ids"][0], results["distances"][0]
        vector_scores = [1 - d for d in distances]
        return list(zip(ids, docs, metadatas, vector_scores))

    def get_fts_candidates(self, query, top_k=30):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        conn.row_factory = sqlite3.Row
        query_clean = re.sub(r"[^\w\s]", "", query)
        fts_query = f'"{query_clean}"'
        sql = f"""
            SELECT c.id, c.doc_name, c.doc_title, c.doc_url, c.chunk_index, c.page_num, c.content,
                   bm25(chunks_fts) AS score
            FROM chunks c
            JOIN chunks_fts fts ON c.id = fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT {top_k}
        """
        cur.execute(sql, (fts_query,))
        rows = cur.fetchall()
        conn.close()
        return [(row["id"], row["content"], dict(row), row["score"]) for row in rows]

    def hybrid_rerank(self, vector_candidates, fts_candidates, top_k=5) -> List[Dict[str, Any]]:
        candidate_dict = {}
        for cid, doc, meta, score in vector_candidates + fts_candidates:
            if cid not in candidate_dict:
                candidate_dict[cid] = {"doc": doc, "meta": meta, "vector_score":0, "fts_score":0, "hybrid_score":0}
            if (cid, doc, meta, score) in vector_candidates:
                candidate_dict[cid]["vector_score"] = score
            if (cid, doc, meta, score) in fts_candidates:
                candidate_dict[cid]["fts_score"] = score

        vec_scores = [v["vector_score"] for v in candidate_dict.values()]
        fts_scores = [v["fts_score"] for v in candidate_dict.values()]
        norm_vec = self.scaler.fit_transform(np.array(vec_scores).reshape(-1,1)).flatten() if vec_scores else []
        norm_fts = self.scaler.fit_transform(np.array(fts_scores).reshape(-1,1)).flatten() if fts_scores else []

        for idx, key in enumerate(candidate_dict.keys()):
            candidate_dict[key]["vector_score"] = norm_vec[idx] if norm_vec.size else 0
            candidate_dict[key]["fts_score"] = norm_fts[idx] if norm_fts.size else 0
            candidate_dict[key]["hybrid_score"] = self.alpha*candidate_dict[key]["vector_score"] + (1-self.alpha)*candidate_dict[key]["fts_score"]

        return sorted(candidate_dict.values(), key=lambda x:x["hybrid_score"], reverse=True)[:top_k]

    def learned_rerank(self, candidates, query):
        features, ids = [], []
        for cid, doc, meta, score in candidates:
            vector_score = meta.get("vector_score", 0)
            fts_score = meta.get("fts_score", 0)
            meta['content'] = doc  # Add content to meta for feature extraction
            feats = extract_features(query, meta, vector_score, fts_score)
            features.append(feats)
            ids.append(cid)

        X = np.array(features)
        if self.clf is None:
            y = np.array([1 if i<len(X)//2 else 0 for i in range(len(X))])
            self.clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            self.clf.fit(X,y)
            with open(self.model_file,"wb") as f:
                pickle.dump(self.clf,f)

        probs = self.clf.predict_proba(X)[:,1]
        reranked = sorted(zip(candidates, probs), key=lambda x:x[1], reverse=True)
        return [(c[0], c[1], c[2], p) for c,p in reranked]

    def query_docs(self, query, top_k=5, use_learned=True):
        query_emb = self.model.encode([query]).tolist()[0]
        vector_candidates = self.get_vector_candidates(query_emb, top_k=top_k)
        fts_candidates = self.get_fts_candidates(query, top_k=30)
        hybrid_candidates = self.hybrid_rerank(vector_candidates, fts_candidates, top_k=top_k)

        if use_learned:
            final = self.learned_rerank(
                [(idx,c["doc"],c["meta"],c["hybrid_score"]) for idx,c in enumerate(hybrid_candidates)],
                query
            )
        else:
            final = [(idx,c["doc"],c["meta"],c["hybrid_score"]) for idx,c in enumerate(hybrid_candidates)]

        return [
            {
                "doc_name": c[2].get("doc_name",""),
                "doc_title": c[2].get("doc_title",""),
                "doc_url": c[2].get("doc_url",""),
                "page_num": c[2].get("page_num"),
                "chunk_index": c[2].get("chunk_index"),
                "score": c[3],
                "content": c[1]
            } for c in final
        ]
