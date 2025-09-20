import os
import re
import logging
from sentence_transformers import SentenceTransformer, util
from methods.baseline import baseline_search
from methods.reranker import DocSearch
from train_model.questions import training_data
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.ERROR)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_DIR, "sql_store", "chunks.db")
CHROMA_PATH = os.path.join(ROOT_DIR, "chromadb_store")
MODEL_PATH = os.path.join(ROOT_DIR, "model", "learned_reranker.pkl")

# Initialize SentenceTransformer model for embeddings
emb_mod = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize DocSearch for hybrid and learned reranking
srch = DocSearch(
    model=emb_mod, db_path=DB_PATH, chroma_path=CHROMA_PATH, model_file=MODEL_PATH
)

def get_answer_and_contexts(query: str, top_k: int, mode: str) -> Dict[str, Any]:
    """
    Replicates the core logic of the /ask endpoint to get answers and contexts.
    """
    m = mode.lower()
    q = query
    k = top_k

    res = []
    if m == "baseline":
        res = baseline_search(model=emb_mod, q=q, top_k=k, chroma_path=CHROMA_PATH)
    elif m == "hybrid":
        res = srch.query_docs(q, top_k=k, ul=False)
    elif m == "learned":
        res = srch.query_docs(q, top_k=k, ul=True)
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
            "content": r.get("content")[:1000] # Truncate content for display
        })

    # Abstention logic
    if not simp or simp[0]['score'] < 0.5:
        return {"answer": None, "reranker_used": m, "contexts": simp, "details": "Could not find a sufficiently relevant document chunk to form an answer."}
    
    tct = simp[0]["content"]
    
    # Extractive answer generation
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

def export_to_csv(results: List[Dict[str, Any]], filename: str = "reranker_comparison.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nComparison table exported to {filename}")

def main():
    results_table = []
    modes = ["baseline", "hybrid", "learned"]
    top_k = 5 # Using default top_k

    print("Generating comparison table for 8 questions across different reranking modes...\n")

    for i, item in enumerate(training_data):
        query = item["query"]
        print(f"Processing Question {i+1}: {query}")
        
        row = {"Question": f"Q{i+1}: {query}"}
        for mode in modes:
            result = get_answer_and_contexts(query, top_k, mode)
            
            answer = result.get("answer", "Abstained")
            top_context = result["contexts"][0] if result["contexts"] else {"doc_name": "N/A", "score": "N/A"}
            
            row[f"{mode.capitalize()} Answer"] = answer
            row[f"{mode.capitalize()} Top Doc"] = f"{top_context['doc_name']} (Score: {top_context['score']:.2f})" if isinstance(top_context['score'], (int, float)) else f"{top_context['doc_name']} (Score: {top_context['score']})"
        results_table.append(row)
        print("-" * 80)

    # Print the table
    print("\n--- Comparison Results Table ---\n")
    
    # Determine column widths
    col_widths = {key: len(key) for key in results_table[0].keys()}
    for row in results_table:
        for key, value in row.items():
            col_widths[key] = max(col_widths[key], len(str(value)))

    # Print header
    header = " | ".join(key.ljust(col_widths[key]) for key in results_table[0].keys())
    print(header)
    print("-+-".join("-" * col_widths[key] for key in results_table[0].keys()))

    # Print rows
    for row in results_table:
        print(" | ".join(str(value).ljust(col_widths[key]) for key, value in row.items()))

    export_to_csv(results_table)

if __name__ == "__main__":
    main()
