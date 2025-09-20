import os
from ingest.embedding import build_chroma
from ingest.pdf_chunker import run_pdf_chunking
from train_model.train_learned_reranker import train_model
from sentence_transformers import SentenceTransformer

def main():
    r_dir = os.path.dirname(os.path.abspath(__file__))
    db_p = r_dir + "\\sql_store\\chunks.db"
    c_path = r_dir + "\\chromadb_store"
    p_dir = r_dir + "\\data\\industrial-safety-pdfs"
    s_file = r_dir + "\\data\\sources.json"
    m_path = r_dir + "\\model\\learned_reranker.pkl"

    if not os.path.exists(db_p) :
        print("Step 1: Chunking PDFs...\n")
        os.makedirs(os.path.dirname(db_p), exist_ok=True)
        run_pdf_chunking(pdf_dir=p_dir, sources_file=s_file, db_path=db_p)
    if not os.path.exists(c_path) :
        print("\nStep 2: Building ChromaDB embeddings...\n")
        mod = SentenceTransformer("all-MiniLM-L6-v2")
        build_chroma(db_path=db_p, chroma_path=c_path, model=mod)
    if not os.path.exists(m_path) :
        print("\nStep 3: Training the model.\n")
        os.makedirs(os.path.dirname(m_path), exist_ok=True)
        train_model(model=mod, chroma_path=c_path, db_path=db_p, model_save_path=m_path)

    print("\nPipeline completed successfully!\n")

if __name__ == "__main__":
    main()
