import os
from ingest.embedding import build_chroma
from ingest.pdf_chunker import run_pdf_chunking
from train_model.train_learned_reranker import train_model
from sentence_transformers import SentenceTransformer

def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(ROOT_DIR, "sql_store", "chunks.db")
    CHROMA_PATH = os.path.join(ROOT_DIR, "chromadb_store")
    PDF_DIR = os.path.join(ROOT_DIR, "data", "industrial-safety-pdfs")
    SOURCES_FILE = os.path.join(ROOT_DIR, "data", "sources.json")
    MODEL_PATH = os.path.join(ROOT_DIR, "model", "learned_reranker.pkl")

    if not os.path.exists(DB_PATH) :
        print("Step 1: Chunking PDFs...")
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        run_pdf_chunking(
            pdf_dir=PDF_DIR,
            sources_file=SOURCES_FILE,
            db_path=DB_PATH
        )
    if not os.path.exists(CHROMA_PATH) :
        print("\nStep 2: Building ChromaDB embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        build_chroma(
            db_path=DB_PATH,
            chroma_path=CHROMA_PATH,
            model=model
        )
    if not os.path.exists(MODEL_PATH) :
        print("\nStep 3: Training the model.")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        train_model(
            model=model,
            chroma_path=CHROMA_PATH,
            db_path=DB_PATH,
            model_save_path=MODEL_PATH
        )

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
