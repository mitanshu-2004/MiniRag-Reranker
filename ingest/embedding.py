import sqlite3
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def fetch_chunks(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, doc_name, doc_title, doc_url, chunk_index, content, is_title, page_num FROM chunks ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def build_chroma(db_path: str, chroma_path: str, model: SentenceTransformer):
    print("Loading chunks from DB...")
    chunks = fetch_chunks(db_path)
    if not chunks:
        raise SystemExit("No chunks found in DB. Run the chunker first.")

    print(f"Loaded {len(chunks)} chunks.")

    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("safety_docs")

    print("Embedding and adding to ChromaDB...")
    for c in tqdm(chunks, desc="Processing chunks"):
        emb = model.encode(c["content"]).tolist()
        collection.add(
            ids=[str(c["id"])],
            embeddings=[emb],
            documents=[c["content"]],
            metadatas=[{
                "doc_name": c["doc_name"],
                "doc_title": c["doc_title"],
                "doc_url": c["doc_url"],
                "page_num": c["page_num"],
                "chunk_index": c["chunk_index"],
                "is_title": c["is_title"]
            }]
        )
    print(f"ChromaDB built at {chroma_path}")
