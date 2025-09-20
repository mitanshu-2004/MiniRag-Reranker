import sqlite3, chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

def fetch_chunks(dp):
    con = sqlite3.connect(dp)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    sql = "SELECT id, doc_name, doc_title, doc_url, chunk_index, content, is_title, page_num FROM chunks ORDER BY id"
    cur.execute(sql)
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]

def build_chroma(dp: str, cp: str, mod: SentenceTransformer):
    print("Loading chunks from DB...\n")
    chunks = fetch_chunks(dp)
    if not chunks:
        raise SystemExit("No chunks found in DB. Run the chunker first.")

    print(f"Loaded {len(chunks)} chunks.\n")

    cli = chromadb.PersistentClient(path=cp, settings=Settings(anonymized_telemetry=False))
    coll = cli.get_or_create_collection("safety_docs")

    print("Embedding and adding to ChromaDB...\n")
    for c in tqdm(chunks, desc="Processing chunks"):
        emb = mod.encode(c["content"]).tolist()
        coll.add(
            ids=[str(c["id"])],
            embeddings=[emb],
            documents=[c["content"]],
            metadatas=[{"doc_name": c["doc_name"], "doc_title": c["doc_title"], "doc_url": c["doc_url"], "page_num": c["page_num"], "chunk_index": c["chunk_index"], "is_title": c["is_title"]}]
        )
    print(f"ChromaDB built at {cp}\n")
