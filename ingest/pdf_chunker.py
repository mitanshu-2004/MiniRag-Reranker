import os
import json
import re
import sqlite3
from tqdm import tqdm
import PyPDF2
from Crypto.Cipher import AES  # ensures AES support for encrypted PDFs

class PDFChunker:
    def __init__(self, pdf_dir, sources_file, db_path, chunk_size=300, chunk_overlap=50):
        self.pdf_dir = pdf_dir
        self.sources_file = sources_file
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sources = self._load_sources()
        self._setup_database()

    def _load_sources(self):
        with open(self.sources_file, 'r') as f:
            sources_list = json.load(f)
        sources_dict = {}
        for s in sources_list:
            filename = s.get("filename") or os.path.basename(s.get("url", ""))
            sources_dict[filename] = {"title": s.get("title", filename), "url": s.get("url", "")}
        return sources_dict

    def _setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_name TEXT,
            doc_title TEXT,
            doc_url TEXT,
            chunk_index INTEGER,
            content TEXT,
            is_title INTEGER DEFAULT 0,
            page_num INTEGER
        )''')
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            doc_name,
            doc_title,
            content='chunks',
            content_rowid='id'
        )''')
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content, doc_name, doc_title) 
            VALUES (new.id, new.content, new.doc_name, new.doc_title);
        END''')
        conn.commit()
        conn.close()

    def _extract_text_from_pdf(self, pdf_path):
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.is_encrypted:
                    try:
                        reader.decrypt('')
                    except Exception as e:
                        print(f"Warning: Could not decrypt {pdf_path}: {e}")
                        return pages
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append((i+1, text))
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return pages

    def _chunk_text(self, text, page_num):
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        chunks = []
        if len(words) <= self.chunk_size:
            return [(page_num, ' '.join(words))]
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i+self.chunk_size])
            if chunk:
                chunks.append((page_num, chunk))
        return chunks

    def process_pdfs(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        cursor.execute("DELETE FROM chunks")
        conn.commit()
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            doc_info = self.sources.get(pdf_file, {"title": pdf_file, "url": ""})
            pages = self._extract_text_from_pdf(pdf_path)
            for page_num, text in pages:
                if page_num == 1:
                    first_para = text.split('\n\n')[0] if '\n\n' in text else text[:200]
                    cursor.execute(
                        "INSERT INTO chunks (doc_name, doc_title, doc_url, chunk_index, content, is_title, page_num) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (pdf_file, doc_info["title"], doc_info["url"], 0, first_para, 1, page_num)
                    )
                chunks = self._chunk_text(text, page_num)
                for i, (page, chunk) in enumerate(chunks):
                    cursor.execute(
                        "INSERT INTO chunks (doc_name, doc_title, doc_url, chunk_index, content, page_num) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (pdf_file, doc_info["title"], doc_info["url"], i+1, chunk, page)
                    )
        conn.commit()
        conn.close()
        print(f"PDF processing completed. Total chunks: {self.get_chunk_count()}")

    def get_chunk_count(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        conn.close()
        return count


def run_pdf_chunking(pdf_dir, sources_file, db_path):
    chunker = PDFChunker(pdf_dir, sources_file, db_path)
    chunker.process_pdfs()
