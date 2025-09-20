import os, json, re, sqlite3, PyPDF2
from tqdm import tqdm
from Crypto.Cipher import AES

class PDFChunker:
    def __init__(self, pd, sf, dp, cs=300, co=50):
        self.pd = pd
        self.sf = sf
        self.dp = dp
        self.cs = cs
        self.co = co
        self.src = self._load_sources()
        self._setup_db()

    def _load_sources(self):
        with open(self.sf, 'r') as f: src_list = json.load(f)
        src_dict = {}
        for s in src_list:
            fn = s.get("filename") or os.path.basename(s.get("url", ""))
            src_dict[fn] = {"title": s.get("title", fn), "url": s.get("url", "")}
        return src_dict

    def _setup_db(self):
        con = sqlite3.connect(self.dp)
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, doc_name TEXT, doc_title TEXT, doc_url TEXT, chunk_index INTEGER, content TEXT, is_title INTEGER DEFAULT 0, page_num INTEGER)''')
        cur.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, doc_name, doc_title, content='chunks', content_rowid='id')''')
        cur.execute('''CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN INSERT INTO chunks_fts(rowid, content, doc_name, doc_title) VALUES (new.id, new.content, new.doc_name, new.doc_title); END''')
        con.commit(); con.close()

    def _extract_text_from_pdf(self, pp):
        pgs = []
        try:
            with open(pp, 'rb') as f: r = PyPDF2.PdfReader(f)
            if r.is_encrypted: r.decrypt('')
            for i, p in enumerate(r.pages): t = p.extract_text();
            if t and t.strip(): pgs.append((i+1, t))
        except Exception as e: print(f"Error extracting text from {pp}: {e}")
        return pgs

    def _chunk_text(self, txt, pn):
        txt = re.sub(r'\s+', ' ', txt).strip()
        wds = txt.split()
        cks = []
        if len(wds) <= self.cs: return [(pn, ' '.join(wds))]
        for i in range(0, len(wds), self.cs - self.co): ck = ' '.join(wds[i:i+self.cs]);
        if ck: cks.append((pn, ck))
        return cks

    def process_pdfs(self):
        con = sqlite3.connect(self.dp); cur = con.cursor()
        pf = [f for f in os.listdir(self.pd) if f.lower().endswith('.pdf')]
        cur.execute("DELETE FROM chunks"); con.commit()
        for p_f in tqdm(pf, desc="Processing PDFs"):
            p_p = os.path.join(self.pd, p_f)
            d_i = self.src.get(p_f, {"title": p_f, "url": ""})
            pgs = self._extract_text_from_pdf(p_p)
            for p_n, txt in pgs:
                if p_n == 1: fp = txt.split('\n\n')[0] if '\n\n' in txt else txt[:200];
                cur.execute("INSERT INTO chunks (doc_name, doc_title, doc_url, chunk_index, content, is_title, page_num) VALUES (?, ?, ?, ?, ?, ?, ?)", (p_f, d_i["title"], d_i["url"], 0, fp, 1, p_n))
                cks = self._chunk_text(txt, p_n)
                for i, (p, ck) in enumerate(cks):
                    cur.execute("INSERT INTO chunks (doc_name, doc_title, doc_url, chunk_index, content, page_num) VALUES (?, ?, ?, ?, ?, ?)", (p_f, d_i["title"], d_i["url"], i+1, ck, p))
        con.commit(); con.close()
        print(f"PDF processing completed. Total chunks: {self.get_chunk_count()}\n")

    def get_chunk_count(self):
        con = sqlite3.connect(self.dp); cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks"); cnt = cur.fetchone()[0]
        con.close(); return cnt


def run_pdf_chunking(pd, sf, dp):
    ckr = PDFChunker(pd, sf, dp)
    ckr.process_pdfs()
