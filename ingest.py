"""
ingest.py — Parse PDFs and index chunks into the local VectorStore
"""

import re
import sys
import pickle
from pathlib import Path

DATASHEETS_DIR = Path(__file__).parent.parent / "datasheets"
STORE_DIR      = Path(__file__).parent.parent / ".rag_store"
EMBED_MODEL    = "all-MiniLM-L6-v2"
CHUNK_SIZE     = 400
CHUNK_STRIDE   = 320   # stride = chunk_size - overlap (guaranteed forward progress)


def extract_pages(pdf_path: Path) -> list:
    import fitz
    doc     = fitz.open(str(pdf_path))
    n       = len(doc)
    pages   = []
    for i in range(n):
        text = doc[i].get_text("text")
        text = re.sub(r'[ \t]{2,}', ' ', text).strip()
        if text:
            pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text: str) -> list:
    """Split into fixed-stride overlapping chunks — no infinite loop possible."""
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        i += CHUNK_STRIDE
    return chunks


def _make_tfidf_embedder(corpus: list, verbose: bool):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    if verbose:
        print(f"    Fitting TF-IDF on {len(corpus)} chunks …")
    vec = TfidfVectorizer(max_features=512, sublinear_tf=True)
    vec.fit(corpus)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STORE_DIR / "tfidf_vocab.pkl", "wb") as f:
        pickle.dump(vec, f)
    if verbose:
        print("    Vocab saved.")

    def encode(text: str) -> list:
        mat  = vec.transform([text]).toarray().astype("float32")
        norm = np.linalg.norm(mat)
        if norm:
            mat /= norm
        return mat[0].tolist()
    return encode


def load_embedder(corpus: list, verbose: bool = True):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        if verbose:
            print(f"    sentence-transformers/{EMBED_MODEL}")
        def encode(text: str) -> list:
            v = model.encode(text)
            return v.tolist() if hasattr(v, "tolist") else list(v)
        return encode
    except Exception as exc:
        if verbose:
            print(f"    [!] sentence-transformers unavailable ({type(exc).__name__})")
            print(f"    [→] Using TF-IDF embedder (offline fallback)")
        return _make_tfidf_embedder(corpus, verbose)


def ingest_pdfs(verbose: bool = True) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from vector_store import VectorStore

    pdfs = list(DATASHEETS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[!] No PDFs found in {DATASHEETS_DIR}\n    Run: python main.py download")
        return 0

    if verbose:
        print(f"[+] Found {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")

    all_docs, all_metas = [], []
    for pdf_path in pdfs:
        if verbose:
            print(f"\n[→] Parsing: {pdf_path.name}")
        pages = extract_pages(pdf_path)
        if verbose:
            print(f"    {len(pages)} page(s) with text")
        for p in pages:
            for chunk in chunk_text(p["text"]):
                all_docs.append(chunk)
                all_metas.append({"source": pdf_path.name, "page_num": p["page_num"]})

    if not all_docs:
        print("[!] No text extracted from PDFs.")
        return 0

    if verbose:
        print(f"\n[+] {len(all_docs)} chunks total")
        print("[+] Loading embedding model …")

    encode = load_embedder(all_docs, verbose=verbose)

    if verbose:
        print(f"\n[+] Embedding …")

    all_embeddings = [encode(doc) for doc in all_docs]

    if verbose:
        print(f"    {len(all_embeddings)} chunks embedded, dim={len(all_embeddings[0])}")

    store = VectorStore(STORE_DIR)
    store.clear()
    store.add(all_embeddings, all_docs, all_metas)
    store.save()

    if verbose:
        print(f"\n[✓] Indexed {store.count()} chunks → {STORE_DIR}")
    return store.count()


if __name__ == "__main__":
    ingest_pdfs()
