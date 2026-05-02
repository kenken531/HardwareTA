"""
query.py — Retrieve relevant passages and generate an answer via Ollama
"""

import sys
import pickle
import subprocess
import textwrap
from pathlib import Path

STORE_DIR    = Path(__file__).parent.parent / ".rag_store"
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 5
OLLAMA_MODEL = "llama3"   # change to: mistral, phi3, gemma2, etc.


# ── Embedder (must match what was used at ingest time) ────────────────────────
def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMBED_MODEL)
        def enc(text):
            v = m.encode(text)
            return v.tolist() if hasattr(v, "tolist") else list(v)
        return enc
    except Exception:
        return _load_tfidf_embedder()


def _load_tfidf_embedder():
    import numpy as np
    vocab_path = STORE_DIR / "tfidf_vocab.pkl"
    if not vocab_path.exists():
        raise RuntimeError("No TF-IDF vocab found. Run: python main.py ingest")
    with open(vocab_path, "rb") as f:
        vec = pickle.load(f)

    def enc(text: str) -> list:
        mat  = vec.transform([text]).toarray().astype("float32")
        norm = np.linalg.norm(mat)
        if norm:
            mat /= norm
        return mat[0].tolist()
    return enc


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(question: str, top_k: int = TOP_K) -> list:
    sys.path.insert(0, str(Path(__file__).parent))
    from vector_store import VectorStore

    encode = load_embedder()
    q_vec  = encode(question)

    store = VectorStore(STORE_DIR)
    if not store.load():
        raise RuntimeError("No index found. Run: python main.py ingest")

    return store.query(q_vec, top_k=top_k)


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(question: str, passages: list) -> str:
    parts = []
    for i, p in enumerate(passages):
        parts.append(
            f"[{i+1}] SOURCE: {p['source']} | PAGE: {p['page_num']}\n{p['text']}"
        )
    return (
        "You are an expert hardware engineer assistant.\n"
        "Answer the question using ONLY the datasheet excerpts below.\n"
        "Cite every claim with [filename, p.N]. If the excerpts lack the info, say so.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"DATASHEET EXCERPTS:\n" + "\n\n---\n\n".join(parts) +
        "\n\nANSWER (with page citations):"
    )


# ── Generation ────────────────────────────────────────────────────────────────
def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return f"[Ollama error] {result.stderr.strip() or 'no output'}"
    except FileNotFoundError:
        return _no_ollama_response(prompt)
    except subprocess.TimeoutExpired:
        return "[Ollama timed out — try: ollama pull phi3]"


def _no_ollama_response(prompt: str) -> str:
    excerpts = prompt.split("DATASHEET EXCERPTS:\n", 1)[-1].split("\n\nANSWER")[0]
    return (
        "[Ollama not installed — showing raw retrieved passages]\n"
        "Install from https://ollama.com  then: ollama pull llama3\n"
        + "─" * 60 + "\n"
        + excerpts
    )


# ── Display ───────────────────────────────────────────────────────────────────
def format_sources(passages: list) -> str:
    W = 58
    lines = [f"┌─ Top {len(passages)} retrieved passages {'─'*W}"]
    for i, p in enumerate(passages):
        snippet = textwrap.shorten(p["text"], width=W + 2, placeholder="…")
        lines.append(f"│ [{i+1}] {p['source']}  p.{p['page_num']}  score={p['score']:.3f}")
        lines.append(f"│     {snippet}")
    lines.append("└" + "─" * (W + 3))
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────
def answer(question: str, verbose: bool = True) -> dict:
    if verbose:
        print(f"\n🔍  {question}\n")
        print("    Retrieving relevant passages …")

    passages    = retrieve(question)
    if verbose:
        print(format_sources(passages))
        print("\n    Generating answer via Ollama …\n")

    prompt      = build_prompt(question, passages)
    answer_text = ask_ollama(prompt)

    W = 68
    if verbose:
        print("═" * W)
        print("  ANSWER")
        print("═" * W)
        for line in answer_text.splitlines():
            print(textwrap.fill(line, width=W) if line.strip() else "")
        print("═" * W)
        print("\n  Page citations:")
        seen = set()
        for p in passages:
            key = (p["source"], p["page_num"])
            if key not in seen:
                print(f"    • {p['source']}  →  page {p['page_num']}")
                seen.add(key)
        print()

    return {"question": question, "answer": answer_text, "passages": passages}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What is the maximum operating voltage and current consumption?"
    answer(q)
