"""
query.py — Retrieve relevant passages and generate an answer via Ollama
"""

import sys
import json
import pickle
import textwrap
import urllib.request
import urllib.error
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
STORE_DIR      = Path(__file__).parent.parent / ".rag_store"
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K          = 5
OLLAMA_MODEL   = None             # None = auto-detect first available model
OLLAMA_URL     = "http://localhost:11434"
OLLAMA_TIMEOUT = 180              # seconds — bump this on slow machines


# ── Ollama helpers ────────────────────────────────────────────────────────────
def _ollama_get(path: str):
    """GET from Ollama REST API, return parsed JSON or None."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}{path}")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _ollama_post(path: str, payload: dict, timeout: int = OLLAMA_TIMEOUT):
    """POST to Ollama REST API, return parsed JSON or raise."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def detect_model() -> str | None:
    """
    Return the model name to use:
      1. OLLAMA_MODEL constant if set
      2. First model returned by /api/tags
      3. None if Ollama is unreachable
    """
    if OLLAMA_MODEL:
        return OLLAMA_MODEL
    data = _ollama_get("/api/tags")
    if data and data.get("models"):
        name = data["models"][0]["name"]
        return name
    return None


def ollama_reachable() -> bool:
    return _ollama_get("/api/tags") is not None


# ── Generation ────────────────────────────────────────────────────────────────
def ask_ollama(prompt: str) -> str:
    """
    Call Ollama via its HTTP REST API (/api/generate).
    Auto-detects the first available model if OLLAMA_MODEL is None.
    Falls back to displaying raw retrieved passages if Ollama is unreachable.
    """
    model = detect_model()

    if model is None:
        return _no_ollama_response(prompt)

    print(f"    Using model: {model}")

    try:
        result = _ollama_post(
            "/api/generate",
            {
                "model":  model,
                "prompt": prompt,
                "stream": False,        # wait for the full response
                "options": {
                    "temperature": 0.1, # low temp = more factual
                    "num_predict": 512,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        return result.get("response", "").strip()

    except urllib.error.URLError as e:
        return f"[Ollama connection error] {e.reason}\nIs 'ollama serve' running?"
    except TimeoutError:
        return (
            f"[Ollama timed out after {OLLAMA_TIMEOUT}s]\n"
            "Try a smaller model: ollama pull phi3\n"
            f"Or increase OLLAMA_TIMEOUT in src/query.py (currently {OLLAMA_TIMEOUT}s)"
        )
    except Exception as e:
        return f"[Ollama error] {type(e).__name__}: {e}"


def _no_ollama_response(prompt: str) -> str:
    excerpts = prompt.split("DATASHEET EXCERPTS:\n", 1)[-1].split("\n\nANSWER")[0]
    return (
        "⚠️  Ollama not reachable at " + OLLAMA_URL + "\n"
        "   • Make sure Ollama is running: ollama serve\n"
        "   • Pull a model if needed:      ollama pull phi3\n"
        "   • Then re-run your question.\n"
        + "─" * 60 + "\n"
        + "[Raw retrieved passages shown below]\n\n"
        + excerpts
    )


# ── Embedder (must match ingest time) ─────────────────────────────────────────
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
        "DATASHEET EXCERPTS:\n" + "\n\n---\n\n".join(parts) +
        "\n\nANSWER (with page citations):"
    )


# ── Display ───────────────────────────────────────────────────────────────────
def format_sources(passages: list) -> str:
    W = 58
    lines = [f"┌─ Top {len(passages)} retrieved passages {'─' * W}"]
    for i, p in enumerate(passages):
        snippet = textwrap.shorten(p["text"], width=W + 2, placeholder="…")
        lines.append(f"│ [{i+1}] {p['source']}  p.{p['page_num']}  score={p['score']:.3f}")
        lines.append(f"│     {snippet}")
    lines.append("└" + "─" * (W + 3))
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────
def answer(question: str, verbose: bool = True) -> dict:
    if verbose:
        # Show Ollama status upfront so there are no surprises
        model = detect_model()
        if model:
            print(f"\n[Ollama ✓] Connected — model: {model}")
        else:
            print(f"\n[Ollama ✗] Not reachable at {OLLAMA_URL}")
            print("           Run 'ollama serve' in another terminal, then retry.\n")

        print(f"\n🔍  {question}\n")
        print("    Retrieving relevant passages …")

    passages    = retrieve(question)

    if verbose:
        print(format_sources(passages))
        print("\n    Generating answer …\n")

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
