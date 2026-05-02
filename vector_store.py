"""
vector_store.py — Lightweight pure-NumPy/JSON vector store

A drop-in replacement for ChromaDB that avoids its heavy native dependencies.
Uses cosine similarity with NumPy matrix multiplication for retrieval.
Persists to two files:
  .rag_store/vectors.npy   — float32 matrix (N × dim)
  .rag_store/metadata.json — list of dicts with text, source, page_num
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path

STORE_DIR = Path(__file__).parent.parent / ".rag_store"


class VectorStore:
    """Minimal cosine-similarity vector store backed by NumPy + JSON."""

    def __init__(self, store_dir: Path = STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._vec_path  = self.store_dir / "vectors.npy"
        self._meta_path = self.store_dir / "metadata.json"
        self._vectors: np.ndarray | None = None   # (N, dim)
        self._metadata: list[dict]        = []

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self):
        if self._vectors is not None:
            np.save(str(self._vec_path), self._vectors)
        with open(self._meta_path, "w") as f:
            json.dump(self._metadata, f)

    def load(self) -> bool:
        if self._vec_path.exists() and self._meta_path.exists():
            self._vectors  = np.load(str(self._vec_path))
            with open(self._meta_path) as f:
                self._metadata = json.load(f)
            return True
        return False

    def clear(self):
        self._vectors  = None
        self._metadata = []
        for p in [self._vec_path, self._meta_path]:
            if p.exists():
                p.unlink()

    # ── Write ─────────────────────────────────────────────────────────────────
    def add(self, embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        mat = np.array(embeddings, dtype="float32")
        # L2-normalise rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

        if self._vectors is None:
            self._vectors = mat
        else:
            self._vectors = np.vstack([self._vectors, mat])

        for doc, meta in zip(documents, metadatas):
            self._metadata.append({"text": doc, **meta})

    # ── Read ──────────────────────────────────────────────────────────────────
    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if self._vectors is None or len(self._metadata) == 0:
            return []
        q = np.array(query_embedding, dtype="float32")
        norm = np.linalg.norm(q)
        if norm:
            q /= norm
        scores = self._vectors @ q              # cosine similarity (N,)
        k      = min(top_k, len(scores))
        idx    = np.argsort(scores)[::-1][:k]
        results = []
        for i in idx:
            results.append({
                **self._metadata[i],
                "score": float(scores[i]),
            })
        return results

    def count(self) -> int:
        return len(self._metadata)
