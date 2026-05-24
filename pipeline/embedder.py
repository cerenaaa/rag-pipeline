"""
Embedding + FAISS vector store for RAG retrieval.
Supports building, saving, loading, and querying the index.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
from dataclasses import dataclass

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from pipeline.chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, model_name: str = "all-mpnet-base-v2", nlist: int = 64):
        self.model = SentenceTransformer(model_name)
        self.nlist = nlist
        self.index = None
        self.chunks: list[Chunk] = []
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=len(texts) > 100).astype("float32")

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vectors = self.embed(texts)

        if FAISS_AVAILABLE:
            quantizer = faiss.IndexFlatIP(self.dim)
            n = len(vectors)
            nlist = min(self.nlist, max(1, n // 10))
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = min(nlist, 8)
        else:
            # Fallback: brute-force numpy
            self.index = vectors
        print(f"Built index: {len(chunks)} chunks, dim={self.dim}")
        return self

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        q = self.embed([query])
        if FAISS_AVAILABLE:
            scores, idxs = self.index.search(q, k)
            return [SearchResult(chunk=self.chunks[i], score=float(s))
                    for s, i in zip(scores[0], idxs[0]) if i >= 0]
        else:
            sims = (self.index @ q.T).flatten()
            top = np.argsort(sims)[::-1][:k]
            return [SearchResult(chunk=self.chunks[i], score=float(sims[i])) for i in top]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if FAISS_AVAILABLE:
            faiss.write_index(self.index, f"{path}/faiss.index")
        else:
            np.save(f"{path}/vectors.npy", self.index)
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Saved index to {path}/")

    def load(self, path: str):
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        if FAISS_AVAILABLE:
            self.index = faiss.read_index(f"{path}/faiss.index")
        else:
            self.index = np.load(f"{path}/vectors.npy")
        return self