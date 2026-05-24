"""
Retrieval with optional cross-encoder reranking.
Two-stage: FAISS recall (top-K) → cross-encoder precision (top-N).
"""
from __future__ import annotations
from dataclasses import dataclass

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

from pipeline.embedder import VectorStore, SearchResult


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranker: bool = True,
        recall_k: int = 20,
        final_k: int = 5,
    ):
        self.store = vector_store
        self.recall_k = recall_k
        self.final_k = final_k
        self.reranker = None
        if use_reranker and RERANKER_AVAILABLE:
            try:
                self.reranker = CrossEncoder(reranker_model)
                print(f"Reranker loaded: {reranker_model}")
            except Exception as e:
                print(f"Reranker unavailable: {e}")

    def retrieve(self, query: str) -> list[SearchResult]:
        # Stage 1: fast embedding-based recall
        candidates = self.store.search(query, k=self.recall_k)
        if not candidates:
            return []

        # Stage 2: cross-encoder reranking
        if self.reranker:
            pairs = [(query, c.chunk.text) for c in candidates]
            scores = self.reranker.predict(pairs)
            reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [SearchResult(chunk=r.chunk, score=float(s))
                    for s, r in reranked[:self.final_k]]

        return candidates[:self.final_k]