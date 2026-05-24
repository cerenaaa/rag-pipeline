"""
Document chunking strategies for RAG pipelines.
Supports recursive character splitting and semantic (embedding-based) chunking.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: int
    metadata: dict


class RecursiveChunker:
    """
    Recursively splits on paragraph → sentence → word boundaries.
    Maintains overlap to preserve context across chunk boundaries.
    """
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split(self, text: str, separators: list[str]) -> list[str]:
        sep = separators[0]
        splits = re.split(f"(?<={re.escape(sep)})", text) if sep else list(text)
        splits = [s for s in splits if s.strip()]

        chunks, current = [], ""
        for split in splits:
            if len(current) + len(split) <= self.chunk_size:
                current += split
            else:
                if current:
                    chunks.append(current)
                if len(split) > self.chunk_size and len(separators) > 1:
                    chunks.extend(self._split(split, separators[1:]))
                    current = ""
                else:
                    current = split
        if current:
            chunks.append(current)
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1 or self.overlap == 0:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.overlap:]
            overlapped.append(tail + chunks[i])
        return overlapped

    def chunk(self, text: str, doc_id: str = "doc_0") -> list[Chunk]:
        raw = self._split(text, self.SEPARATORS)
        raw = self._add_overlap(raw)
        return [Chunk(text=c.strip(), doc_id=doc_id, chunk_id=i, metadata={})
                for i, c in enumerate(raw) if c.strip()]


class SemanticChunker:
    """
    Semantic chunking: splits at embedding-space discontinuities.
    Groups sentences whose embeddings are cosine-similar above a threshold.
    """
    def __init__(self, embed_fn: Callable, similarity_threshold: float = 0.85,
                 max_chunk_size: int = 800):
        self.embed_fn = embed_fn
        self.threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def chunk(self, text: str, doc_id: str = "doc_0") -> list[Chunk]:
        sentences = self._sentences(text)
        if not sentences:
            return []
        embeddings = self.embed_fn(sentences)  # (N, dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        groups, current_group = [], [sentences[0]]
        for i in range(1, len(sentences)):
            sim = float(embeddings[i - 1] @ embeddings[i])
            total_len = sum(len(s) for s in current_group) + len(sentences[i])
            if sim >= self.threshold and total_len <= self.max_chunk_size:
                current_group.append(sentences[i])
            else:
                groups.append(" ".join(current_group))
                current_group = [sentences[i]]
        if current_group:
            groups.append(" ".join(current_group))

        return [Chunk(text=g, doc_id=doc_id, chunk_id=i, metadata={})
                for i, g in enumerate(groups) if g.strip()]