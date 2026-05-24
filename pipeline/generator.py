"""
Citation-aware answer generation using Claude.
Grounds responses strictly in retrieved context.
"""
from __future__ import annotations
from dataclasses import dataclass
import anthropic
from pipeline.retriever import Retriever


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    context_used: list[str]


class RAGGenerator:
    def __init__(self, retriever: Retriever, model: str = "claude-sonnet-4-20250514"):
        self.retriever = retriever
        self.client = anthropic.Anthropic()
        self.model = model

    def _build_context(self, results) -> str:
        parts = []
        for i, r in enumerate(results):
            parts.append(f"[Source {i+1}] (doc: {r.chunk.doc_id})\n{r.chunk.text}")
        return "\n\n".join(parts)

    def answer(self, question: str) -> RAGResponse:
        results = self.retriever.retrieve(question)
        context = self._build_context(results)

        prompt = f"""Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so — do not hallucinate.
Cite sources as [Source N] inline.

Context:
{context}

Question: {question}

Answer:"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        return RAGResponse(
            answer=response.content[0].text,
            sources=[{"doc_id": r.chunk.doc_id, "score": r.score, "text": r.chunk.text[:200]}
                     for r in results],
            context_used=[r.chunk.text for r in results],
        )