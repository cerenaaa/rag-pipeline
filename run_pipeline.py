"""End-to-end RAG pipeline demo."""
import os
from data.synthetic_corpus import generate_corpus
from pipeline.chunker import RecursiveChunker
from pipeline.embedder import VectorStore
from pipeline.retriever import Retriever
from pipeline.generator import RAGGenerator


def main():
    print("Building corpus...")
    docs = generate_corpus(n_docs=100)

    print("Chunking documents...")
    chunker = RecursiveChunker(chunk_size=400, overlap=50)
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc["text"], doc_id=doc["doc_id"])
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")

    print("Building vector index...")
    store = VectorStore()
    store.build(all_chunks)

    retriever = Retriever(store, use_reranker=False, recall_k=10, final_k=3)

    if os.environ.get("ANTHROPIC_API_KEY"):
        generator = RAGGenerator(retriever)
        questions = [
            "How does gradient boosting handle overfitting?",
            "What is retrieval-augmented generation?",
            "How is price elasticity estimated?",
        ]
        for q in questions:
            print(f"\nQ: {q}")
            response = generator.answer(q)
            print(f"A: {response.answer[:300]}...")
            print(f"Sources: {[s['doc_id'] for s in response.sources]}")
    else:
        print("\nSet ANTHROPIC_API_KEY to test generation. Retrieval-only demo:")
        results = retriever.retrieve("What is price elasticity?")
        for r in results:
            print(f"  [{r.score:.3f}] {r.chunk.text[:100]}...")


if __name__ == "__main__":
    main()