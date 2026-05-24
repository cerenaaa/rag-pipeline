# Production RAG Pipeline

[![CI](https://github.com/cerenaaa/rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/rag-pipeline/actions)

End-to-end Retrieval-Augmented Generation pipeline with chunking strategies, embedding, FAISS vector search, cross-encoder reranking, and RAGAS-style evaluation.

## Architecture

```
Query → Embedding → FAISS Retrieval → Cross-Encoder Rerank → Claude Generation → Answer
                         ↑
              Documents → Chunking → Embedding → Index
```

## Components

| Component | Method | Notes |
|---|---|---|
| Chunking | Recursive character + semantic | Overlap-aware, respects sentence boundaries |
| Embedding | sentence-transformers (all-mpnet-base-v2) | 768-dim, normalized |
| Vector store | FAISS (IVFFlat) | Configurable nprobe for recall/speed tradeoff |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Improves precision@5 by ~15% |
| Generation | Claude claude-sonnet-4-20250514 | Grounded, citation-aware prompting |
| Evaluation | Faithfulness, context recall, answer relevancy | RAGAS-inspired, LLM-judged |

## Structure
```
rag-pipeline/
├── pipeline/
│   ├── chunker.py          # Recursive + semantic chunking strategies
│   ├── embedder.py         # Sentence-transformer embedding + FAISS index
│   ├── retriever.py        # Retrieval + cross-encoder reranking
│   └── generator.py        # Claude-backed answer generation with citations
├── evaluation/
│   └── ragas_eval.py       # Faithfulness, context recall, answer relevancy
├── serving/
│   └── api.py              # FastAPI Q&A endpoint
├── data/
│   └── synthetic_corpus.py # Synthetic document corpus generator
└── run_pipeline.py         # End-to-end demo
```

## Quickstart
```bash
pip install -r requirements.txt
python run_pipeline.py
uvicorn serving.api:app --reload
```
