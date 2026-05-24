"""Synthetic document corpus for RAG pipeline testing."""
import random

TOPICS = {
    "machine_learning": [
        "Gradient boosting methods like XGBoost and LightGBM use an ensemble of weak learners trained sequentially, where each tree corrects residuals from the previous. Regularization parameters control overfitting.",
        "Cross-validation is essential for model selection. Stratified k-fold preserves class balance across folds. Time-series data requires temporal splits to prevent data leakage.",
        "Feature importance in tree models can be computed as mean decrease in impurity (MDI) or via permutation. SHAP values provide consistent, game-theoretically grounded attributions.",
    ],
    "pricing": [
        "Price elasticity of demand measures responsiveness of quantity demanded to price changes. A log-log OLS regression estimates constant elasticity: log(Q) = α + ε·log(P).",
        "Dynamic pricing adjusts prices in real-time based on demand signals, competitor pricing, and inventory levels. Reinforcement learning enables adaptive pricing with delayed reward signals.",
    ],
    "nlp": [
        "Transformer models use self-attention to capture long-range dependencies in text. BERT uses bidirectional attention trained with masked language modeling and next sentence prediction.",
        "Retrieval-Augmented Generation (RAG) combines parametric knowledge in LLMs with non-parametric retrieval from a document corpus, reducing hallucination and enabling domain adaptation without fine-tuning.",
        "Named entity recognition (NER) classifies tokens into categories like person, organization, and location. Fine-tuning pre-trained transformers achieves state-of-the-art performance with minimal labeled data.",
    ],
}


def generate_corpus(n_docs: int = 100, seed: int = 42) -> list[dict]:
    random.seed(seed)
    docs = []
    all_passages = [(topic, p) for topic, passages in TOPICS.items() for p in passages]
    for i in range(n_docs):
        topic, base = random.choice(all_passages)
        noise = f" This is document {i} covering {topic} in detail."
        docs.append({"doc_id": f"doc_{i:04d}", "topic": topic, "text": base + noise})
    print(f"Generated {len(docs)} documents")
    return docs


if __name__ == "__main__":
    corpus = generate_corpus()
    for doc in corpus[:3]:
        print(f"[{doc['doc_id']}] {doc['text'][:80]}...")