from load_data import load_squad_sample
from build_corpus import build_corpus
from retriever_tfidf import TfidfRetriever
from retriever_embedding import EmbeddingRetriever
from lexical_overlap import compute_overlap

def recall_at_k_by_bucket(retriever, qa_pairs, corpus, k=5):
    """
    Same bucketing logic as before, but reusable for any retriever so we
    can run it once for TF-IDF and once for embeddings, on the identical
    questions and buckets - that's what makes this a fair comparison.
    """
    buckets = {"low (<0.3)": [], "medium (0.3-0.6)": [], "high (>0.6)": []}

    for pair in qa_pairs:
        context = corpus[pair["correct_context_id"]]
        overlap = compute_overlap(pair["question"], context)
        top_k = retriever.retrieve(pair["question"], k=k)
        correct = pair["correct_context_id"] in top_k

        if overlap < 0.3:
            buckets["low (<0.3)"].append(correct)
        elif overlap < 0.6:
            buckets["medium (0.3-0.6)"].append(correct)
        else:
            buckets["high (>0.6)"].append(correct)

    return {name: sum(v) / len(v) if v else 0 for name, v in buckets.items()}


if __name__ == "__main__":
    dataset = load_squad_sample()
    corpus, qa_pairs = build_corpus(dataset)

    print("Building TF-IDF retriever...")
    tfidf = TfidfRetriever(corpus)
    tfidf_results = recall_at_k_by_bucket(tfidf, qa_pairs, corpus)

    print("Building embedding retriever (downloads model on first run)...")
    embed = EmbeddingRetriever(corpus)
    embed_results = recall_at_k_by_bucket(embed, qa_pairs, corpus)

    print("\n=== Recall@5 by lexical overlap bucket ===")
    print(f"{'Bucket':<20}{'TF-IDF':<12}{'Embeddings':<12}")
    for bucket in tfidf_results:
        print(f"{bucket:<20}{tfidf_results[bucket]:<12.3f}{embed_results[bucket]:<12.3f}")