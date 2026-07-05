import re
from load_data import load_squad_sample
from build_corpus import build_corpus
from retriever_tfidf import TfidfRetriever

def compute_overlap(question, context):
    """
    Computes what fraction of the question's (non-trivial) words also
    appear in the context. Lowercased, punctuation-stripped, simple
    whitespace split - deliberately simple since we just need a rough
    signal, not a precise linguistic measure.
    """
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return set(text.split())

    # words too short/common to be meaningful signal (a, is, of, the...)
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "of", "in",
                 "on", "at", "to", "for", "and", "or", "did", "does", "do"}

    q_words = tokenize(question) - stopwords
    c_words = tokenize(context) - stopwords

    if not q_words:
        return 0.0

    shared = q_words & c_words
    return len(shared) / len(q_words)


if __name__ == "__main__":
    dataset = load_squad_sample()
    corpus, qa_pairs = build_corpus(dataset)
    retriever = TfidfRetriever(corpus)

    results = []
    for pair in qa_pairs:
        context = corpus[pair["correct_context_id"]]
        overlap = compute_overlap(pair["question"], context)
        top5 = retriever.retrieve(pair["question"], k=5)
        correct = pair["correct_context_id"] in top5
        results.append({"overlap": overlap, "correct": correct})

    # Bucket questions into low/medium/high overlap and check success rate per bucket
    buckets = {"low (<0.3)": [], "medium (0.3-0.6)": [], "high (>0.6)": []}
    for r in results:
        if r["overlap"] < 0.3:
            buckets["low (<0.3)"].append(r["correct"])
        elif r["overlap"] < 0.6:
            buckets["medium (0.3-0.6)"].append(r["correct"])
        else:
            buckets["high (>0.6)"].append(r["correct"])

    print("Recall@5 by lexical overlap bucket:\n")
    for name, values in buckets.items():
        if values:
            recall = sum(values) / len(values)
            print(f"{name}: n={len(values)}, Recall@5={recall:.3f}")
        else:
            print(f"{name}: n=0 (empty bucket)")