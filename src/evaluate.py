from load_data import load_squad_sample
from build_corpus import build_corpus
from retriever_tfidf import TfidfRetriever

def evaluate_recall_at_k(retriever, qa_pairs, k=5):
    """
    Computes Recall@k: the fraction of questions for which the correct
    context appears somewhere in the retriever's top-k guesses.

    This is the standard way to evaluate a retriever - it doesn't check if
    we guessed the #1 result correctly, just whether the right answer was
    in the shortlist at all. That mirrors how retrieval is used in practice
    (e.g. feeding top-k results to a downstream QA/reader model).
    """
    correct = 0
    for pair in qa_pairs:
        top_k = retriever.retrieve(pair["question"], k=k)
        if pair["correct_context_id"] in top_k:
            correct += 1
    return correct / len(qa_pairs)

if __name__ == "__main__":
    dataset = load_squad_sample()
    corpus, qa_pairs = build_corpus(dataset)
    retriever = TfidfRetriever(corpus)

    for k in [1, 3, 5, 10]:
        recall = evaluate_recall_at_k(retriever, qa_pairs, k=k)
        print(f"Recall@{k}: {recall:.3f}")