from load_data import load_squad_sample
from build_corpus import build_corpus
from retriever_tfidf import TfidfRetriever
from lexical_overlap import compute_overlap

if __name__ == "__main__":
    dataset = load_squad_sample()
    corpus, qa_pairs = build_corpus(dataset)
    retriever = TfidfRetriever(corpus)

    failures = []
    for pair in qa_pairs:
        context = corpus[pair["correct_context_id"]]
        overlap = compute_overlap(pair["question"], context)
        top5 = retriever.retrieve(pair["question"], k=5)
        correct = pair["correct_context_id"] in top5

        # We only care about low-overlap questions that ALSO failed
        if overlap < 0.3 and not correct:
            failures.append({
                "question": pair["question"],
                "answer": pair["answer"],
                "overlap": overlap,
                "correct_context": context,
                "wrongly_retrieved": corpus[top5[0]]
            })

    print(f"Total low-overlap failures: {len(failures)}\n")

    # Print 3 concrete examples
    for i, f in enumerate(failures[:3]):
        print(f"--- Example {i+1} ---")
        print(f"Question: {f['question']}")
        print(f"Correct answer: {f['answer']}")
        print(f"Overlap score: {f['overlap']:.2f}")
        print(f"Correct context (first 150 chars): {f['correct_context'][:150]}")
        print(f"What TF-IDF retrieved instead (first 150 chars): {f['wrongly_retrieved'][:150]}")
        print()