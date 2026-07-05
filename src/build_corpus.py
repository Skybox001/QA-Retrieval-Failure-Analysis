from load_data import load_squad_sample

def build_corpus(dataset):
    """
    Turns raw SQuAD examples into:
    - a deduplicated list of unique contexts (our 'search corpus')
    - a list of (question, correct_context_index, answer_text) tuples

    We deduplicate contexts because SQuAD asks multiple questions per
    paragraph — without deduplication our search index would have
    duplicate documents, which distorts retrieval evaluation.
    """
    context_to_id = {}
    corpus = []
    qa_pairs = []

    for example in dataset:
        context = example["context"]
        question = example["question"]
        answer_text = example["answers"]["text"][0]

        if context not in context_to_id:
            context_to_id[context] = len(corpus)
            corpus.append(context)

        correct_id = context_to_id[context]
        qa_pairs.append({
            "question": question,
            "correct_context_id": correct_id,
            "answer": answer_text
        })

    return corpus, qa_pairs

if __name__ == "__main__":
    dataset = load_squad_sample()
    corpus, qa_pairs = build_corpus(dataset)
    print(f"Total questions: {len(qa_pairs)}")
    print(f"Unique contexts (corpus size): {len(corpus)}")
    print("\nExample QA pair:", qa_pairs[0])