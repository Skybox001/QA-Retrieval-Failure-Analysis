from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TfidfRetriever:
    """
    A retriever that ranks documents by lexical (word-overlap) similarity
    to a query, using TF-IDF weighting + cosine similarity.
    """

    def __init__(self, corpus):
        self.corpus = corpus
        # stop_words='english' removes common words (the, is, at...) that
        # carry no distinguishing signal for retrieval
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query, k=5):
        """
        Returns the indices of the top-k most similar contexts to the query,
        ordered from most to least similar.
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        return top_k_indices.tolist()


if __name__ == "__main__":
    from load_data import load_squad_sample
    from build_corpus import build_corpus

    dataset = load_squad_sample()
    corpus, qa_pairs = build_corpus(dataset)

    retriever = TfidfRetriever(corpus)

    example = qa_pairs[0]
    top_k = retriever.retrieve(example["question"], k=5)
    print("Question:", example["question"])
    print("Correct context id:", example["correct_context_id"])
    print("Retrieved top-5 ids:", top_k)
    print("Correct answer in top-5?", example["correct_context_id"] in top_k)