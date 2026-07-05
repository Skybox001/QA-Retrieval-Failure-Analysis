from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EmbeddingRetriever:
    """
    A retriever that ranks documents by semantic similarity using
    sentence embeddings, instead of raw word overlap. This should handle
    cases where paragraphs share the same vocabulary/proper nouns but
    differ in actual meaning/content.
    """

    def __init__(self, corpus):
        self.corpus = corpus
        # A small, fast, well-regarded general-purpose embedding model -
        # good default choice for CPU-only, no-training-needed use cases
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_vectors = self.model.encode(corpus)

    def retrieve(self, query, k=5):
        query_vector = self.model.encode([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        return top_k_indices.tolist()