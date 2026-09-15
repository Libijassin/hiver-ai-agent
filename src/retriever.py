"""
Historical conversation retriever using TF-IDF cosine similarity.
"""
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RETRIEVAL_TOP_K, RETRIEVAL_MIN_SIMILARITY


class HistoricalRetriever:
    """
    TF-IDF-based retriever over historical SpotifyCares conversations.
    Given a customer message, returns the most similar historical exchanges.
    """

    def __init__(self, reference_pairs=None):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
        )
        self.reference_pairs = []
        self.tfidf_matrix = None

        if reference_pairs is not None:
            self.fit(reference_pairs)

    def fit(self, reference_pairs):
        """
        Build the TF-IDF index from reference conversation pairs.
        """
        self.reference_pairs = reference_pairs
        customer_messages = [p['customer_message'] for p in reference_pairs]

        if len(customer_messages) == 0:
            return self

        self.tfidf_matrix = self.vectorizer.fit_transform(customer_messages)
        return self

    def retrieve(self, query, top_k=None, min_similarity=None):
        """
        Retrieve the top-k most similar historical conversations.

        Returns list of dicts:
        {
            'customer_message': str,
            'brand_response': str,
            'similarity': float,
            'customer_tweet_id': str,
            'brand_tweet_id': str,
        }
        """
        top_k = top_k or RETRIEVAL_TOP_K
        min_similarity = min_similarity or RETRIEVAL_MIN_SIMILARITY

        if self.tfidf_matrix is None or len(self.reference_pairs) == 0:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < min_similarity:
                continue
            pair = self.reference_pairs[idx]
            results.append({
                'customer_message': pair['customer_message'],
                'brand_response': pair['brand_response'],
                'similarity': sim,
                'customer_tweet_id': pair.get('customer_tweet_id', ''),
                'brand_tweet_id': pair.get('brand_tweet_id', ''),
            })

        return results

    def get_max_similarity(self, query):
        """Get the maximum similarity score for a query."""
        if self.tfidf_matrix is None:
            return 0.0
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return float(similarities.max()) if len(similarities) > 0 else 0.0

    def save(self, path):
        """Save the retriever to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'retriever.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'reference_pairs': self.reference_pairs,
                'tfidf_matrix': self.tfidf_matrix,
            }, f)

    @classmethod
    def load(cls, path):
        """Load a saved retriever from disk."""
        path = Path(path)
        with open(path / 'retriever.pkl', 'rb') as f:
            data = pickle.load(f)
        retriever = cls()
        retriever.vectorizer = data['vectorizer']
        retriever.reference_pairs = data['reference_pairs']
        retriever.tfidf_matrix = data['tfidf_matrix']
        return retriever


def load_reference_pairs(path=None):
    """Load reference pairs from processed data directory."""
    path = path or (PROCESSED_DIR / "reference_pairs.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    pairs = load_reference_pairs()
    print(f"Loaded {len(pairs)} reference pairs")

    retriever = HistoricalRetriever(pairs)
    print("Built TF-IDF index")

    # Test retrieval
    test_queries = [
        "I can't log into my account",
        "I was charged twice",
        "My playlist disappeared",
    ]
    for query in test_queries:
        results = retriever.retrieve(query)
        print(f"\nQuery: {query}")
        for r in results[:2]:
            print(f"  [{r['similarity']:.3f}] Customer: {r['customer_message'][:60]}...")
            print(f"           Brand: {r['brand_response'][:60]}...")
