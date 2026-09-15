"""
Simple Baseline: TF-IDF + Logistic Regression classifier.
Trained on the golden set labels (or a portion thereof) via cross-validation.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleBaseline:
    """TF-IDF + Logistic Regression intent classifier."""

    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
                class_weight='balanced',
            )),
        ])
        self.is_fitted = False

    def fit(self, golden_set):
        """
        Train on golden set examples.
        Each example must have 'customer_message' and 'intent'.
        """
        messages = [ex['customer_message'] for ex in golden_set]
        labels = [ex['intent'] for ex in golden_set]

        self.pipeline.fit(messages, labels)
        self.is_fitted = True
        return self

    def predict(self, customer_message):
        """Predict intent for a single message."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        intent = self.pipeline.predict([customer_message])[0]

        # Get confidence from max probability
        probas = self.pipeline.predict_proba([customer_message])[0]
        confidence = float(max(probas))

        return {
            'predicted_intent': intent,
            'confidence': confidence,
            'generated_reply': f"Thank you for reaching out about {intent.lower()}. Please DM us your details so we can help.",
            'action': 'AUTO_HANDLED' if confidence >= 0.5 else 'ESCALATE',
            'escalation_reason': f'Low confidence ({confidence:.2f})' if confidence < 0.5 else '',
            'escalation_factors': [f'Low confidence ({confidence:.2f})'] if confidence < 0.5 else [],
            'retrieved_examples': [],
            'mode': 'simple_baseline',
        }

    def predict_batch(self, golden_set):
        """Predict for all examples in the golden set."""
        return [self.predict(ex['customer_message']) for ex in golden_set]

    def predict_cv(self, golden_set, n_splits=5):
        """
        Generate out-of-fold predictions using Stratified K-Fold cross-validation.
        This provides a realistic baseline accuracy without train-test leakage.
        """
        from sklearn.model_selection import StratifiedKFold

        messages = np.array([ex['customer_message'] for ex in golden_set])
        labels = np.array([ex['intent'] for ex in golden_set])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_predictions = [None] * len(golden_set)

        for train_idx, val_idx in skf.split(messages, labels):
            fold_clf = SimpleBaseline()
            fold_clf.pipeline.fit(messages[train_idx], labels[train_idx])
            fold_clf.is_fitted = True

            for idx in val_idx:
                oof_predictions[idx] = fold_clf.predict(messages[idx])

        return oof_predictions


if __name__ == "__main__":
    # Quick test
    golden = [
        {'customer_message': "I can't log in to my account", 'intent': 'Account / Login'},
        {'customer_message': 'Password reset not working', 'intent': 'Account / Login'},
        {'customer_message': 'I was charged twice', 'intent': 'Billing / Subscription'},
        {'customer_message': 'Double charge on my card', 'intent': 'Billing / Subscription'},
        {'customer_message': 'Songs keep buffering', 'intent': 'Playback / Streaming'},
        {'customer_message': 'Music stops playing', 'intent': 'Playback / Streaming'},
    ]
    baseline = SimpleBaseline()
    baseline.fit(golden)
    result = baseline.predict("my account is locked")
    print(f"Prediction: {result['predicted_intent']} ({result['confidence']:.2f})")
