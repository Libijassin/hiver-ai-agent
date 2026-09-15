"""
Trivial Baseline: Majority-class classifier.
Always predicts the most common intent. Always auto-handles.
"""
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TrivialBaseline:
    """Majority-class classifier: predicts the most frequent intent for everything."""

    def __init__(self):
        self.majority_intent = None
        self.intent_distribution = {}

    def fit(self, golden_set):
        """Learn the majority intent from golden set labels."""
        intents = [ex['intent'] for ex in golden_set]
        counts = Counter(intents)
        self.intent_distribution = dict(counts)
        self.majority_intent = counts.most_common(1)[0][0]
        return self

    def predict(self, customer_message):
        """Always predict the majority intent."""
        return {
            'predicted_intent': self.majority_intent,
            'confidence': 1.0,
            'generated_reply': f"Thank you for reaching out! Please DM us your details so we can assist you.",
            'action': 'AUTO_HANDLED',
            'escalation_reason': '',
            'escalation_factors': [],
            'retrieved_examples': [],
            'mode': 'trivial_baseline',
        }

    def predict_batch(self, golden_set):
        """Predict for all examples in the golden set."""
        return [self.predict(ex['customer_message']) for ex in golden_set]


if __name__ == "__main__":
    # Quick test
    golden = [
        {'customer_message': 'test', 'intent': 'Account / Login'},
        {'customer_message': 'test2', 'intent': 'Account / Login'},
        {'customer_message': 'test3', 'intent': 'Billing / Subscription'},
    ]
    baseline = TrivialBaseline()
    baseline.fit(golden)
    print(f"Majority intent: {baseline.majority_intent}")
    print(f"Distribution: {baseline.intent_distribution}")
    result = baseline.predict("any message")
    print(f"Prediction: {result['predicted_intent']}")
