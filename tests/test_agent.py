"""
Unit tests for Taxonomy, Classifier, Retriever, Escalation Engine, and Support Agent.
"""
import unittest
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.taxonomy import INTENT_TAXONOMY, INTENT_NAMES, validate_intent
from src.classifier import classify_intent_llm, classify_intent_mock
from src.retriever import HistoricalRetriever
from src.escalation import decide_escalation
from src.agent import SupportAgent


class TestTaxonomy(unittest.TestCase):

    def test_intent_names(self):
        self.assertIn("Account / Login", INTENT_NAMES)
        self.assertIn("Billing / Subscription", INTENT_NAMES)
        self.assertIn("Other", INTENT_NAMES)
        self.assertEqual(len(INTENT_NAMES), 9)

    def test_validate_intent(self):
        self.assertTrue(validate_intent("Account / Login"))
        self.assertFalse(validate_intent("Invalid Intent Category"))


class TestClassifier(unittest.TestCase):

    def test_mock_classifier(self):
        res = classify_intent_mock("I cannot login to my account")
        self.assertIn('intent', res)
        self.assertIn('confidence', res)
        self.assertGreaterEqual(res['confidence'], 0.0)
        self.assertLessEqual(res['confidence'], 1.0)


class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.reference_pairs = [
            {
                'customer_tweet_id': '1',
                'customer_message': 'How do I reset my password?',
                'brand_response': 'You can reset your password at spotify.com/reset',
                'created_at': '2017-10-01',
            },
            {
                'customer_tweet_id': '2',
                'customer_message': 'I was double charged on my premium account',
                'brand_response': 'Please check your billing history at spotify.com/account',
                'created_at': '2017-10-02',
            },
            {
                'customer_tweet_id': '3',
                'customer_message': 'The music keeps pausing every few seconds',
                'brand_response': 'Try clearing your app cache in settings',
                'created_at': '2017-10-03',
            },
        ]

    def test_retrieval(self):
        retriever = HistoricalRetriever()
        retriever.fit(self.reference_pairs)
        results = retriever.retrieve("forgot my password cannot login", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn('similarity', results[0])
        self.assertIn('brand_response', results[0])


class TestEscalation(unittest.TestCase):

    def test_account_takeover_escalation(self):
        res = decide_escalation(
            intent="Account / Login",
            confidence=0.9,
            retrieved_examples=[{'similarity': 0.8}],
            customer_message="Someone hacked my account and changed the email",
        )
        self.assertEqual(res['action'], 'ESCALATE')
        self.assertTrue(any('Sensitive content detected' in f for f in res['escalation_factors']))

    def test_low_confidence_escalation(self):
        res = decide_escalation(
            intent="Other",
            confidence=0.2,
            retrieved_examples=[],
            customer_message="something weird is happening",
        )
        self.assertEqual(res['action'], 'ESCALATE')
        self.assertTrue(any('Low classification confidence' in f for f in res['escalation_factors']))

    def test_auto_handle(self):
        res = decide_escalation(
            intent="Account / Login",
            confidence=0.85,
            retrieved_examples=[{'similarity': 0.85}],
            customer_message="How do I change my password?",
        )
        self.assertEqual(res['action'], 'AUTO_HANDLED')


class TestAgent(unittest.TestCase):

    def test_full_pipeline(self):
        reference_pairs = [
            {
                'customer_tweet_id': '1',
                'customer_message': 'How do I reset my password?',
                'brand_response': 'Head to spotify.com/reset to get a password reset link!',
                'created_at': '2017-10-01',
            }
        ]
        retriever = HistoricalRetriever()
        retriever.fit(reference_pairs)

        agent = SupportAgent(retriever=retriever, use_mock=True)
        response = agent.process("I forgot my password")

        self.assertIn('predicted_intent', response)
        self.assertIn('generated_reply', response)
        self.assertIn('action', response)
        self.assertIn('retrieved_examples', response)
        self.assertIn(response['action'], ['AUTO_HANDLED', 'ESCALATE'])


if __name__ == '__main__':
    unittest.main()
