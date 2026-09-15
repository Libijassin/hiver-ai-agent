"""
Unit tests for data preprocessing and cleaning.
"""
import unittest
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import clean_text, is_usable_pair, preprocess_pairs, split_data


class TestPreprocessing(unittest.TestCase):

    def test_clean_text_strips_handles_and_urls(self):
        text = "Hey @SpotifyCares my app is broken! Check https://example.com/help"
        cleaned = clean_text(text)
        self.assertNotIn("@SpotifyCares", cleaned)
        self.assertNotIn("https://example.com", cleaned)
        self.assertIn("my app is broken", cleaned)

    def test_clean_text_normalizes_whitespace(self):
        text = "  too    many   spaces   "
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "too many spaces")

    def test_is_usable_pair(self):
        valid_pair = {
            'customer_message': 'How do I cancel my account?',
            'brand_response': 'We can help with that! DM us your email.',
        }
        invalid_pair_short = {
            'customer_message': 'hi',
            'brand_response': 'hello',
        }
        self.assertTrue(is_usable_pair(valid_pair))
        self.assertFalse(is_usable_pair(invalid_pair_short))

    def test_split_data_no_leakage(self):
        pairs = [
            {'customer_tweet_id': f'id_{i}', 'customer_message': f'msg {i}', 'brand_response': f'resp {i}'}
            for i in range(100)
        ]
        ref, eval_set = split_data(pairs, eval_ratio=0.2, seed=42)
        self.assertEqual(len(ref) + len(eval_set), 100)
        self.assertGreater(len(ref), 0)
        self.assertGreater(len(eval_set), 0)

        ref_ids = set(p['customer_tweet_id'] for p in ref)
        eval_ids = set(p['customer_tweet_id'] for p in eval_set)
        self.assertEqual(len(ref_ids & eval_ids), 0)


if __name__ == '__main__':
    unittest.main()
