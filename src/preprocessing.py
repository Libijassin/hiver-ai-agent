"""
Preprocessing: cleans text, handles malformed records, creates train/eval splits.
"""
import re
import json
import hashlib
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, EVAL_SPLIT_RATIO, RANDOM_SEED, BRAND_NAME
)


def clean_text(text):
    """
    Clean a single tweet text:
    - Remove @mentions (anonymized author IDs like @115712)
    - Normalize whitespace
    - Remove URLs
    - Strip leading/trailing whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove @mentions (anonymized as @author_id numbers or brand names)
    text = re.sub(r'@\w+', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_usable_pair(pair):
    """
    Check if a customer→brand pair is usable for our purposes.
    Filters out:
    - Empty customer messages or brand responses
    - Very short messages (< 5 chars after cleaning)
    - Messages that are just links or mentions
    - Brand responses that are just "DM us"
    """
    cust = clean_text(pair.get('customer_message', ''))
    brand = clean_text(pair.get('brand_response', ''))

    if len(cust) < 5 or len(brand) < 5:
        return False

    # Skip if customer message is just a number or gibberish
    if re.match(r'^[\d\s\.\,]+$', cust):
        return False

    return True


def preprocess_pairs(pairs):
    """
    Clean and filter conversation pairs.
    Returns list of cleaned, usable pairs.
    """
    cleaned = []
    for pair in pairs:
        if not is_usable_pair(pair):
            continue

        cleaned_pair = {
            'customer_tweet_id': pair['customer_tweet_id'],
            'customer_message': clean_text(pair['customer_message']),
            'customer_message_raw': pair['customer_message'],
            'customer_author_id': pair['customer_author_id'],
            'brand_tweet_id': pair['brand_tweet_id'],
            'brand_response': clean_text(pair['brand_response']),
            'brand_response_raw': pair['brand_response'],
            'created_at': pair.get('created_at', ''),
        }
        cleaned.append(cleaned_pair)

    return cleaned


def split_data(pairs, eval_ratio=None, seed=None):
    """
    Split pairs into reference (for retrieval) and evaluation sets.

    Uses deterministic hashing on customer_tweet_id for reproducibility.
    The evaluation set is used to draw golden examples from.
    The reference set is used as the retrieval corpus.

    Returns: (reference_pairs, eval_pairs)
    """
    eval_ratio = eval_ratio or EVAL_SPLIT_RATIO
    seed = seed or RANDOM_SEED

    # Deterministic split using hash of tweet_id
    reference = []
    evaluation = []

    for pair in pairs:
        # Hash the customer tweet ID for deterministic assignment
        h = hashlib.md5(
            f"{pair['customer_tweet_id']}_{seed}".encode()
        ).hexdigest()
        hash_val = int(h[:8], 16) / (16**8)

        if hash_val < eval_ratio:
            evaluation.append(pair)
        else:
            reference.append(pair)

    return reference, evaluation


def save_processed_data(reference_pairs, eval_pairs, output_dir=None):
    """Save processed data splits to disk."""
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = output_dir / "reference_pairs.json"
    eval_path = output_dir / "eval_pairs.json"
    meta_path = output_dir / "split_metadata.json"

    with open(ref_path, 'w', encoding='utf-8') as f:
        json.dump(reference_pairs, f, indent=2, ensure_ascii=False)

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_pairs, f, indent=2, ensure_ascii=False)

    metadata = {
        'brand': BRAND_NAME,
        'total_pairs': len(reference_pairs) + len(eval_pairs),
        'reference_pairs': len(reference_pairs),
        'eval_pairs': len(eval_pairs),
        'eval_ratio': EVAL_SPLIT_RATIO,
        'random_seed': RANDOM_SEED,
        'split_method': 'deterministic_hash_on_customer_tweet_id',
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return ref_path, eval_path


if __name__ == "__main__":
    from data_loader import load_raw_dataset, build_conversation_pairs

    print("Loading and preprocessing...")
    df = load_raw_dataset()
    pairs = build_conversation_pairs(df)
    print(f"Raw pairs: {len(pairs)}")

    cleaned = preprocess_pairs(pairs)
    print(f"After cleaning: {len(cleaned)}")

    reference, evaluation = split_data(cleaned)
    print(f"Reference set: {len(reference)}")
    print(f"Evaluation set: {len(evaluation)}")

    save_processed_data(reference, evaluation)
    print("Saved to processed directory.")
