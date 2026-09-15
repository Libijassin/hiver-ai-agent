"""
Script: Prepare data — load, filter, preprocess, and split the dataset.
"""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import BRAND_NAME, RAW_DATA_PATH, PROCESSED_DIR
from src.data_loader import load_raw_dataset, filter_brand_conversations, build_conversation_pairs, get_brand_stats
from src.preprocessing import preprocess_pairs, split_data, save_processed_data


def main():
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    # Load raw dataset
    print(f"\n[1/5] Loading raw dataset from {RAW_DATA_PATH}...")
    df = load_raw_dataset()
    print(f"  Total rows: {len(df):,}")

    # Show brand stats
    print(f"\n[2/5] Brand statistics:")
    stats = get_brand_stats(df)
    print(f"  Total unique brands: {len(stats)}")
    print(f"  Top 10 brands by outbound messages:")
    for brand, count in sorted(stats.items(), key=lambda x: -x[1])[:10]:
        marker = " <- SELECTED" if brand == BRAND_NAME else ""
        print(f"    {brand}: {count:,}{marker}")

    # Filter for selected brand
    print(f"\n[3/5] Filtering for {BRAND_NAME}...")
    brand_df = filter_brand_conversations(df)
    print(f"  Brand-related messages: {len(brand_df):,}")

    # Build conversation pairs
    print(f"\n[4/5] Building customer→brand pairs...")
    pairs = build_conversation_pairs(df)
    print(f"  Raw pairs: {len(pairs):,}")

    # Preprocess and clean
    cleaned = preprocess_pairs(pairs)
    print(f"  After cleaning: {len(cleaned):,}")
    print(f"  Removed: {len(pairs) - len(cleaned):,} unusable pairs")

    # Split data
    print(f"\n[5/5] Splitting data...")
    reference, evaluation = split_data(cleaned)
    print(f"  Reference set (for retrieval): {len(reference):,}")
    print(f"  Evaluation set (for golden sampling): {len(evaluation):,}")

    # Save
    ref_path, eval_path = save_processed_data(reference, evaluation)
    print(f"\n  Saved reference pairs to: {ref_path}")
    print(f"  Saved evaluation pairs to: {eval_path}")

    # Verify no leakage
    ref_ids = set(p['customer_tweet_id'] for p in reference)
    eval_ids = set(p['customer_tweet_id'] for p in evaluation)
    overlap = ref_ids & eval_ids
    print(f"\n  Data leakage check: {len(overlap)} overlapping IDs")
    assert len(overlap) == 0, "DATA LEAKAGE DETECTED!"
    print("  ✓ No data leakage detected")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
