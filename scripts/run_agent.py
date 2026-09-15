"""
Script: Run the AI agent on the golden evaluation set.
"""
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import GOLDEN_DIR, PROCESSED_DIR, RESULTS_DIR, USE_MOCK
from src.retriever import HistoricalRetriever, load_reference_pairs
from src.agent import SupportAgent


def main():
    print("=" * 60)
    print("STEP 3-5: RUN AI AGENT")
    print("=" * 60)
    print(f"  Mode: {'MOCK (no API key)' if USE_MOCK else 'LLM (Gemini)'}")

    # Load reference pairs for retriever
    print(f"\n[1/4] Loading reference pairs...")
    reference_pairs = load_reference_pairs()
    print(f"  Loaded {len(reference_pairs)} reference pairs")

    # Build retriever
    print(f"\n[2/4] Building TF-IDF retriever...")
    retriever = HistoricalRetriever(reference_pairs)
    print(f"  Index built with {len(reference_pairs)} documents")

    # Save retriever for later use
    retriever.save(PROCESSED_DIR / "retriever")
    print(f"  Retriever saved to {PROCESSED_DIR / 'retriever'}")

    # Load golden set
    print(f"\n[3/4] Loading golden evaluation set...")
    golden_path = GOLDEN_DIR / "golden_eval_set.json"
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)
    print(f"  Loaded {len(golden_set)} golden examples")

    # Run agent
    print(f"\n[4/4] Running agent on golden set...")
    agent = SupportAgent(retriever)

    predictions = []
    for i, example in enumerate(golden_set):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(golden_set)}...")

        result = agent.process(example['customer_message'])
        result['golden_id'] = example['id']
        result['customer_message'] = example['customer_message']

        # Trim retrieved examples for storage (keep just essentials)
        trimmed_retrieved = []
        for ex in result.get('retrieved_examples', []):
            trimmed_retrieved.append({
                'customer_message': ex['customer_message'][:200],
                'brand_response': ex['brand_response'][:200],
                'similarity': ex['similarity'],
            })
        result['retrieved_examples'] = trimmed_retrieved

        predictions.append(result)

    # Save predictions
    pred_path = RESULTS_DIR / "predictions_agent.json"
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(predictions)} predictions to: {pred_path}")

    # Quick summary
    from collections import Counter
    intent_counts = Counter(p['predicted_intent'] for p in predictions)
    action_counts = Counter(p['action'] for p in predictions)

    print(f"\n  Intent distribution:")
    for intent, count in intent_counts.most_common():
        print(f"    {intent}: {count}")

    print(f"\n  Action distribution:")
    for action, count in action_counts.most_common():
        print(f"    {action}: {count}")

    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"\n  Average confidence: {avg_confidence:.3f}")

    print("\n" + "=" * 60)
    print("AGENT INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
