"""
Script: Run failure analysis on agent predictions.
"""
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import GOLDEN_DIR, RESULTS_DIR
from src.failure_analysis import analyze_failures, save_failure_analysis


def main():
    print("=" * 60)
    print("STEP 11: FAILURE ANALYSIS")
    print("=" * 60)

    # Load golden set
    golden_path = GOLDEN_DIR / "golden_eval_set.json"
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    # Load predictions
    pred_path = RESULTS_DIR / "predictions_agent.json"
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    print(f"  Analyzing {len(predictions)} predictions against golden set")

    # Run failure analysis
    analysis = analyze_failures(predictions, golden_set)

    # Display results
    summary = analysis['summary']
    print(f"\n  Summary:")
    print(f"    Total examples: {summary['total_examples']}")
    print(f"    Total errors: {summary['total_errors']}")
    print(f"    Error rate: {summary['error_rate']:.4f}")
    print(f"    Intent errors: {summary['intent_errors']}")
    print(f"    Escalation errors: {summary['escalation_errors']}")
    print(f"    Over-escalation: {summary['over_escalation_count']}")
    print(f"    Under-escalation: {summary['under_escalation_count']}")

    print(f"\n{'═' * 60}")
    print("TOP 5 FAILURE MODES")
    print(f"{'═' * 60}")

    for i, mode in enumerate(analysis['failure_modes'], 1):
        print(f"\n  Failure Mode #{i}: {mode['name']}")
        print(f"  Count: {mode['frequency']}")
        print(f"  Type: {mode['type']}")

        if 'example' in mode:
            ex = mode['example']
            print(f"  Example:")
            msg = ex.get('customer_message', '')
            print(f"    Message: \"{msg[:120]}{'...' if len(msg) > 120 else ''}\"")
            if 'expected_intent' in ex:
                print(f"    Expected: {ex['expected_intent']}")
                print(f"    Predicted: {ex['predicted_intent']}")
            if 'expected_action' in ex:
                print(f"    Expected: {ex['expected_action']}")
                print(f"    Predicted: {ex['predicted_action']}")

        print(f"  Hypothesis: {mode['hypothesis'][:150]}...")
        print(f"  Improvement: {mode['improvement'][:150]}...")

    # Save
    path = save_failure_analysis(analysis)
    print(f"\n  Saved failure analysis to: {path}")

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
