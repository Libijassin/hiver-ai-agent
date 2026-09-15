"""
Script: Run automated evaluation on agent predictions.
"""
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import GOLDEN_DIR, RESULTS_DIR
from src.evaluation import (
    compute_intent_metrics,
    compute_escalation_metrics,
    save_evaluation_report,
)


def main():
    print("=" * 60)
    print("STEP 7: AUTOMATED EVALUATION")
    print("=" * 60)

    # Load golden set
    golden_path = GOLDEN_DIR / "golden_eval_set.json"
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    # Load predictions
    pred_path = RESULTS_DIR / "predictions_agent.json"
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    print(f"  Golden set: {len(golden_set)} examples")
    print(f"  Predictions: {len(predictions)} examples")
    assert len(golden_set) == len(predictions), "Size mismatch!"

    golden_intents = [ex['intent'] for ex in golden_set]
    golden_actions = [ex['expected_action'] for ex in golden_set]
    pred_intents = [p['predicted_intent'] for p in predictions]
    pred_actions = [p['action'] for p in predictions]

    # ── Intent Classification Metrics ────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("INTENT CLASSIFICATION METRICS")
    print(f"{'─' * 40}")

    intent_metrics = compute_intent_metrics(pred_intents, golden_intents)

    print(f"  Accuracy: {intent_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {intent_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {intent_metrics['weighted_f1']:.4f}")

    print(f"\n  Per-Intent Breakdown:")
    print(f"  {'Intent':<30} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'─' * 56}")
    for intent, m in sorted(intent_metrics['per_intent'].items()):
        if m['support'] > 0:
            print(f"  {intent:<30} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8}")

    # ── Escalation Metrics ───────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("ESCALATION METRICS")
    print(f"{'─' * 40}")

    esc_metrics = compute_escalation_metrics(pred_actions, golden_actions)

    print(f"  Accuracy: {esc_metrics['accuracy']:.4f}")
    print(f"  Escalation Precision: {esc_metrics['escalation_precision']:.4f}")
    print(f"  Escalation Recall: {esc_metrics['escalation_recall']:.4f}")
    print(f"  Escalation F1: {esc_metrics['escalation_f1']:.4f}")
    print(f"  Predicted escalation rate: {esc_metrics['escalation_rate']:.4f}")
    print(f"  Golden escalation rate: {esc_metrics['golden_escalation_rate']:.4f}")
    print(f"  Confusion: TP={esc_metrics['tp']} FP={esc_metrics['fp']} FN={esc_metrics['fn']} TN={esc_metrics['tn']}")

    # Save full evaluation
    save_evaluation_report(
        intent_metrics, esc_metrics, {},
        predictions, model_name="agent"
    )

    # ── Load baselines for comparison ────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("COMPARISON: Agent vs Baselines")
    print(f"{'═' * 60}")

    comp_path = RESULTS_DIR / "baseline_comparison.json"
    if comp_path.exists():
        with open(comp_path, 'r') as f:
            baselines = json.load(f)

        trivial = baselines.get('trivial_baseline', {})
        simple = baselines.get('simple_baseline', {})

        print(f"{'Metric':<30} {'Trivial':>10} {'Simple':>10} {'Agent':>10}")
        print(f"{'─' * 60}")
        print(f"{'Accuracy':<30} {trivial.get('accuracy', 0):>10.4f} {simple.get('accuracy', 0):>10.4f} {intent_metrics['accuracy']:>10.4f}")
        print(f"{'Macro F1':<30} {trivial.get('macro_f1', 0):>10.4f} {simple.get('macro_f1', 0):>10.4f} {intent_metrics['macro_f1']:>10.4f}")
        print(f"{'Weighted F1':<30} {trivial.get('weighted_f1', 0):>10.4f} {simple.get('weighted_f1', 0):>10.4f} {intent_metrics['weighted_f1']:>10.4f}")
        print(f"{'Escalation Accuracy':<30} {trivial.get('escalation_accuracy', 0):>10.4f} {simple.get('escalation_accuracy', 0):>10.4f} {esc_metrics['accuracy']:>10.4f}")
        print(f"{'Escalation F1':<30} {trivial.get('escalation_f1', 0):>10.4f} {simple.get('escalation_f1', 0):>10.4f} {esc_metrics['escalation_f1']:>10.4f}")

        # Update comparison
        baselines['agent'] = {
            'accuracy': intent_metrics['accuracy'],
            'macro_f1': intent_metrics['macro_f1'],
            'weighted_f1': intent_metrics['weighted_f1'],
            'escalation_accuracy': esc_metrics['accuracy'],
            'escalation_f1': esc_metrics['escalation_f1'],
        }
        with open(comp_path, 'w') as f:
            json.dump(baselines, f, indent=2)
    else:
        print("  (Baseline results not found. Run run_baselines.py first.)")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
