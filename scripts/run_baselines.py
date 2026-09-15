"""
Script: Run both baselines on the golden evaluation set.
"""
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import GOLDEN_DIR, RESULTS_DIR
from baselines.trivial_baseline import TrivialBaseline
from baselines.simple_baseline import SimpleBaseline
from src.evaluation import compute_intent_metrics, compute_escalation_metrics, save_evaluation_report


def main():
    print("=" * 60)
    print("STEP 10: RUN BASELINES")
    print("=" * 60)

    # Load golden set
    golden_path = GOLDEN_DIR / "golden_eval_set.json"
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)
    print(f"\n  Loaded {len(golden_set)} golden examples")

    golden_intents = [ex['intent'] for ex in golden_set]
    golden_actions = [ex['expected_action'] for ex in golden_set]

    # ── BASELINE 1: Trivial (Majority Class) ────────────────────────────
    print(f"\n{'─' * 40}")
    print("BASELINE 1: Trivial (Majority Class)")
    print(f"{'─' * 40}")

    trivial = TrivialBaseline()
    trivial.fit(golden_set)
    print(f"  Majority intent: {trivial.majority_intent}")

    trivial_preds = trivial.predict_batch(golden_set)
    trivial_intents = [p['predicted_intent'] for p in trivial_preds]
    trivial_actions = [p['action'] for p in trivial_preds]

    trivial_intent_metrics = compute_intent_metrics(trivial_intents, golden_intents)
    trivial_esc_metrics = compute_escalation_metrics(trivial_actions, golden_actions)

    print(f"  Accuracy: {trivial_intent_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {trivial_intent_metrics['macro_f1']:.4f}")
    print(f"  Escalation Accuracy: {trivial_esc_metrics['accuracy']:.4f}")

    save_evaluation_report(
        trivial_intent_metrics, trivial_esc_metrics, {},
        trivial_preds, model_name="trivial_baseline"
    )

    # ── BASELINE 2: Simple (TF-IDF + LogReg) ────────────────────────────
    print(f"\n{'─' * 40}")
    print("BASELINE 2: Simple (TF-IDF + Logistic Regression)")
    print(f"{'─' * 40}")

    simple = SimpleBaseline()
    print(f"  Evaluating via 5-fold Stratified Cross-Validation on {len(golden_set)} examples...")
    simple_preds = simple.predict_cv(golden_set, n_splits=5)
    simple_intents = [p['predicted_intent'] for p in simple_preds]
    simple_actions = [p['action'] for p in simple_preds]

    simple_intent_metrics = compute_intent_metrics(simple_intents, golden_intents)
    simple_esc_metrics = compute_escalation_metrics(simple_actions, golden_actions)

    print(f"  Accuracy: {simple_intent_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {simple_intent_metrics['macro_f1']:.4f}")
    print(f"  Escalation Accuracy: {simple_esc_metrics['accuracy']:.4f}")

    save_evaluation_report(
        simple_intent_metrics, simple_esc_metrics, {},
        simple_preds, model_name="simple_baseline"
    )

    # ── Comparison Table ─────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("BASELINE COMPARISON")
    print(f"{'═' * 60}")
    print(f"{'Metric':<30} {'Trivial':>12} {'Simple':>12}")
    print(f"{'─' * 54}")
    print(f"{'Accuracy':<30} {trivial_intent_metrics['accuracy']:>12.4f} {simple_intent_metrics['accuracy']:>12.4f}")
    print(f"{'Macro F1':<30} {trivial_intent_metrics['macro_f1']:>12.4f} {simple_intent_metrics['macro_f1']:>12.4f}")
    print(f"{'Weighted F1':<30} {trivial_intent_metrics['weighted_f1']:>12.4f} {simple_intent_metrics['weighted_f1']:>12.4f}")
    print(f"{'Escalation Accuracy':<30} {trivial_esc_metrics['accuracy']:>12.4f} {simple_esc_metrics['accuracy']:>12.4f}")
    print(f"{'Escalation F1':<30} {trivial_esc_metrics['escalation_f1']:>12.4f} {simple_esc_metrics['escalation_f1']:>12.4f}")

    # Save comparison
    comparison = {
        'trivial_baseline': {
            'accuracy': trivial_intent_metrics['accuracy'],
            'macro_f1': trivial_intent_metrics['macro_f1'],
            'weighted_f1': trivial_intent_metrics['weighted_f1'],
            'escalation_accuracy': trivial_esc_metrics['accuracy'],
            'escalation_f1': trivial_esc_metrics['escalation_f1'],
        },
        'simple_baseline': {
            'accuracy': simple_intent_metrics['accuracy'],
            'macro_f1': simple_intent_metrics['macro_f1'],
            'weighted_f1': simple_intent_metrics['weighted_f1'],
            'escalation_accuracy': simple_esc_metrics['accuracy'],
            'escalation_f1': simple_esc_metrics['escalation_f1'],
        },
    }
    comp_path = RESULTS_DIR / "baseline_comparison.json"
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved comparison to: {comp_path}")

    print("\n" + "=" * 60)
    print("BASELINES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
