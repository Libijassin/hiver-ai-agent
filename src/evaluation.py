"""
Evaluation harness: computes metrics for intent classification, escalation, and reply quality.
"""
import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR
from src.taxonomy import INTENT_NAMES


def compute_intent_metrics(predictions, golden_labels):
    """
    Compute intent classification metrics.

    Args:
        predictions: list of predicted intent strings
        golden_labels: list of golden intent strings

    Returns dict with accuracy, per-intent P/R/F1, macro F1, confusion matrix.
    """
    assert len(predictions) == len(golden_labels), "Length mismatch"

    n = len(predictions)
    correct = sum(1 for p, g in zip(predictions, golden_labels) if p == g)
    accuracy = correct / n if n > 0 else 0.0

    # Per-intent metrics
    all_intents = sorted(set(golden_labels) | set(predictions))
    per_intent = {}
    for intent in all_intents:
        tp = sum(1 for p, g in zip(predictions, golden_labels) if p == intent and g == intent)
        fp = sum(1 for p, g in zip(predictions, golden_labels) if p == intent and g != intent)
        fn = sum(1 for p, g in zip(predictions, golden_labels) if p != intent and g == intent)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for g in golden_labels if g == intent)

        per_intent[intent] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': support,
            'tp': tp, 'fp': fp, 'fn': fn,
        }

    # Macro F1
    f1_scores = [m['f1'] for m in per_intent.values() if m['support'] > 0]
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0

    # Weighted F1
    weighted_f1_num = sum(m['f1'] * m['support'] for m in per_intent.values())
    weighted_f1_den = sum(m['support'] for m in per_intent.values())
    weighted_f1 = weighted_f1_num / weighted_f1_den if weighted_f1_den > 0 else 0.0

    # Confusion matrix
    confusion = {}
    for g_intent in all_intents:
        confusion[g_intent] = {}
        for p_intent in all_intents:
            confusion[g_intent][p_intent] = sum(
                1 for p, g in zip(predictions, golden_labels)
                if g == g_intent and p == p_intent
            )

    return {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'per_intent': per_intent,
        'confusion_matrix': confusion,
        'total': n,
        'correct': correct,
    }


def compute_escalation_metrics(predicted_actions, golden_actions):
    """
    Compute escalation decision metrics.

    Args:
        predicted_actions: list of 'AUTO_HANDLED' or 'ESCALATE'
        golden_actions: list of expected actions

    Returns dict with accuracy, precision/recall for ESCALATE, etc.
    """
    n = len(predicted_actions)
    correct = sum(1 for p, g in zip(predicted_actions, golden_actions) if p == g)
    accuracy = correct / n if n > 0 else 0.0

    # ESCALATE as positive class
    tp = sum(1 for p, g in zip(predicted_actions, golden_actions)
             if p == 'ESCALATE' and g == 'ESCALATE')
    fp = sum(1 for p, g in zip(predicted_actions, golden_actions)
             if p == 'ESCALATE' and g == 'AUTO_HANDLED')
    fn = sum(1 for p, g in zip(predicted_actions, golden_actions)
             if p == 'AUTO_HANDLED' and g == 'ESCALATE')
    tn = sum(1 for p, g in zip(predicted_actions, golden_actions)
             if p == 'AUTO_HANDLED' and g == 'AUTO_HANDLED')

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': round(accuracy, 4),
        'escalation_precision': round(precision, 4),
        'escalation_recall': round(recall, 4),
        'escalation_f1': round(f1, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'total': n,
        'escalation_rate': round((tp + fp) / n if n > 0 else 0, 4),
        'golden_escalation_rate': round((tp + fn) / n if n > 0 else 0, 4),
    }


def compute_reply_quality_metrics(judge_scores):
    """
    Aggregate LLM judge scores.

    Args:
        judge_scores: list of dicts with 'helpfulness', 'correctness', etc.

    Returns aggregated metrics.
    """
    if not judge_scores:
        return {}

    dimensions = ['helpfulness', 'correctness', 'grounding', 'relevance', 'overall']
    metrics = {}

    for dim in dimensions:
        scores = [s.get(dim, 0) for s in judge_scores if dim in s]
        if scores:
            metrics[dim] = {
                'mean': round(np.mean(scores), 2),
                'std': round(np.std(scores), 2),
                'min': min(scores),
                'max': max(scores),
                'median': round(np.median(scores), 2),
            }

    return metrics


def generate_confusion_csv(confusion_matrix, output_path):
    """Write confusion matrix as CSV."""
    intents = sorted(confusion_matrix.keys())

    lines = ["actual\\predicted," + ",".join(intents)]
    for actual in intents:
        row = [actual]
        for predicted in intents:
            row.append(str(confusion_matrix[actual].get(predicted, 0)))
        lines.append(",".join(row))

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def save_evaluation_report(
    intent_metrics,
    escalation_metrics,
    reply_metrics,
    predictions,
    output_dir=None,
    model_name="agent",
):
    """Save all evaluation results to disk."""
    output_dir = Path(output_dir) if output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main evaluation report
    report = {
        'model': model_name,
        'intent_classification': intent_metrics,
        'escalation': escalation_metrics,
        'reply_quality': reply_metrics,
    }

    report_path = output_dir / f"evaluation_{model_name}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Predictions
    pred_path = output_dir / f"predictions_{model_name}.json"
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # Confusion matrix CSV
    if intent_metrics and 'confusion_matrix' in intent_metrics:
        cm_path = output_dir / f"confusion_matrix_{model_name}.csv"
        generate_confusion_csv(intent_metrics['confusion_matrix'], cm_path)

    return report_path


if __name__ == "__main__":
    # Quick sanity test
    preds = ['Account / Login', 'Billing / Subscription', 'Other', 'Account / Login']
    golds = ['Account / Login', 'Billing / Subscription', 'Complaint', 'Other']

    metrics = compute_intent_metrics(preds, golds)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Macro F1: {metrics['macro_f1']}")

    esc_preds = ['AUTO_HANDLED', 'ESCALATE', 'AUTO_HANDLED', 'ESCALATE']
    esc_golds = ['AUTO_HANDLED', 'ESCALATE', 'ESCALATE', 'AUTO_HANDLED']

    esc_metrics = compute_escalation_metrics(esc_preds, esc_golds)
    print(f"Escalation accuracy: {esc_metrics['accuracy']}")
