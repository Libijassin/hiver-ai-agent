"""
Failure analysis: identifies top failure modes from evaluation results.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR


def analyze_failures(predictions, golden_set):
    """
    Analyze prediction failures against the golden evaluation set.

    Args:
        predictions: list of agent output dicts (with predicted_intent, action, etc.)
        golden_set: list of golden example dicts (with intent, expected_action, etc.)

    Returns list of failure mode dicts, each containing:
        - name, count, frequency, examples, expected, actual, hypothesis, improvement
    """
    assert len(predictions) == len(golden_set), "Length mismatch"

    # Collect all errors
    intent_errors = []
    escalation_errors = []
    all_errors = []

    for i, (pred, gold) in enumerate(zip(predictions, golden_set)):
        pred_intent = pred.get('predicted_intent', 'Other')
        gold_intent = gold.get('intent', 'Other')
        pred_action = pred.get('action', 'AUTO_HANDLED')
        gold_action = gold.get('expected_action', 'AUTO_HANDLED')

        has_intent_error = pred_intent != gold_intent
        has_escalation_error = pred_action != gold_action

        if has_intent_error or has_escalation_error:
            error = {
                'index': i,
                'customer_message': gold.get('customer_message', ''),
                'gold_intent': gold_intent,
                'pred_intent': pred_intent,
                'gold_action': gold_action,
                'pred_action': pred_action,
                'intent_error': has_intent_error,
                'escalation_error': has_escalation_error,
                'confidence': pred.get('confidence', 0),
                'notes': gold.get('notes', ''),
            }
            all_errors.append(error)
            if has_intent_error:
                intent_errors.append(error)
            if has_escalation_error:
                escalation_errors.append(error)

    # Cluster intent errors by confusion pair
    confusion_pairs = Counter()
    confusion_examples = defaultdict(list)
    for err in intent_errors:
        pair = (err['gold_intent'], err['pred_intent'])
        confusion_pairs[pair] += 1
        confusion_examples[pair].append(err)

    # Build failure modes from top confusion pairs
    failure_modes = []

    # Top intent confusion pairs
    for (gold, pred), count in confusion_pairs.most_common(5):
        examples = confusion_examples[(gold, pred)]
        best_example = examples[0]

        # Generate hypothesis
        hypothesis = _generate_hypothesis(gold, pred, examples)
        improvement = _suggest_improvement(gold, pred)

        failure_modes.append({
            'name': f"Confusing '{gold}' with '{pred}'",
            'type': 'intent_misclassification',
            'count': count,
            'frequency': f"{count}/{len(golden_set)} ({100*count/len(golden_set):.1f}%)",
            'example': {
                'customer_message': best_example['customer_message'],
                'expected_intent': gold,
                'predicted_intent': pred,
                'confidence': best_example['confidence'],
            },
            'hypothesis': hypothesis,
            'improvement': improvement,
        })

    # Escalation errors (over-escalation and under-escalation)
    over_escalation = [e for e in escalation_errors if e['pred_action'] == 'ESCALATE' and e['gold_action'] == 'AUTO_HANDLED']
    under_escalation = [e for e in escalation_errors if e['pred_action'] == 'AUTO_HANDLED' and e['gold_action'] == 'ESCALATE']

    if over_escalation and len(failure_modes) < 5:
        ex = over_escalation[0]
        failure_modes.append({
            'name': 'Over-escalation (false positives)',
            'type': 'escalation_error',
            'count': len(over_escalation),
            'frequency': f"{len(over_escalation)}/{len(golden_set)} ({100*len(over_escalation)/len(golden_set):.1f}%)",
            'example': {
                'customer_message': ex['customer_message'],
                'expected_action': 'AUTO_HANDLED',
                'predicted_action': 'ESCALATE',
                'intent': ex['pred_intent'],
            },
            'hypothesis': 'Conservative escalation thresholds cause too many false escalations, especially for straightforward queries that happen to contain sensitive keywords or have low retrieval similarity.',
            'improvement': 'Tune escalation thresholds on validation data; add keyword context analysis to reduce false triggers on innocuous uses of sensitive words.',
        })

    if under_escalation and len(failure_modes) < 5:
        ex = under_escalation[0]
        failure_modes.append({
            'name': 'Under-escalation (false negatives)',
            'type': 'escalation_error',
            'count': len(under_escalation),
            'frequency': f"{len(under_escalation)}/{len(golden_set)} ({100*len(under_escalation)/len(golden_set):.1f}%)",
            'example': {
                'customer_message': ex['customer_message'],
                'expected_action': 'ESCALATE',
                'predicted_action': 'AUTO_HANDLED',
                'intent': ex['pred_intent'],
            },
            'hypothesis': 'Some messages require escalation for reasons not captured by keyword matching or confidence thresholds (e.g., implicit urgency, multi-issue messages).',
            'improvement': 'Add contextual escalation signals beyond keywords: message sentiment analysis, repeated contact detection, multi-issue detection.',
        })

    # Pad to 5 if needed
    while len(failure_modes) < 5:
        if all_errors:
            err = all_errors[len(failure_modes)]
            failure_modes.append({
                'name': f"Miscellaneous error on '{err['gold_intent']}' messages",
                'type': 'miscellaneous',
                'count': 1,
                'frequency': f"1/{len(golden_set)} ({100/len(golden_set):.1f}%)",
                'example': {
                    'customer_message': err['customer_message'],
                    'expected_intent': err['gold_intent'],
                    'predicted_intent': err['pred_intent'],
                },
                'hypothesis': 'Ambiguous message that could reasonably belong to multiple categories.',
                'improvement': 'Refine intent boundaries or add multi-label support for ambiguous cases.',
            })
        else:
            break

    # Summary stats
    summary = {
        'total_examples': len(golden_set),
        'total_errors': len(all_errors),
        'intent_errors': len(intent_errors),
        'escalation_errors': len(escalation_errors),
        'error_rate': round(len(all_errors) / len(golden_set), 4) if golden_set else 0,
        'over_escalation_count': len(over_escalation),
        'under_escalation_count': len(under_escalation),
    }

    return {
        'failure_modes': failure_modes[:5],
        'summary': summary,
        'all_errors': all_errors,
    }


def _generate_hypothesis(gold_intent, pred_intent, examples):
    """Generate a hypothesis for why this confusion occurs."""
    hypotheses = {
        ('Account / Login', 'Billing / Subscription'): 'Customers mentioning "account" in billing contexts triggers Account / Login classification. The word "account" is shared between both intents.',
        ('Billing / Subscription', 'Account / Login'): 'Billing messages that mention "my account was charged" get classified as Account issues due to the "account" keyword overlap.',
        ('Billing / Subscription', 'Complaint'): 'Frustrated billing messages with emotional language get classified as Complaints. The frustration signal overpowers the billing topic signal.',
        ('Complaint', 'Billing / Subscription'): 'Billing complaints that focus on the charge amount rather than emotional language get classified as Billing instead of Complaint.',
        ('Playback / Streaming', 'App / Technical Issue'): 'Playback issues caused by app bugs blur the line between these categories. "App won\'t play" could be either.',
        ('App / Technical Issue', 'Playback / Streaming'): 'Technical issues that manifest as playback failures are classified as Playback. The symptom (can\'t play) is prioritized over the cause (app bug).',
        ('Content / Library', 'Playback / Streaming'): 'Missing content issues ("song won\'t play") may appear as playback issues when the real problem is content availability.',
        ('General Inquiry', 'Other'): 'Short, simple questions without clear intent keywords get classified as Other instead of General Inquiry.',
        ('Other', 'General Inquiry'): 'Off-topic messages that contain question words get misclassified as General Inquiry.',
        ('Device / Connectivity', 'App / Technical Issue'): 'Device connection issues that involve app interaction get classified as App / Technical Issue.',
    }

    key = (gold_intent, pred_intent)
    if key in hypotheses:
        return hypotheses[key]

    return f"Messages intended as '{gold_intent}' contain keywords or patterns that trigger '{pred_intent}' classification. The semantic boundary between these categories is ambiguous for edge cases."


def _suggest_improvement(gold_intent, pred_intent):
    """Suggest an improvement for a confusion pair."""
    return (
        f"Add disambiguation examples to the taxonomy for '{gold_intent}' vs '{pred_intent}'. "
        f"Consider using multi-signal classification (keywords + context + LLM reasoning) "
        f"and adding specific exclusion rules for the '{pred_intent}' category."
    )


def save_failure_analysis(failure_analysis, output_dir=None, model_name="agent"):
    """Save failure analysis to disk."""
    output_dir = Path(output_dir) if output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"failure_analysis_{model_name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(failure_analysis, f, indent=2, ensure_ascii=False)
    return path
