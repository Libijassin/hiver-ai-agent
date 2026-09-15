"""
Escalation logic: decides AUTO_HANDLED vs ESCALATE with reasons.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIDENCE_THRESHOLD, RETRIEVAL_SIM_THRESHOLD


# Sensitive topic keywords that warrant escalation
SENSITIVE_KEYWORDS = [
    'hack', 'hacked', 'stolen', 'fraud', 'unauthorized',
    'legal', 'lawyer', 'lawsuit', 'police',
    'threat', 'suicide', 'harm', 'emergency',
    'data breach', 'privacy', 'gdpr',
]


def decide_escalation(
    intent,
    confidence,
    retrieved_examples,
    customer_message,
):
    """
    Decide whether a message should be auto-handled or escalated.

    Args:
        intent: predicted intent name
        confidence: classification confidence (0-1)
        retrieved_examples: list of retrieved historical conversations
        customer_message: the customer's message text

    Returns:
        dict with:
            'action': 'AUTO_HANDLED' or 'ESCALATE'
            'reason': str (empty for AUTO_HANDLED, explanation for ESCALATE)
            'escalation_factors': list of specific factors that triggered escalation
    """
    escalation_factors = []

    # Factor 1: Low classification confidence
    if confidence < CONFIDENCE_THRESHOLD:
        escalation_factors.append(
            f"Low classification confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD})"
        )

    # Factor 2: Insufficient historical evidence
    if not retrieved_examples or len(retrieved_examples) == 0:
        escalation_factors.append("No historical examples found for retrieval")
    else:
        max_sim = max(ex['similarity'] for ex in retrieved_examples)
        if max_sim < RETRIEVAL_SIM_THRESHOLD:
            escalation_factors.append(
                f"Low retrieval similarity (max={max_sim:.2f} < {RETRIEVAL_SIM_THRESHOLD})"
            )

    # Factor 3: Sensitive content
    message_lower = customer_message.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in message_lower:
            escalation_factors.append(f"Sensitive content detected: '{keyword}'")
            break

    # Factor 4: Complaint intent with low retrieval confidence
    if intent == 'Complaint':
        if not retrieved_examples or len(retrieved_examples) < 2:
            escalation_factors.append(
                "Complaint with insufficient historical resolution examples"
            )

    # Factor 5: Conflicting historical resolutions
    if retrieved_examples and len(retrieved_examples) >= 2:
        responses = [ex['brand_response'].lower() for ex in retrieved_examples[:3]]
        # Check if some say "DM us" and others give direct solutions
        has_dm = any('dm' in r or 'direct message' in r for r in responses)
        has_direct = any(len(r) > 100 and 'dm' not in r for r in responses)
        if has_dm and has_direct:
            escalation_factors.append(
                "Conflicting historical resolutions (some DM, some direct)"
            )

    # Decision
    if len(escalation_factors) >= 1:
        reason = "; ".join(escalation_factors)
        return {
            'action': 'ESCALATE',
            'reason': reason,
            'escalation_factors': escalation_factors,
        }
    else:
        return {
            'action': 'AUTO_HANDLED',
            'reason': '',
            'escalation_factors': [],
        }


if __name__ == "__main__":
    # Test cases
    print("Test 1: Low confidence")
    result = decide_escalation('Other', 0.3, [], "Something weird happened")
    print(f"  {result['action']}: {result['reason']}\n")

    print("Test 2: Good match")
    result = decide_escalation(
        'Playback / Streaming', 0.85,
        [{'similarity': 0.6, 'brand_response': 'Try clearing cache', 'customer_message': 'test'}],
        "Songs won't play"
    )
    print(f"  {result['action']}: {result['reason']}\n")

    print("Test 3: Sensitive content")
    result = decide_escalation(
        'Account / Login', 0.9,
        [{'similarity': 0.5, 'brand_response': 'DM us', 'customer_message': 'test'}],
        "My account was hacked and someone stole my data"
    )
    print(f"  {result['action']}: {result['reason']}\n")
