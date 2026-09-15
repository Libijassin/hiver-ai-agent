"""
LLM-as-judge: evaluates generated replies using a structured rubric.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import USE_MOCK, GEMINI_API_KEY, GEMINI_MODEL


def _get_client():
    if USE_MOCK:
        return None
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


JUDGE_PROMPT = """You are an expert evaluator for customer support quality.

Evaluate the following AI-generated customer support reply on these dimensions.
Score each from 1 (worst) to 5 (best).

## Scoring Rubric

**Helpfulness** (1-5): Does the reply address the customer's issue and provide actionable guidance?
- 1: Completely unhelpful, ignores the issue
- 3: Partially helpful, addresses the issue but vaguely
- 5: Extremely helpful, provides clear actionable steps

**Correctness** (1-5): Is the information in the reply accurate and not misleading?
- 1: Contains incorrect or harmful information
- 3: Mostly correct but has minor inaccuracies
- 5: Fully correct and appropriate

**Grounding** (1-5): Is the reply consistent with how the brand historically handles similar issues?
- 1: Completely inconsistent with historical brand behavior
- 3: Somewhat consistent but deviates in tone or approach
- 5: Perfectly aligned with historical brand responses

**Relevance** (1-5): Is the reply relevant to the specific customer message and intent?
- 1: Completely off-topic
- 3: Somewhat relevant but misses key aspects
- 5: Directly and specifically addresses the customer's issue

**Completeness** (1-5): Does the reply cover all aspects the customer needs?
- 1: Misses the main issue entirely
- 3: Covers the main issue but misses secondary aspects
- 5: Comprehensively addresses all aspects

**Auto-send Safety** (1-5): Would it be safe to automatically send this reply without human review?
- 1: Dangerous to send automatically (could cause harm or PR issues)
- 3: Acceptable but would benefit from review
- 5: Perfectly safe to auto-send

## Context

**Customer Message:** "{customer_message}"

**Predicted Intent:** {intent}

**Historical Examples Used:**
{historical_examples}

**Generated Reply:** "{generated_reply}"

## Instructions
Respond with ONLY valid JSON in this exact format:
{{"helpfulness": <int>, "correctness": <int>, "grounding": <int>, "relevance": <int>, "completeness": <int>, "auto_send_safety": <int>, "overall": <int>, "reason": "<brief explanation>"}}

The "overall" score should be your holistic assessment (1-5).
"""


def judge_reply_llm(customer_message, intent, retrieved_examples, generated_reply, client=None):
    """
    Use Gemini to judge the quality of a generated reply.
    """
    if client is None:
        client = _get_client()

    # Format historical examples
    hist_text = ""
    if retrieved_examples:
        for i, ex in enumerate(retrieved_examples[:3], 1):
            hist_text += f"  Example {i}: Customer: {ex.get('customer_message', '')[:150]}\n"
            hist_text += f"            Brand: {ex.get('brand_response', '')[:150]}\n"
    else:
        hist_text = "  No historical examples were provided."

    prompt = JUDGE_PROMPT.format(
        customer_message=customer_message,
        intent=intent,
        historical_examples=hist_text,
        generated_reply=generated_reply,
    )

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        config={"temperature": 0.1},
    )

    response_text = interaction.output_text.strip()

    try:
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response_text)

        # Validate and clamp scores
        for key in ['helpfulness', 'correctness', 'grounding', 'relevance',
                     'completeness', 'auto_send_safety', 'overall']:
            if key in result:
                result[key] = max(1, min(5, int(result[key])))

        return result

    except (json.JSONDecodeError, ValueError):
        return {
            'helpfulness': 3, 'correctness': 3, 'grounding': 3,
            'relevance': 3, 'completeness': 3, 'auto_send_safety': 3,
            'overall': 3, 'reason': 'Failed to parse judge response',
            'parse_error': True,
        }


def judge_reply_mock(customer_message, intent, retrieved_examples, generated_reply):
    """
    Mock judge: scores based on heuristic signals.
    Deterministic fallback when no API key is available.
    """
    scores = {}

    # Helpfulness: based on reply length and content
    reply_len = len(generated_reply)
    if reply_len > 100:
        scores['helpfulness'] = 4
    elif reply_len > 50:
        scores['helpfulness'] = 3
    else:
        scores['helpfulness'] = 2

    # Correctness: assume reasonable if we have historical evidence
    scores['correctness'] = 4 if retrieved_examples else 3

    # Grounding: based on similarity scores
    if retrieved_examples:
        max_sim = max(ex.get('similarity', 0) for ex in retrieved_examples)
        if max_sim > 0.3:
            scores['grounding'] = 4
        elif max_sim > 0.15:
            scores['grounding'] = 3
        else:
            scores['grounding'] = 2
    else:
        scores['grounding'] = 1

    # Relevance: check if intent-related keywords appear in reply
    intent_keywords = {
        'Account / Login': ['account', 'login', 'password', 'dm'],
        'Billing / Subscription': ['billing', 'charge', 'subscription', 'payment', 'dm'],
        'Playback / Streaming': ['play', 'cache', 'stream', 'quality'],
        'App / Technical Issue': ['app', 'reinstall', 'update', 'device'],
        'Content / Library': ['library', 'playlist', 'song', 'album'],
        'Device / Connectivity': ['device', 'bluetooth', 'connect', 'speaker'],
        'General Inquiry': ['help', 'support', 'info', 'check'],
        'Complaint': ['sorry', 'understand', 'help', 'dm'],
        'Other': ['help', 'dm', 'reach out'],
    }
    keywords = intent_keywords.get(intent, [])
    reply_lower = generated_reply.lower()
    keyword_matches = sum(1 for kw in keywords if kw in reply_lower)
    scores['relevance'] = min(5, 2 + keyword_matches)

    # Completeness
    scores['completeness'] = 3 if reply_len > 50 else 2

    # Auto-send safety
    scores['auto_send_safety'] = 3 if scores['correctness'] >= 3 else 2

    # Overall
    all_scores = [scores[k] for k in ['helpfulness', 'correctness', 'grounding', 'relevance']]
    scores['overall'] = round(sum(all_scores) / len(all_scores))

    scores['reason'] = f"Mock evaluation (reply length={reply_len}, evidence={len(retrieved_examples or [])} examples)"

    return scores


def judge_reply(customer_message, intent, retrieved_examples, generated_reply, use_mock=None):
    """Main entry point for LLM-as-judge."""
    if use_mock is None:
        use_mock = USE_MOCK

    if use_mock:
        return judge_reply_mock(customer_message, intent, retrieved_examples, generated_reply)
    else:
        return judge_reply_llm(customer_message, intent, retrieved_examples, generated_reply)


if __name__ == "__main__":
    result = judge_reply(
        customer_message="I can't log into my account",
        intent="Account / Login",
        retrieved_examples=[{
            'customer_message': 'locked out of account',
            'brand_response': "We're sorry about that! DM us your details /JR",
            'similarity': 0.4,
        }],
        generated_reply="We're sorry about that! DM us your account details so we can help you get back in /JR",
    )
    print(json.dumps(result, indent=2))
