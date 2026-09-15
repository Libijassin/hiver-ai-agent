"""
Intent classifier: uses Gemini LLM for classification with mock fallback.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import USE_MOCK, GEMINI_API_KEY, GEMINI_MODEL
from src.taxonomy import INTENT_TAXONOMY, INTENT_NAMES, get_taxonomy_prompt


def _get_client():
    """Get Gemini client (lazy init)."""
    if USE_MOCK:
        return None
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


CLASSIFICATION_PROMPT = """You are an intent classifier for SpotifyCares customer support.

Given a customer message, classify it into exactly ONE of the following intents.
Also provide a confidence score between 0.0 and 1.0.

{taxonomy}

Respond with ONLY valid JSON in this exact format:
{{"intent": "<intent name>", "confidence": <float>}}

Do not include any other text. The intent must be one of: {intent_list}

Customer message: "{message}"
"""


def classify_intent_llm(message, client=None):
    """
    Classify a customer message using Gemini LLM.

    Returns: dict with 'intent' and 'confidence'
    """
    if client is None:
        client = _get_client()

    prompt = CLASSIFICATION_PROMPT.format(
        taxonomy=get_taxonomy_prompt(),
        intent_list=", ".join(INTENT_NAMES),
        message=message,
    )

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        config={"temperature": 0.1},  # Low temperature for consistency
    )

    response_text = interaction.output_text.strip()

    # Parse JSON response
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response_text)

        intent = result.get('intent', 'Other')
        confidence = float(result.get('confidence', 0.5))

        # Validate intent
        if intent not in INTENT_NAMES:
            # Try fuzzy matching
            for name in INTENT_NAMES:
                if name.lower() in intent.lower() or intent.lower() in name.lower():
                    intent = name
                    break
            else:
                intent = 'Other'
                confidence = 0.3

        return {'intent': intent, 'confidence': min(max(confidence, 0.0), 1.0)}

    except (json.JSONDecodeError, ValueError, AttributeError):
        return {'intent': 'Other', 'confidence': 0.3}


def classify_intent_mock(message):
    """
    Mock classifier using keyword matching.
    Deterministic fallback when no API key is available.
    """
    message_lower = message.lower()

    # Keyword-based rules (ordered by specificity)
    rules = [
        ('Account / Login', [
            'login', 'log in', 'sign in', 'password', 'locked out',
            'can\'t access', 'account hacked', 'reset password',
            'email change', 'account recovery', 'forgot password'
        ]),
        ('Billing / Subscription', [
            'charge', 'charged', 'billing', 'payment', 'subscription',
            'premium', 'cancel', 'refund', 'student discount', 'family plan',
            'free trial', 'upgrade', 'downgrade', 'invoice', 'price'
        ]),
        ('Playback / Streaming', [
            'play', 'playing', 'pause', 'buffer', 'skip', 'stream',
            'offline', 'download', 'audio quality', 'sound quality',
            'won\'t play', 'not playing', 'stops playing', 'keeps pausing'
        ]),
        ('App / Technical Issue', [
            'crash', 'crashing', 'bug', 'error', 'update', 'install',
            'slow', 'laggy', 'freeze', 'frozen', 'not working',
            'glitch', 'broken', 'fix', 'issue'
        ]),
        ('Content / Library', [
            'playlist', 'library', 'song missing', 'album missing',
            'disappeared', 'saved songs', 'discover weekly',
            'recommendation', 'not available', 'removed'
        ]),
        ('Device / Connectivity', [
            'bluetooth', 'speaker', 'connect', 'car', 'alexa',
            'google home', 'chromecast', 'smart', 'device',
            'spotify connect', 'cast'
        ]),
        ('Complaint', [
            'worst', 'terrible', 'horrible', 'ridiculous', 'unacceptable',
            'frustrated', 'angry', 'switch to', 'apple music',
            'cancelling', 'done with', 'hate', 'useless'
        ]),
        ('General Inquiry', [
            'how do i', 'how to', 'can i', 'is there', 'feature',
            'when will', 'does spotify', 'what is', 'where is'
        ]),
    ]

    for intent_name, keywords in rules:
        for keyword in keywords:
            if keyword in message_lower:
                # Confidence based on keyword specificity
                confidence = 0.65 if len(keyword) > 5 else 0.55
                return {'intent': intent_name, 'confidence': confidence}

    return {'intent': 'Other', 'confidence': 0.4}


def classify_intent(message, use_mock=None):
    """
    Main classification entry point.
    Uses LLM if available, falls back to mock.
    """
    if use_mock is None:
        use_mock = USE_MOCK

    if use_mock:
        return classify_intent_mock(message)
    else:
        return classify_intent_llm(message)


if __name__ == "__main__":
    test_messages = [
        "I can't log into my account",
        "I was charged twice this month",
        "Songs keep buffering",
        "The app crashes on startup",
        "My playlist disappeared",
        "Can't connect to my Bluetooth speaker",
        "This is the worst service ever",
        "How do I share a playlist?",
        "Thanks for helping!",
    ]

    print(f"Mode: {'MOCK' if USE_MOCK else 'LLM'}\n")
    for msg in test_messages:
        result = classify_intent(msg)
        print(f"  [{result['confidence']:.2f}] {result['intent']:25s} ← {msg}")
