"""
Reply generator: generates grounded responses using retrieved historical examples.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import USE_MOCK, GEMINI_API_KEY, GEMINI_MODEL, BRAND_NAME


def _get_client():
    """Get Gemini client (lazy init)."""
    if USE_MOCK:
        return None
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


GENERATION_PROMPT = """You are a customer support agent for {brand_name} on Twitter.

Generate a helpful, empathetic reply to the customer's message.
Your reply MUST be grounded in how {brand_name} historically handled similar issues.

## Customer Message
"{customer_message}"

## Predicted Intent
{intent}

## Historical Examples (how {brand_name} handled similar issues)
{historical_examples}

## Instructions
1. Use the tone and style from the historical examples above.
2. Address the customer's specific issue.
3. If historical examples suggest asking the customer to DM, follow that pattern.
4. Keep the reply concise (tweet-length, ~280 chars max).
5. Be empathetic and professional.
6. Do NOT make up solutions not supported by the historical examples.

Reply as {brand_name}:"""


def format_historical_examples(retrieved_examples):
    """Format retrieved examples for the generation prompt."""
    if not retrieved_examples:
        return "No similar historical examples found."

    lines = []
    for i, ex in enumerate(retrieved_examples[:5], 1):
        lines.append(f"Example {i} (similarity: {ex['similarity']:.2f}):")
        lines.append(f"  Customer: {ex['customer_message'][:200]}")
        lines.append(f"  {BRAND_NAME}: {ex['brand_response'][:200]}")
        lines.append("")
    return "\n".join(lines)


def generate_reply_llm(customer_message, intent, retrieved_examples, client=None):
    """
    Generate a reply using Gemini LLM, grounded in historical examples.
    """
    if client is None:
        client = _get_client()

    prompt = GENERATION_PROMPT.format(
        brand_name=BRAND_NAME,
        customer_message=customer_message,
        intent=intent,
        historical_examples=format_historical_examples(retrieved_examples),
    )

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        config={"temperature": 0.7, "max_output_tokens": 300},
    )

    reply = interaction.output_text.strip()

    # Clean up: remove quotes if the model wrapped the reply
    if reply.startswith('"') and reply.endswith('"'):
        reply = reply[1:-1]

    return reply


def generate_reply_mock(customer_message, intent, retrieved_examples):
    """
    Mock reply generator: uses the most similar historical response
    as a template. Deterministic fallback.
    """
    if retrieved_examples and len(retrieved_examples) > 0:
        best_match = retrieved_examples[0]
        # Use the historical brand response directly (adapted slightly)
        response = best_match['brand_response']

        # If it's very short, add a generic helpful prefix
        if len(response) < 20:
            response = f"Hi there! {response}"

        return response
    else:
        # No historical examples found — generic fallback by intent
        fallbacks = {
            'Account / Login': f"Hi there! We'd love to help with your account. Could you DM us your details so we can look into this? /LM",
            'Billing / Subscription': f"Hey! We understand billing concerns are important. Please DM us your account info and we'll sort this out. /AB",
            'Playback / Streaming': f"Sorry to hear about the playback issues! Could you try clearing the cache in Settings? If that doesn't help, DM us. /JR",
            'App / Technical Issue': f"We're sorry about the trouble! Try reinstalling the app. If the issue persists, send us a DM with your device info. /KS",
            'Content / Library': f"We understand how frustrating it is when content goes missing. DM us your account details and we'll investigate. /PD",
            'Device / Connectivity': f"Sorry about the connectivity issues! Try disconnecting and reconnecting the device. DM us if you need more help. /TM",
            'General Inquiry': f"Great question! You can find more info at https://support.spotify.com. DM us if you need anything else! /LM",
            'Complaint': f"We're really sorry to hear about your experience. We want to make this right. Please DM us so we can help. /AB",
            'Other': f"Thanks for reaching out! DM us with more details and we'll be happy to help. /JR",
        }
        return fallbacks.get(intent, fallbacks['Other'])


def generate_reply(customer_message, intent, retrieved_examples, use_mock=None):
    """
    Main reply generation entry point.
    """
    if use_mock is None:
        use_mock = USE_MOCK

    if use_mock:
        return generate_reply_mock(customer_message, intent, retrieved_examples)
    else:
        return generate_reply_llm(customer_message, intent, retrieved_examples)


if __name__ == "__main__":
    # Quick test
    test_examples = [
        {
            'customer_message': 'my account got hacked',
            'brand_response': "That's not cool! Let's get this sorted. DM us your username and email so we can look into it /JR",
            'similarity': 0.45,
        }
    ]

    reply = generate_reply(
        "Someone hacked into my Spotify account and changed my password",
        "Account / Login",
        test_examples,
    )
    print(f"Generated reply: {reply}")
