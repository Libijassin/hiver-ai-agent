"""
Intent taxonomy for SpotifyCares customer support.
Derived from inspection of actual SpotifyCares conversations.

Each intent has:
- name: canonical identifier
- description: what this intent covers
- examples: 3 real-world-style examples
- inclusion_criteria: when to assign this intent
- exclusion_criteria: when NOT to assign this intent
"""

INTENT_TAXONOMY = [
    {
        "name": "Account / Login",
        "description": "Issues related to account access, login problems, password resets, account recovery, account settings, or profile management.",
        "examples": [
            "I can't log into my Spotify account, it keeps saying wrong password",
            "My account was hacked and someone changed my email",
            "How do I change the email address on my Spotify account?"
        ],
        "inclusion_criteria": "Customer mentions login failure, password issues, account lockout, account recovery, profile changes, or account settings.",
        "exclusion_criteria": "Do NOT use for billing/subscription issues even if they mention 'account'. If the core issue is a charge or plan change, use Billing / Subscription."
    },
    {
        "name": "Billing / Subscription",
        "description": "Issues related to charges, payments, subscription plans, free trial, premium upgrades/downgrades, family plan management, or unexpected charges.",
        "examples": [
            "I was charged twice this month for my premium subscription",
            "How do I cancel my premium and go back to free?",
            "My student discount isn't being applied anymore"
        ],
        "inclusion_criteria": "Customer mentions charges, payments, subscription plan changes, billing errors, refund requests related to billing, premium/free tier, family plan, or student plan.",
        "exclusion_criteria": "Do NOT use for general account access issues. If the customer can't log in but isn't talking about charges, use Account / Login."
    },
    {
        "name": "Playback / Streaming",
        "description": "Issues with playing music or podcasts: songs won't play, buffering, quality issues, offline playback problems, or streaming errors.",
        "examples": [
            "Songs keep pausing every few seconds, my internet is fine",
            "I downloaded songs for offline but they won't play on the train",
            "The audio quality sounds terrible even on Very High setting"
        ],
        "inclusion_criteria": "Customer mentions playback failures, buffering, audio quality, offline download playback, streaming errors, or songs skipping.",
        "exclusion_criteria": "Do NOT use for missing content (use Content / Library) or device connectivity (use Device / Connectivity)."
    },
    {
        "name": "App / Technical Issue",
        "description": "App crashes, bugs, update problems, installation issues, or general technical problems with the Spotify application.",
        "examples": [
            "The app keeps crashing every time I open it after the latest update",
            "Spotify won't install on my new phone",
            "The app is super slow and laggy on my laptop"
        ],
        "inclusion_criteria": "Customer mentions app crashes, bugs, update issues, installation failures, performance problems, or error messages.",
        "exclusion_criteria": "Do NOT use for playback issues (use Playback / Streaming) or device-specific connectivity (use Device / Connectivity)."
    },
    {
        "name": "Content / Library",
        "description": "Issues with music library, playlists, missing songs or albums, recommendations, Discover Weekly, or content availability.",
        "examples": [
            "An album I saved disappeared from my library",
            "My playlist lost all its songs after the update",
            "Why isn't [artist]'s new album on Spotify?"
        ],
        "inclusion_criteria": "Customer mentions missing songs/albums, playlist issues, library sync problems, saved music disappearing, or content availability.",
        "exclusion_criteria": "Do NOT use for playback problems (songs exist but won't play → Playback / Streaming)."
    },
    {
        "name": "Device / Connectivity",
        "description": "Issues connecting Spotify to external devices: Bluetooth speakers, smart speakers, car systems, Chromecast, Spotify Connect, or multi-device sync.",
        "examples": [
            "Spotify won't connect to my Bluetooth speaker anymore",
            "I can't get Spotify to play on my Google Home",
            "Spotify Connect keeps dropping when I switch to my car"
        ],
        "inclusion_criteria": "Customer mentions device connectivity, Bluetooth issues, smart speaker problems, car integration, Spotify Connect, or casting.",
        "exclusion_criteria": "Do NOT use for app crashes (use App / Technical Issue) or general playback issues not device-related."
    },
    {
        "name": "General Inquiry",
        "description": "General questions about Spotify features, how-to questions, feature requests, or positive feedback.",
        "examples": [
            "How do I share a playlist with my friend?",
            "Is there a way to see my listening stats?",
            "It would be great if Spotify added a sleep timer"
        ],
        "inclusion_criteria": "Customer is asking how to do something, requesting a feature, giving feedback, or asking a general question.",
        "exclusion_criteria": "Do NOT use when there is a clear problem or complaint. If the customer is frustrated, use Complaint."
    },
    {
        "name": "Complaint",
        "description": "Expressions of frustration, dissatisfaction, threats to cancel, negative experiences, or unresolved issues where the customer is emotionally upset.",
        "examples": [
            "This is ridiculous, I've been trying to fix this for weeks!",
            "Worst customer service ever, I'm switching to Apple Music",
            "I'm so frustrated, nothing works and nobody helps"
        ],
        "inclusion_criteria": "Customer expresses strong frustration, anger, threatens to leave, or describes ongoing unresolved issues with emotional language.",
        "exclusion_criteria": "Do NOT use for calm problem reports. A customer saying 'my app crashes' without frustration should be App / Technical Issue, not Complaint."
    },
    {
        "name": "Other",
        "description": "Messages that don't fit any of the above categories: greetings, thank-you messages, off-topic, or ambiguous messages.",
        "examples": [
            "Thanks for the help!",
            "Hello?",
            "Can you guys sponsor our event?"
        ],
        "inclusion_criteria": "Message doesn't fit any other category, is a greeting/farewell, is off-topic, or is too ambiguous to classify.",
        "exclusion_criteria": "Prefer a specific category whenever possible. Only use Other as a last resort."
    }
]

# Quick lookups
INTENT_NAMES = [intent["name"] for intent in INTENT_TAXONOMY]
INTENT_MAP = {intent["name"]: intent for intent in INTENT_TAXONOMY}


def get_taxonomy_prompt():
    """Format taxonomy as a prompt string for LLM classification."""
    lines = ["Here are the intent categories:\n"]
    for i, intent in enumerate(INTENT_TAXONOMY, 1):
        lines.append(f"{i}. **{intent['name']}**")
        lines.append(f"   Description: {intent['description']}")
        lines.append(f"   Examples: {'; '.join(intent['examples'])}")
        lines.append(f"   Include when: {intent['inclusion_criteria']}")
        lines.append(f"   Exclude when: {intent['exclusion_criteria']}")
        lines.append("")
    return "\n".join(lines)


def validate_intent(intent_name):
    """Check if an intent name is valid."""
    return intent_name in INTENT_NAMES


if __name__ == "__main__":
    print(f"Intent Taxonomy ({len(INTENT_TAXONOMY)} intents):")
    for intent in INTENT_TAXONOMY:
        print(f"  - {intent['name']}: {intent['description'][:60]}...")
    print(f"\nTaxonomy prompt length: {len(get_taxonomy_prompt())} chars")
