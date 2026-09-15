"""
Agent orchestrator: ties together classifier → retriever → generator → escalation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import USE_MOCK
from src.classifier import classify_intent
from src.retriever import HistoricalRetriever
from src.generator import generate_reply
from src.escalation import decide_escalation


class SupportAgent:
    """
    AI Customer Support Agent for SpotifyCares.

    Pipeline:
        Customer Message → Intent Classification → Historical Retrieval
        → Reply Generation → Escalation Decision → Final Output
    """

    def __init__(self, retriever, use_mock=None):
        """
        Args:
            retriever: HistoricalRetriever instance (already fitted)
            use_mock: if True, use mock LLM; if None, auto-detect
        """
        self.retriever = retriever
        self.use_mock = use_mock if use_mock is not None else USE_MOCK

    def process(self, customer_message):
        """
        Process a single customer message through the full pipeline.

        Returns dict with:
            - predicted_intent: str
            - confidence: float
            - generated_reply: str
            - action: 'AUTO_HANDLED' or 'ESCALATE'
            - escalation_reason: str
            - retrieved_examples: list of historical examples used
            - mode: 'llm' or 'mock'
        """
        # Step 1: Intent Classification
        classification = classify_intent(customer_message, use_mock=self.use_mock)
        intent = classification['intent']
        confidence = classification['confidence']

        # Step 2: Historical Retrieval
        retrieved = self.retriever.retrieve(customer_message)

        # Step 3: Reply Generation
        reply = generate_reply(
            customer_message, intent, retrieved, use_mock=self.use_mock
        )

        # Step 4: Escalation Decision
        escalation = decide_escalation(
            intent, confidence, retrieved, customer_message
        )

        return {
            'predicted_intent': intent,
            'confidence': confidence,
            'generated_reply': reply,
            'action': escalation['action'],
            'escalation_reason': escalation['reason'],
            'escalation_factors': escalation['escalation_factors'],
            'retrieved_examples': retrieved,
            'mode': 'mock' if self.use_mock else 'llm',
        }

    def process_batch(self, messages, progress_callback=None):
        """
        Process a batch of customer messages.

        Args:
            messages: list of customer message strings
            progress_callback: optional callable(i, total) for progress

        Returns: list of result dicts
        """
        results = []
        total = len(messages)
        for i, msg in enumerate(messages):
            result = self.process(msg)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)
        return results


if __name__ == "__main__":
    import json
    from src.retriever import load_reference_pairs

    # Load retriever
    print("Loading reference pairs...")
    pairs = load_reference_pairs()
    print(f"Loaded {len(pairs)} reference pairs")

    retriever = HistoricalRetriever(pairs)
    print("Built retriever index")

    # Create agent
    agent = SupportAgent(retriever)
    print(f"Agent mode: {'MOCK' if agent.use_mock else 'LLM'}\n")

    # Test
    test_msgs = [
        "I can't log into my Spotify account, it says wrong password",
        "I was charged $9.99 but I cancelled last month",
        "Songs keep pausing every 30 seconds",
        "My account was hacked!!!",
    ]

    for msg in test_msgs:
        result = agent.process(msg)
        print(f"Customer: {msg}")
        print(f"  Intent: {result['predicted_intent']} ({result['confidence']:.2f})")
        print(f"  Action: {result['action']}")
        if result['escalation_reason']:
            print(f"  Escalation: {result['escalation_reason']}")
        print(f"  Reply: {result['generated_reply'][:100]}...")
        print(f"  Evidence: {len(result['retrieved_examples'])} examples")
        print()
