"""
Response system for the NLP Intent Classification System.

Maps predicted intents to human-readable response strings.
Responses are loaded from responses.json and selected randomly
for natural conversation variety.

Fully decoupled from the prediction engine — receives an intent
string and returns a response string. No model dependency.

Usage:
    from src.response import load_responses, get_response

    responses = load_responses()
    reply = get_response("booking_request", responses)

Author: EasyWay Logistics AI Team
"""

import os
import json
import random


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSES_PATH = os.path.join(PROJECT_ROOT, "data", "responses.json")

# Default fallback message — used when the intent is unknown, missing,
# or explicitly set to "fallback" by the prediction engine.
FALLBACK_RESPONSE = (
    "I'm not sure I understood that. Let me connect you to support."
)


# ---------------------------------------------------------------------------
# STEP 1: LOAD RESPONSES (one-time)
# ---------------------------------------------------------------------------

def load_responses(filepath: str = None) -> dict:
    """
    Load the intent-to-responses mapping from a JSON file.

    The file structure is:
        {
            "booking_request": ["response 1", "response 2", ...],
            "greeting": ["response 1", "response 2", ...],
            ...
        }

    Args:
        filepath: Path to responses.json.
                  Defaults to data/responses.json relative to project root.

    Returns:
        Dictionary mapping intent strings to lists of response strings.

    Raises:
        FileNotFoundError: If the responses file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """

    if filepath is None:
        filepath = RESPONSES_PATH

    with open(filepath, "r", encoding="utf-8") as f:
        responses = json.load(f)

    return responses


# ---------------------------------------------------------------------------
# STEPS 2–4: GET RESPONSE WITH FALLBACK
# ---------------------------------------------------------------------------

def get_response(intent: str, responses: dict) -> str:
    """
    Select a random response for the given intent.

    Behavior:
        - Valid intent with responses → random.choice() from the list
        - intent == "fallback"        → fallback message
        - intent not in responses     → fallback message
        - Empty/None intent           → fallback message

    Args:
        intent:    Predicted intent string (e.g., "booking_request").
        responses: Dictionary from load_responses().

    Returns:
        A single response string for the chatbot to display.
    """

    # Guard: handle None, empty string, or non-string input
    if not intent or not isinstance(intent, str):
        return FALLBACK_RESPONSE

    # Explicit fallback from the prediction engine (low confidence)
    if intent == "fallback":
        return FALLBACK_RESPONSE

    # Look up intent in the responses dictionary
    if intent in responses:
        intent_responses = responses[intent]

        # Guard: empty response list (shouldn't happen, but defensive)
        if not intent_responses:
            return FALLBACK_RESPONSE

        # Random selection for natural conversation variety
        return random.choice(intent_responses)

    # Intent not recognized in responses.json
    return FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# SELF-TEST: Validate response selection
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("RESPONSE SYSTEM — TEST")
    print("=" * 65)

    # Load responses once
    responses = load_responses()
    print(f"\nLoaded responses for {len(responses)} intents.")
    print(f"Intents: {list(responses.keys())}\n")

    # Test with known intents, fallback, and edge cases
    test_intents = [
        "booking_request",
        "price_inquiry",
        "complaint",
        "greeting",
        "thanks",
        "fallback",
        "unknown_intent",
        "",
        None,
    ]

    print(f"{'Intent':<25} {'Response'}")
    print("-" * 90)

    for intent in test_intents:
        display = repr(intent) if intent is None or intent == "" else intent
        reply = get_response(intent, responses)
        print(f"{str(display):<25} {reply}")

    # Demonstrate randomization — same intent, different responses
    print("\n" + "-" * 90)
    print("Randomization check (booking_request × 5):\n")
    for i in range(5):
        reply = get_response("booking_request", responses)
        print(f"  [{i+1}] {reply}")

    print("\n" + "=" * 65)
