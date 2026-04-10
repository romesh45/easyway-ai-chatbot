"""
Main chatbot application for the NLP Intent Classification System.

Integrates the full pipeline:
    User input → preprocess → predict intent → select response → display

Runs as an interactive CLI loop. Type 'exit' or 'quit' to end.

Usage:
    python src/chatbot.py

Author: EasyWay Logistics AI Team
"""

import os
import sys

# ---------------------------------------------------------------------------
# Resolve project paths for imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_artifacts, predict_intent
from src.response import load_responses, get_response

# ADDED LOGGING
from src.logger import log_query, log_unknown, log_error


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BOT_NAME = "EasyWay AI"
EXIT_COMMANDS = {"exit", "quit", "bye", "q"}

WELCOME_MESSAGE = """
╔══════════════════════════════════════════════════════════╗
║           🚛  EasyWay AI Assistant  🚛                  ║
║                                                          ║
║   Your intelligent logistics & transport helper.         ║
║   Ask me about bookings, tracking, pricing, and more.    ║
║                                                          ║
║   Type 'exit' or 'quit' to end the conversation.         ║
╚══════════════════════════════════════════════════════════╝
"""

GOODBYE_MESSAGE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Thank you for using EasyWay AI. Have a great day! 👋
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ---------------------------------------------------------------------------
# STEP 1: SYSTEM INITIALIZATION
# ---------------------------------------------------------------------------

def initialize_system() -> tuple:
    """
    Load all system components required for the chatbot.

    Loads:
        - Model artifacts (classifier, vectorizer, encoder, abbreviations)
        - Response templates (intent → response list mapping)

    Returns:
        Tuple of (artifacts dict, responses dict).

    Raises:
        FileNotFoundError: If model files are missing (not trained yet).
    """

    print("Loading system components...")

    artifacts = load_artifacts()
    responses = load_responses()

    print("System ready.\n")

    return artifacts, responses


# ---------------------------------------------------------------------------
# STEP 2–3: PROCESS A SINGLE USER QUERY
# ---------------------------------------------------------------------------

def process_query(user_input: str, artifacts: dict, responses: dict) -> dict:
    """
    Run the full pipeline on a single user query.

    Pipeline:
        raw input → predict_intent() → get_response()

    Args:
        user_input: Raw text from the user.
        artifacts:  Model artifacts dict from load_artifacts().
        responses:  Response templates dict from load_responses().

    Returns:
        Dictionary with:
            - "intent":     Predicted intent string
            - "confidence": Float confidence score
            - "response":   Selected response string
    """

    # Predict intent with confidence score
    prediction = predict_intent(user_input, artifacts)

    intent     = prediction["intent"]
    confidence = prediction["confidence"]

    # Select a response based on the predicted intent
    reply = get_response(intent, responses)

    # ADDED LOGGING
    log_query(user_input, intent, confidence)
    if intent == "fallback":
        log_unknown(user_input)

    return {
        "intent":     intent,
        "confidence": confidence,
        "response":   reply,
    }


# ---------------------------------------------------------------------------
# STEP 4: FORMAT OUTPUT
# ---------------------------------------------------------------------------

def display_response(result: dict) -> None:
    """
    Print the bot response and prediction metadata in a clean format.

    Output:
        Bot: <response text>
        (Intent: <intent> | Confidence: <score>)

    Args:
        result: Dictionary from process_query().
    """

    intent     = result["intent"]
    confidence = result["confidence"]
    response   = result["response"]

    # Display the bot's reply
    print(f"\n  {BOT_NAME}: {response}")

    # Display prediction metadata (useful for debugging / transparency)
    if intent == "fallback":
        print(f"  ⚠️  (Intent: {intent} | Confidence: {confidence:.4f})")
    else:
        print(f"  📌 (Intent: {intent} | Confidence: {confidence:.4f})")

    print()


# ---------------------------------------------------------------------------
# STEPS 5–8: MAIN CHAT LOOP
# ---------------------------------------------------------------------------

def chat_loop(artifacts: dict, responses: dict) -> None:
    """
    Run the interactive chatbot loop.

    Handles:
        - Normal queries → predict → respond
        - Exit commands  → graceful shutdown
        - Empty input    → prompt to try again
        - Ctrl+C         → graceful shutdown

    Args:
        artifacts: Model artifacts dict.
        responses: Response templates dict.
    """

    while True:
        try:
            # ---- Get user input ----
            user_input = input("  You: ").strip()

            # ---- Handle empty input ----
            if not user_input:
                print(f"\n  {BOT_NAME}: Please type something so I can help you!\n")
                continue

            # ---- Handle exit commands ----
            if user_input.lower() in EXIT_COMMANDS:
                print(GOODBYE_MESSAGE)
                break

            # ---- Process and respond ----
            result = process_query(user_input, artifacts, responses)
            display_response(result)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print(GOODBYE_MESSAGE)
            break

        except Exception as e:
            # Catch unexpected errors without crashing the chatbot
            print(f"\n  {BOT_NAME}: Sorry, something went wrong. Please try again.")
            print(f"  ❌ Error: {e}\n")
            
            # ADDED LOGGING
            log_error(str(e))


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """
    Initialize the system and start the chatbot.
    """

    print(WELCOME_MESSAGE)

    try:
        artifacts, responses = initialize_system()
    except FileNotFoundError as e:
        print(f"❌ Initialization failed: {e}")
        print("   Run 'python src/train.py' first to train the model.")
        sys.exit(1)

    # Separator before chat begins
    print("━" * 58)
    print("  Chat started. Type your message below.\n")

    chat_loop(artifacts, responses)


if __name__ == "__main__":
    main()
