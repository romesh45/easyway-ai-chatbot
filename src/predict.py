"""
Prediction engine for the NLP Intent Classification System.

Loads trained model artifacts and provides a clean predict_intent() API
for classifying user queries into intents with confidence scores.

Usage:
    # As a module
    from src.predict import predict_intent
    result = predict_intent("need truck tomorrow")

    # Direct test
    python src/predict.py

Author: EasyWay Logistics AI Team
"""

import os
import sys
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Resolve project paths for imports and artifact loading
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import normalize_text, load_abbreviations


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODELS_DIR         = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH         = os.path.join(MODELS_DIR, "intent_classifier.pkl")
VECTORIZER_PATH    = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH       = os.path.join(MODELS_DIR, "label_encoder.pkl")
ABBREVIATIONS_PATH = os.path.join(PROJECT_ROOT, "data", "abbreviations.json")

# Confidence threshold — below this, the prediction is treated as unreliable
# and routed to the fallback handler instead.
#
# With 132 samples across 11 intents, the model distributes probability mass
# thinly. Random baseline = 1/11 ≈ 0.09, so 0.15 is a meaningful signal.
# Recommended thresholds by dataset size:
#   132 samples  → 0.15
#   500 samples  → 0.30
#   1000+ samples → 0.40
CONFIDENCE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# STEP 1–2: LOAD ALL ARTIFACTS (one-time initialization)
# ---------------------------------------------------------------------------

def load_artifacts() -> dict:
    """
    Load all trained model artifacts and abbreviation mappings from disk.

    This should be called ONCE at startup. The returned dict is passed
    to predict_intent() to avoid reloading on every call.

    Returns:
        Dictionary containing:
            - model:      Trained LogisticRegression classifier
            - vectorizer: Fitted TfidfVectorizer
            - encoder:    Fitted LabelEncoder
            - abbr_dict:  Abbreviation expansion mappings

    Raises:
        FileNotFoundError: If any artifact file is missing (model not trained).
    """

    # Validate all files exist before loading
    for path, name in [
        (MODEL_PATH, "Model"),
        (VECTORIZER_PATH, "Vectorizer"),
        (ENCODER_PATH, "Label Encoder"),
        (ABBREVIATIONS_PATH, "Abbreviations"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at '{path}'. "
                "Run 'python src/train.py' first to generate model artifacts."
            )

    artifacts = {
        "model":      joblib.load(MODEL_PATH),
        "vectorizer": joblib.load(VECTORIZER_PATH),
        "encoder":    joblib.load(ENCODER_PATH),
        "abbr_dict":  load_abbreviations(ABBREVIATIONS_PATH),
    }

    return artifacts


# ---------------------------------------------------------------------------
# STEP 3–5: PREDICT INTENT
# ---------------------------------------------------------------------------

def predict_intent(text: str, artifacts: dict) -> dict:
    """
    Classify a single user query into an intent with a confidence score.

    Pipeline:
        raw text → normalize_text() → TF-IDF transform → predict_proba()
        → decode label → apply confidence threshold

    Args:
        text:      Raw user input string.
        artifacts: Dictionary from load_artifacts() containing model,
                   vectorizer, encoder, and abbr_dict.

    Returns:
        Dictionary with keys:
            - "input":      Original raw text
            - "cleaned":    Preprocessed text
            - "intent":     Predicted intent string (or "fallback")
            - "confidence": Float confidence score (0.0 – 1.0)

    Edge cases:
        - Empty/None input → returns fallback with 0.0 confidence
    """

    # ---- Edge case: empty or invalid input ----
    if not text or not isinstance(text, str) or text.strip() == "":
        return {
            "input":      text,
            "cleaned":    "",
            "intent":     "fallback",
            "confidence": 0.0,
        }

    # ---- Step 3: Preprocess (same pipeline as training) ----
    cleaned = normalize_text(text, artifacts["abbr_dict"])

    # Guard: if preprocessing reduces text to empty string
    if not cleaned:
        return {
            "input":      text,
            "cleaned":    "",
            "intent":     "fallback",
            "confidence": 0.0,
        }

    # ---- Step 4: Vectorize + Predict ----
    model      = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    encoder    = artifacts["encoder"]

    # Transform cleaned text using the SAME fitted vectorizer from training
    text_vector = vectorizer.transform([cleaned])

    # Get probability distribution across all intents
    probabilities = model.predict_proba(text_vector)[0]

    # Find the intent with the highest probability
    max_index      = np.argmax(probabilities)
    max_confidence = float(probabilities[max_index])
    predicted_label = encoder.inverse_transform([max_index])[0]

    # ---- Step 5: Apply confidence threshold ----
    if max_confidence < CONFIDENCE_THRESHOLD:
        return {
            "input":      text,
            "cleaned":    cleaned,
            "intent":     "fallback",
            "confidence": round(max_confidence, 4),
        }

    return {
        "input":      text,
        "cleaned":    cleaned,
        "intent":     str(predicted_label),
        "confidence": round(max_confidence, 4),
    }


# ---------------------------------------------------------------------------
# SELF-TEST: Validate prediction on sample queries
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("INTENT CLASSIFIER — PREDICTION ENGINE TEST")
    print("=" * 65)

    # Load artifacts once
    print("\nLoading model artifacts...")
    artifacts = load_artifacts()
    print("Artifacts loaded successfully.\n")

    # Test queries — mix of clear intents, slang, and out-of-domain
    test_queries = [
        "need truck tomorrow",
        "payment failed refund",
        "where is my order",
        "hello",
        "thanks a lot",
        "cancel my booking pls",
        "what is quantum computing",
        "asjkdhaskjd random gibberish",
        "",
        None,
    ]

    print(f"{'Query':<45} {'Intent':<25} {'Confidence':>10}")
    print("-" * 82)

    for query in test_queries:
        result = predict_intent(query, artifacts)

        display = repr(query) if query is None or (isinstance(query, str) and query.strip() == "") else query
        intent = result["intent"]
        conf   = result["confidence"]

        # Color-code confidence levels in terminal
        if intent == "fallback":
            tag = "⚠️  FALLBACK"
        elif conf >= 0.70:
            tag = "✅ HIGH"
        else:
            tag = "🔶 MEDIUM"

        print(f"{str(display):<45} {intent:<25} {conf:>8.4f}  {tag}")

    print("\n" + "=" * 65)
