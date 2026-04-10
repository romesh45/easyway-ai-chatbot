"""
Training pipeline for the NLP Intent Classification System.

Loads dataset → preprocesses → vectorizes → trains Logistic Regression
→ evaluates → saves all artifacts to models/ directory.

Usage:
    python src/train.py

Author: EasyWay Logistics AI Team
"""

import os
import sys
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Resolve project paths so imports and file loading work correctly
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import normalize_text, load_abbreviations


# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

INTENTS_FILE       = os.path.join(DATA_DIR, "intents.json")
ABBREVIATIONS_FILE = os.path.join(DATA_DIR, "abbreviations.json")

MODEL_PATH    = os.path.join(MODELS_DIR, "intent_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH  = os.path.join(MODELS_DIR, "label_encoder.pkl")


# ---------------------------------------------------------------------------
# STEP 1: LOAD DATASET
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> tuple:
    """
    Load the intent dataset from a JSON file.

    Args:
        filepath: Absolute path to intents.json.

    Returns:
        Tuple of (texts, labels) — both as Python lists.
    """

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts  = [entry["text"] for entry in data]
    labels = [entry["intent"] for entry in data]

    return texts, labels


# ---------------------------------------------------------------------------
# STEP 2–3: PREPROCESS ALL TEXTS
# ---------------------------------------------------------------------------

def preprocess_dataset(texts: list, abbr_dict: dict) -> list:
    """
    Apply the full normalization pipeline to every text in the dataset.

    Uses the same normalize_text() function that will be used at inference
    time, preventing train/serve skew.

    Args:
        texts:     List of raw text strings.
        abbr_dict: Abbreviation expansion dictionary.

    Returns:
        List of cleaned, normalized text strings.
    """

    cleaned = [normalize_text(text, abbr_dict) for text in texts]
    return cleaned


# ---------------------------------------------------------------------------
# STEP 4: ENCODE LABELS
# ---------------------------------------------------------------------------

def encode_labels(labels: list) -> tuple:
    """
    Convert string intent labels to numeric values using LabelEncoder.

    Args:
        labels: List of string labels (e.g., ["booking_request", "greeting"]).

    Returns:
        Tuple of (encoded_labels as numpy array, fitted LabelEncoder).
    """

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)

    return encoded, encoder


# ---------------------------------------------------------------------------
# STEP 5: TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------

def split_data(X: list, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into training and test sets with stratification.

    Stratification ensures each intent is proportionally represented
    in both splits — critical when some intents have only 11 examples.

    Args:
        X:            List of preprocessed text strings.
        y:            Numpy array of encoded labels.
        test_size:    Fraction reserved for testing (default: 20%).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# STEP 6: TF-IDF VECTORIZATION
# ---------------------------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """
    Create a configured TF-IDF vectorizer for short logistics queries.

    Configuration rationale:
        - max_features=1500:  Headroom for domain vocabulary growth.
        - ngram_range=(1,2):  Captures bigrams like "payment failed", "book lorry".
        - min_df=1:           Retain all terms — dataset is small (132 samples),
                              so even single-occurrence words carry intent signal.
                              Increase to 2 when dataset exceeds 500 samples.
        - sublinear_tf=True:  Dampens the impact of repeated terms within a query.

    Returns:
        Configured (unfitted) TfidfVectorizer instance.
    """

    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    return vectorizer


# ---------------------------------------------------------------------------
# STEP 7: MODEL TRAINING
# ---------------------------------------------------------------------------

def build_model() -> LogisticRegression:
    """
    Create a configured Logistic Regression classifier.

    Configuration rationale:
        - max_iter=1000:        Ensures convergence on small datasets.
        - solver="lbfgs":       Efficient for small-to-medium datasets.
        - C=1.0:                Default regularization — good baseline.

    Returns:
        Configured (unfitted) LogisticRegression instance.
    """

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42
    )

    return model


# ---------------------------------------------------------------------------
# STEP 8: EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, encoder: LabelEncoder) -> float:
    """
    Evaluate the trained model on the test set and print metrics.

    Prints:
        - Overall accuracy
        - Per-intent precision, recall, F1 (classification report)
        - Confusion matrix

    Args:
        model:   Trained LogisticRegression model.
        X_test:  TF-IDF transformed test features.
        y_test:  Encoded test labels.
        encoder: Fitted LabelEncoder for decoding label names.

    Returns:
        Accuracy score as a float.
    """

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Decode label names for readable output
    label_names = encoder.classes_

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)\n")

    print("-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    print("-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return accuracy


# ---------------------------------------------------------------------------
# STEP 9: SAVE ARTIFACTS
# ---------------------------------------------------------------------------

def save_artifacts(model, vectorizer, encoder) -> None:
    """
    Save all trained artifacts to the models/ directory using joblib.

    Artifacts saved:
        - intent_classifier.pkl   (trained Logistic Regression model)
        - tfidf_vectorizer.pkl    (fitted TF-IDF vectorizer)
        - label_encoder.pkl       (fitted LabelEncoder)

    Args:
        model:      Trained LogisticRegression.
        vectorizer: Fitted TfidfVectorizer.
        encoder:    Fitted LabelEncoder.
    """

    # Create models/ directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print(f"\n{'=' * 60}")
    print("ARTIFACTS SAVED")
    print(f"{'=' * 60}")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Vectorizer: {VECTORIZER_PATH}")
    print(f"  Encoder:    {ENCODER_PATH}")


# ---------------------------------------------------------------------------
# MAIN: FULL TRAINING PIPELINE
# ---------------------------------------------------------------------------

def main():
    """
    Execute the complete training pipeline:
        Load → Preprocess → Encode → Split → Vectorize → Train → Evaluate → Save
    """

    print("=" * 60)
    print("INTENT CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # ---- Step 1: Load dataset ----
    print("\n[1/9] Loading dataset...")
    texts, labels = load_dataset(INTENTS_FILE)
    print(f"      Loaded {len(texts)} examples across {len(set(labels))} intents.")

    # ---- Step 2: Load abbreviations ----
    print("[2/9] Loading abbreviation mappings...")
    abbr_dict = load_abbreviations(ABBREVIATIONS_FILE)
    print(f"      Loaded {len(abbr_dict)} abbreviation mappings.")

    # ---- Step 3: Preprocess ----
    print("[3/9] Preprocessing texts...")
    cleaned_texts = preprocess_dataset(texts, abbr_dict)
    print(f"      Sample: \"{texts[0]}\"")
    print(f"           → \"{cleaned_texts[0]}\"")

    # ---- Step 4: Encode labels ----
    print("[4/9] Encoding labels...")
    y_encoded, encoder = encode_labels(labels)
    print(f"      Labels: {list(encoder.classes_)}")

    # ---- Step 5: Train/test split ----
    print("[5/9] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = split_data(cleaned_texts, y_encoded)
    print(f"      Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ---- Step 6: TF-IDF vectorization ----
    print("[6/9] Vectorizing with TF-IDF...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"      Feature matrix shape: {X_train_tfidf.shape}")
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")

    # ---- Step 7: Model training ----
    print("[7/9] Training Logistic Regression...")
    model = build_model()
    model.fit(X_train_tfidf, y_train)
    print("      Training complete.")

    # ---- Step 8: Evaluation ----
    print("[8/9] Evaluating on test set...")
    accuracy = evaluate_model(model, X_test_tfidf, y_test, encoder)

    # ---- Step 9: Save artifacts ----
    print("\n[9/9] Saving model artifacts...")
    save_artifacts(model, vectorizer, encoder)

    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — Final accuracy: {accuracy * 100:.1f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
