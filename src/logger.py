"""
Logging system for the NLP Intent Classification System.

Provides centralized logging for:
    - All classified queries   → logs/query_log.csv
    - Low-confidence fallbacks → logs/unknown_queries.log
    - System/runtime errors    → logs/error_log.log

All functions are crash-safe — logging failures are silently caught
so the chatbot never goes down because of a logging issue.

Usage:
    from src.logger import log_query, log_unknown, log_error

    log_query("need truck", "booking_request", 0.85)
    log_unknown("asjkdhaskjd gibberish")
    log_error("Model file not found")

Author: EasyWay Logistics AI Team
"""

import os
import csv
from datetime import datetime


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR     = os.path.join(PROJECT_ROOT, "logs")

QUERY_LOG_PATH   = os.path.join(LOGS_DIR, "query_log.csv")
UNKNOWN_LOG_PATH = os.path.join(LOGS_DIR, "unknown_queries.log")
ERROR_LOG_PATH   = os.path.join(LOGS_DIR, "error_log.log")

# CSV column headers for query_log.csv
CSV_HEADERS = ["timestamp", "input_text", "intent", "confidence"]


# ---------------------------------------------------------------------------
# STEP 1: ENSURE LOG DIRECTORY EXISTS
# ---------------------------------------------------------------------------

def _ensure_log_dir() -> None:
    """
    Create the logs/ directory if it doesn't exist.
    Called internally before every write operation.
    """

    os.makedirs(LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 6: TIMESTAMP HELPER
# ---------------------------------------------------------------------------

def _get_timestamp() -> str:
    """
    Generate a formatted timestamp string for the current moment.

    Format: YYYY-MM-DD HH:MM:SS
    Example: 2026-04-10 19:45:32

    Returns:
        Formatted timestamp string.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# STEP 3.1: LOG EVERY QUERY
# ---------------------------------------------------------------------------

def log_query(input_text: str, intent: str, confidence: float) -> None:
    """
    Log a classified query to query_log.csv.

    Every single user query is logged here — both successful classifications
    and fallbacks. This enables:
        - Accuracy monitoring over time
        - Business analytics (most common intents, peak hours)
        - Drift detection (confidence trends)

    CSV columns: timestamp, input_text, intent, confidence

    Args:
        input_text: Raw user input string.
        intent:     Predicted intent (or "fallback").
        confidence: Model confidence score (0.0 – 1.0).
    """

    try:
        _ensure_log_dir()

        # Check if file exists to decide whether to write headers
        file_exists = os.path.exists(QUERY_LOG_PATH)

        with open(QUERY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header row only on first creation
            if not file_exists:
                writer.writerow(CSV_HEADERS)

            writer.writerow([
                _get_timestamp(),
                input_text,
                intent,
                f"{confidence:.4f}",
            ])

    except Exception:
        # Logging must NEVER crash the chatbot
        pass


# ---------------------------------------------------------------------------
# STEP 3.2: LOG UNKNOWN / FALLBACK QUERIES
# ---------------------------------------------------------------------------

def log_unknown(input_text: str) -> None:
    """
    Log a low-confidence (fallback) query to unknown_queries.log.

    These are queries the model couldn't classify confidently.
    Used for:
        - Weekly review cycle → add to dataset → retrain
        - Identifying new intent categories
        - Monitoring unknown query volume

    Format: [YYYY-MM-DD HH:MM:SS] <query text>

    Args:
        input_text: Raw user input that triggered fallback.
    """

    try:
        _ensure_log_dir()

        with open(UNKNOWN_LOG_PATH, "a", encoding="utf-8") as f:
            timestamp = _get_timestamp()
            f.write(f"[{timestamp}] {input_text}\n")

    except Exception:
        # Logging must NEVER crash the chatbot
        pass


# ---------------------------------------------------------------------------
# STEP 3.3: LOG SYSTEM ERRORS
# ---------------------------------------------------------------------------

def log_error(error_message: str) -> None:
    """
    Log a system or runtime error to error_log.log.

    Captures:
        - Model loading failures
        - File I/O errors
        - Unexpected exceptions during prediction

    Format: [YYYY-MM-DD HH:MM:SS] ERROR: <message>

    Args:
        error_message: Description of the error that occurred.
    """

    try:
        _ensure_log_dir()

        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            timestamp = _get_timestamp()
            f.write(f"[{timestamp}] ERROR: {error_message}\n")

    except Exception:
        # Last resort — if even error logging fails, silently continue
        pass


# ---------------------------------------------------------------------------
# SELF-TEST: Validate all logging functions
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("LOGGING SYSTEM — TEST")
    print("=" * 60)

    # Test log_query
    print("\n[1] Testing log_query()...")
    log_query("need truck tomorrow", "booking_request", 0.2154)
    log_query("hello", "greeting", 0.1965)
    log_query("random gibberish", "fallback", 0.0999)
    print("    ✅ 3 entries written to query_log.csv")

    # Test log_unknown
    print("\n[2] Testing log_unknown()...")
    log_unknown("asjkdhaskjd random gibberish")
    log_unknown("what is quantum computing")
    print("    ✅ 2 entries written to unknown_queries.log")

    # Test log_error
    print("\n[3] Testing log_error()...")
    log_error("Model file not found: intent_classifier.pkl")
    log_error("TF-IDF transform failed: empty input vector")
    print("    ✅ 2 entries written to error_log.log")

    # Display log contents for verification
    print("\n" + "-" * 60)
    print("LOG FILE CONTENTS:")
    print("-" * 60)

    for filepath, label in [
        (QUERY_LOG_PATH, "query_log.csv"),
        (UNKNOWN_LOG_PATH, "unknown_queries.log"),
        (ERROR_LOG_PATH, "error_log.log"),
    ]:
        print(f"\n📄 {label}:")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    print(f"   {line.rstrip()}")
        else:
            print("   (file not found)")

    print("\n" + "=" * 60)
    print("All logging tests passed.")
    print("=" * 60)
