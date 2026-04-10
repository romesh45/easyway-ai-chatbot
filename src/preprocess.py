"""
Preprocessing module for the NLP Intent Classification System.

Handles text cleaning, abbreviation expansion, normalization, and tokenization.
Designed to be used in both training and inference pipelines to prevent
train/serve skew.

Author: EasyWay Logistics AI Team
"""

import re
import json
import os


# ---------------------------------------------------------------------------
# 1. TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Perform base-level text cleaning on raw user input.

    Steps applied (in order):
        1. Guard against None / non-string input
        2. Convert to lowercase
        3. Replace digit sequences with <NUM> token
        4. Strip punctuation (keep alphanumeric, spaces, and <NUM> token)
        5. Collapse multiple whitespace into a single space
        6. Strip leading/trailing whitespace

    Args:
        text: Raw user input string.

    Returns:
        Cleaned string ready for further processing.
    """

    # Guard: return empty string for None or non-string input
    if not text or not isinstance(text, str):
        return ""

    # Lowercase everything — "URGENT Book Lorry" → "urgent book lorry"
    text = text.lower()

    # Replace digit sequences with a safe placeholder before punctuation
    # removal. We use __num__ (all lowercase, no special chars) so it
    # survives the regex below. Restored to <NUM> at the end.
    # Example: "order id 88721" → "order id __num__"
    text = re.sub(r'\d+', ' __num__ ', text)

    # Remove punctuation — keep only lowercase letters, underscores, and spaces
    text = re.sub(r'[^a-z_\s]', '', text)

    # Collapse multiple whitespace into a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Restore the placeholder to the final <NUM> token
    text = text.replace('__num__', '<NUM>')

    return text


# ---------------------------------------------------------------------------
# 2. ABBREVIATION EXPANSION
# ---------------------------------------------------------------------------

def expand_abbreviations(text: str, abbr_dict: dict) -> str:
    """
    Replace slang and abbreviations with their full forms.

    Uses word-boundary matching to avoid partial replacements
    (e.g., "urgent" should NOT become "yourgent" from the "u" → "you" rule).

    Args:
        text:      Cleaned text string.
        abbr_dict: Dictionary mapping abbreviations to expansions.
                   Example: {"tmrw": "tomorrow", "pls": "please"}

    Returns:
        Text with all known abbreviations expanded.
    """

    # Guard: return as-is if inputs are invalid
    if not text or not isinstance(text, str):
        return ""
    if not abbr_dict or not isinstance(abbr_dict, dict):
        return text

    # Replace each abbreviation using word-boundary regex.
    # We sort by length (longest first) to prevent shorter abbreviations
    # from matching substrings of longer ones.
    for abbr in sorted(abbr_dict.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, abbr_dict[abbr], text)

    return text


# ---------------------------------------------------------------------------
# 3. TEXT NORMALIZATION (full pipeline)
# ---------------------------------------------------------------------------

def normalize_text(text: str, abbr_dict: dict) -> str:
    """
    Full preprocessing pipeline: clean → expand abbreviations.

    This is the single entry point used by both `train.py` and `predict.py`
    to ensure identical preprocessing in training and inference.

    Args:
        text:      Raw user input string.
        abbr_dict: Abbreviation expansion dictionary.

    Returns:
        Fully normalized text ready for vectorization.
    """

    text = clean_text(text)
    text = expand_abbreviations(text, abbr_dict)

    return text


# ---------------------------------------------------------------------------
# 4. TOKENIZATION
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """
    Split text into a list of word tokens using whitespace.

    Designed for short logistics queries (4-8 words on average).
    Complex tokenizers add overhead with no accuracy benefit for this
    domain, so simple whitespace split is the right choice.

    Args:
        text: Normalized text string.

    Returns:
        List of word tokens. Empty list if input is invalid.
    """

    if not text or not isinstance(text, str):
        return []

    return text.split()


# ---------------------------------------------------------------------------
# 5. UTILITY: LOAD ABBREVIATIONS FROM FILE
# ---------------------------------------------------------------------------

def load_abbreviations(filepath: str = None) -> dict:
    """
    Load the abbreviation mapping dictionary from a JSON file.

    Args:
        filepath: Path to abbreviations.json.
                  Defaults to data/abbreviations.json relative to project root.

    Returns:
        Dictionary of abbreviation → expansion mappings.

    Raises:
        FileNotFoundError: If the abbreviations file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """

    if filepath is None:
        # Resolve path relative to project root (two levels up from src/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(project_root, "data", "abbreviations.json")

    with open(filepath, "r", encoding="utf-8") as f:
        abbr_dict = json.load(f)

    return abbr_dict


# ---------------------------------------------------------------------------
# SELF-TEST: Quick validation when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Load abbreviations
    abbr = load_abbreviations()
    print(f"Loaded {len(abbr)} abbreviation mappings.\n")

    # Test cases — realistic logistics queries
    test_inputs = [
        "Need Lorry ASAP for 500kg load tmrw!!!",
        "truck avl tmrw morning?",
        "pls check my order id 88721 delivery status",
        "rate kya hai mumbai to pune lorry ka",
        "   ",
        "",
        None,
        "bro can u arrange one tempo for evening?",
        "PAYMENT DEDUCTED TWICE PLS CHECK & REFUND 2500rs",
    ]

    print(f"{'Input':<55} → {'Normalized Output'}")
    print("-" * 110)

    for raw in test_inputs:
        display = repr(raw) if raw is None or (isinstance(raw, str) and raw.strip() == "") else raw
        normalized = normalize_text(raw, abbr)
        tokens = tokenize(normalized)
        print(f"{str(display):<55} → {normalized}")
        print(f"{'':55}   tokens: {tokens}\n")
