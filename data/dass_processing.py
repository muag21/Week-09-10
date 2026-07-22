# ============================================================
#  MindPulse — DASS-21 Questionnaire Processing
#  Author  : Puja (ML Engineer)
#  File    : data/dass_processing.py
#  Purpose : Score DASS-21 responses and assign clinical labels
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ── DASS-21 Official Item Assignments ────────────────────────
# Each subscale uses 7 specific questions from the 21-item form

DEPRESSION_ITEMS = [3, 5, 10, 13, 16, 17, 21]
ANXIETY_ITEMS    = [2, 4,  7,  9, 15, 19, 20]
STRESS_ITEMS     = [1, 6,  8, 11, 12, 14, 18]

# ── Clinical Thresholds (moderate severity) ──────────────────
# Source: Lovibond & Lovibond (1995) — official DASS-21 manual
THRESHOLDS = {
    "depression" : 14,   # Score >= 14 → depressed
    "anxiety"    : 10,   # Score >= 10 → anxious
    "stress"     : 19,   # Score >= 19 → stressed
}

MAX_SCORE = 42   # Maximum possible for each subscale (7 items × 3 × 2)

# Label mapping for the model
LABEL_MAP = {
    "control"    : 0,
    "depression" : 1,
    "anxiety"    : 2,
    "stress"     : 3,
}


def compute_subscale_scores(answers: Dict[str, int]) -> Dict[str, int]:
    """
    Compute the three DASS-21 subscale scores from raw answers.

    How scoring works:
        Each item is answered 0, 1, 2, or 3.
        The subscale score = sum of its 7 items, multiplied by 2.
        Multiplying by 2 scales from the 21-item version to match
        the original 42-item DASS norms (clinical standard).

    Args:
        answers: Dictionary like {"q1": 2, "q2": 0, ..., "q21": 1}
                 Values must be integers 0, 1, 2, or 3.

    Returns:
        Dictionary with keys "depression", "anxiety", "stress"
        and integer score values (0 to 42 each).

    Example:
        answers = {f"q{i}": 1 for i in range(1, 22)}
        scores = compute_subscale_scores(answers)
        # → {"depression": 14, "anxiety": 14, "stress": 14}
    """
    depression_score = sum(
        answers.get(f"q{i}", 0) for i in DEPRESSION_ITEMS
    ) * 2

    anxiety_score = sum(
        answers.get(f"q{i}", 0) for i in ANXIETY_ITEMS
    ) * 2

    stress_score = sum(
        answers.get(f"q{i}", 0) for i in STRESS_ITEMS
    ) * 2

    return {
        "depression" : depression_score,
        "anxiety"    : anxiety_score,
        "stress"     : stress_score,
    }


# REVIEW [Logic/readability]: Docstring vs implementation mismatch: the docstring below says
# "Confidence = how far above threshold / max possible" (i.e. (score - threshold) / (max -
# threshold)), but the code actually computes `scores[label] / MAX_SCORE`, i.e. the raw score
# over 42, which is a different number. That's not necessarily the wrong choice, but right now
# the comment and the code tell two different stories about what "confidence" means clinically,
# so it's worth fixing whichever one doesn't reflect the intended meaning so the next person
# (or your dissertation write-up) doesn't have to guess.
#
# REVIEW [Logic/readability]: This scoring/classification logic is duplicated with a slight
# behavioural difference in app/streamlit_app.py's classify_dass() and again in
# mindpulse_complete.py's classify_dass_label(). The "control" branch already disagrees between
# copies: this function computes a real confidence from the max score, while
# streamlit_app.py's classify_dass() just hardcodes confidence=0.85 for control. Consider
# having the app import this module instead of re-implementing the same thresholds, so the
# three copies can't quietly drift further apart.
def classify_label(scores: Dict[str, int]) -> Tuple[str, float]:
    """
    Apply clinical thresholds to assign a label and confidence.

    Logic:
        1. Check which subscales are above their threshold
        2. If multiple are elevated, pick the highest one
        3. If none are elevated, label is "control"
        4. Confidence = how far above threshold / max possible

    Args:
        scores: Output of compute_subscale_scores()

    Returns:
        label      : "depression", "anxiety", "stress", or "control"
        confidence : Float 0.0 to 1.0

    Example:
        scores = {"depression": 18, "anxiety": 8, "stress": 12}
        label, conf = classify_label(scores)
        # → ("depression", 0.43)
        # depression is the only one above its threshold (14)
    """
    elevated = {
        name: score
        for name, score in scores.items()
        if score >= THRESHOLDS[name]
    }

    if not elevated:
        # All subscales below threshold — person is in normal range
        max_score = max(scores.values())
        confidence = 1.0 - (max_score / MAX_SCORE)
        return "control", round(confidence, 3)

    # Pick the most elevated subscale
    label = max(elevated, key=elevated.get)
    confidence = min(scores[label] / MAX_SCORE, 1.0)
    return label, round(confidence, 3)


def normalise_scores(scores: Dict[str, int]) -> Dict[str, float]:
    """
    Normalise subscale scores to the range [0.0, 1.0].

    Neural networks train better when inputs are in a small,
    consistent range. Dividing by MAX_SCORE (42) achieves this.

    Args:
        scores: Raw subscale scores (0–42 each)

    Returns:
        Same keys but float values between 0.0 and 1.0
    """
    return {k: round(v / MAX_SCORE, 4) for k, v in scores.items()}


def build_feature_vector(answers: Dict[str, int]) -> np.ndarray:
    """
    Convert a full set of DASS-21 answers into a feature vector
    for the neural network.

    Feature vector contents (24 features total):
        - 21 raw item scores (one per question)
        - 3 normalised subscale scores (depression, anxiety, stress)

    Args:
        answers: {"q1": 0..3, ..., "q21": 0..3}

    Returns:
        numpy array of shape (24,)

    Example:
        answers = {f"q{i}": 1 for i in range(1, 22)}
        vec = build_feature_vector(answers)
        # vec.shape → (24,)
    """
    # 21 raw item scores in order
    raw_items = np.array([
        float(answers.get(f"q{i}", 0)) for i in range(1, 22)
    ])

    # 3 normalised subscale scores
    scores = compute_subscale_scores(answers)
    norm   = normalise_scores(scores)
    subscales = np.array([
        norm["depression"],
        norm["anxiety"],
        norm["stress"],
    ])

    return np.concatenate([raw_items, subscales])


# REVIEW [Security/performance]: No validation that df["q1".."q21"] actually contain integers
# 0-3 before scoring: `int(row.get(...))` will raise an uncaught ValueError on a malformed
# CSV (e.g. a blank cell or a stray string), and out-of-range values (e.g. 9) would silently
# produce an invalid "elevated" score instead of being rejected. Worth clamping/validating
# input range given this feeds a clinical-style classification.
#
# REVIEW [Coding standards/performance]: Iterates row-by-row with df.iterrows(), which is a
# well-known pandas anti-pattern for anything beyond questionnaire-sized batches (each row
# access with iterrows() is far slower than vectorised operations). Fine for a handful of
# responses; if this is ever run over a bulk dataset, a vectorised version (matrix-multiply
# the answer columns against a one-hot item-to-subscale mapping) would scale much better.
def process_dass_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a full DASS-21 dataset DataFrame.

    Expects columns q1 through q21 (integer 0–3).
    Adds computed columns: depression_score, anxiety_score,
    stress_score, label, confidence, label_int.

    Args:
        df: DataFrame with columns q1..q21

    Returns:
        Same DataFrame with new computed columns added

    Example:
        df = pd.read_csv("dass21_dataset.csv")
        df = process_dass_dataframe(df)
        print(df["label"].value_counts())
    """
    results = []
    for _, row in df.iterrows():
        answers = {f"q{i}": int(row.get(f"q{i}", 0)) for i in range(1, 22)}
        scores  = compute_subscale_scores(answers)
        label, conf = classify_label(scores)

        results.append({
            **scores,
            "label"      : label,
            "confidence" : conf,
            "label_int"  : LABEL_MAP.get(label, 0),
        })

    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Testing DASS-21 Processing")
    print("=" * 50)

    # Test 1: High anxiety scenario
    anxious_answers = {f"q{i}": 0 for i in range(1, 22)}
    for q in ANXIETY_ITEMS:
        anxious_answers[f"q{q}"] = 2   # All anxiety items = 2
    scores = compute_subscale_scores(anxious_answers)
    label, conf = classify_label(scores)
    print(f"\nTest 1 — High anxiety:")
    print(f"  Scores : {scores}")
    print(f"  Label  : {label} ({conf*100:.0f}% confidence)")
    assert label == "anxiety", "Test 1 FAILED"
    print("  PASSED")

    # Test 2: All zeros = control
    zero_answers = {f"q{i}": 0 for i in range(1, 22)}
    scores2 = compute_subscale_scores(zero_answers)
    label2, _ = classify_label(scores2)
    print(f"\nTest 2 — All zeros (control):")
    print(f"  Scores : {scores2}")
    print(f"  Label  : {label2}")
    assert label2 == "control", "Test 2 FAILED"
    print("  PASSED")

    # Test 3: Feature vector shape
    vec = build_feature_vector(anxious_answers)
    print(f"\nTest 3 — Feature vector shape: {vec.shape}")
    assert vec.shape == (24,), "Test 3 FAILED"
    print("  PASSED")

    print("\nAll DASS-21 tests passed!")
