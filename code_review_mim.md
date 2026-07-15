# Code Review — MindPulse (Week 09–10 EDA)

**Reviewed:** `data/`, `models/`, `app/streamlit_app.py`, `mindpulse_complete.py`, `tests/`
**Status at time of review:** Phase 2 — Data Preparation & Model Development (per README)

Overall this is a solid piece of team work for where the project is — the preprocessing pipelines are genuinely well thought out (the overlapping signal windows so you don't cut a stress event in half is a nice touch), the docstrings are some of the clearest I've seen in a student project, and the test suite actually tests behaviour rather than just "does it run." That said, there are a handful of things worth fixing before this goes further, and one that I'd treat as a must-fix given the subject matter of the app.

---

## 1. The single biggest thing: `mindpulse_complete.py` duplicates the entire codebase

This 1,800-line file re-implements everything that already exists in `data/`, `models/`, and `app/` — text preprocessing, DASS scoring, signal processing, the LSTM, the BERT fine-tuning loop, the database models, and the whole Streamlit app — all pasted into one script with its own copies of the constants (`THRESHOLDS`, `DEPRESSION_ITEMS`, `WESAD_SAMPLING_RATES`, etc.).

This is the classic "combine everything into one submittable file" move, and I get why it happens, but it's a real maintenance risk going forward:
- Any bug fix or tweak now has to be made in two places, and it's easy to fix one and forget the other (see point 3 below — the DASS confidence logic already exists in three slightly-diverging copies: `dass_processing.py`, `streamlit_app.py`, and `mindpulse_complete.py`).
- It roughly doubles the surface area a reviewer or new teammate has to read to understand the system.

If you need a single-file version for submission, consider generating it with a small script that concatenates the module files at build time, rather than hand-maintaining a second copy.

## 2. Password hashing is home-rolled and not suitable for production

In `mindpulse_complete.py`, `Database.hash_password()` does:

```python
salt   = secrets.token_hex(16)
hashed = hashlib.sha256((password + salt).encode()).hexdigest()
```

Salting is good, but plain salted SHA-256 is a fast hash — it's designed to be computed quickly, which is exactly the wrong property for password storage. It's crackable at billions of guesses/second on commodity GPUs. For an app that's collecting mental health data tied to user accounts, this is worth fixing properly rather than shipping as-is: swap in `bcrypt`, `argon2-cid` (via `argon2-cffi`), or Werkzeug's `generate_password_hash`, all of which are deliberately slow and tunable. This is a small code change (a few lines in `hash_password`/`verify_password`) for a meaningful security improvement, and worth doing before any real user data touches this table.

## 3. DASS-21 confidence score doesn't match its own docstring

In `dass_processing.py`, `classify_label()`'s docstring says:

> Confidence = how far above threshold / max possible

But the actual code is:

```python
confidence = min(scores[label] / MAX_SCORE, 1.0)
```

That's *raw score ÷ 42*, not *(score − threshold) ÷ (max − threshold)*. It's not necessarily wrong as a design choice, but right now the comment and the code are telling two different stories, which will confuse whoever touches this next (including future-you). Worth either fixing the docstring to describe what's actually happening, or changing the formula to match the stated intent — whichever reflects what you actually want the confidence number to mean clinically.

Same logic (and same discrepancy) is duplicated in `streamlit_app.py`'s `classify_dass()` and again in `mindpulse_complete.py` — a good candidate to consolidate into one shared function that both the app and the offline processing script import, rather than three copies that can quietly drift apart (the "control" case already has: `dass_processing.py` computes a real confidence from the max score, while `streamlit_app.py` just hardcodes `0.85`).

## 4. LF/HF ratio is being computed on EDA, not just ECG

In `signal_preprocessing.py`, `build_feature_vector()` calls `extract_freq_features()` on both the ECG *and* the EDA window, using the same 0.04–0.15 Hz / 0.15–0.40 Hz bands. LF/HF is a heart-rate-variability concept — it's meaningful when applied to the R-R interval series derived from ECG, because it reflects the balance of sympathetic/parasympathetic nervous activity on heartbeats. Applying the same band split to raw EDA (skin conductance) doesn't have that established interpretation — EDA is generally characterised by tonic level (SCL) and phasic peaks (SCRs) rather than an LF/HF ratio. It'll still run and produce numbers, but it's worth double-checking with whoever's writing up the feature-engineering rationale that this is intentional and not just copy-pasted from the ECG feature extractor, since it currently reads that way.

## 5. `transformers.AdamW` is deprecated

`models/bert_model.py` does:

```python
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
```

`AdamW` was removed from newer `transformers` releases in favour of `torch.optim.AdamW`, which is functionally the same optimizer. With `transformers==4.35.0` pinned in `requirements.txt` this will likely still emit a deprecation warning at best or fail at worst depending on the exact patch version. Simple fix:

```python
from torch.optim import AdamW
```

## 6. A few smaller things

- **`BertTokenizer` vs `AutoTokenizer`** — not a bug, but `AutoTokenizer.from_pretrained(...)` is the more future-proof convention (works if you ever swap `bert-base-uncased` for a different checkpoint without changing the import).
- **`process_dass_dataframe()`** iterates the DataFrame row-by-row with `df.iterrows()`. Fine at questionnaire-sized volumes, but if this ever runs over a large batch dataset, a vectorised version (matrix-multiply the answers against a one-hot item-to-subscale mapping) would be considerably faster.
- **Streamlit `load_models()`** silently swallows *any* exception when loading LSTM/BERT (`except Exception: models["lstm"] = None`). That's reasonable for "model not trained yet," but it'll just as quietly hide a real bug (corrupted file, wrong TF version, out-of-memory) behind the same "not trained yet" state. Worth at least logging the exception (`st.warning(str(e))` behind a debug flag, or a `logging.exception(...)` call) so a genuine failure isn't mistaken for "haven't trained it yet."
- **Text/physio pages return hardcoded placeholder predictions** (`label = "anxiety"; confidence = 0.74`) rather than calling the loaded models — this is clearly flagged in a comment and matches the README's "in progress" status, so not a bug, just flagging so it doesn't get forgotten once BERT/hybrid training finishes.
- **`MAX_LENGTH = 512` for BERT** combined with `BATCH_SIZE = 16` — reasonable defaults, but on Reddit-length posts (SMHD posts are often much shorter than 512 tokens) you may get a lot of wasted padding compute. Worth checking the actual token-length distribution of the SMHD subset and possibly dropping `MAX_LENGTH` to something like 256 for speed, unless there's a reason to expect very long posts.

## What's working well

- `SignalPreprocessor` and `TextPreprocessor` are genuinely nice — clear separation between "clean," "validate," "encode/segment," and "featurize" steps, and the docstrings explain *why*, not just *what* (e.g. the note on keeping the last N tokens of a post because Reddit posts tend to end with the most emotionally relevant content).
- Class imbalance is properly handled in LSTM training via `compute_class_weight("balanced", ...)`, and both LSTM and BERT report macro-F1 alongside accuracy — the right call for a 3-class clinical-ish task where naive accuracy can be misleading.
- Z-score normalisation correctly warns about fitting on training data only to avoid leakage.
- The Streamlit app has a persistent ethics/crisis-line banner on every results page — a genuinely good practice for this kind of tool, and something a lot of student projects in this space forget entirely.
- The test suite (`tests/`) checks actual behavioural properties (window counts, array shapes, no-NaN outputs, feature-key presence) rather than just "does it not throw" — that's a good habit and will catch real regressions.

---

**Summary:** nothing here is a "start over" situation — it's a well-structured Phase 2 codebase with the kind of rough edges you'd expect at this stage. The two things I'd genuinely prioritise before moving further are consolidating the duplicated logic in `mindpulse_complete.py` (so bugs like #3 can't silently diverge across three copies) and swapping out the password hashing before any real user accounts are created against it.
