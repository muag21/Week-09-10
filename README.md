# MindPulse — AI Mental Wellbeing System

An AI-powered system that detects, monitors, and predicts mental health conditions
(depression, anxiety, stress) using NLP and physiological signals.

## Team
| Name | Role |
|------|------|
| Sahar | Group Leader |
| Ahmad (Osama) | Deputy Leader — Database & Backend |
| Puja | ML Engineer — LSTM & Hybrid Model |
| Hanzla | NLP Specialist & QA Lead — BERT |
| Usama | App Developer — Streamlit |
| Irfan | Ethics & Bias Lead |

## Datasets
- SMHD — Reddit posts (self-reported mental health)
- DASS-21 — Questionnaire responses
- WESAD — Physiological signals (ECG, EDA, temperature)
- Custom Simulated Dataset

## Models
1. LSTM Neural Network (text classification)
2. Fine-tuned BERT (NLP understanding)
3. Hybrid Model (NLP + physiological signal fusion)

## Tech Stack
- Python 3.10
- TensorFlow / Keras — LSTM
- HuggingFace Transformers — BERT
- Streamlit — Web application
- SQLAlchemy + SQLite — Database
- Pytest — Testing

## Project Structure
```
mindpulse/
├── data/
│   ├── text_preprocessing.py      # NLP cleaning pipeline (Hanzla)
│   ├── signal_preprocessing.py    # WESAD signal processing (Puja)
│   └── dass_processing.py         # DASS-21 scoring (Puja)
├── models/
│   └── lstm_model.py              # LSTM architecture (Puja)
├── app/
│   ├── streamlit_app.py           # Main web app (Usama)
│   └── database.py                # DB models & CRUD (Ahmad)
<!-- REVIEW [Coding standards]: app/database.py does not exist in the repo, the
     User/Session/Prediction models and CRUD Database class currently only live inside
     mindpulse_complete.py. Either add the standalone file this diagram promises (and wire
     app/streamlit_app.py up to it, see the review comment on page_history() in that file),
     or update this diagram so a new contributor navigating by the README isn't sent
     looking for a file that isn't there. -->

<!-- REVIEW [Process note]: No test plan, coding standards, or pipeline workflow document was
     included in this repo alongside the code (only this README), so this review was carried
     out against general software engineering best practice rather than against the reviewed
     team's own stated criteria. If a test plan exists elsewhere, linking it here would let
     future reviewers check the test suite against what was actually planned to be covered
     (e.g. does it match the "30+ tests" claim below, and are the failing tests noted in
     tests/test_preprocessing.py part of a documented scope) rather than reviewers guessing
     at intent from the code alone. -->
├── tests/
│   └── test_preprocessing.py      # Unit tests (Hanzla)
└── README.md
```

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOURUSERNAME/mindpulse-ai.git
cd mindpulse-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/streamlit_app.py

# 4. Run tests
pytest tests/ -v
```

## Progress Status
- [x] Project planning and architecture design
- [x] Text preprocessing pipeline (SMHD dataset)
- [x] Physiological signal preprocessing (WESAD dataset)
- [x] DASS-21 scoring and feature engineering
- [x] LSTM model architecture defined
- [x] Database schema and CRUD operations
- [x] Streamlit app skeleton with all pages
- [x] Unit test suite (30+ tests)
- [ ] BERT fine-tuning (in progress)
- [ ] Hybrid model training (in progress)
- [ ] Model integration with app (upcoming)
- [ ] Deployment (upcoming)

## Status: Phase 2 — Data Preparation & Model Development
