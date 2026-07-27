# Hiring Recommendation System

FastAPI backend + simple HTML/JS frontend that scores candidates on:
- **Stability** (based on job durations)
- **Long-term retention probability** — computed two ways so you can compare:
  - **Rule-based**: fixed thresholds on average tenure
  - **ML-based**: the trained `RandomForestClassifier` in `long_term_model.pkl`
- **Monetary motivation** (based on salary jump)
- **Cultural fit** (based on skill overlap)
- **Final weighted score** (30/30/20/20)

## Project structure

```
hiring_recommendation_system/
├── Models/
│   ├── cultural_fit.py
│   ├── monetary_motivation.py
│   ├── stability_score.py
│   ├── long_term_prediction.py       # rule-based
│   ├── ml_long_term_predictor.py     # loads long_term_model.pkl
│   └── recommendation_engine.py
├── app.py                            # FastAPI app
├── long_term_model.pkl               # trained model (from train_model.py)
├── requirements.txt
├── frontend/
│   └── index.html                    # standalone HTML/JS UI
└── README.md
```

## 1. Setup

```bash
cd hiring_recommendation_system
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the API

```bash
uvicorn app:app --reload --port 8000
```

- API docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/health

## 3. Run the frontend

Just open `frontend/index.html` directly in a browser (double-click it, or
`open frontend/index.html` / drag into a browser tab). It talks to the API
at `http://localhost:8000` by default — you can change the API URL at the
top of the page if you host the API elsewhere.

No build step needed — it's plain HTML/CSS/JS.

## API reference

### `POST /score`

**Request body:**
```json
{
  "job_durations": [12, 18, 24],
  "companies": 3,
  "experience": 4.5,
  "current_salary": 8,
  "expected_salary": 12,
  "candidate_skills": ["Python", "FastAPI", "PostgreSQL"],
  "company_skills": ["Python", "FastAPI", "Git", "Docker"]
}
```

**Response:** `rule_based` and `ml_based` score breakdowns (stability,
long_term_probability, long_term_recommendation, monetary, culture,
final_score), plus the derived `avg_tenure_months` and `salary_jump_percent`.

## Note on the two long-term engines

The rule-based engine and the ML model can disagree — they were built from
different logic (the model was trained on `employee_data.csv`, which itself
uses somewhat different rules than `long_term_prediction.py`). The API
returns both so you can compare them side by side rather than silently
picking one. If you want a single source of truth going forward, consider
retraining the model on real historical outcome data rather than the
synthetic dataset from `generate_dataset.py`.

## Retraining the model

`train_model.py` (not modified here) regenerates `long_term_model.pkl` from
`employee_data.csv`. Re-run it if you update the dataset, then replace the
`.pkl` file in this project's root.
