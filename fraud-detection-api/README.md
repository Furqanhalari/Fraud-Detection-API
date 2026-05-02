# Fraud Detection API

A FastAPI-based fraud detection service with ML inference, backtesting, drift monitoring, and a built-in dashboard.

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI (Python 3.11) |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| Database | SQLite (dev) via SQLAlchemy + Alembic |
| Monitoring | Custom PSI / feature drift stored in DB |
| Dashboard | Jinja2 templates + Chart.js |

## Project Structure

```
app/
  api/routes/         # prediction, backtest, monitoring endpoints
  core/               # config (env vars) and DB session setup
  models/             # Pydantic schemas and SQLAlchemy table definitions
  ml/                 # training pipeline, inference, feature engineering, ensemble
  monitoring/         # drift tracking (PSI + feature drift)
  dashboard/          # Jinja2 templates and static assets
  main.py             # FastAPI entry point
alembic.ini           # Alembic migration config
requirements.txt      # Pinned dependencies
.env.example          # Required environment variables (copy to .env)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Health check: `GET /api/health`

## Training the Models

1. Download the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`.
2. Run the training pipeline:

```bash
cd fraud-detection-api
python -m app.ml.train
```

This will:
- Engineer features (cyclical time encoding, RobustScaler normalization)
- Train XGBoost (300 estimators, class-imbalance corrected via `scale_pos_weight`)
- Train Isolation Forest (unsupervised, full dataset)
- Print a full classification report
- Save artifacts to `app/ml/artifacts/`:
  - `scaler.pkl`
  - `xgboost_model.pkl`
  - `isolation_forest.pkl`
  - `metrics.json`

Override the dataset path: `CREDITCARD_CSV=/path/to/file.csv python -m app.ml.train`

## Ensemble Scoring

| Component | Weight | Method |
|---|---|---|
| XGBoost | 70% | Supervised fraud probability (0–1) |
| Isolation Forest | 30% | Anomaly flag converted to 0/1 |

`final_score = 0.7 × xgb_prob + 0.3 × iso_binary`  
`is_fraud = final_score ≥ FRAUD_THRESHOLD` (default: 0.5)

## Environment Variables

See `.env.example` for all required variables with descriptions.
