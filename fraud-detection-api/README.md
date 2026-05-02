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

## Environment Variables

See `.env.example` for all required variables with descriptions.
