# Fraud Detection API — Production Deployment Guide (Replit)

## Table of Contents
1. [Required Environment Variables](#1-required-environment-variables)
2. [One-Time Setup](#2-one-time-setup)
3. [Database Migration](#3-database-migration)
4. [Seed the Database](#4-seed-the-database)
5. [Train the Models](#5-train-the-models)
6. [Start the Server](#6-start-the-server)
7. [Health Check](#7-health-check)
8. [Access the Dashboard](#8-access-the-dashboard)
9. [Calling the API](#9-calling-the-api)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Required Environment Variables

Set each of these in **Replit → Secrets** (the padlock icon in the sidebar).  
Never put secrets in `.env` on Replit — use Replit Secrets instead.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | Yes | `sqlite:///./fraud_detection.db` | SQLAlchemy connection string. Use the SQLite default for single-node Replit or a PostgreSQL URL for persistent/multi-worker deployments. |
| `API_KEY` | Yes (production) | *(blank = auth disabled)* | X-API-Key header value required on all POST endpoints. Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `MODEL_PATH` | No | `./app/ml/artifacts/` | Directory containing `.pkl` model files written by `train.py`. |
| `FRAUD_THRESHOLD` | No | `0.5` | Ensemble score cutoff (0–1). Transactions scoring above this are flagged fraud. Tune with `/api/v1/backtest`. |
| `ANOMALY_CONTAMINATION` | No | `0.01` | Isolation Forest expected-outlier fraction. Typical production fraud rate is 0.001–0.005. |
| `LOG_LEVEL` | No | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `ALLOWED_ORIGINS` | No | localhost ports | Comma-separated browser origins allowed by CORS. Set to your Replit app domain in production, e.g. `https://your-app.replit.app` |
| `PORT` | No | `8000` | Replit sets this automatically. Only override manually in non-Replit environments. |

---

## 2. One-Time Setup

```bash
# From the fraud-detection-api/ directory:

# Install dependencies
pip install -r requirements.txt

# Verify imports are clean
python -c "from app.main import app; print('imports ok')"
```

---

## 3. Database Migration

Run the Alembic upgrade every time you deploy. It is a no-op if the schema is already current.

```bash
# From fraud-detection-api/
DATABASE_URL="sqlite:///./fraud_detection.db" alembic upgrade head
```

**PostgreSQL:**
```bash
DATABASE_URL="postgresql://user:pass@host:5432/fraud_detection" alembic upgrade head
```

What this creates:
- `transactions` — prediction log with indexes on `created_at` and `user_id`
- `model_versions` — trained model registry
- `drift_snapshots` — PSI feature drift history
- `backtest_runs` — threshold sweep results

---

## 4. Seed the Database

The seed script inserts 1 000 synthetic transactions (10 labelled as fraud) and one active `model_versions` row. Safe to run multiple times — it skips rows that already exist.

```bash
# From fraud-detection-api/
DATABASE_URL="sqlite:///./fraud_detection.db" python scripts/seed.py
```

Expected output:
```
[seed] DATABASE_URL=sqlite:///./fraud_detection.db
[seed] Alembic migrations applied.
[seed] Created model version: <uuid> (tag=v1.0-seed)
[seed] Done. inserted=1000 skipped=0 fraud=10 legit=990
```

---

## 5. Train the Models

Training requires the Kaggle Credit Card Fraud dataset (`data/creditcard.csv`).  
Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

```bash
# From fraud-detection-api/
python -m app.ml.train
```

This writes to `app/ml/artifacts/`:
- `xgboost_model.pkl`
- `isolation_forest.pkl`
- `scaler.pkl`
- `train_distributions.json`
- `metrics.json`

Training takes ~2–5 minutes on a standard CPU. The server must be restarted after training so the lifespan hook reloads the new model files.

> **If you don't have the dataset:** The API starts without models (returns 503 on predict). Use the seed script to populate the DB, then call the monitoring and backtest endpoints. Models can be trained later.

---

## 6. Start the Server

### On Replit (Deployments)

Set the **Run command** in your Replit deployment settings to:

```
cd fraud-detection-api && uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
```

> Use `--workers 1` with SQLite. Switch to `--workers 2` or more only with PostgreSQL.

### Local / manual start

```bash
cd fraud-detection-api
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 7. Health Check

The health endpoint requires no API key and is suitable for Replit's deployment health probe.

```bash
curl https://your-app.replit.app/api/health
```

Expected response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "db_connected": true,
  "timestamp": 1746230400.123
}
```

| Field | Meaning |
|---|---|
| `status` | Always `"ok"` if the server is reachable |
| `models_loaded` | `true` once `train.py` has been run and models are on disk |
| `db_connected` | `true` if the database responded to a test query |
| `timestamp` | Unix epoch seconds (UTC) |

---

## 8. Access the Dashboard

The dashboard is a read-only Jinja2 page — no API key required.

```
https://your-app.replit.app/dashboard
```

It auto-refreshes every 30 seconds and shows:
- Today's transaction volume and fraud rate
- Average fraud score
- Hourly volume chart (fraud vs legit)
- Precision-Recall chart
- Feature drift table (PSI scores)

---

## 9. Calling the API

All `POST` endpoints require the `X-API-Key` header.

### Predict

```bash
curl -X POST https://your-app.replit.app/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "transaction_id": "txn-abc-001",
    "amount": 149.99,
    "hour_of_day": 14,
    "day_of_week": 2
  }'
```

### Run a backtest

```bash
curl -X POST https://your-app.replit.app/api/v1/backtest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"dataset_label": "production-2026-05"}'
```

### Trigger drift snapshot

```bash
curl -X POST https://your-app.replit.app/api/v1/monitoring/drift-snapshot \
  -H "X-API-Key: YOUR_API_KEY"
```

### Read monitoring data (no key needed)

```bash
curl https://your-app.replit.app/api/v1/monitoring/summary
curl https://your-app.replit.app/api/v1/monitoring/volume
curl https://your-app.replit.app/api/v1/monitoring/drift-history
curl https://your-app.replit.app/api/v1/backtest/history
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `{"models_loaded": false}` | `train.py` not run yet | Run `python -m app.ml.train` and restart the server |
| `503 model_unavailable` | Same as above | Same fix |
| `401 unauthorized` | Missing or wrong `X-API-Key` | Set `API_KEY` in Replit Secrets and pass it in the header |
| Dashboard shows no data | DB empty | Run `python scripts/seed.py` |
| `alembic.util.exc.CommandError` | Schema already at head | Normal — no action needed |
| `sqlite3.OperationalError: database is locked` | Multiple writers on SQLite | Use `--workers 1` or switch to PostgreSQL |
| CORS error in browser | `ALLOWED_ORIGINS` not set | Add your domain to `ALLOWED_ORIGINS` in Replit Secrets |
