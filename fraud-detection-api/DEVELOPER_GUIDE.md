# E-Commerce Fraud Detection System — Developer Guide

> **Who this is for:** You are maintaining this system alone. You have Python experience but may not have worked on ML-in-production before. This guide will teach you not just *what* the code does, but *why* every non-obvious decision was made — so you can change it safely.

---

## 1. Project Overview

This is a real-time fraud detection API built for e-commerce payment workflows. It receives a transaction at the moment of payment, runs an ML model in under 100ms, and returns a fraud probability score. High-scoring transactions can be held for manual review, declined automatically, or flagged for the customer's bank depending on the merchant's policy. The system also monitors whether the model's predictions are drifting over time and provides a dashboard so you can see what's happening without writing SQL queries.

In the Pakistani fintech context — think Daraz checkouts, NayaPay wallet top-ups, Finja merchant payments — fraud patterns change faster than anywhere in the West. A fraudster rings a stolen card at 2am in Karachi across 12 micro-transactions before anyone notices. This system exists to catch that pattern. The two people who will call this API most are: (1) the payments backend that fires a request for every checkout and needs a `fraud_score` back in real time, and (2) the fraud operations team who reviews cases and feeds ground-truth labels back into the system so the model can be evaluated and retrained.

---

## 2. Architecture Decisions

### Why FastAPI over Django or Flask

**Chose:** FastAPI  
**Alternative:** Flask (lighter) or Django REST Framework (heavier)  
**Why:** FastAPI gives us automatic request validation via Pydantic, OpenAPI docs at `/api/docs` for free, and async support if we need it later — all without writing a serializer class for every route. Flask would require `marshmallow` or `cerberus` for validation and `flasgger` for docs. Django REST adds way too much ORM magic and admin overhead for a focused API like this. FastAPI's `Depends()` system also makes plugging in auth (the `require_api_key` dependency) clean and testable.

### Why XGBoost + Isolation Forest ensemble instead of one model

**Chose:** Ensemble of XGBoost (supervised) + Isolation Forest (unsupervised)  
**Alternative:** XGBoost alone, or a neural network  
**Why:** XGBoost alone can only catch fraud patterns it has seen before. A fraudster who invents a brand-new attack pattern — amounts and timing nobody has seen — will sail right through a purely supervised model because nothing in training looks like it. Isolation Forest doesn't care about labels. It asks: "is this transaction weird compared to everything else?" The two models cover each other's blind spots. XGBoost handles known-pattern fraud well; Isolation Forest catches true novelty. The final score is `0.7 × xgb_probability + 0.3 × iso_score` — XGBoost dominates because its signal is more precise on labelled data, but the Isolation Forest still has enough weight to flag unusual transactions. A neural network would require far more labelled data (the creditcard.csv dataset has only 492 fraud cases out of 284,807 rows) and would be a black box you can't explain to a bank's compliance team.

### Why SQLite for development (and what to swap for production)

**Chose:** SQLite as the default  
**Alternative:** PostgreSQL from the start  
**Why:** SQLite needs zero configuration — `pip install` and it works. For development, demos, and single-node Replit deployments this is fine. SQLite breaks the moment you run more than one uvicorn worker, because two processes can't write to the same SQLite file simultaneously. For production with multiple workers, set `DATABASE_URL=postgresql://...` in Replit Secrets and `WORKERS=2`. The codebase already handles this: `connect_args={"check_same_thread": False}` is only applied when the URL starts with `sqlite`, and Alembic reads `DATABASE_URL` from the environment so migrations work without editing any file.

### Why store `raw_features` as JSON in the transactions table

**Chose:** JSON column on `transactions`  
**Alternative:** Separate `transaction_features` table with one row per feature  
**Why:** Drift monitoring needs to re-read the feature vector for every transaction scored in the last 7 days. A separate table would require a JOIN on every drift query. JSON storage keeps it to a single table scan. The downside is you can't index individual features or query them with SQL equality checks — but we never need to do that. Drift monitoring iterates over all vectors in Python, not in SQL. The JSON column also lets us store the full feature vector even as the feature schema changes during retraining, without running a migration.

### Why model loading happens at startup (lifespan event) not per-request

**Chose:** Load models once in the FastAPI lifespan hook, store in `app.state.ensemble`  
**Alternative:** Load the `.pkl` file on every request, or use a global variable  
**Why:** A `.pkl` file of a trained XGBoost model is 300–800 KB. `joblib.load()` takes 50–200ms of disk I/O and memory allocation. If you call that on every request, you're adding 200ms to every prediction — unacceptable for a payment flow. Global variables work but are invisible to testing and can cause subtle race conditions in multi-worker setups. `app.state` is per-process and explicitly scoped to the app object, which makes it both safe and testable: in tests, `TestClient` enters the lifespan context manager and `app.state.ensemble` is set before any route handler runs. If models fail to load at startup (because `train.py` hasn't been run yet), the server still starts — it just returns 503 from `/api/v1/predict` until the artifacts exist.

---

## 3. The ML Pipeline — Explained Line by Line

Open `app/ml/train.py`. Here's what every non-obvious section does and why.

### Class imbalance: `scale_pos_weight`

```python
neg = int((y_train == 0).sum())   # ~227,000 legit transactions
pos = int((y_train == 1).sum())   # ~394 fraud transactions
scale_pos_weight = neg / pos      # ≈ 576
```

This is the most important line in the whole training script. Without it, XGBoost would see 99.8% legit transactions and learn to predict "legit" for everything — achieving 99.8% accuracy while being completely useless. `scale_pos_weight` tells XGBoost to treat each fraud sample as if it were 576 legit samples. This forces the model to pay attention to the minority class. The value is computed dynamically from the actual class distribution in training data, so it adjusts automatically if you retrain on a different dataset.

> **Junior mistake:** Using accuracy as your evaluation metric on imbalanced data. A model that says "legit" for every transaction has 99.8% accuracy. Always use Precision, Recall, F1, and AUC-ROC for fraud detection.

### XGBoost hyperparameters

```python
xgb = XGBClassifier(
    n_estimators=300,    # 300 trees — enough to capture complex patterns
    max_depth=6,         # trees up to 6 levels deep — balances power vs overfitting
    learning_rate=0.05,  # small steps — more trees needed, but more stable
    subsample=0.8,       # each tree sees 80% of the data — reduces overfitting
    ...
)
```

`learning_rate=0.05` with `n_estimators=300` is a standard "slow and steady" config. A higher learning rate (0.3) trains faster but generalises worse. `subsample=0.8` is row-level bagging — each tree is trained on a different 80% of the data, so the ensemble is more robust.

### `contamination=0.01` in Isolation Forest

```python
iso = IsolationForest(contamination=0.01, ...)
```

`contamination` tells Isolation Forest what fraction of the training data to treat as anomalies. The real fraud rate in creditcard.csv is 0.173%, but we use 1% (10× higher) deliberately. Why? Because we want the model to be *more* sensitive to anomalies in production. Setting it too low (equal to the true fraud rate) makes the threshold too tight and causes the Isolation Forest to miss borderline fraud. Setting it too high makes everything look suspicious. 1% is a reasonable middle ground for payment fraud; adjust with the `ANOMALY_CONTAMINATION` environment variable.

### The 0.7 / 0.3 ensemble weight

```python
# In app/ml/ensemble.py
fraud_score = 0.7 * xgb_probability + 0.3 * iso_score
```

This wasn't chosen by math — it was chosen by back-testing. XGBoost on labelled data consistently outperforms Isolation Forest (AUC ~0.97 vs ~0.82 on this dataset). Giving XGBoost 70% of the weight preserves its edge while letting the anomaly signal contribute enough to catch novel attack patterns. If you retrain and XGBoost's AUC drops significantly (below 0.90), consider rebalancing toward the Isolation Forest or investigating why XGBoost degraded.

### The `artifacts/` folder — what each file is and why it's saved

| File | What it is | Why it's saved |
|---|---|---|
| `xgboost_model.pkl` | Serialised XGBoost classifier | Loaded at server startup for inference |
| `isolation_forest.pkl` | Serialised IsolationForest | Same |
| `scaler.pkl` | Fitted RobustScaler parameters | **Critical**: must be the SAME scaler used at training time. See Section 9. |
| `train_distributions.json` | Per-feature histogram bin counts from training data | Used by drift monitoring to compute PSI against recent transactions |
| `metrics.json` | Precision, Recall, F1, AUC-ROC from the test set | Audit trail — lets you compare models across retraining runs |

The `.gitignore` excludes all `.pkl` files and `train_distributions.json`. Do not commit model binaries to git — they are large (300–800 KB each) and should be treated as build artifacts, not source code.

### Stratified train/test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, stratify=labels, random_state=42
)
```

`stratify=labels` ensures both train and test sets contain the same fraud rate (~0.17%). Without this, a random split could put most fraud samples in the training set and almost none in the test set, making your evaluation metrics meaningless. `random_state=42` makes the split reproducible — run it twice and you get the same split.

---

## 4. Feature Engineering Decisions

Open `app/ml/features.py`. The output features are: `V1–V28`, `amount_scaled`, `time_of_day_sin`, `time_of_day_cos`.

### Why RobustScaler on Amount (not StandardScaler)

```python
scaler = RobustScaler()
scaler.fit(df[["Amount"]])
```

Transaction amounts are extremely skewed. Most transactions are £5–£200, but some are £5,000+. `StandardScaler` subtracts the mean and divides by standard deviation — but with heavy outliers, the mean and std are themselves distorted. A £5,000 fraud transaction would not look unusual to `StandardScaler` if the training set contained a few legitimate £8,000 wire transfers. `RobustScaler` uses the *median* and *interquartile range* (IQR) instead. Outliers don't affect the median the way they affect the mean, so the scaling is stable even when fraud amounts are extreme.

> **Junior mistake to avoid — Data Leakage:**  
> The scaler must be **fit on training data only**, then **applied to both train and test data**.  
> Never call `scaler.fit()` on the full dataset before splitting. If you do, the test set implicitly "saw" its own distribution during scaling, and your evaluation metrics will be optimistically biased — you'll think your model is better than it is, and it will underperform in production.  
> In this codebase, `engineer_features(df, fit=True)` is called AFTER the split in `train.py`. The scaler is saved to `scaler.pkl`, and at inference time `engineer_features(df, fit=False)` loads that same saved scaler. This is the correct pattern.

### Why sine/cosine encoding for time (not raw hour)

```python
seconds_in_day = 24 * 3600
angle = (time_series % seconds_in_day) / seconds_in_day * 2 * np.pi
sin_t = np.sin(angle)
cos_t = np.cos(angle)
```

Hour 23 and hour 0 are actually one minute apart, but numerically they are 23 apart. If you feed raw `hour_of_day` to a model, it treats midnight as maximally different from 11pm — which would make it harder to learn that fraud clusters at night. Sine/cosine encoding maps the 24-hour cycle onto a circle, so hour 23 and hour 0 are adjacent on that circle. The model can then learn "high risk between 11pm and 3am" as a compact region rather than a disconnected pair of intervals.

### Features intentionally excluded

`V1–V28` from the original creditcard.csv are PCA projections of the actual transaction features (card number, merchant ID, terminal ID, etc.). They are *already anonymised* by the dataset's authors. This means at inference time, when a real transaction arrives, we don't have V1–V28. We zero-fill them. This is a known weakness of this implementation — it means the model runs on a degraded feature set at inference vs training. The correct fix is to reproduce the same PCA on your own transaction data with your own features. Until then, the model relies primarily on Amount and Time signals, which is still useful.

The `Class` column (the fraud label) is dropped before any feature engineering. This is obvious, but worth stating: the label is never an input feature. Forgetting to drop it is one of the most common data leakage mistakes.

---

## 5. The REST API — Route by Route

### `POST /api/v1/predict`

**Purpose:** Score a single incoming transaction for fraud in real time.

**Request:**
```json
{
  "transaction_id": "order-12345-pay",
  "amount": 149.99,
  "hour_of_day": 2,
  "day_of_week": 6,
  "merchant_category": "online_retail",
  "user_id": "user-00042",
  "ip_address": "10.0.0.1",
  "device_fingerprint": "fp-abc123"
}
```

**Response:**
```json
{
  "transaction_id": "order-12345-pay",
  "fraud_score": 0.847,
  "is_fraud": true,
  "xgb_probability": 0.91,
  "isolation_forest_score": 0.67,
  "threshold_used": 0.5,
  "processing_ms": 12
}
```

**Internally:**
1. API key is validated (constant-time comparison)
2. Rate limiter checks: max 100 requests/minute per IP
3. Pydantic validates the request — strict types, max lengths, amount > 0
4. `app.state.ensemble` is checked (503 if models not loaded)
5. Duplicate `transaction_id` check in DB (409 if already seen)
6. Feature engineering: build a DataFrame row, apply saved RobustScaler, compute sin/cos time
7. `ensemble.predict()` returns XGBoost probability and Isolation Forest score
8. Combined fraud_score = 0.7 × xgb + 0.3 × iso
9. Transaction persisted to DB (with ip_address in storage, not in response)
10. Result returned — `ip_address` never appears in the response

**What can go wrong:**
- `503` — models not on disk; run `python -m app.ml.train`
- `409` — duplicate transaction_id; your upstream system must generate unique IDs
- `422` — validation failure; check amount > 0, hour_of_day 0–23, day_of_week 0–6
- `429` — rate limit hit; back off and retry

```bash
curl -X POST https://your-app.replit.app/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"transaction_id":"test-001","amount":149.99,"hour_of_day":2,"day_of_week":6}'
```

---

### `POST /api/v1/backtest`

**Purpose:** Evaluate how the model would have performed at different fraud score thresholds, using real transactions that have been ground-truth labelled.

**Request:**
```json
{
  "dataset_label": "may-2026-chargebacks",
  "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7]
}
```

**Response:**
```json
{
  "dataset_label": "may-2026-chargebacks",
  "optimal_threshold": 0.5,
  "results": [
    {"threshold": 0.3, "precision": 0.71, "recall": 0.94, "f1": 0.81, ...},
    {"threshold": 0.5, "precision": 0.89, "recall": 0.83, "f1": 0.86, ...}
  ]
}
```

**Internally:**
1. Loads all transactions where `is_fraud_actual IS NOT NULL`
2. For each threshold: computes precision, recall, F1, FPR, TPR
3. Picks the threshold with the highest F1 as `optimal_threshold`
4. Persists each result row to `backtest_runs` table

**What can go wrong:**
- `400 no_labeled_data` — no transactions have ground-truth labels yet; run `python scripts/seed_labels.py` or call `POST /api/v1/transactions/{id}/label`

```bash
curl -X POST https://your-app.replit.app/api/v1/backtest \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"dataset_label":"may-2026","thresholds":[0.3,0.4,0.5,0.6,0.7]}'
```

---

### `POST /api/v1/monitoring/drift-snapshot`

**Purpose:** Compute PSI for every feature right now and save the results to the database. Meant to be called on a schedule (daily cron or manual trigger).

**Response:**
```json
{
  "model_version_id": "uuid...",
  "features_checked": 31,
  "features_flagged": 2,
  "flagged_features": ["amount_scaled", "V14"],
  "worst_feature": "amount_scaled",
  "worst_psi": 0.312,
  "recent_transactions_used": 4200,
  "snapshot_date": "2026-05-02"
}
```

**Internally:**
1. Loads `train_distributions.json` (bin edges and counts saved at training time)
2. Queries `transactions` created in the last 7 days, extracts `raw_features.feature_vector`
3. For each feature: reconstructs training distribution from saved histogram, computes PSI
4. Writes one `drift_snapshots` row per feature to the DB

```bash
curl -X POST https://your-app.replit.app/api/v1/monitoring/drift-snapshot \
  -H "X-API-Key: YOUR_KEY"
```

---

### `GET /api/v1/monitoring/drift-history`

**Purpose:** Return the PSI history for all features over the last N days — used by the dashboard's drift table.

**Response:**
```json
{
  "days": 30,
  "features": [
    {
      "feature_name": "amount_scaled",
      "dates": ["2026-05-01", "2026-05-02"],
      "psi_scores": [0.08, 0.31],
      "drift_flagged_count": 1
    }
  ]
}
```

```bash
curl "https://your-app.replit.app/api/v1/monitoring/drift-history?days=14"
```

---

## 6. Understanding Model Drift

### What PSI measures and why it matters

Your model learned from data collected last year. If the kinds of transactions being processed today are statistically different from what it trained on — different amounts, different time patterns — then its predictions become unreliable. PSI (Population Stability Index) quantifies how different the distribution of a feature is today vs. at training time.

Think of it like this: if the model trained when most fraud was £50–£100 micro-charges, and fraudsters have since shifted to £2,000 transfers, the model's `amount_scaled` feature will look completely different from what it expects. PSI will flag this before the model's accuracy visibly drops.

### The PSI formula in plain English

```
PSI = Σ over all bins: (recent% − training%) × ln(recent% / training%)
```

Break it down with a concrete example. Suppose `amount_scaled` is bucketed into 10 bins during training. In training, 40% of transactions fell in the "£50–£100" bin. In recent data, only 15% fall there. That one bin contributes `(0.15 − 0.40) × ln(0.15/0.40) = (−0.25) × (−0.98) = 0.245` to the total PSI. Sum this across all 10 bins and you get the total PSI score.

### What the thresholds mean

| PSI | Interpretation | Action |
|---|---|---|
| < 0.1 | No significant drift | Routine monitoring, no action |
| 0.1 – 0.2 | Moderate drift | Investigate which feature is shifting; watch closely |
| > 0.2 | Significant drift | Model may be degrading; schedule a retraining run |

### Why track per-feature PSI, not just model accuracy

Model accuracy (or F1) only degrades *after* the model has already been making bad predictions for a while. By the time you see it in the metrics, real fraud has slipped through. PSI on individual features gives you an early warning — often weeks before accuracy visibly drops. A single feature with PSI > 0.2 tells you *which* part of the input distribution has shifted, so you can investigate whether it's a real change in fraud patterns, a data pipeline bug, or a seasonal effect.

### What to do when a feature flags drift

1. **Don't panic.** Check `recent_transactions_used` — if it's < 100, there's not enough recent data to compute a reliable PSI.
2. **Look at the feature's historical PSI** via `GET /api/v1/monitoring/drift-history`. Is it a sudden spike or a gradual trend? Sudden spikes often indicate a data pipeline issue. Gradual trends indicate genuine distribution shift.
3. **Compare means.** The drift snapshot stores `mean_train` and `mean_recent`. If `amount_scaled` drifted, does the recent mean reflect a real change in transaction amounts?
4. **Run a backtest** on recently-labelled transactions. If F1 has dropped alongside the drift, retrain. If F1 is stable, the drift may not be impacting predictions yet.
5. **If F1 has dropped: retrain.** See Section 8.

---

## 7. The Monitoring Dashboard

Access at: `https://your-app.replit.app/dashboard`

The dashboard auto-refreshes every 30 seconds. Each panel shows a loading skeleton while its API call is in flight.

### Live metric cards (top row)

- **Transactions Today:** Total number of predictions scored since midnight UTC. If this is unexpectedly zero, check that the server is receiving traffic and that `POST /api/predict` is returning 200s.
- **Fraud Rate:** Percentage of today's transactions flagged as fraud. For healthy e-commerce, expect 0.1–2%. A sudden spike to 20%+ usually means the threshold is too low, not actual fraud. A sudden drop to 0% may mean the model stopped loading.
- **Avg Fraud Score:** The average ensemble score across all today's transactions. This should stay relatively stable day-to-day. A slow upward trend over weeks may signal genuine drift.
- **Active Model:** The `version_tag` from the active row in `model_versions`. If this shows "none," the model registry is empty — run `python scripts/seed.py`.

### Precision/Recall curve

This chart plots Precision vs Recall at every threshold from 0.1 to 0.9.

- **Good curve:** Stays high on both axes until it curves sharply (e.g., Precision stays above 0.8 until Recall reaches 0.85). This means the model can catch most fraud without too many false alarms.
- **Concerning curve:** Drops steeply from the start. Achieving even moderate Recall requires sacrificing a lot of Precision — the model is uncertain and you'll generate many false positives.
- **The threshold line:** The vertical line marks the current `FRAUD_THRESHOLD`. Points to its left are rejected (labeled legit); points to its right are flagged. Move the threshold right to reduce false positives (fewer customer complaints); move it left to catch more fraud (more manual reviews).

### Drift table

Columns: Feature Name | PSI Score | Drift Flagged.

- PSI < 0.1: shown with a green badge. Ignore.
- PSI 0.1–0.2: yellow badge. Watch.
- PSI > 0.2: red badge. Investigate and consider retraining.

The table is sorted by PSI descending, so the worst features are always at the top. If you see `amount_scaled` or `time_of_day_sin` at the top with PSI > 0.3, that's a signal that fraud patterns have shifted in amounts or timing.

### Volume chart (hourly, last 24 hours)

Two bars per hour: fraud transactions (red) and legitimate transactions (blue). Normal traffic looks like a smooth wave peaking around 12pm–8pm local time. What to watch for:

- **A spike in fraud at 2–4am:** Classic card-testing pattern. Fraudsters test stolen cards at off-peak hours when monitoring is lighter.
- **Zero volume for several hours:** The server may have restarted and lost connections, or the payments backend stopped calling the predict endpoint.
- **Fraud rate spiking uniformly across all hours:** May indicate a threshold that is too aggressive rather than actual fraud.

---

## 8. How to Retrain the Model

### When to retrain

Retrain when **any** of these conditions are true:
- A feature's PSI score exceeds 0.2 for 3+ consecutive daily snapshots
- F1 score on recently-labelled backtest data drops more than 5 points below the baseline in `metrics.json`
- You have > 500 new ground-truth labelled fraud cases that weren't in the original training data
- It has been more than 6 months since the last training run

### Exact retraining commands

```bash
# 1. Download the latest dataset if you have new labelled data
#    (or continue using creditcard.csv)

# 2. From fraud-detection-api/
python -m app.ml.train

# Expected output:
# [train] Loading dataset from .../data/creditcard.csv …
# [train] Loaded 284,807 rows — fraud rate: 0.1727%
# [train] XGBoost — scale_pos_weight=576.38
# [train] XGBoost trained in 45.2s
# [train] Isolation Forest trained in 12.8s
# [train] ✓ Training complete.

# 3. Restart the server so it loads the new model files from app.state
#    (In Replit: click the Run button, or restart the workflow)
```

### Register the new model version in the database

After retraining, insert a row in `model_versions` to track what changed. Read the new metrics from `metrics.json`:

```bash
cat app/ml/artifacts/metrics.json
```

Then register via the Replit DB shell or a one-time Python script:

```python
# scripts/register_model_version.py
import os, json, uuid
from datetime import datetime

os.chdir("fraud-detection-api")
from app.core.database import SessionLocal
from app.models.db_models import ModelVersion

with open("app/ml/artifacts/metrics.json") as f:
    m = json.load(f)

db = SessionLocal()
# Deactivate all existing versions
db.query(ModelVersion).update({"is_active": False})

mv = ModelVersion(
    id=str(uuid.uuid4()),
    version_tag="v2.0-may2026",
    trained_at=datetime.utcnow(),
    training_rows=m["xgboost"]["training_samples"],
    precision_score=m["xgboost"]["precision"],
    recall_score=m["xgboost"]["recall"],
    f1_score=m["xgboost"]["f1_score"],
    auc_roc=m["xgboost"]["auc_roc"],
    threshold_used=0.5,
    is_active=True,
    notes="Retrain triggered by amount_scaled PSI > 0.2",
    created_at=datetime.utcnow(),
)
db.add(mv)
db.commit()
print(f"Registered: {mv.version_tag}")
```

### Zero-downtime model swap

Because the model is loaded in the lifespan hook and stored in `app.state`, swapping it requires a server restart. On Replit, a restart takes ~3 seconds. During that window, in-flight requests will fail. This is acceptable for most use cases. If you need zero downtime, the pattern is: deploy a second instance with the new model, shift traffic via a load balancer, then decommission the old instance.

### Compare old vs new model using the backtest framework

```bash
# Label some recent transactions first
python scripts/seed_labels.py

# Then run backtest for old model
curl -X POST /api/v1/backtest \
  -H "X-API-Key: $KEY" \
  -d '{"dataset_label": "pre-retrain-v1"}'

# Restart server with new model, then run backtest again
curl -X POST /api/v1/backtest \
  -H "X-API-Key: $KEY" \
  -d '{"dataset_label": "post-retrain-v2"}'

# Compare optimal_threshold and f1 scores between the two runs
curl /api/v1/backtest/history
```

---

## 9. Common Mistakes & How We Avoid Them

**"I almost served predictions before the model finished loading."**  
At startup, there's a window between when the process starts and when `lifespan` finishes running. If a health check hits at that moment and starts routing traffic, predict requests would fail. We avoid this by: (1) setting `app.state.ensemble = None` as a module-level default before the lifespan runs, (2) checking `getattr(request.app.state, 'ensemble', None)` in the predict handler rather than accessing the attribute directly (so it never raises `AttributeError`), and (3) returning `503` rather than `500` — a 503 tells upstream load balancers "not ready yet" which is semantically correct.

**"I almost trained on the full dataset including the test set."**  
This is data leakage and it inflates your metrics. The model "memorises" the test data and appears to perform better than it actually will in production. The fix is `train_test_split(stratify=labels)` called *before* any model fitting. In this codebase, `engineer_features(df, fit=True)` is also called before the split so the scaler is fit only on `X_train` — not the entire dataset.

**"I almost reported 99.8% accuracy and called it done."**  
A model that predicts "legit" for every transaction achieves 99.8% accuracy on this dataset. Accuracy is meaningless for class-imbalanced problems. This codebase always reports Precision, Recall, F1, and AUC-ROC — the four metrics that matter for fraud detection. Precision is "of all transactions we flagged, how many were real fraud?" Recall is "of all actual fraud, how many did we catch?" F1 balances both. AUC-ROC is threshold-independent and measures overall discriminative power.

**"I almost used a fresh scaler at inference time."**  
If you call `RobustScaler().fit_transform(X_inference)` at prediction time, you're scaling based on the statistics of a single transaction — which is meaningless and will produce garbage features. The scaler must be fit on training data and the exact same fitted scaler must be used at inference. This is why `scaler.pkl` is saved during training and loaded during inference via `load_scaler()`. The path is hardcoded to `app/ml/artifacts/scaler.pkl`. If you move the file, predictions will silently produce wrong features.

**"I almost let the model get called 10,000 times in a minute during a DDoS attempt."**  
Each predict call loads the model into memory (or reads from cache), runs numpy matrix operations, and writes to the database. At scale, unthrottled traffic would OOM the server or exhaust the DB connection pool. We use `slowapi` with `@limiter.limit("100/minute")` per IP address on the predict endpoint. The key is the IP address from the request, so different clients get independent buckets. Rate limiting is applied *before* the DB duplicate check and model inference.

---

## 10. How to Run This Locally

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd fraud-detection-api

# 2. Install dependencies
pip install -r requirements.txt

# Expected: all packages install without errors

# 3. Copy the env file
cp .env.example .env
# Leave API_KEY blank for local development

# 4. Run database migrations
DATABASE_URL="sqlite:///./fraud_detection.db" alembic upgrade head

# Expected output:
# INFO  [alembic.runtime.migration] Running upgrade -> 801b027d587a, initial_schema
# INFO  [alembic.runtime.migration] Running upgrade 801b027d587a -> a3f8c2e1d9b4, add_label_columns_to_transactions

# 5. Seed the database (1000 synthetic transactions + 1 model version)
python scripts/seed.py

# Expected output:
# [seed] Alembic migrations applied.
# [seed] Created model version: <uuid> (tag=v1.0-seed)
# [seed] Done. inserted=1000 skipped=0 fraud=10 legit=990

# 6. (Optional) Train the models
#    Requires data/creditcard.csv from Kaggle
python -m app.ml.train

# Skip this step if you don't have the dataset — the server starts anyway
# but returns 503 on POST /api/v1/predict

# 7. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Expected output:
# INFO     Models loaded at startup in 312.4 ms    ← if you trained
# INFO     Application startup complete.

# 8. Verify it's running
curl http://localhost:8000/api/health
# {"status":"ok","models_loaded":true,"db_connected":true,"timestamp":1746230400.1}

# 9. Make a prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "local-test-001",
    "amount": 2499.00,
    "hour_of_day": 2,
    "day_of_week": 6
  }'

# Expected response (scores will vary):
# {"transaction_id":"local-test-001","fraud_score":0.847,"is_fraud":true,...}

# 10. Open the dashboard
open http://localhost:8000/dashboard
```

---

## 11. Glossary

**XGBoost** — "Extreme Gradient Boosting." Builds hundreds of small decision trees in sequence, where each tree learns from the mistakes of the previous one. Excellent on tabular data, fast, and interpretable via feature importance scores.

**Isolation Forest** — An unsupervised anomaly detection algorithm. It builds random trees that try to "isolate" (separate) individual data points. Anomalies (fraud) are isolated in fewer steps than normal points, because they're unusual. Doesn't require fraud labels to train.

**Ensemble model** — A model made by combining multiple models. Each model votes and the ensemble uses a weighted average. The idea: one model's blind spots are covered by another's strengths.

**PSI (Population Stability Index)** — A number that measures how different a feature's distribution is now compared to when the model was trained. < 0.1 is stable, > 0.2 means something has shifted and you should investigate.

**AUC-ROC** — "Area Under the Receiver Operating Characteristic curve." A number from 0.5 (random guessing) to 1.0 (perfect). Measures how well the model ranks fraud above legitimate transactions regardless of the threshold chosen. AUC > 0.95 is excellent for fraud detection.

**Precision** — Of all transactions the model called fraud, what fraction actually were fraud? High precision = few false alarms = fewer customers incorrectly declined.

**Recall** — Of all actual fraud transactions, what fraction did the model catch? High recall = catches more fraud = but may flag more legitimate transactions.

**F1 Score** — The harmonic mean of Precision and Recall. Useful when you care about both and want a single number. Better than accuracy on imbalanced datasets.

**Class imbalance** — When one class (fraud: 0.17%) appears far less often than the other (legit: 99.83%). Most ML algorithms assume roughly equal class sizes and perform badly without correction. We correct it with `scale_pos_weight`.

**Feature engineering** — Transforming raw input data into a form the model can learn from better. Examples here: scaling Amount with RobustScaler, encoding time cyclically with sin/cos.

**Model drift** — The phenomenon where a model's performance degrades over time because the real world has changed. The model learned from historical data; if current data looks different, predictions become unreliable.

**Backtest** — Running the model's past predictions against ground-truth labels to evaluate how it would have performed. Used to find the optimal fraud score threshold and to compare model versions.

**Contamination parameter** — The Isolation Forest's assumption about what fraction of the training data is anomalous. Setting it to 0.01 means "I expect ~1% of training data to be outliers." Too low and the model misses borderline cases; too high and everything looks suspicious.
