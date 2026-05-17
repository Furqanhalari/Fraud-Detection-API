# Fraud-Detection-API Monorepo

A comprehensive, production-grade fraud detection platform featuring a Python/FastAPI backend, Machine Learning (ML) ensemble scoring, real-time feature/drift monitoring, interactive dashboard, and a modern TypeScript-based monorepo workspace structure for generating type-safe API clients.

---

## 🏗️ Monorepo Architecture

This workspace is managed as a multi-language monorepo powered by **Python 3.11** and **Node.js (pnpm)**. It contains the following modules:

```
├── fraud-detection-api/       # FastAPI + Python ML Backend
│   ├── app/                   # Core application logic
│   │   ├── api/routes/        # Prediction, backtest, and monitoring endpoints
│   │   ├── ml/                # ML Pipeline (XGBoost + Isolation Forest)
│   │   ├── monitoring/        # Feature & PSI drift tracking
│   │   └── dashboard/         # Jinja2 + Chart.js dashboard
│   ├── alembic/               # Database migration schemas
│   └── scripts/               # Seeding and utility scripts
├── lib/
│   ├── db/                    # Shared Drizzle ORM database layer (TypeScript)
│   ├── api-spec/              # OpenAPI specifications and Orval client generator config
│   ├── api-zod/               # Generated Zod validation schemas
│   └── api-client-react/      # Generated React API query hooks
├── package.json               # Monorepo Workspace configuration
└── pyproject.toml / uv.lock   # Python dependencies and packaging configuration
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.11**
- **Node.js (24+) & pnpm**

---

### 2. Backend Setup (`fraud-detection-api`)

Navigate to the `fraud-detection-api` folder to perform the backend setup:

```bash
cd fraud-detection-api
```

#### A. Install Dependencies
Set up your virtual environment and install the required libraries:
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

#### B. Setup Environment Configuration
Copy the sample environment variables:
```bash
cp .env.example .env
```
*(Configure the keys in `.env` as needed. If deploying on Replit, set these inside the Secrets manager.)*

#### C. Database Setup (Migrations & Seeding)
Apply SQLAlchemy database migrations via Alembic and run the seeder:
```bash
# Run database migrations
alembic upgrade head

# Seed synthetic transaction and model version data
python scripts/seed.py
```

#### D. Train ML Ensemble Models
To enable fraud predictions, download the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`, then run:
```bash
python -m app.ml.train
```
This trains an ensemble composed of:
*   **XGBoost (70%)** - Supervised classification
*   **Isolation Forest (30%)** - Unsupervised anomaly detection

#### E. Spin Up the Backend Server
Run the FastAPI development server:
```bash
uvicorn app.main:app --reload
```
- API Health Check: `GET http://localhost:8000/api/health`
- Interactive Swagger UI: `http://localhost:8000/docs`
- Live Dashboard: `http://localhost:8000/dashboard`

---

### 3. Frontend & Type-Safe Libraries Setup (TypeScript)

From the root directory, install all Node.js workspace dependencies and compile type definitions:

```bash
# From workspace root
pnpm install
pnpm build
```

#### Generate API Client & Zod Schemas
If you modify the backend API routes, you can automatically regenerate your TypeScript clients and Zod schemas:
```bash
cd lib/api-spec
pnpm run generate
```

---

## 📊 Core Features

*   **Ensemble ML Inference:** Combines supervised and unsupervised methods for highly accurate predictions (`final_score = 0.7 × XGBoost + 0.3 × Isolation Forest`).
*   **Drift Monitoring:** Automatic population stability index (PSI) and feature drift tracking.
*   **Backtest Engine:** Test model thresholds against historical transaction batches to minimize false positives and false negatives.
*   **Built-in Live Dashboard:** Real-time visual metrics, hourly transaction volumes, and performance curves.

---

## 📖 Deep-Dive Documentation

For detailed information, refer to these guides inside the `fraud-detection-api/` directory:
*   **[Backend README](file:///c:/Users/dell/Desktop/Fraud-Detection-API/fraud-detection-api/README.md)** — Core FastAPI backend overview.
*   **[Deployment Guide](file:///c:/Users/dell/Desktop/Fraud-Detection-API/fraud-detection-api/DEPLOY.md)** — Production checklist, database tuning, and deployment settings.
*   **[Developer Guide](file:///c:/Users/dell/Desktop/Fraud-Detection-API/fraud-detection-api/DEVELOPER_GUIDE.md)** — Inside details about the ML pipelines, training, drift formulas, and database schemas.

---

## 🔒 Security

All prediction, backtesting, and administrative API endpoints require the `X-API-Key` header in production. To generate a secure key:
```bash
python -m secrets
```
