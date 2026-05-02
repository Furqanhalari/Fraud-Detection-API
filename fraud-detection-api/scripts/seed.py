"""
Seed script — generates realistic synthetic data for development and production testing.

Usage (from fraud-detection-api/):
    python scripts/seed.py

What it does:
  1. Runs all Alembic migrations to ensure schema is current
  2. Inserts one model_versions row (is_active=True) if none exists
  3. Generates 1 000 synthetic transactions with realistic distributions
  4. Marks exactly 10 of them as is_fraud_actual=True (fraud rate ≈ 1%)
  5. Assigns a fraud_score drawn from realistic score distributions
  6. Skips rows that already exist (idempotent by transaction_id)
"""

import os
import sys
import uuid
from datetime import datetime, timedelta
import random

# Allow running from repo root or from fraud-detection-api/
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db_models import Base, ModelVersion, Transaction

# ── Config ────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")
TOTAL_TRANSACTIONS = 1_000
FRAUD_COUNT = 10
RANDOM_SEED = 42

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail", "travel",
    "entertainment", "healthcare", "utilities", "atm", "clothing",
]

# ── Setup ─────────────────────────────────────────────────────────────────────

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# Run Alembic migrations programmatically so seed can be run standalone
def _run_migrations():
    try:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config(os.path.join(_ROOT, "alembic.ini"))
        alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
        alembic_cfg.set_main_option("script_location", os.path.join(_ROOT, "alembic"))
        command.upgrade(alembic_cfg, "head")
        print("[seed] Alembic migrations applied.")
    except Exception as exc:
        print(f"[seed] Alembic migration failed — falling back to Base.metadata.create_all: {exc}")
        Base.metadata.create_all(bind=engine)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_transaction(
    rng: random.Random,
    np_rng: np.random.Generator,
    is_fraud: bool,
    base_time: datetime,
    idx: int,
) -> Transaction:
    """Generate one synthetic transaction row."""
    # Amounts: fraud transactions tend to be higher or very low (card-testing)
    if is_fraud:
        amount = float(rng.choice([
            round(np_rng.uniform(1.0, 5.0), 2),       # micro-charge card-testing
            round(np_rng.uniform(300.0, 2000.0), 2),   # large fraudulent purchase
        ]))
        fraud_score = round(float(np_rng.uniform(0.72, 0.99)), 4)
        merchant_category = rng.choice(["online_retail", "atm", "travel"])
        hour = rng.choice([0, 1, 2, 3, 22, 23])  # unusual hours
    else:
        amount = round(float(np_rng.lognormal(mean=3.5, sigma=1.2)), 2)
        amount = max(0.50, min(amount, 500.0))
        fraud_score = round(float(np_rng.beta(1.2, 8.0)), 4)  # skewed low
        merchant_category = rng.choice(MERCHANT_CATEGORIES)
        hour = rng.randint(7, 22)

    created_at = base_time - timedelta(
        hours=rng.randint(0, 72 * 24),
        minutes=rng.randint(0, 59),
        seconds=rng.randint(0, 59),
    )

    return Transaction(
        id=str(uuid.uuid4()),
        transaction_id=f"seed-{idx:05d}-{uuid.uuid4().hex[:8]}",
        amount=amount,
        merchant_category=merchant_category,
        hour_of_day=hour,
        day_of_week=created_at.weekday(),
        user_id=f"user-{rng.randint(1, 200):04d}",
        ip_address="",   # not stored in seeds — kept blank intentionally
        device_fingerprint=f"fp-{uuid.uuid4().hex[:16]}",
        raw_features={
            "amount": amount,
            "merchant_category": merchant_category,
            "hour_of_day": hour,
            "day_of_week": created_at.weekday(),
            "feature_vector": np_rng.standard_normal(30).tolist(),
        },
        fraud_score=fraud_score,
        is_fraud_predicted=fraud_score >= 0.5,
        is_fraud_actual=is_fraud,
        created_at=created_at,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[seed] DATABASE_URL={DATABASE_URL}")
    _run_migrations()

    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)

    db = SessionLocal()
    try:
        # ── 1. Model version ──────────────────────────────────────────────────
        existing_mv = db.query(ModelVersion.id).filter(ModelVersion.is_active == True).first()
        if existing_mv:
            model_version_id = str(existing_mv.id)
            print(f"[seed] Active model version already exists: {model_version_id}")
        else:
            mv = ModelVersion(
                id=str(uuid.uuid4()),
                version_tag="v1.0-seed",
                trained_at=datetime.utcnow(),
                training_rows=284_807,   # canonical creditcard.csv size
                precision_score=0.91,
                recall_score=0.78,
                f1_score=0.84,
                auc_roc=0.97,
                threshold_used=0.5,
                is_active=True,
                notes="Seeded model version for development and production testing.",
                created_at=datetime.utcnow(),
            )
            db.add(mv)
            db.commit()
            db.refresh(mv)
            model_version_id = str(mv.id)
            print(f"[seed] Created model version: {model_version_id} (tag=v1.0-seed)")

        # ── 2. Transactions ───────────────────────────────────────────────────
        existing_count = db.query(Transaction.id).count()
        if existing_count >= TOTAL_TRANSACTIONS:
            print(f"[seed] {existing_count} transactions already exist — skipping insertion.")
            return

        base_time = datetime.utcnow()

        # Build index list: first FRAUD_COUNT are fraud, rest are legit
        indices = list(range(TOTAL_TRANSACTIONS))
        fraud_indices = set(rng.sample(indices, FRAUD_COUNT))

        inserted = 0
        skipped = 0
        for i in indices:
            is_fraud = i in fraud_indices
            txn = _make_transaction(rng, np_rng, is_fraud, base_time, i)
            # Idempotency: skip if transaction_id already exists
            exists = (
                db.query(Transaction.id)
                .filter(Transaction.transaction_id == txn.transaction_id)
                .first()
            )
            if exists:
                skipped += 1
                continue
            db.add(txn)
            inserted += 1

            # Flush in batches to avoid large memory accumulation
            if inserted % 200 == 0:
                db.flush()

        db.commit()
        print(
            f"[seed] Done. inserted={inserted} skipped={skipped} "
            f"fraud={FRAUD_COUNT} legit={inserted - FRAUD_COUNT}"
        )

    except Exception as exc:
        db.rollback()
        print(f"[seed] ERROR: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
