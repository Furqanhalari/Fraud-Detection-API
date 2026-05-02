"""
Seed labels script — applies synthetic ground-truth labels to existing transactions.

Usage (from fraud-detection-api/):
    python scripts/seed_labels.py

What it does:
  1. Fetches the 100 most recent transactions from the DB
  2. Labels them synthetically:
       fraud_score > 0.7  →  is_fraud_actual = True,  label_source = "synthetic"
       fraud_score <= 0.7 →  is_fraud_actual = False, label_source = "synthetic"
  3. Sets labeled_at to the current UTC timestamp
  4. Prints a summary: total labeled, fraud count, non-fraud count

Purpose: instantly populate ground truth for demo/testing without manual work.
         After running this, POST /api/v1/backtest will have data to evaluate.
"""

import os
import sys
from datetime import datetime

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db_models import Transaction

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")
FRAUD_SCORE_THRESHOLD = 0.7
LIMIT = 100

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def main():
    print(f"[seed_labels] DATABASE_URL={DATABASE_URL}")
    db = SessionLocal()
    try:
        rows = (
            db.query(Transaction)
            .order_by(Transaction.created_at.desc())
            .limit(LIMIT)
            .all()
        )

        if not rows:
            print("[seed_labels] No transactions found. Run scripts/seed.py first.")
            return

        now = datetime.utcnow()
        fraud_count = 0
        legit_count = 0

        for txn in rows:
            score = txn.fraud_score or 0.0
            is_fraud = score > FRAUD_SCORE_THRESHOLD
            txn.is_fraud_actual = is_fraud
            txn.label_source = "synthetic"
            txn.label_notes = f"Auto-labeled by seed_labels.py (score={score:.4f}, threshold={FRAUD_SCORE_THRESHOLD})"
            txn.labeled_at = now
            if is_fraud:
                fraud_count += 1
            else:
                legit_count += 1

        db.commit()
        total = fraud_count + legit_count
        print(
            f"[seed_labels] Done. total_labeled={total} fraud={fraud_count} "
            f"legit={legit_count} threshold={FRAUD_SCORE_THRESHOLD}"
        )
        print(
            "[seed_labels] You can now run POST /api/v1/backtest to evaluate "
            "model thresholds against this ground truth."
        )

    except Exception as exc:
        db.rollback()
        print(f"[seed_labels] ERROR: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
