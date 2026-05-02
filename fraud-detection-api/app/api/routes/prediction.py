"""
POST /api/v1/predict — Real-time fraud prediction endpoint.

Pipeline:
  1. Validate input
  2. Check for duplicate transaction_id (409)
  3. Build feature vector via features.py
  4. Score with ensemble (XGBoost + Isolation Forest)
  5. Persist transaction record to DB
  6. Return structured response
"""

import json
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.ml.ensemble import get_ensemble
from app.ml.features import engineer_features
from app.models.db_models import Transaction
from app.models.schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/api/v1", tags=["prediction"])


def _build_feature_row(req: PredictRequest) -> pd.DataFrame:
    """
    Construct a single-row DataFrame that matches the training schema:
      Time, V1–V28, Amount, Class

    In a real deployment V1–V28 are PCA projections from raw transaction
    features (card network, terminal, velocity counters, etc.).
    At inference time we set them to 0.0 — the ensemble's Isolation Forest
    and the anomaly signals from Amount / Time still contribute.

    Approximation: map hour_of_day back to seconds within the day so the
    cyclical time features are populated from the request.
    """
    row: dict = {
        "Time": req.hour_of_day * 3600.0,
        "Amount": req.amount,
        "Class": 0,
    }
    for i in range(1, 29):
        row[f"V{i}"] = 0.0

    return pd.DataFrame([row])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    responses={
        409: {"description": "Duplicate transaction_id"},
        422: {"description": "Validation error"},
        503: {"description": "Models not loaded"},
    },
)
def predict(req: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    t_start = time.monotonic()

    # ── 1. Duplicate check ──────────────────────────────────────────────────
    existing = (
        db.query(Transaction)
        .filter(Transaction.transaction_id == req.transaction_id)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"transaction_id '{req.transaction_id}' already exists.",
        )

    # ── 2. Feature engineering ──────────────────────────────────────────────
    try:
        raw_df = _build_feature_row(req)
        feature_df = engineer_features(raw_df, fit=False)
        feature_vector = feature_df.values[0]
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Scaler artifact missing — run train.py first. ({exc})",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature engineering failed: {exc}",
        )

    # ── 3. Ensemble inference ───────────────────────────────────────────────
    try:
        ensemble = get_ensemble()
        result = ensemble.predict(feature_vector)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact missing — run train.py first. ({exc})",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble inference failed: {exc}",
        )

    processing_ms = int((time.monotonic() - t_start) * 1000)

    # ── 4. Persist to DB ────────────────────────────────────────────────────
    raw_features_snapshot = {
        "amount": req.amount,
        "merchant_category": req.merchant_category,
        "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week,
        "ip_address": req.ip_address,
        "device_fingerprint": req.device_fingerprint,
        "feature_vector": feature_vector.tolist(),
    }

    txn = Transaction(
        id=str(uuid.uuid4()),
        transaction_id=req.transaction_id,
        amount=req.amount,
        merchant_category=req.merchant_category,
        hour_of_day=req.hour_of_day,
        day_of_week=req.day_of_week,
        user_id=req.user_id,
        ip_address=req.ip_address,
        device_fingerprint=req.device_fingerprint,
        raw_features=raw_features_snapshot,
        fraud_score=result["fraud_score"],
        is_fraud_predicted=result["is_fraud"],
        is_fraud_actual=None,
        created_at=datetime.utcnow(),
    )

    db.add(txn)
    db.commit()

    # ── 5. Response ─────────────────────────────────────────────────────────
    return PredictResponse(
        transaction_id=req.transaction_id,
        fraud_score=result["fraud_score"],
        is_fraud=result["is_fraud"],
        xgb_probability=result["xgb_prob"],
        isolation_forest_score=result["iso_score"],
        threshold_used=settings.fraud_threshold,
        processing_ms=processing_ms,
    )
