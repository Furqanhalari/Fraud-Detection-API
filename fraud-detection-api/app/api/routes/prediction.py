"""
POST /api/v1/predict — Real-time fraud prediction endpoint.

Security:
  - X-API-Key header required (require_api_key dependency)
  - Rate limited: 100 req/min per IP
  - ip_address stored in DB but never returned in response
  - No feature vectors logged

Pipeline:
  1. API key validation
  2. Rate-limit check (100 req/min per IP)
  3. Validate & sanitize input (Pydantic strict types + model_validator)
  4. Check models are loaded (503 if startup failed)
  5. Duplicate transaction_id check (409)
  6. Feature engineering
  7. Ensemble inference via request.app.state.ensemble (loaded at startup)
  8. Persist to DB
  9. Return structured response (ip_address excluded)
"""

import time
import uuid
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.limiter import limiter
from app.core.logger import get_logger
from app.core.security import require_api_key
from app.ml.features import engineer_features
from app.models.db_models import Transaction
from app.models.schemas import PredictRequest, PredictResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _structured_error(error: str, message: str, detail: str = "") -> dict:
    return {"error": error, "message": message, "detail": detail}


def _build_feature_row(req: PredictRequest) -> pd.DataFrame:
    """
    Construct a single-row DataFrame matching the training schema.
    V1–V28 are PCA projections unavailable at real-time inference; they are
    zeroed so the ensemble relies on Amount/Time signals.
    """
    row: dict = {"Time": req.hour_of_day * 3600.0, "Amount": req.amount, "Class": 0}
    for i in range(1, 29):
        row[f"V{i}"] = 0.0
    return pd.DataFrame([row])


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"model": dict, "description": "Missing or invalid API key"},
        409: {"model": dict, "description": "Duplicate transaction_id"},
        422: {"model": dict, "description": "Validation / feature error"},
        429: {"model": dict, "description": "Rate limit exceeded"},
        503: {"model": dict, "description": "ML models not available"},
    },
)
@limiter.limit("100/minute")
def predict(request: Request, req: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    t_start = time.monotonic()

    # ── 0. Check models loaded at startup ─────────────────────────────────────
    ensemble = getattr(request.app.state, "ensemble", None)
    if ensemble is None:
        logger.error("[predict] Ensemble is None — models did not load at startup")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_structured_error(
                "model_unavailable",
                "ML models are not loaded. Run train.py to generate model files and restart the server.",
                "app.state.ensemble is None",
            ),
        )

    # ── 1. Duplicate check ────────────────────────────────────────────────────
    try:
        existing = (
            db.query(Transaction.id)
            .filter(Transaction.transaction_id == req.transaction_id)
            .first()
        )
    except Exception as exc:
        logger.error("[predict] DB duplicate check failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("db_error", "Database error during duplicate check.", str(exc)),
        )

    if existing:
        logger.warning("[predict] Duplicate transaction_id=%s", req.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_structured_error(
                "duplicate_transaction",
                f"transaction_id '{req.transaction_id}' already exists.",
                "Submit a unique transaction_id for each request.",
            ),
        )

    # ── 2. Feature engineering ────────────────────────────────────────────────
    try:
        raw_df = _build_feature_row(req)
        feature_df = engineer_features(raw_df, fit=False)
        feature_vector = feature_df.values[0]
    except FileNotFoundError as exc:
        logger.error("[predict] Scaler artifact missing: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_structured_error(
                "model_unavailable",
                "Scaler artifact not found. Run train.py to generate model files.",
                str(exc),
            ),
        )
    except (TypeError, ValueError) as exc:
        logger.warning("[predict] Feature engineering type/value error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_structured_error(
                "feature_engineering_error",
                "Input values produced invalid feature types during engineering.",
                str(exc),
            ),
        )
    except Exception as exc:
        logger.error("[predict] Feature engineering unexpected error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error(
                "feature_engineering_error",
                "Unexpected error during feature engineering.",
                str(exc),
            ),
        )

    # ── 3. Ensemble inference (uses startup-loaded model from app.state) ───────
    try:
        result = ensemble.predict(feature_vector)
    except RuntimeError as exc:
        logger.error("[predict] Ensemble runtime error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_structured_error("model_unavailable", "Ensemble model is not loaded.", str(exc)),
        )
    except Exception as exc:
        logger.error("[predict] Ensemble inference failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("inference_error", "Unexpected error during model inference.", str(exc)),
        )

    processing_ms = int((time.monotonic() - t_start) * 1000)

    # ── 4. Persist to DB ──────────────────────────────────────────────────────
    # NOTE: feature_vector is stored in raw_features for drift monitoring
    # but is NOT logged (avoid leaking model internals to log files).
    raw_features_snapshot = {
        "amount": req.amount,
        "merchant_category": req.merchant_category,
        "hour_of_day": req.hour_of_day,
        "day_of_week": req.day_of_week,
        # ip_address stored for internal audit only — never returned in responses
        "ip_address": req.ip_address,
        "device_fingerprint": req.device_fingerprint,
        "feature_vector": feature_vector.tolist(),
    }

    try:
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
    except Exception as exc:
        db.rollback()
        logger.error("[predict] DB write failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error(
                "db_error",
                "Prediction succeeded but could not be persisted to the database.",
                str(exc),
            ),
        )

    # Log outcome WITHOUT feature vector or ip_address
    logger.info(
        "[predict] txn=%s fraud_score=%.4f is_fraud=%s ms=%d",
        req.transaction_id,
        result["fraud_score"],
        result["is_fraud"],
        processing_ms,
    )

    # PredictResponse schema excludes ip_address by design
    return PredictResponse(
        transaction_id=req.transaction_id,
        fraud_score=result["fraud_score"],
        is_fraud=result["is_fraud"],
        xgb_probability=result["xgb_prob"],
        isolation_forest_score=result["iso_score"],
        threshold_used=settings.fraud_threshold,
        processing_ms=processing_ms,
    )
