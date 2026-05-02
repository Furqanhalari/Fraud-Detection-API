"""
POST /api/v1/monitoring/drift-snapshot  — Trigger a drift snapshot for the active model.
GET  /api/v1/monitoring/drift-history   — PSI history grouped by feature.
GET  /api/v1/monitoring/summary         — Live today metrics for the dashboard.
GET  /api/v1/monitoring/volume          — Hourly transaction counts (last 24 h).
"""

from collections import defaultdict
from datetime import datetime, timedelta, date, time as dt_time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.core.logger import get_logger
from app.models.db_models import DriftSnapshot, ModelVersion, Transaction
from app.monitoring.drift import run_drift_snapshot

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


# ── Shared helper ─────────────────────────────────────────────────────────────

def _structured_error(error: str, message: str, detail: str = "") -> dict:
    return {"error": error, "message": message, "detail": detail}


# ── Pydantic response schemas ─────────────────────────────────────────────────

class DriftSnapshotResponse(BaseModel):
    model_version_id: str
    features_checked: int
    features_flagged: int
    flagged_features: list
    worst_feature: str
    worst_psi: float
    recent_transactions_used: int
    snapshot_date: str


class FeatureDriftHistory(BaseModel):
    feature_name: str
    dates: List[str]
    psi_scores: List[float]
    drift_flagged_count: int


class DriftHistoryResponse(BaseModel):
    days: int
    features: List[FeatureDriftHistory]


class SummaryResponse(BaseModel):
    total_transactions_today: int
    fraud_rate_today: float
    avg_fraud_score_today: float
    active_model_version: str
    threshold_used: float


class VolumeResponse(BaseModel):
    hours: List[int]
    fraud_counts: List[int]
    legit_counts: List[int]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_active_model_version(db: Session) -> str:
    mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    if mv:
        return str(mv.id)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=_structured_error(
            "no_active_model",
            "No active model version found in the database.",
            "Run train.py to train a model and register a ModelVersion first.",
        ),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/drift-snapshot",
    response_model=DriftSnapshotResponse,
    status_code=status.HTTP_200_OK,
    responses={
        503: {"model": dict, "description": "Model or training distributions not ready"},
        500: {"model": dict, "description": "Drift computation error"},
    },
)
def trigger_drift_snapshot(db: Session = Depends(get_db)) -> DriftSnapshotResponse:
    """
    Compute PSI for every feature against the training distribution,
    persist one drift_snapshots row per feature, and return the summary.
    """
    model_version_id = _resolve_active_model_version(db)

    try:
        summary = run_drift_snapshot(model_version_id=model_version_id, db=db)
    except FileNotFoundError as exc:
        logger.error("[drift_snapshot] train_distributions.json missing: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_structured_error(
                "distributions_unavailable",
                "Training distributions file (train_distributions.json) not found.",
                "Run train.py to generate model artifacts before triggering drift monitoring.",
            ),
        )
    except Exception as exc:
        logger.error("[drift_snapshot] Drift computation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error(
                "drift_computation_error",
                "Unexpected error while computing feature drift.",
                str(exc),
            ),
        )

    logger.info(
        "[drift_snapshot] model=%s features=%d drifted=%d",
        model_version_id,
        summary.get("features_evaluated", 0),
        summary.get("drifted_features", 0),
    )
    return DriftSnapshotResponse(model_version_id=model_version_id, **summary)


@router.get(
    "/drift-history",
    response_model=DriftHistoryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": dict, "description": "Database error"},
    },
)
def drift_history(
    days: int = Query(default=30, ge=1, le=365, description="Look-back window in days"),
    db: Session = Depends(get_db),
) -> DriftHistoryResponse:
    """
    Return drift_snapshots rows for the last N days, grouped by feature_name.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = (
            db.query(DriftSnapshot)
            .filter(DriftSnapshot.created_at >= cutoff)
            .order_by(DriftSnapshot.feature_name, DriftSnapshot.snapshot_date)
            .all()
        )
    except Exception as exc:
        logger.error("[drift_history] DB query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error(
                "db_error",
                "Failed to load drift history from the database.",
                str(exc),
            ),
        )

    by_feature: dict[str, list[DriftSnapshot]] = defaultdict(list)
    for row in rows:
        by_feature[row.feature_name].append(row)

    features: List[FeatureDriftHistory] = []
    for feature_name, snapshots in sorted(by_feature.items()):
        features.append(
            FeatureDriftHistory(
                feature_name=feature_name,
                dates=[str(s.snapshot_date) for s in snapshots],
                psi_scores=[round(s.psi_score or 0.0, 6) for s in snapshots],
                drift_flagged_count=sum(1 for s in snapshots if s.drift_flagged),
            )
        )

    return DriftHistoryResponse(days=days, features=features)


@router.get(
    "/summary",
    response_model=SummaryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": dict, "description": "Database error"},
    },
)
def summary(db: Session = Depends(get_db)) -> SummaryResponse:
    """Live today metrics for the dashboard cards."""
    try:
        today_start = datetime.combine(datetime.utcnow().date(), dt_time.min)
        today_txns = (
            db.query(Transaction)
            .filter(Transaction.created_at >= today_start)
            .all()
        )
    except Exception as exc:
        logger.error("[summary] DB query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error(
                "db_error",
                "Failed to load today's transactions from the database.",
                str(exc),
            ),
        )

    total = len(today_txns)
    fraud_count = sum(1 for t in today_txns if t.is_fraud_predicted)
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0.0
    avg_score = (
        sum(t.fraud_score or 0.0 for t in today_txns) / total if total > 0 else 0.0
    )

    try:
        mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        version_tag = mv.version_tag if mv else "none"
    except Exception as exc:
        logger.warning("[summary] Could not resolve active model version: %s", exc)
        version_tag = "unknown"

    return SummaryResponse(
        total_transactions_today=total,
        fraud_rate_today=round(fraud_rate, 2),
        avg_fraud_score_today=round(avg_score, 4),
        active_model_version=version_tag,
        threshold_used=settings.fraud_threshold,
    )


@router.get(
    "/volume",
    response_model=VolumeResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": dict, "description": "Database error"},
    },
)
def volume(db: Session = Depends(get_db)) -> VolumeResponse:
    """Hourly transaction counts (fraud vs legit) for the last 24 hours."""
    try:
        cutoff = datetime.utcnow() - timedelta(hours=24)
        rows = (
            db.query(Transaction)
            .filter(Transaction.created_at >= cutoff)
            .all()
        )
    except Exception as exc:
        logger.error("[volume] DB query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error(
                "db_error",
                "Failed to load transaction volume data from the database.",
                str(exc),
            ),
        )

    fraud_by_hour: dict[int, int] = defaultdict(int)
    legit_by_hour: dict[int, int] = defaultdict(int)

    for t in rows:
        h = t.created_at.hour
        if t.is_fraud_predicted:
            fraud_by_hour[h] += 1
        else:
            legit_by_hour[h] += 1

    hours = list(range(24))
    return VolumeResponse(
        hours=hours,
        fraud_counts=[fraud_by_hour[h] for h in hours],
        legit_counts=[legit_by_hour[h] for h in hours],
    )
