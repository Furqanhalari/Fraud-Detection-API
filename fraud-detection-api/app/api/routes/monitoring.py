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
from app.models.db_models import DriftSnapshot, ModelVersion, Transaction
from app.monitoring.drift import run_drift_snapshot

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class DriftSnapshotResponse(BaseModel):
    model_version_id: str
    features_checked: int
    features_flagged: int
    flagged_features: List[str]
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_active_model_version(db: Session) -> str:
    """Return the active ModelVersion id, or any existing one, or raise 503."""
    mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    if mv:
        return str(mv.id)
    mv = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
    if mv:
        return str(mv.id)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "No model version found in the database. "
            "Run train.py and register a ModelVersion first."
        ),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/drift-snapshot",
    response_model=DriftSnapshotResponse,
    status_code=status.HTTP_200_OK,
    responses={
        503: {"description": "Models or training distributions not ready"},
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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Training distributions missing — run train.py first. ({exc})",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift computation failed: {exc}",
        )

    return DriftSnapshotResponse(
        model_version_id=model_version_id,
        **summary,
    )


@router.get(
    "/drift-history",
    response_model=DriftHistoryResponse,
    status_code=status.HTTP_200_OK,
)
def drift_history(
    days: int = Query(default=30, ge=1, le=365, description="Look-back window in days"),
    db: Session = Depends(get_db),
) -> DriftHistoryResponse:
    """
    Return drift_snapshots rows for the last N days, grouped by feature_name.
    Each feature entry includes a parallel arrays of dates and PSI scores,
    plus a count of how many days were flagged as drifted.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    rows = (
        db.query(DriftSnapshot)
        .filter(DriftSnapshot.created_at >= cutoff)
        .order_by(DriftSnapshot.feature_name, DriftSnapshot.snapshot_date)
        .all()
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


# ── Summary ───────────────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    total_transactions_today: int
    fraud_rate_today: float
    avg_fraud_score_today: float
    active_model_version: str
    threshold_used: float


@router.get("/summary", response_model=SummaryResponse)
def summary(db: Session = Depends(get_db)) -> SummaryResponse:
    """Live today metrics for the dashboard cards."""
    today_start = datetime.combine(datetime.utcnow().date(), dt_time.min)

    today_txns = (
        db.query(Transaction)
        .filter(Transaction.created_at >= today_start)
        .all()
    )

    total = len(today_txns)
    fraud_count = sum(1 for t in today_txns if t.is_fraud_predicted)
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0.0
    avg_score = (
        sum(t.fraud_score or 0.0 for t in today_txns) / total if total > 0 else 0.0
    )

    mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    version_tag = mv.version_tag if mv else "none"

    return SummaryResponse(
        total_transactions_today=total,
        fraud_rate_today=round(fraud_rate, 2),
        avg_fraud_score_today=round(avg_score, 4),
        active_model_version=version_tag,
        threshold_used=settings.fraud_threshold,
    )


# ── Volume ────────────────────────────────────────────────────────────────────

class VolumeResponse(BaseModel):
    hours: List[int]
    fraud_counts: List[int]
    legit_counts: List[int]


@router.get("/volume", response_model=VolumeResponse)
def volume(db: Session = Depends(get_db)) -> VolumeResponse:
    """Hourly transaction counts (fraud vs legit) for the last 24 hours."""
    cutoff = datetime.utcnow() - timedelta(hours=24)

    rows = (
        db.query(Transaction)
        .filter(Transaction.created_at >= cutoff)
        .all()
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
