"""
POST /api/v1/monitoring/drift-snapshot  — Trigger a drift snapshot for the active model.
GET  /api/v1/monitoring/drift-history   — Retrieve PSI history grouped by feature.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.db_models import DriftSnapshot, ModelVersion
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
