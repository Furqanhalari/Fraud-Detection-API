"""
POST /api/v1/backtest  — Threshold sweep evaluation over labelled transactions.
GET  /api/v1/backtest/history — All past backtest runs grouped by dataset_label.
"""

import uuid
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.db_models import BacktestRun, ModelVersion, Transaction

router = APIRouter(prefix="/api/v1", tags=["backtest"])


# ── Pydantic schemas (local to this module) ──────────────────────────────────

class BacktestRequest(BaseModel):
    dataset_label: str
    thresholds: List[float] = Field(
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        min_length=1,
    )


class ThresholdResult(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positive_rate: float


class BacktestResponse(BaseModel):
    dataset_label: str
    results: List[ThresholdResult]
    optimal_threshold: float


class BacktestRunRecord(BaseModel):
    id: str
    model_version_id: str
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positive_rate: float
    dataset_label: str
    run_at: Optional[datetime]
    created_at: datetime


class BacktestHistoryResponse(BaseModel):
    groups: dict[str, List[BacktestRunRecord]]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_or_create_model_version(db: Session) -> str:
    """
    Return the id of the active ModelVersion.
    Falls back to any existing version, then creates a placeholder so the
    NOT NULL FK on backtest_runs is always satisfied.
    SQLite does not enforce FK constraints by default, but we satisfy
    the ORM nullable=False constraint at the Python level.
    """
    mv = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    if mv:
        return str(mv.id)

    mv = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
    if mv:
        return str(mv.id)

    placeholder = ModelVersion(
        id=str(uuid.uuid4()),
        version_tag="auto-placeholder",
        trained_at=datetime.utcnow(),
        is_active=False,
        notes="Auto-created by backtest runner (no trained version exists yet).",
        created_at=datetime.utcnow(),
    )
    db.add(placeholder)
    db.commit()
    db.refresh(placeholder)
    return str(placeholder.id)


def _compute_threshold_metrics(
    y_true: np.ndarray,
    fraud_scores: np.ndarray,
    threshold: float,
) -> ThresholdResult:
    """Apply a threshold and return classification metrics."""
    y_pred = (fraud_scores >= threshold).astype(int)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))

    # FPR = FP / (FP + TN),  TPR = TP / (TP + FN)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return ThresholdResult(
        threshold=round(threshold, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        true_positive_rate=round(tpr, 4),
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post(
    "/backtest",
    response_model=BacktestResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Validation error"},
        404: {"description": "No labelled transactions found"},
    },
)
def run_backtest(req: BacktestRequest, db: Session = Depends(get_db)) -> BacktestResponse:
    # ── 1. Load labelled transactions ──────────────────────────────────────
    rows = (
        db.query(Transaction)
        .filter(Transaction.is_fraud_actual.isnot(None))
        .filter(Transaction.fraud_score.isnot(None))
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No labelled transactions found. "
                "Populate is_fraud_actual on existing transaction rows first."
            ),
        )

    y_true = np.array([int(r.is_fraud_actual) for r in rows])
    fraud_scores = np.array([float(r.fraud_score) for r in rows])

    # ── 2. Resolve model version for FK ───────────────────────────────────
    model_version_id = _get_or_create_model_version(db)

    # ── 3. Threshold sweep ─────────────────────────────────────────────────
    thresholds = sorted(set(req.thresholds))
    run_at = datetime.utcnow()
    results: List[ThresholdResult] = []

    for t in thresholds:
        metrics = _compute_threshold_metrics(y_true, fraud_scores, t)
        results.append(metrics)

        db.add(
            BacktestRun(
                id=str(uuid.uuid4()),
                model_version_id=model_version_id,
                threshold=metrics.threshold,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                false_positive_rate=metrics.false_positive_rate,
                true_positive_rate=metrics.true_positive_rate,
                dataset_label=req.dataset_label,
                run_at=run_at,
                created_at=run_at,
            )
        )

    db.commit()

    # ── 4. Best F1 threshold ───────────────────────────────────────────────
    optimal = max(results, key=lambda r: r.f1)

    return BacktestResponse(
        dataset_label=req.dataset_label,
        results=sorted(results, key=lambda r: r.threshold),
        optimal_threshold=optimal.threshold,
    )


@router.get(
    "/backtest/history",
    response_model=BacktestHistoryResponse,
    status_code=status.HTTP_200_OK,
)
def backtest_history(db: Session = Depends(get_db)) -> BacktestHistoryResponse:
    """Return all past backtest_runs rows grouped by dataset_label."""
    rows = (
        db.query(BacktestRun)
        .order_by(BacktestRun.dataset_label, BacktestRun.run_at, BacktestRun.threshold)
        .all()
    )

    groups: dict[str, List[BacktestRunRecord]] = defaultdict(list)
    for r in rows:
        groups[r.dataset_label].append(
            BacktestRunRecord(
                id=str(r.id),
                model_version_id=str(r.model_version_id),
                threshold=r.threshold,
                precision=r.precision,
                recall=r.recall,
                f1=r.f1,
                false_positive_rate=r.false_positive_rate,
                true_positive_rate=r.true_positive_rate,
                dataset_label=r.dataset_label,
                run_at=r.run_at,
                created_at=r.created_at,
            )
        )

    return BacktestHistoryResponse(groups=dict(groups))
