"""
POST /api/v1/backtest         — Threshold sweep evaluation over labelled transactions.
GET  /api/v1/backtest/history — Paginated past backtest runs grouped by dataset_label.

Security: POST route requires X-API-Key header.

NOTE: POST /api/v1/backtest requires transactions with is_fraud_actual populated.
Use POST /api/v1/transactions/{transaction_id}/label (or /bulk-label) to apply
ground-truth labels before running a backtest. The 400 error response includes
total_transactions so callers know predictions exist but are unlabeled.
"""

import math
import uuid
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator
from sklearn.metrics import f1_score, precision_score, recall_score
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logger import get_logger
from app.core.security import require_api_key
from app.models.db_models import BacktestRun, ModelVersion, Transaction

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["backtest"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _structured_error(error: str, message: str, detail: str = "") -> dict:
    return {"error": error, "message": message, "detail": detail}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    dataset_label: str = Field(..., max_length=128)
    thresholds: List[float] = Field(
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        min_length=1,
        max_length=50,
    )

    @model_validator(mode="before")
    @classmethod
    def strip_strings(cls, values):
        if isinstance(values, dict):
            for k, v in values.items():
                if isinstance(v, str):
                    values[k] = v.strip()
        return values


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
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_or_create_model_version(db: Session) -> str:
    mv = db.query(ModelVersion.id).filter(ModelVersion.is_active == True).first()
    if mv:
        return str(mv.id)
    mv = db.query(ModelVersion.id).order_by(ModelVersion.created_at.desc()).first()
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
    y_pred = (fraud_scores >= threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
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


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/backtest",
    response_model=BacktestResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    responses={
        400: {"model": dict, "description": "No labelled transactions available"},
        401: {"model": dict, "description": "Missing or invalid API key"},
        422: {"model": dict, "description": "Validation error"},
        500: {"model": dict, "description": "Internal error during evaluation"},
    },
)
def run_backtest(req: BacktestRequest, db: Session = Depends(get_db)) -> BacktestResponse:
    # ── 1. Load labelled transactions (explicit column selection) ──────────────
    try:
        rows = (
            db.query(Transaction.is_fraud_actual, Transaction.fraud_score)
            .filter(Transaction.is_fraud_actual.isnot(None))
            .filter(Transaction.fraud_score.isnot(None))
            .all()
        )
    except Exception as exc:
        logger.error("[backtest] DB query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("db_error", "Failed to load transactions from the database.", str(exc)),
        )

    if not rows:
        logger.warning("[backtest] No labelled transactions found for dataset_label=%s", req.dataset_label)
        # Count total transactions so the caller knows predictions exist but are unlabeled
        try:
            total_transactions = db.query(Transaction.id).count()
        except Exception:
            total_transactions = 0
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "no_labeled_data",
                "message": (
                    "No transactions with ground truth labels found. "
                    "Use POST /api/v1/transactions/{id}/label first."
                ),
                "labeled_count": 0,
                "total_transactions": total_transactions,
            },
        )

    try:
        y_true = np.array([int(r.is_fraud_actual) for r in rows])
        fraud_scores = np.array([float(r.fraud_score) for r in rows])
    except Exception as exc:
        logger.error("[backtest] Label extraction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error(
                "data_error",
                "Could not extract labels or scores from transaction records.",
                str(exc),
            ),
        )

    # ── 2. Resolve model version ───────────────────────────────────────────────
    try:
        model_version_id = _get_or_create_model_version(db)
    except Exception as exc:
        logger.error("[backtest] Model version resolution failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("db_error", "Could not resolve or create a model version record.", str(exc)),
        )

    # ── 3. Threshold sweep ─────────────────────────────────────────────────────
    thresholds = sorted(set(req.thresholds))
    run_at = datetime.utcnow()
    results: List[ThresholdResult] = []

    try:
        for t in thresholds:
            if not (0.0 <= t <= 1.0):
                logger.warning("[backtest] Threshold %.2f out of [0,1] range, skipping", t)
                continue
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
    except Exception as exc:
        db.rollback()
        logger.error("[backtest] Threshold sweep failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("evaluation_error", "Error during threshold sweep computation.", str(exc)),
        )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_structured_error(
                "no_valid_thresholds",
                "All provided thresholds were out of the valid range [0.0, 1.0].",
                f"Received: {req.thresholds}",
            ),
        )

    optimal = max(results, key=lambda r: r.f1)
    logger.info(
        "[backtest] label=%s rows=%d thresholds=%d optimal=%.2f f1=%.4f",
        req.dataset_label, len(rows), len(results), optimal.threshold, optimal.f1,
    )

    return BacktestResponse(
        dataset_label=req.dataset_label,
        results=sorted(results, key=lambda r: r.threshold),
        optimal_threshold=optimal.threshold,
    )


@router.get(
    "/backtest/history",
    response_model=BacktestHistoryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": dict, "description": "Invalid pagination parameters"},
        500: {"model": dict, "description": "Database error"},
    },
)
def backtest_history(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=50, ge=1, le=200, description="Records per page (max 200)"),
    db: Session = Depends(get_db),
) -> BacktestHistoryResponse:
    """Return past backtest_runs rows grouped by dataset_label, paginated. No API key required."""
    try:
        total = db.query(BacktestRun.id).count()
        rows = (
            db.query(
                BacktestRun.id,
                BacktestRun.model_version_id,
                BacktestRun.threshold,
                BacktestRun.precision,
                BacktestRun.recall,
                BacktestRun.f1,
                BacktestRun.false_positive_rate,
                BacktestRun.true_positive_rate,
                BacktestRun.dataset_label,
                BacktestRun.run_at,
                BacktestRun.created_at,
            )
            .order_by(BacktestRun.dataset_label, BacktestRun.run_at, BacktestRun.threshold)
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
    except Exception as exc:
        logger.error("[backtest_history] DB query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("db_error", "Failed to load backtest history from the database.", str(exc)),
        )

    total_pages = math.ceil(total / page_size) if total > 0 else 1

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

    return BacktestHistoryResponse(
        groups=dict(groups),
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
