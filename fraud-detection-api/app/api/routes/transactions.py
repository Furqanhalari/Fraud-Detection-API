"""
GET  /api/v1/transactions                        — List transactions with filters and pagination.
GET  /api/v1/transactions/{transaction_id}       — Fetch a single transaction by external ID.
POST /api/v1/transactions/{transaction_id}/label — Apply a ground-truth fraud label.
POST /api/v1/transactions/bulk-label             — Apply up to 500 labels in one request.

Security: POST routes require X-API-Key header. GET routes are public.

The label endpoints are the recommended way to populate is_fraud_actual before
running POST /api/v1/backtest to evaluate model threshold performance.

Route ordering note: /bulk-label and / (list) are declared before /{transaction_id}
so FastAPI's router matches them before treating the path segment as a path param.
"""

import math
from datetime import datetime, date
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logger import get_logger
from app.core.security import require_api_key
from app.models.db_models import Transaction

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])

LabelSource = Literal["chargeback", "investigation", "manual", "synthetic"]
SortField  = Literal["created_at", "amount", "fraud_score", "labeled_at"]
SortDir    = Literal["asc", "desc"]

_SORT_COLUMNS = {
    "created_at":  Transaction.created_at,
    "amount":      Transaction.amount,
    "fraud_score": Transaction.fraud_score,
    "labeled_at":  Transaction.labeled_at,
}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class LabelRequest(BaseModel):
    is_fraud: bool
    label_source: LabelSource
    notes: str = Field(default="", max_length=500)

    @model_validator(mode="before")
    @classmethod
    def strip_strings(cls, values):
        if isinstance(values, dict):
            for k, v in values.items():
                if isinstance(v, str):
                    values[k] = v.strip()
        return values


class LabelResponse(BaseModel):
    transaction_id: str
    is_fraud_actual: bool
    is_fraud_predicted: bool
    fraud_score: float
    label_source: str
    was_correct: bool          # is_fraud_actual == is_fraud_predicted
    labeled_at: str            # ISO 8601 timestamp


class BulkLabelItem(BaseModel):
    transaction_id: str = Field(..., max_length=64)
    is_fraud: bool
    label_source: LabelSource

    @model_validator(mode="before")
    @classmethod
    def strip_strings(cls, values):
        if isinstance(values, dict):
            for k, v in values.items():
                if isinstance(v, str):
                    values[k] = v.strip()
        return values


class BulkLabelRequest(BaseModel):
    labels: List[BulkLabelItem] = Field(..., min_length=1, max_length=500)


class AccuracySnapshot(BaseModel):
    correct_predictions: int
    total_labeled: int
    accuracy: float


class BulkLabelResponse(BaseModel):
    processed: int
    succeeded: int
    failed: int
    failures: List[dict]
    accuracy_snapshot: AccuracySnapshot


class TransactionSummary(BaseModel):
    """Lightweight row returned by the list endpoint — excludes raw_features."""
    transaction_id: str
    amount: float
    merchant_category: Optional[str]
    hour_of_day: Optional[int]
    day_of_week: Optional[int]
    fraud_score: Optional[float]
    is_fraud_predicted: Optional[bool]
    is_fraud_actual: Optional[bool]
    label_source: Optional[str]
    labeled_at: Optional[str]
    created_at: str


class TransactionListResponse(BaseModel):
    items: List[TransactionSummary]
    total: int
    page: int
    page_size: int
    total_pages: int
    filters_applied: dict


class TransactionDetail(BaseModel):
    transaction_id: str
    amount: float
    merchant_category: Optional[str]
    hour_of_day: Optional[int]
    day_of_week: Optional[int]
    fraud_score: Optional[float]
    is_fraud_predicted: Optional[bool]
    is_fraud_actual: Optional[bool]
    label_source: Optional[str]
    label_notes: Optional[str]
    labeled_at: Optional[str]
    created_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _structured_error(error: str, message: str, detail: str = "") -> dict:
    return {"error": error, "message": message, "detail": detail}


def _fetch_transaction(transaction_id: str, db: Session) -> Transaction:
    """Look up by the external transaction_id string field, not the UUID PK."""
    row = (
        db.query(Transaction)
        .filter(Transaction.transaction_id == transaction_id)
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "transaction_id": transaction_id},
        )
    return row


def _apply_label(
    txn: Transaction,
    is_fraud: bool,
    label_source: str,
    notes: str,
    now: datetime,
) -> None:
    """Mutate the ORM object in place. Caller is responsible for commit."""
    txn.is_fraud_actual = is_fraud
    txn.label_source = label_source
    txn.label_notes = notes or None
    txn.labeled_at = now


def _to_label_response(txn: Transaction) -> LabelResponse:
    return LabelResponse(
        transaction_id=txn.transaction_id,
        is_fraud_actual=bool(txn.is_fraud_actual),
        is_fraud_predicted=bool(txn.is_fraud_predicted),
        fraud_score=float(txn.fraud_score or 0.0),
        label_source=txn.label_source or "",
        was_correct=bool(txn.is_fraud_actual) == bool(txn.is_fraud_predicted),
        labeled_at=txn.labeled_at.isoformat() if txn.labeled_at else "",
    )


def _to_summary(txn) -> TransactionSummary:
    return TransactionSummary(
        transaction_id=txn.transaction_id,
        amount=txn.amount,
        merchant_category=txn.merchant_category,
        hour_of_day=txn.hour_of_day,
        day_of_week=txn.day_of_week,
        fraud_score=txn.fraud_score,
        is_fraud_predicted=txn.is_fraud_predicted,
        is_fraud_actual=txn.is_fraud_actual,
        label_source=txn.label_source,
        labeled_at=txn.labeled_at.isoformat() if txn.labeled_at else None,
        created_at=txn.created_at.isoformat(),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=TransactionListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": dict, "description": "Invalid date format"},
        500: {"model": dict, "description": "Database error"},
    },
)
def list_transactions(
    # ── Filters ───────────────────────────────────────────────────────────────
    label_source: Optional[LabelSource] = Query(
        default=None,
        description="Filter by label origin: chargeback | investigation | manual | synthetic",
    ),
    is_fraud_actual: Optional[bool] = Query(
        default=None,
        description="Filter by ground-truth label. true=fraud, false=legit, omit=all",
    ),
    is_fraud_predicted: Optional[bool] = Query(
        default=None,
        description="Filter by model prediction. true=flagged, false=not flagged, omit=all",
    ),
    labeled_only: bool = Query(
        default=False,
        description="When true, only return transactions where is_fraud_actual is set",
    ),
    date_from: Optional[str] = Query(
        default=None,
        description="ISO date (YYYY-MM-DD). Include transactions created on or after this date. Example: 2026-01-01",
    ),
    date_to: Optional[str] = Query(
        default=None,
        description="ISO date (YYYY-MM-DD). Include transactions created on or before this date. Example: 2026-12-31",
    ),
    min_fraud_score: Optional[float] = Query(
        default=None, ge=0.0, le=1.0,
        description="Only return transactions with fraud_score >= this value",
    ),
    max_fraud_score: Optional[float] = Query(
        default=None, ge=0.0, le=1.0,
        description="Only return transactions with fraud_score <= this value",
    ),
    # ── Sorting ───────────────────────────────────────────────────────────────
    sort_by: SortField = Query(
        default="created_at",
        description="Field to sort results by",
    ),
    sort_dir: SortDir = Query(
        default="desc",
        description="asc or desc",
    ),
    # ── Pagination ────────────────────────────────────────────────────────────
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page (max 200)"),
    db: Session = Depends(get_db),
) -> TransactionListResponse:
    """
    List transactions with optional filtering, sorting, and pagination.
    Use this to audit labeled data, inspect model predictions, or build
    reports without direct database access.

    All filters are ANDed together. Omit a filter to include all values.
    ip_address is intentionally excluded from all list responses.
    """
    # ── Parse date strings ─────────────────────────────────────────────────────
    dt_from: Optional[datetime] = None
    dt_to: Optional[datetime] = None

    if date_from:
        try:
            dt_from = datetime.combine(date.fromisoformat(date_from), datetime.min.time())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_structured_error(
                    "invalid_date",
                    f"date_from '{date_from}' is not a valid ISO date (YYYY-MM-DD).",
                ),
            )

    if date_to:
        try:
            dt_to = datetime.combine(date.fromisoformat(date_to), datetime.max.time())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_structured_error(
                    "invalid_date",
                    f"date_to '{date_to}' is not a valid ISO date (YYYY-MM-DD).",
                ),
            )

    # ── Build query (explicit column selection — no raw_features, no ip_address) ──
    try:
        q = db.query(
            Transaction.transaction_id,
            Transaction.amount,
            Transaction.merchant_category,
            Transaction.hour_of_day,
            Transaction.day_of_week,
            Transaction.fraud_score,
            Transaction.is_fraud_predicted,
            Transaction.is_fraud_actual,
            Transaction.label_source,
            Transaction.labeled_at,
            Transaction.created_at,
        )

        # Apply filters
        if label_source is not None:
            q = q.filter(Transaction.label_source == label_source)

        if is_fraud_actual is not None:
            q = q.filter(Transaction.is_fraud_actual == is_fraud_actual)

        if is_fraud_predicted is not None:
            q = q.filter(Transaction.is_fraud_predicted == is_fraud_predicted)

        if labeled_only:
            q = q.filter(Transaction.is_fraud_actual.isnot(None))

        if dt_from is not None:
            q = q.filter(Transaction.created_at >= dt_from)

        if dt_to is not None:
            q = q.filter(Transaction.created_at <= dt_to)

        if min_fraud_score is not None:
            q = q.filter(Transaction.fraud_score >= min_fraud_score)

        if max_fraud_score is not None:
            q = q.filter(Transaction.fraud_score <= max_fraud_score)

        # Count before pagination (same filters, no SELECT *)
        total = q.count()

        # Apply sort
        sort_col = _SORT_COLUMNS[sort_by]
        q = q.order_by(sort_col.desc() if sort_dir == "desc" else sort_col.asc())

        # Paginate
        rows = q.offset((page - 1) * page_size).limit(page_size).all()

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[list_transactions] DB query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("db_error", "Failed to query transactions.", str(exc)),
        )

    total_pages = math.ceil(total / page_size) if total > 0 else 1

    # Capture which filters were actually applied (for transparency)
    filters_applied = {}
    if label_source is not None:
        filters_applied["label_source"] = label_source
    if is_fraud_actual is not None:
        filters_applied["is_fraud_actual"] = is_fraud_actual
    if is_fraud_predicted is not None:
        filters_applied["is_fraud_predicted"] = is_fraud_predicted
    if labeled_only:
        filters_applied["labeled_only"] = True
    if date_from:
        filters_applied["date_from"] = date_from
    if date_to:
        filters_applied["date_to"] = date_to
    if min_fraud_score is not None:
        filters_applied["min_fraud_score"] = min_fraud_score
    if max_fraud_score is not None:
        filters_applied["max_fraud_score"] = max_fraud_score
    filters_applied["sort_by"] = sort_by
    filters_applied["sort_dir"] = sort_dir

    items = [
        TransactionSummary(
            transaction_id=r.transaction_id,
            amount=r.amount,
            merchant_category=r.merchant_category,
            hour_of_day=r.hour_of_day,
            day_of_week=r.day_of_week,
            fraud_score=r.fraud_score,
            is_fraud_predicted=r.is_fraud_predicted,
            is_fraud_actual=r.is_fraud_actual,
            label_source=r.label_source,
            labeled_at=r.labeled_at.isoformat() if r.labeled_at else None,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]

    return TransactionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        filters_applied=filters_applied,
    )


@router.post(
    "/bulk-label",
    response_model=BulkLabelResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"model": dict, "description": "Missing or invalid API key"},
        422: {"model": dict, "description": "Validation error (max 500 labels)"},
        500: {"model": dict, "description": "Database error"},
    },
)
def bulk_label(req: BulkLabelRequest, db: Session = Depends(get_db)) -> BulkLabelResponse:
    """
    Apply ground-truth labels to multiple transactions in one request.
    Processing continues on individual failures — partial success is reported.
    Maximum 500 labels per request (enforced by Pydantic max_length on the list).
    """
    now = datetime.utcnow()
    failures: List[dict] = []
    succeeded = 0
    correct_count = 0
    total_labeled = 0

    for item in req.labels:
        try:
            txn = (
                db.query(Transaction)
                .filter(Transaction.transaction_id == item.transaction_id)
                .first()
            )
            if txn is None:
                failures.append({
                    "transaction_id": item.transaction_id,
                    "reason": "not_found",
                })
                continue

            _apply_label(txn, item.is_fraud, item.label_source, notes="", now=now)
            db.flush()
            succeeded += 1
            total_labeled += 1
            if bool(txn.is_fraud_actual) == bool(txn.is_fraud_predicted):
                correct_count += 1

        except Exception as exc:
            logger.error(
                "[bulk_label] Failed to label txn=%s: %s",
                item.transaction_id, exc, exc_info=True,
            )
            failures.append({
                "transaction_id": item.transaction_id,
                "reason": str(exc),
            })

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("[bulk_label] Commit failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("db_error", "Failed to commit labels to the database.", str(exc)),
        )

    accuracy = round(correct_count / total_labeled, 4) if total_labeled > 0 else 0.0
    logger.info(
        "[bulk_label] processed=%d succeeded=%d failed=%d accuracy=%.4f",
        len(req.labels), succeeded, len(failures), accuracy,
    )

    return BulkLabelResponse(
        processed=len(req.labels),
        succeeded=succeeded,
        failed=len(failures),
        failures=failures,
        accuracy_snapshot=AccuracySnapshot(
            correct_predictions=correct_count,
            total_labeled=total_labeled,
            accuracy=accuracy,
        ),
    )


@router.get(
    "/{transaction_id}",
    response_model=TransactionDetail,
    status_code=status.HTTP_200_OK,
    responses={404: {"model": dict, "description": "Transaction not found"}},
)
def get_transaction(transaction_id: str, db: Session = Depends(get_db)) -> TransactionDetail:
    """
    Return all fields for a single transaction including its current label, if any.
    Use this to inspect the label state before overwriting it.
    ip_address is stored internally but excluded from this response.
    """
    txn = _fetch_transaction(transaction_id, db)
    return TransactionDetail(
        transaction_id=txn.transaction_id,
        amount=txn.amount,
        merchant_category=txn.merchant_category,
        hour_of_day=txn.hour_of_day,
        day_of_week=txn.day_of_week,
        fraud_score=txn.fraud_score,
        is_fraud_predicted=txn.is_fraud_predicted,
        is_fraud_actual=txn.is_fraud_actual,
        label_source=txn.label_source,
        label_notes=txn.label_notes,
        labeled_at=txn.labeled_at.isoformat() if txn.labeled_at else None,
        created_at=txn.created_at.isoformat(),
    )


@router.post(
    "/{transaction_id}/label",
    response_model=LabelResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"model": dict, "description": "Missing or invalid API key"},
        404: {"model": dict, "description": "Transaction not found"},
        422: {"model": dict, "description": "Validation error"},
        500: {"model": dict, "description": "Database error"},
    },
)
def label_transaction(
    transaction_id: str,
    req: LabelRequest,
    db: Session = Depends(get_db),
) -> LabelResponse:
    """
    Apply a ground-truth fraud label to a single transaction by its external ID.

    This populates is_fraud_actual, which is required before running
    POST /api/v1/backtest to evaluate model threshold performance.
    """
    txn = _fetch_transaction(transaction_id, db)
    now = datetime.utcnow()

    try:
        _apply_label(txn, req.is_fraud, req.label_source, req.notes, now)
        db.commit()
        db.refresh(txn)
    except Exception as exc:
        db.rollback()
        logger.error("[label] DB write failed for txn=%s: %s", transaction_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_structured_error("db_error", "Failed to persist label to the database.", str(exc)),
        )

    logger.info(
        "[label] %s labeled as fraud=%s source=%s",
        transaction_id, req.is_fraud, req.label_source,
    )

    return _to_label_response(txn)
