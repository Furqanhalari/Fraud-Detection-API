"""
GET  /api/v1/transactions/{transaction_id}       — Fetch a single transaction by external ID.
POST /api/v1/transactions/{transaction_id}/label — Apply a ground-truth fraud label.
POST /api/v1/transactions/bulk-label             — Apply up to 500 labels in one request.

Security: POST routes require X-API-Key header. GET is public.

The label endpoint is the recommended way to populate is_fraud_actual before
running POST /api/v1/backtest.
"""

from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logger import get_logger
from app.core.security import require_api_key
from app.models.db_models import Transaction

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])

LabelSource = Literal["chargeback", "investigation", "manual", "synthetic"]


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


# ── Routes ────────────────────────────────────────────────────────────────────

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
    "/bulk-label",
    response_model=BulkLabelResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    responses={
        400: {"model": dict, "description": "More than 500 labels submitted"},
        401: {"model": dict, "description": "Missing or invalid API key"},
        422: {"model": dict, "description": "Validation error"},
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
