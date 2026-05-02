from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Annotated strict-type aliases ────────────────────────────────────────────
# strict=True prevents silent coercion (e.g. "12" -> 12 for int fields).

StrictInt = Annotated[int, Field(strict=True)]
StrictFloat = Annotated[float, Field(strict=True)]


class PredictRequest(BaseModel):
    model_config = {"strict": False}   # allow normal JSON but enforce our validators

    transaction_id: str = Field(..., max_length=64)
    amount: StrictFloat
    merchant_category: str = Field(default="", max_length=128)
    hour_of_day: StrictInt = Field(..., ge=0, le=23)
    day_of_week: StrictInt = Field(..., ge=0, le=6)
    user_id: str = Field(default="", max_length=128)
    ip_address: str = Field(default="", max_length=45)   # stored only, never returned
    device_fingerprint: str = Field(default="", max_length=256)

    @model_validator(mode="before")
    @classmethod
    def strip_strings(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for key, val in values.items():
                if isinstance(val, str):
                    values[key] = val.strip()
        return values

    @field_validator("amount", mode="after")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be greater than 0")
        return v

    @field_validator("transaction_id", mode="after")
    @classmethod
    def transaction_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("transaction_id must not be empty")
        return v


class PredictResponse(BaseModel):
    """
    ip_address is intentionally excluded — it is stored in the DB but
    must never be returned in any API response.
    """
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    xgb_probability: float
    isolation_forest_score: float
    threshold_used: float
    processing_ms: int


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: str = ""
