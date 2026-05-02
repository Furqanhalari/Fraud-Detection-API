from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    transaction_id: str
    amount: float
    merchant_category: str = ""
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    user_id: str = ""
    ip_address: str = ""
    device_fingerprint: str = ""

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be greater than 0")
        return v


class PredictResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    xgb_probability: float
    isolation_forest_score: float
    threshold_used: float
    processing_ms: int


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
