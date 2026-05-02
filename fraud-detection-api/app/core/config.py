import os
from functools import lru_cache


class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")
    model_path: str = os.getenv("MODEL_PATH", "./app/ml/artifacts/")
    fraud_threshold: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    anomaly_contamination: float = float(os.getenv("ANOMALY_CONTAMINATION", "0.01"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
