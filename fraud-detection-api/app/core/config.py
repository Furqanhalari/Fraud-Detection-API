import os
from functools import lru_cache


class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")
    model_path: str = os.getenv("MODEL_PATH", "./app/ml/artifacts/")
    fraud_threshold: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    anomaly_contamination: float = float(os.getenv("ANOMALY_CONTAMINATION", "0.01"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_key: str = os.getenv("API_KEY", "")

    # CORS — comma-separated origins; defaults to localhost dev ports
    allowed_origins: list[str] = [
        o.strip()
        for o in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:8000,http://localhost:5173,http://127.0.0.1:8000",
        ).split(",")
        if o.strip()
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
