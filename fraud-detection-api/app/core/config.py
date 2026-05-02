import os
from functools import lru_cache

# Load .env file if present (no-op in production when env vars are set directly)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass  # python-dotenv is optional; env vars must be set by the host


class Settings:
    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")

    # ── ML artifacts ──────────────────────────────────────────────────────────
    model_path: str = os.getenv("MODEL_PATH", "./app/ml/artifacts/")

    # ── Fraud scoring ──────────────────────────────────────────────────────────
    fraud_threshold: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    anomaly_contamination: float = float(os.getenv("ANOMALY_CONTAMINATION", "0.01"))

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Security ──────────────────────────────────────────────────────────────
    # Leave blank to disable API key auth in local development
    api_key: str = os.getenv("API_KEY", "")

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed browser origins.
    # Defaults to localhost dev ports when ALLOWED_ORIGINS is unset.
    allowed_origins: list[str] = [
        o.strip()
        for o in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:8000,http://localhost:5173,http://127.0.0.1:8000",
        ).split(",")
        if o.strip()
    ]

    # ── Server ─────────────────────────────────────────────────────────────────
    # Replit sets PORT automatically; falls back to 8000 for local dev
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "1"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
