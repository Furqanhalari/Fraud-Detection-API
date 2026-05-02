import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import settings
from app.core.database import check_db_connection
from app.core.limiter import limiter
from app.core.logger import get_logger
from app.core.security import install_ip_mask_filter
from app.ml.ensemble import FraudEnsemble

logger = get_logger(__name__)

# Install IP-masking log filter before any log output
install_ip_mask_filter()


# ── Lifespan: load models ONCE at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()
    try:
        ensemble = FraudEnsemble()
        ensemble.load_models()
        app.state.ensemble = ensemble
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("Models loaded at startup in %.1f ms", elapsed_ms)
    except FileNotFoundError as exc:
        logger.warning("Models not found at startup — predict will return 503: %s", exc)
        app.state.ensemble = None

    yield  # application runs here

    logger.info("Application shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Set safe defaults — lifespan overwrites ensemble after startup completes
app.state.ensemble = None

# ── CORS ──────────────────────────────────────────────────────────────────────
# Configurable via ALLOWED_ORIGINS env var (comma-separated).
# Defaults to localhost dev ports; set to your production domain(s) in Replit Secrets.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Static files ──────────────────────────────────────────────────────────────
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "dashboard", "static")
os.makedirs(_STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.routes.prediction import router as prediction_router
from app.api.routes.backtest import router as backtest_router
from app.api.routes.monitoring import router as monitoring_router
from app.api.routes.dashboard import router as dashboard_router

app.include_router(prediction_router)
app.include_router(backtest_router)
app.include_router(monitoring_router)
app.include_router(dashboard_router)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        content = detail
    else:
        content = {"error": "http_error", "message": str(detail), "detail": ""}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc, exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "detail": str(exc),
        },
    )


# ── Health (no API key required — public) ─────────────────────────────────────

@app.get("/api/health")
def health(request: Request):
    """
    Public health-check endpoint. Safe to use as a Replit deployment probe.

    Returns:
      status       — always "ok" when the server is reachable
      models_loaded — true once train.py has been run and models are on disk
      db_connected  — true if the database responded to SELECT 1
      timestamp     — Unix epoch seconds (UTC)
    """
    return {
        "status": "ok",
        "models_loaded": request.app.state.ensemble is not None,
        "db_connected": check_db_connection(),
        "timestamp": time.time(),
    }
