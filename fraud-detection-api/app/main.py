import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.limiter import limiter
from app.core.logger import get_logger
from app.api.routes.prediction import router as prediction_router
from app.api.routes.backtest import router as backtest_router
from app.api.routes.monitoring import router as monitoring_router
from app.api.routes.dashboard import router as dashboard_router

logger = get_logger(__name__)

app = FastAPI(title="Fraud Detection API", version="0.1.0")

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Static files ──────────────────────────────────────────────────────────────
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "dashboard", "static")
os.makedirs(_STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(prediction_router)
app.include_router(backtest_router)
app.include_router(monitoring_router)
app.include_router(dashboard_router)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Reformat all HTTPException details into the canonical error envelope:
      {"error": "...", "message": "...", "detail": "..."}
    If the handler already raised with a dict detail, pass it through unchanged.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        content = detail
    else:
        content = {
            "error": "http_error",
            "message": str(detail),
            "detail": "",
        }
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "detail": str(exc),
        },
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}
