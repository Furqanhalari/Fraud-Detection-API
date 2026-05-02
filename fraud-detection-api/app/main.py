from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes.prediction import router as prediction_router
from app.api.routes.backtest import router as backtest_router
from app.api.routes.monitoring import router as monitoring_router

app = FastAPI(title="Fraud Detection API", version="0.1.0")

app.include_router(prediction_router)
app.include_router(backtest_router)
app.include_router(monitoring_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
