"""
main.py
-------
FastAPI application entry point for TradeSense AI.

Registers all routers, configures CORS, creates DB tables on startup,
and exposes a /health endpoint.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_backtest, routes_market, routes_news, routes_signals
from app.config import settings
from app.models.database import create_tables
from app.models.schemas import HealthResponse
from app.utils.logger import logger

# ── App Instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "TradeSense AI — Stock decision-support system combining "
        "technical analysis and LLM news sentiment. "
        "For educational/portfolio use only. Not financial advice."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    create_tables()
    logger.info("Database tables ready")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Liveness check. Returns 200 if the API is running.

    Example:
        GET /health
    """
    return HealthResponse(status="ok", version=settings.APP_VERSION)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(routes_market.router)
app.include_router(routes_news.router)
app.include_router(routes_signals.router)
app.include_router(routes_backtest.router)

logger.info("All routers registered")
