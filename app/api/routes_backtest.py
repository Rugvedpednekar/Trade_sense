"""
routes_backtest.py
------------------
FastAPI routes for backtesting.

POST /backtest/{ticker} — Run a historical backtest for a ticker
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import BacktestRequest, BacktestResult
from app.services.backtester import run_backtest
from app.utils.logger import logger

router = APIRouter(prefix="/backtest", tags=["Backtesting"])


@router.post("/{ticker}", response_model=BacktestResult)
def backtest_ticker(ticker: str, request: BacktestRequest | None = None):
    """
    Run a simplified backtest for the given ticker over a historical period.

    Uses technical-only signals (no LLM) for speed.
    For educational use only — not for real trading decisions.

    Example body (optional, defaults shown):
        {
          "ticker": "AAPL",
          "start_date": "2023-01-01",
          "end_date": "2024-01-01",
          "initial_capital": 10000.0
        }

    Example:
        POST /backtest/AAPL
    """
    if request is None:
        request = BacktestRequest(ticker=ticker)
    else:
        request.ticker = ticker.upper()

    try:
        return run_backtest(request)
    except Exception as exc:
        logger.error(f"Backtest error for {ticker}: {exc}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")
