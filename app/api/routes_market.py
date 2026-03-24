"""
routes_market.py
----------------
FastAPI routes for market data and technical indicators.

GET /market/{ticker}           — OHLCV bars for the last 3 months
GET /market/{ticker}/indicators — Latest technical indicator values
"""

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import IndicatorValues, MarketDataResponse
from app.services.indicators import get_latest_indicators
from app.services.market_data import fetch_price_data, get_market_data
from app.utils.logger import logger

router = APIRouter(prefix="/market", tags=["Market Data"])


@router.get("/{ticker}", response_model=MarketDataResponse)
def get_market(
    ticker: str,
    period: str = Query(default="3mo", description="yfinance period: 1mo, 3mo, 6mo, 1y"),
):
    """
    Return historical OHLCV bars for the given ticker.

    Example:
        GET /market/AAPL
        GET /market/TSLA?period=6mo
    """
    try:
        return get_market_data(ticker.upper(), period=period)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/{ticker}/indicators", response_model=IndicatorValues)
def get_indicators(
    ticker: str,
    period: str = Query(default="3mo", description="More data = more reliable indicators"),
):
    """
    Compute and return the latest technical indicator values for a ticker.

    Example:
        GET /market/AAPL/indicators
    """
    try:
        df = fetch_price_data(ticker.upper(), period=period)
        return get_latest_indicators(ticker.upper(), df)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
