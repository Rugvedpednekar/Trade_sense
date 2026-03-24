"""
market_data.py
--------------
Fetches historical OHLCV price data using yfinance.
Returns a clean DataFrame and a structured Pydantic response.
"""

from datetime import datetime
import pandas as pd
import yfinance as yf

from app.config import settings
from app.models.schemas import MarketDataResponse, OHLCVBar
from app.utils.logger import logger


def fetch_price_data(ticker: str, period: str | None = None, interval: str | None = None) -> pd.DataFrame:
    """
    Download historical OHLCV data for a ticker.

    Args:
        ticker:   Stock symbol, e.g. "AAPL"
        period:   yfinance period string, e.g. "3mo", "6mo", "1y"
        interval: Bar size, e.g. "1d", "1h"

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date.
        Raises ValueError if no data is returned.
    """
    ticker = ticker.upper().strip()
    period = period or settings.MARKET_DATA_PERIOD
    interval = interval or settings.MARKET_DATA_INTERVAL

    logger.info(f"Fetching market data: ticker={ticker} period={period} interval={interval}")

    try:
        raw: pd.DataFrame = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception as exc:
        logger.exception(f"yfinance download failed for {ticker}: {exc}")
        raise RuntimeError(f"Failed to download data for {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        logger.warning(f"No market data returned for ticker={ticker}")
        raise ValueError(f"No market data returned for ticker '{ticker}'. Check the symbol.")

    # Flatten multi-level columns if needed
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in cols_needed if col not in raw.columns]
    if missing_cols:
        logger.error(f"Missing expected columns for {ticker}: {missing_cols}. Columns received: {list(raw.columns)}")
        raise RuntimeError(f"Incomplete market data returned for {ticker}. Missing columns: {missing_cols}")

    raw = raw[cols_needed].copy()
    raw.dropna(subset=["Close"], inplace=True)

    if raw.empty:
        logger.warning(f"All rows dropped after cleaning for ticker={ticker}")
        raise ValueError(f"No usable market data returned for ticker '{ticker}'.")

    raw.index.name = "Date"

    logger.info(f"Market data fetched successfully: {len(raw)} bars for {ticker}")
    return raw


def build_market_response(ticker: str, df: pd.DataFrame, period: str) -> MarketDataResponse:
    """
    Convert a price DataFrame into a structured API response.
    """
    bars: list[OHLCVBar] = []
    for date_idx, row in df.iterrows():
        bars.append(
            OHLCVBar(
                date=str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx),
                open=round(float(row["Open"]), 4),
                high=round(float(row["High"]), 4),
                low=round(float(row["Low"]), 4),
                close=round(float(row["Close"]), 4),
                volume=float(row["Volume"]),
            )
        )

    latest = df.iloc[-1]

    return MarketDataResponse(
        ticker=ticker.upper(),
        period=period,
        bars=bars,
        latest_close=round(float(latest["Close"]), 4),
        latest_volume=float(latest["Volume"]),
        fetched_at=datetime.utcnow().isoformat(),
    )


def get_market_data(ticker: str, period: str | None = None) -> MarketDataResponse:
    """
    High-level entry point: fetch + format market data.
    """
    period = period or settings.MARKET_DATA_PERIOD
    df = fetch_price_data(ticker, period=period)
    return build_market_response(ticker, df, period)
