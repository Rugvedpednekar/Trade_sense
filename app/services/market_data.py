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
    period = period or settings.MARKET_DATA_PERIOD
    interval = interval or settings.MARKET_DATA_INTERVAL

    logger.info(f"Fetching market data: ticker={ticker} period={period} interval={interval}")

    try:
        raw: pd.DataFrame = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,   # adjusts for splits & dividends
            progress=False,
        )
    except Exception as exc:
        logger.error(f"yfinance download failed for {ticker}: {exc}")
        raise RuntimeError(f"Failed to download data for {ticker}: {exc}") from exc

    if raw.empty:
        raise ValueError(f"No market data returned for ticker '{ticker}'. Check the symbol.")

    # Flatten multi-level columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only the columns we care about
    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    raw = raw[cols_needed].copy()

    # Drop rows where Close is NaN (e.g. weekends in intraday data edge cases)
    raw.dropna(subset=["Close"], inplace=True)

    # Ensure index is named Date
    raw.index.name = "Date"

    logger.info(f"Market data fetched: {len(raw)} bars for {ticker}")
    return raw


def build_market_response(ticker: str, df: pd.DataFrame, period: str) -> MarketDataResponse:
    """
    Convert a price DataFrame into a structured API response.

    Args:
        ticker: Stock symbol
        df:     Clean OHLCV DataFrame from fetch_price_data
        period: Period string used for the fetch

    Returns:
        MarketDataResponse Pydantic model
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

    Args:
        ticker: Stock symbol
        period: Optional override for the fetch period

    Returns:
        MarketDataResponse
    """
    period = period or settings.MARKET_DATA_PERIOD
    df = fetch_price_data(ticker, period=period)
    return build_market_response(ticker, df, period)
