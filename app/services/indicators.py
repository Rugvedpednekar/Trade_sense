import pandas as pd
import ta

from app.models.schemas import IndicatorValues
from app.utils.helpers import safe_round
from app.utils.logger import logger

VOLUME_SPIKE_MULTIPLIER = 1.5


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        logger.warning("Less than 50 bars — some indicators may be NaN")

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["RSI_14"] = ta.momentum.RSIIndicator(
        close=df["Close"], window=14
    ).rsi()

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_indicator = ta.trend.MACD(
        close=df["Close"], window_slow=26, window_fast=12, window_sign=9
    )
    df["MACD"]        = macd_indicator.macd()
    df["MACD_signal"] = macd_indicator.macd_signal()
    df["MACD_hist"]   = macd_indicator.macd_diff()

    # ── ATR ───────────────────────────────────────────────────────────────────
    df["ATR_14"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()

    # ── Volume Spike ──────────────────────────────────────────────────────────
    df["Avg_Volume_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Spike"]  = df["Volume"] > (
        df["Avg_Volume_20"] * VOLUME_SPIKE_MULTIPLIER
    )

    return df


def get_latest_indicators(ticker: str, df: pd.DataFrame) -> IndicatorValues:
    df = compute_indicators(df.copy())
    latest  = df.iloc[-1]
    date_str = (
        str(df.index[-1].date())
        if hasattr(df.index[-1], "date")
        else str(df.index[-1])
    )

    logger.debug(
        f"Indicators for {ticker} on {date_str}: "
        f"SMA20={safe_round(latest.get('SMA_20'))} "
        f"RSI={safe_round(latest.get('RSI_14'))} "
        f"MACD={safe_round(latest.get('MACD'))}"
    )

    return IndicatorValues(
        ticker=ticker.upper(),
        date=date_str,
        close=round(float(latest["Close"]), 4),
        sma_20=safe_round(latest.get("SMA_20")),
        sma_50=safe_round(latest.get("SMA_50")),
        rsi_14=safe_round(latest.get("RSI_14")),
        macd=safe_round(latest.get("MACD")),
        macd_signal=safe_round(latest.get("MACD_signal")),
        macd_hist=safe_round(latest.get("MACD_hist")),
        atr_14=safe_round(latest.get("ATR_14")),
        volume=float(latest["Volume"]),
        avg_volume_20=safe_round(latest.get("Avg_Volume_20")),
        volume_spike=bool(latest.get("Volume_Spike", False)),
    )