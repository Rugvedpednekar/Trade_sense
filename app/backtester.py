"""
backtester.py
-------------
MVP backtester: replays the signal engine day-by-day over historical data
and tracks simulated trade outcomes.

This is intentionally simple for v1:
  - One position at a time (no concurrent trades)
  - Long only (no short selling)
  - Entry at next day's open, exit at stop-loss or target (checked against H/L)
  - No transaction costs or slippage in v1

TODOs for v2:
  - Short selling support
  - Position sizing by risk %
  - Transaction costs + slippage
  - Walk-forward validation
  - Per-trade P&L breakdown
  - Sharpe/Sortino/Max-DD metrics
"""

from datetime import datetime

import pandas as pd

from app.models.schemas import BacktestRequest, BacktestResult
from app.services.indicators import compute_indicators
from app.services.market_data import fetch_price_data
from app.utils.logger import logger


def run_backtest(request: BacktestRequest) -> BacktestResult:
    """
    Run a simplified signal-replay backtest over a historical period.

    Args:
        request: BacktestRequest with ticker, date range, and initial capital

    Returns:
        BacktestResult with summary statistics
    """
    ticker = request.ticker.upper()
    logger.info(f"Running backtest: {ticker} {request.start_date} → {request.end_date}")

    # ── 1. Fetch full historical data ─────────────────────────────────────────
    try:
        # yfinance supports start/end directly via download
        import yfinance as yf
        raw = yf.download(
            ticker,
            start=request.start_date,
            end=request.end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.dropna(subset=["Close"], inplace=True)
    except Exception as exc:
        logger.error(f"Backtest data fetch failed: {exc}")
        return BacktestResult(
            ticker=ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            note=f"Data fetch failed: {exc}",
        )

    if len(raw) < 60:
        return BacktestResult(
            ticker=ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            note="Insufficient data for backtest (need at least 60 bars).",
        )

    # ── 2. Compute indicators across full history ─────────────────────────────
    df = compute_indicators(raw.copy())

    # ── 3. Simulate trades ────────────────────────────────────────────────────
    capital = request.initial_capital
    peak_capital = capital
    max_drawdown = 0.0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0

    # Use rolling window: need at least 50 bars of history before generating signals
    warm_up = 50
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    target = 0.0

    for i in range(warm_up, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if in_trade:
            # Check if stop or target hit during next bar
            low = float(next_row["Low"])
            high = float(next_row["High"])
            open_price = float(next_row["Open"])

            hit_stop = low <= stop_loss
            hit_target = high >= target

            if hit_stop and hit_target:
                # Assume stop hit first (conservative)
                pnl_pct = (stop_loss - entry_price) / entry_price
            elif hit_stop:
                pnl_pct = (stop_loss - entry_price) / entry_price
            elif hit_target:
                pnl_pct = (target - entry_price) / entry_price
            else:
                continue  # still in trade

            capital *= (1 + pnl_pct)
            total_trades += 1
            if pnl_pct > 0:
                winning_trades += 1
            else:
                losing_trades += 1

            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital * 100
            max_drawdown = max(max_drawdown, drawdown)

            in_trade = False

        else:
            # Generate a simplified technical signal (no LLM in backtest for speed)
            close = float(row["Close"])
            sma20 = row.get("SMA_20")
            sma50 = row.get("SMA_50")
            rsi = row.get("RSI_14")
            macd = row.get("MACD")
            macd_sig = row.get("MACD_signal")
            atr = row.get("ATR_14")
            vol_spike = bool(row.get("Volume_Spike", False))

            if any(pd.isna(x) for x in [sma20, sma50, rsi, macd, macd_sig, atr]):
                continue

            score = 0
            if close > sma20:
                score += 1
            if sma20 > sma50:
                score += 1
            if 50 <= rsi <= 65:
                score += 1
            if macd > macd_sig:
                score += 1
            if vol_spike:
                score += 1
            if rsi > 75:
                score -= 1

            if score >= 3 and not in_trade:
                entry_price = float(next_row["Open"])
                atr_val = float(atr)
                stop_loss = entry_price - 1.5 * atr_val
                target = entry_price + 2 * (entry_price - stop_loss)
                in_trade = True

    # ── 4. Summarize ──────────────────────────────────────────────────────────
    total_return_pct = round((capital - request.initial_capital) / request.initial_capital * 100, 2)
    win_rate = round(winning_trades / total_trades * 100, 1) if total_trades > 0 else 0.0

    logger.info(
        f"Backtest complete: {ticker} trades={total_trades} win_rate={win_rate}% "
        f"return={total_return_pct}% max_dd={round(max_drawdown, 2)}%"
    )

    return BacktestResult(
        ticker=ticker,
        start_date=request.start_date,
        end_date=request.end_date,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_return_pct=total_return_pct,
        max_drawdown_pct=round(max_drawdown, 2),
        note=(
            "Technical-only backtest (no LLM signals for speed). "
            "No transaction costs. Long only. For educational use only."
        ),
    )
