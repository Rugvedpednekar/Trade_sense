"""
risk_manager.py
---------------
Computes entry, stop-loss, and target prices for a given signal
using ATR-based position sizing logic.

Also applies a minimum confidence gate — signals below the threshold
are marked as non-tradeable.
"""

from app.config import settings
from app.models.schemas import IndicatorValues, RiskParameters, SignalResponse
from app.utils.logger import logger


def compute_risk(
    signal_response: SignalResponse,
    indicators: IndicatorValues,
) -> RiskParameters:
    """
    Calculate risk parameters for a trade based on the signal and ATR.

    BUY:
        entry      = latest close
        stop_loss  = entry - ATR_MULTIPLIER * ATR
        target     = entry + RISK_REWARD_RATIO * (entry - stop_loss)

    SELL (short):
        entry      = latest close
        stop_loss  = entry + ATR_MULTIPLIER * ATR
        target     = entry - RISK_REWARD_RATIO * (stop_loss - entry)

    HOLD:
        No trade parameters computed; tradeable=False.

    Args:
        signal_response: SignalResponse from signal_engine
        indicators:      Latest indicator values (needs close + ATR)

    Returns:
        RiskParameters Pydantic model
    """
    signal = signal_response.signal
    confidence = signal_response.confidence
    entry = indicators.close
    atr = indicators.atr_14

    # Gate: if signal is HOLD or confidence too low, return non-tradeable params
    if signal == "HOLD" or confidence < settings.MIN_CONFIDENCE or atr is None:
        reason = (
            "HOLD signal"
            if signal == "HOLD"
            else (
                f"low confidence ({confidence})"
                if confidence < settings.MIN_CONFIDENCE
                else "ATR unavailable"
            )
        )
        logger.info(f"[{signal_response.ticker}] Non-tradeable: {reason}")
        return RiskParameters(
            signal=signal,
            entry=entry,
            stop_loss=entry,
            target=entry,
            risk_per_share=0.0,
            reward_per_share=0.0,
            risk_reward_ratio=0.0,
            atr_used=atr or 0.0,
            confidence=confidence,
            tradeable=False,
        )

    atr_offset = settings.ATR_STOP_MULTIPLIER * atr

    if signal == "BUY":
        stop_loss = round(entry - atr_offset, 4)
        risk_per_share = round(entry - stop_loss, 4)
        reward_per_share = round(settings.RISK_REWARD_RATIO * risk_per_share, 4)
        target = round(entry + reward_per_share, 4)

    else:  # SELL
        stop_loss = round(entry + atr_offset, 4)
        risk_per_share = round(stop_loss - entry, 4)
        reward_per_share = round(settings.RISK_REWARD_RATIO * risk_per_share, 4)
        target = round(entry - reward_per_share, 4)

    rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0.0

    logger.info(
        f"[{signal_response.ticker}] Risk params: "
        f"entry={entry} stop={stop_loss} target={target} "
        f"R:R={rr_ratio} ATR={atr}"
    )

    return RiskParameters(
        signal=signal,
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        risk_reward_ratio=rr_ratio,
        atr_used=round(atr, 4),
        confidence=confidence,
        tradeable=True,
    )
