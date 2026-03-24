"""
signal_engine.py
----------------
Combines technical indicators and LLM news sentiment into a
final BUY / SELL / HOLD signal using a weighted scoring approach.

Scoring rules are kept explicit and easy to tune via constants.
"""

from app.config import settings
from app.models.schemas import (
    IndicatorValues,
    LLMAnalysisResponse,
    NewsSignalBreakdown,
    SignalResponse,
    TechnicalSignalBreakdown,
)
from app.utils.helpers import clamp
from app.utils.logger import logger


# ── Technical Scoring ─────────────────────────────────────────────────────────

# Max possible raw technical score (for normalization)
TECH_MAX_SCORE = 5   # five positive signals
TECH_MIN_SCORE = -1  # one negative signal (RSI overbought)


def score_technical(ind: IndicatorValues) -> TechnicalSignalBreakdown:
    """
    Apply rule-based scoring to the latest indicator values.

    Returns a TechnicalSignalBreakdown with individual signal flags
    and a normalized score in [-1, 1].
    """
    score = 0

    # ── Positive signals ──────────────────────────────────────────────────────
    close_above_sma20 = (
        ind.sma_20 is not None and ind.close > ind.sma_20
    )
    if close_above_sma20:
        score += 1

    sma20_above_sma50 = (
        ind.sma_20 is not None
        and ind.sma_50 is not None
        and ind.sma_20 > ind.sma_50
    )
    if sma20_above_sma50:
        score += 1

    rsi_healthy = (
        ind.rsi_14 is not None and 50 <= ind.rsi_14 <= 65
    )
    if rsi_healthy:
        score += 1

    macd_bullish = (
        ind.macd is not None
        and ind.macd_signal is not None
        and ind.macd > ind.macd_signal
    )
    if macd_bullish:
        score += 1

    if ind.volume_spike:
        score += 1

    # ── Negative signals ──────────────────────────────────────────────────────
    rsi_overbought = (
        ind.rsi_14 is not None and ind.rsi_14 > 75
    )
    if rsi_overbought:
        score -= 1

    # ── Normalize to [-1, 1] ──────────────────────────────────────────────────
    span = TECH_MAX_SCORE - TECH_MIN_SCORE  # = 6
    normalized = clamp((score - TECH_MIN_SCORE) / span * 2 - 1, -1.0, 1.0)

    logger.debug(
        f"Technical score: raw={score} "
        f"[sma20={close_above_sma20}, sma_cross={sma20_above_sma50}, "
        f"rsi_ok={rsi_healthy}, rsi_ob={rsi_overbought}, "
        f"macd={macd_bullish}, vol={ind.volume_spike}] "
        f"normalized={round(normalized, 3)}"
    )

    return TechnicalSignalBreakdown(
        close_above_sma20=close_above_sma20,
        sma20_above_sma50=sma20_above_sma50,
        rsi_healthy=rsi_healthy,
        rsi_overbought=rsi_overbought,
        macd_bullish=macd_bullish,
        volume_spike=ind.volume_spike,
        raw_score=score,
        normalized_score=round(normalized, 4),
    )


# ── News Scoring ──────────────────────────────────────────────────────────────

# Max possible raw news score per article
NEWS_ARTICLE_MAX = 2
NEWS_ARTICLE_MIN = -2


def score_news(llm: LLMAnalysisResponse) -> NewsSignalBreakdown:
    """
    Convert LLM article analyses into a normalized news score.

    Each article contributes a vote:
        bullish + impact_strength > 0.6  → +2
        bullish + impact_strength <= 0.6 → +1
        neutral                          →  0
        bearish + impact_strength <= 0.6 → -1
        bearish + impact_strength > 0.6  → -2

    Returns a NewsSignalBreakdown with a normalized score in [-1, 1].
    """
    if not llm.analyses:
        return NewsSignalBreakdown(num_articles=0, raw_score=0.0, normalized_score=0.0)

    raw_total = 0.0
    for analysis in llm.analyses:
        label = analysis.sentiment_label
        strength = analysis.impact_strength

        if label == "bullish":
            raw_total += 2 if strength > 0.6 else 1
        elif label == "bearish":
            raw_total += -2 if strength > 0.6 else -1
        # neutral → 0

    # Normalize: divide by max possible score across all articles
    max_possible = len(llm.analyses) * NEWS_ARTICLE_MAX
    min_possible = len(llm.analyses) * NEWS_ARTICLE_MIN
    span = max_possible - min_possible

    if span == 0:
        normalized = 0.0
    else:
        normalized = clamp((raw_total - min_possible) / span * 2 - 1, -1.0, 1.0)

    logger.debug(
        f"News score: raw={raw_total} articles={len(llm.analyses)} normalized={round(normalized, 3)}"
    )

    return NewsSignalBreakdown(
        num_articles=len(llm.analyses),
        raw_score=round(raw_total, 2),
        normalized_score=round(normalized, 4),
    )


# ── Signal Fusion ─────────────────────────────────────────────────────────────

NEWS_WEIGHT = 0.6
TECH_WEIGHT = 0.4


def generate_signal(
    ticker: str,
    indicators: IndicatorValues,
    llm_analysis: LLMAnalysisResponse,
) -> SignalResponse:
    """
    Combine technical + news scores into a final BUY / SELL / HOLD signal.

    Formula:
        final_score = 0.6 * news_normalized + 0.4 * tech_normalized

    Thresholds (from settings):
        BUY  if final_score >= BUY_THRESHOLD
        SELL if final_score <= SELL_THRESHOLD
        HOLD otherwise

    Confidence:
        Defined as how far the score is from 0, clamped to [0, 1].
        A confidence below MIN_CONFIDENCE forces HOLD.

    Args:
        ticker:       Stock symbol
        indicators:   Latest indicator values
        llm_analysis: LLM analysis response

    Returns:
        SignalResponse
    """
    tech_breakdown = score_technical(indicators)
    news_breakdown = score_news(llm_analysis)

    final_score = round(
        NEWS_WEIGHT * news_breakdown.normalized_score
        + TECH_WEIGHT * tech_breakdown.normalized_score,
        4,
    )

    # Confidence = absolute magnitude of the score, normalized to [0, 1]
    confidence = round(min(abs(final_score), 1.0), 4)

    # Determine signal
    if confidence < settings.MIN_CONFIDENCE:
        signal = "HOLD"
        logger.info(f"[{ticker}] Confidence too low ({confidence}) — forcing HOLD")
    elif final_score >= settings.BUY_THRESHOLD:
        signal = "BUY"
    elif final_score <= settings.SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    logger.info(
        f"[{ticker}] Signal={signal} score={final_score} confidence={confidence} "
        f"(news={news_breakdown.normalized_score}, tech={tech_breakdown.normalized_score})"
    )

    return SignalResponse(
        ticker=ticker.upper(),
        signal=signal,
        final_score=final_score,
        confidence=confidence,
        technical=tech_breakdown,
        news=news_breakdown,
    )
