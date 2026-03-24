"""
test_services.py
----------------
Unit tests for core TradeSense AI services.
Run with: pytest tests/ -v

These tests do NOT call external APIs — they use local fixtures or mocks.
"""

import pandas as pd
import pytest

from app.models.schemas import (
    IndicatorValues,
    LLMAnalysisResponse,
    LLMArticleAnalysis,
    NewsArticle,
    NewsResponse,
)
from app.services.indicators import compute_indicators, get_latest_indicators
from app.services.risk_manager import compute_risk
from app.services.signal_engine import generate_signal, score_news, score_technical
from app.utils.helpers import clamp, normalize, safe_round


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate a simple rising price DataFrame for testing."""
    import numpy as np
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    closes = 150 + np.cumsum(np.random.normal(0.1, 1.0, n))
    df = pd.DataFrame({
        "Open": closes - 0.5,
        "High": closes + 1.0,
        "Low": closes - 1.0,
        "Close": closes,
        "Volume": np.random.randint(5_000_000, 20_000_000, n).astype(float),
    }, index=dates)
    df.index.name = "Date"
    return df


def make_indicator_values(**kwargs) -> IndicatorValues:
    defaults = dict(
        ticker="TEST",
        date="2024-01-01",
        close=150.0,
        sma_20=145.0,
        sma_50=140.0,
        rsi_14=55.0,
        macd=0.5,
        macd_signal=0.3,
        macd_hist=0.2,
        atr_14=2.5,
        volume=10_000_000.0,
        avg_volume_20=8_000_000.0,
        volume_spike=False,
    )
    defaults.update(kwargs)
    return IndicatorValues(**defaults)


def make_llm_analysis(sentiment_label="bullish", sentiment_score=0.7,
                       impact_strength=0.7, trade_bias="buy") -> LLMAnalysisResponse:
    article = LLMArticleAnalysis(
        company="Test Corp",
        ticker="TEST",
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        event_type="earnings",
        impact_strength=impact_strength,
        impact_horizon="short_term",
        trade_bias=trade_bias,
        summary="Test summary",
        explanation="Test explanation",
        headline="Test headline",
    )
    return LLMAnalysisResponse(
        ticker="TEST",
        analyses=[article],
        aggregate_sentiment=sentiment_score,
        aggregate_trade_bias=trade_bias,
    )


# ── Helper Tests ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_safe_round_normal(self):
        assert safe_round(3.14159, 2) == 3.14

    def test_safe_round_none(self):
        assert safe_round(None) is None

    def test_safe_round_nan(self):
        import math
        assert safe_round(float("nan")) is None

    def test_clamp_within(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below(self):
        assert clamp(-2.0, -1.0, 1.0) == -1.0

    def test_clamp_above(self):
        assert clamp(2.0, -1.0, 1.0) == 1.0

    def test_normalize_midpoint(self):
        result = normalize(5.0, 0.0, 10.0)
        assert result == 0.0

    def test_normalize_max(self):
        result = normalize(10.0, 0.0, 10.0)
        assert result == 1.0

    def test_normalize_min(self):
        result = normalize(0.0, 0.0, 10.0)
        assert result == -1.0


# ── Indicator Tests ───────────────────────────────────────────────────────────

class TestIndicators:
    def test_compute_indicators_returns_df(self):
        df = make_ohlcv(100)
        result = compute_indicators(df)
        assert "SMA_20" in result.columns
        assert "SMA_50" in result.columns
        assert "RSI_14" in result.columns
        assert "MACD" in result.columns
        assert "ATR_14" in result.columns
        assert "Volume_Spike" in result.columns

    def test_get_latest_indicators_structure(self):
        df = make_ohlcv(100)
        ind = get_latest_indicators("TEST", df)
        assert ind.ticker == "TEST"
        assert ind.close > 0
        assert ind.sma_20 is not None
        assert ind.sma_50 is not None
        assert isinstance(ind.volume_spike, bool)

    def test_insufficient_data_does_not_crash(self):
        """With < 50 bars, SMA_50 will be NaN — should not raise."""
        df = make_ohlcv(30)
        ind = get_latest_indicators("SHORT", df)
        assert ind.sma_50 is None  # not enough data


# ── Signal Engine Tests ───────────────────────────────────────────────────────

class TestSignalEngine:
    def test_technical_score_bullish_setup(self):
        """Close > SMA20 > SMA50, healthy RSI, MACD bullish → high score."""
        ind = make_indicator_values(
            close=155.0, sma_20=150.0, sma_50=145.0,
            rsi_14=58.0, macd=1.0, macd_signal=0.5, volume_spike=True
        )
        breakdown = score_technical(ind)
        assert breakdown.close_above_sma20 is True
        assert breakdown.sma20_above_sma50 is True
        assert breakdown.rsi_healthy is True
        assert breakdown.macd_bullish is True
        assert breakdown.volume_spike is True
        assert breakdown.raw_score == 5
        assert breakdown.normalized_score > 0.5

    def test_technical_score_bearish_setup(self):
        """Close < SMA20, RSI overbought → low score."""
        ind = make_indicator_values(
            close=140.0, sma_20=145.0, sma_50=142.0,
            rsi_14=80.0, macd=-0.5, macd_signal=0.3, volume_spike=False
        )
        breakdown = score_technical(ind)
        assert breakdown.close_above_sma20 is False
        assert breakdown.rsi_overbought is True
        assert breakdown.raw_score < 0

    def test_news_score_bullish_strong(self):
        """Single strong bullish article → positive news score."""
        llm = make_llm_analysis(
            sentiment_label="bullish", impact_strength=0.8, trade_bias="buy"
        )
        breakdown = score_news(llm)
        assert breakdown.num_articles == 1
        assert breakdown.raw_score == 2.0
        assert breakdown.normalized_score > 0

    def test_news_score_bearish_strong(self):
        llm = make_llm_analysis(
            sentiment_label="bearish", sentiment_score=-0.8,
            impact_strength=0.9, trade_bias="sell"
        )
        breakdown = score_news(llm)
        assert breakdown.raw_score == -2.0
        assert breakdown.normalized_score < 0

    def test_news_score_empty(self):
        empty_llm = LLMAnalysisResponse(
            ticker="TEST", analyses=[], aggregate_sentiment=0.0, aggregate_trade_bias="hold"
        )
        breakdown = score_news(empty_llm)
        assert breakdown.normalized_score == 0.0
        assert breakdown.num_articles == 0

    def test_generate_signal_buy(self):
        """Strongly bullish inputs → BUY signal."""
        ind = make_indicator_values(
            close=155.0, sma_20=150.0, sma_50=145.0,
            rsi_14=58.0, macd=1.0, macd_signal=0.5, volume_spike=True
        )
        llm = make_llm_analysis(
            sentiment_label="bullish", sentiment_score=0.9, impact_strength=0.9
        )
        result = generate_signal("TEST", ind, llm)
        assert result.signal == "BUY"
        assert result.final_score > 0

    def test_generate_signal_low_confidence_forces_hold(self):
        """Near-zero score → confidence too low → HOLD."""
        ind = make_indicator_values(
            close=150.0, sma_20=150.5, sma_50=148.0,  # close barely below sma20
            rsi_14=50.0, macd=0.01, macd_signal=0.01, volume_spike=False
        )
        empty_llm = LLMAnalysisResponse(
            ticker="TEST", analyses=[], aggregate_sentiment=0.0, aggregate_trade_bias="hold"
        )
        result = generate_signal("TEST", ind, empty_llm)
        # With near-zero scores, confidence should be low → HOLD
        assert result.signal == "HOLD"


# ── Risk Manager Tests ────────────────────────────────────────────────────────

class TestRiskManager:
    def test_buy_risk_params(self):
        ind = make_indicator_values(close=150.0, atr_14=2.0)

        from app.models.schemas import SignalResponse, TechnicalSignalBreakdown, NewsSignalBreakdown
        tech = TechnicalSignalBreakdown(
            close_above_sma20=True, sma20_above_sma50=True, rsi_healthy=True,
            rsi_overbought=False, macd_bullish=True, volume_spike=False,
            raw_score=4, normalized_score=0.83
        )
        news = NewsSignalBreakdown(num_articles=1, raw_score=2, normalized_score=1.0)
        sig = SignalResponse(
            ticker="TEST", signal="BUY", final_score=0.9, confidence=0.9,
            technical=tech, news=news
        )

        risk = compute_risk(sig, ind)
        assert risk.tradeable is True
        assert risk.entry == 150.0
        assert risk.stop_loss < risk.entry
        assert risk.target > risk.entry
        assert risk.risk_reward_ratio == 2.0

    def test_hold_is_not_tradeable(self):
        ind = make_indicator_values(close=150.0, atr_14=2.0)

        from app.models.schemas import SignalResponse, TechnicalSignalBreakdown, NewsSignalBreakdown
        tech = TechnicalSignalBreakdown(
            close_above_sma20=False, sma20_above_sma50=False, rsi_healthy=False,
            rsi_overbought=False, macd_bullish=False, volume_spike=False,
            raw_score=0, normalized_score=0.0
        )
        news = NewsSignalBreakdown(num_articles=0, raw_score=0, normalized_score=0.0)
        sig = SignalResponse(
            ticker="TEST", signal="HOLD", final_score=0.0, confidence=0.0,
            technical=tech, news=news
        )

        risk = compute_risk(sig, ind)
        assert risk.tradeable is False

    def test_sell_risk_params(self):
        """Stop loss is above entry for SELL signals."""
        ind = make_indicator_values(close=150.0, atr_14=2.0)

        from app.models.schemas import SignalResponse, TechnicalSignalBreakdown, NewsSignalBreakdown
        tech = TechnicalSignalBreakdown(
            close_above_sma20=False, sma20_above_sma50=False, rsi_healthy=False,
            rsi_overbought=True, macd_bullish=False, volume_spike=False,
            raw_score=-1, normalized_score=-0.67
        )
        news = NewsSignalBreakdown(num_articles=1, raw_score=-2, normalized_score=-1.0)
        sig = SignalResponse(
            ticker="TEST", signal="SELL", final_score=-0.87, confidence=0.87,
            technical=tech, news=news
        )

        risk = compute_risk(sig, ind)
        assert risk.tradeable is True
        assert risk.stop_loss > risk.entry
        assert risk.target < risk.entry


# ── LLM Analyzer Tests (no API calls) ────────────────────────────────────────

class TestLLMAnalyzer:
    def test_neutral_fallback_on_no_api_key(self):
        """When OpenAI key is missing, should return neutral fallbacks."""
        import os
        original = os.environ.get("OPENAI_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = ""

        from app.services import llm_analyzer
        llm_analyzer.settings.OPENAI_API_KEY = ""

        news = NewsResponse(
            ticker="TEST",
            articles=[
                NewsArticle(headline="Test headline", source="Reuters")
            ],
            count=1,
        )

        result = llm_analyzer.analyze_news("TEST", news)

        # Restore
        os.environ["OPENAI_API_KEY"] = original
        llm_analyzer.settings.OPENAI_API_KEY = original

        assert result.ticker == "TEST"
        assert len(result.analyses) == 1
        assert result.analyses[0].sentiment_label == "neutral"
        assert result.analyses[0].trade_bias == "hold"

    def test_json_extraction_clean(self):
        from app.services.llm_analyzer import _extract_json
        raw = '{"sentiment_label": "bullish", "sentiment_score": 0.8}'
        result = _extract_json(raw)
        assert result["sentiment_label"] == "bullish"

    def test_json_extraction_with_markdown_fences(self):
        from app.services.llm_analyzer import _extract_json
        raw = '```json\n{"sentiment_label": "bearish"}\n```'
        result = _extract_json(raw)
        assert result["sentiment_label"] == "bearish"

    def test_validate_and_coerce_clamps_score(self):
        from app.services.llm_analyzer import _validate_and_coerce
        data = {
            "company": "Test", "ticker": "TEST",
            "sentiment_label": "bullish",
            "sentiment_score": 5.0,   # out of range
            "event_type": "earnings",
            "impact_strength": -0.5,  # out of range
            "impact_horizon": "short_term",
            "trade_bias": "buy",
            "summary": "s", "explanation": "e"
        }
        article = NewsArticle(headline="test")
        result = _validate_and_coerce(data, "TEST", article)
        assert result.sentiment_score == 1.0   # clamped
        assert result.impact_strength == 0.0   # clamped

    def test_validate_and_coerce_invalid_label_defaults_to_neutral(self):
        from app.services.llm_analyzer import _validate_and_coerce
        data = {
            "company": "Test", "ticker": "TEST",
            "sentiment_label": "very_positive",  # invalid
            "sentiment_score": 0.5,
            "event_type": "other",
            "impact_strength": 0.5,
            "impact_horizon": "not_real",  # invalid
            "trade_bias": "maybe",         # invalid
            "summary": "s", "explanation": "e"
        }
        article = NewsArticle(headline="test")
        result = _validate_and_coerce(data, "TEST", article)
        assert result.sentiment_label == "neutral"
        assert result.impact_horizon == "short_term"
        assert result.trade_bias == "hold"
