"""
schemas.py
----------
All Pydantic v2 schemas used as request/response models across the API.
Keeping them in one place makes it easy to see the full data contract.
"""

from __future__ import annotations
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ── Market Data ───────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataResponse(BaseModel):
    ticker: str
    period: str
    bars: list[OHLCVBar]
    latest_close: float
    latest_volume: float
    fetched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Technical Indicators ─────────────────────────────────────────────────────

class IndicatorValues(BaseModel):
    ticker: str
    date: str
    close: float
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    atr_14: Optional[float] = None
    volume: float
    avg_volume_20: Optional[float] = None
    volume_spike: bool = False
    computed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── News ─────────────────────────────────────────────────────────────────────

class NewsArticle(BaseModel):
    headline: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None  # snippet or description


class NewsResponse(BaseModel):
    ticker: str
    articles: list[NewsArticle]
    count: int
    fetched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── LLM Analysis ─────────────────────────────────────────────────────────────

class LLMArticleAnalysis(BaseModel):
    company: str
    ticker: str
    sentiment_label: str          # bullish | bearish | neutral
    sentiment_score: float         # -1.0 to 1.0
    event_type: str               # earnings | acquisition | macro | product | regulatory | other
    impact_strength: float         # 0.0 to 1.0
    impact_horizon: str           # intraday | short_term | medium_term | long_term
    trade_bias: str               # buy | sell | hold
    summary: str
    explanation: str
    headline: str                 # original headline for reference
    source: Optional[str] = None


class LLMAnalysisResponse(BaseModel):
    ticker: str
    analyses: list[LLMArticleAnalysis]
    aggregate_sentiment: float    # average sentiment_score across articles
    aggregate_trade_bias: str     # majority vote
    analyzed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Signal Engine ─────────────────────────────────────────────────────────────

class TechnicalSignalBreakdown(BaseModel):
    close_above_sma20: bool
    sma20_above_sma50: bool
    rsi_healthy: bool             # RSI between 50-65
    rsi_overbought: bool          # RSI > 75 (negative signal)
    macd_bullish: bool
    volume_spike: bool
    raw_score: int                # sum of +1/-1 votes
    normalized_score: float       # -1 to 1


class NewsSignalBreakdown(BaseModel):
    num_articles: int
    raw_score: float
    normalized_score: float       # -1 to 1


class SignalResponse(BaseModel):
    ticker: str
    signal: str                   # BUY | SELL | HOLD
    final_score: float            # -1 to 1
    confidence: float             # 0 to 1
    technical: TechnicalSignalBreakdown
    news: NewsSignalBreakdown
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Risk Management ──────────────────────────────────────────────────────────

class RiskParameters(BaseModel):
    signal: str
    entry: float
    stop_loss: float
    target: float
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float
    atr_used: float
    confidence: float
    tradeable: bool               # False if confidence too low


# ── Full Analysis (combined) ──────────────────────────────────────────────────

class FullAnalysisResponse(BaseModel):
    ticker: str
    market: MarketDataResponse
    indicators: IndicatorValues
    news: NewsResponse
    llm_analysis: LLMAnalysisResponse
    signal: SignalResponse
    risk: RiskParameters
    analyzed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Backtesting (stub) ────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 10000.0


class BacktestResult(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float] = None
    note: str = "Backtesting stub — implement signal replay logic here."


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
