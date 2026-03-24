"""
database.py
-----------
SQLAlchemy ORM models and session management for SQLite.
Tables: news_articles, llm_analysis, technical_indicators, trade_signals, backtest_results
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from app.config import settings


# ── Engine & Session ──────────────────────────────────────────────────────────

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI threads
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class NewsArticleDB(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    headline = Column(Text, nullable=False)
    source = Column(String, nullable=True)
    published_at = Column(String, nullable=True)
    url = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class LLMAnalysisDB(Base):
    __tablename__ = "llm_analysis"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    headline = Column(Text, nullable=True)
    sentiment_label = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    event_type = Column(String, nullable=True)
    impact_strength = Column(Float, nullable=True)
    impact_horizon = Column(String, nullable=True)
    trade_bias = Column(String, nullable=False)
    summary = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow)


class TechnicalIndicatorDB(Base):
    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(String, nullable=False)
    close = Column(Float, nullable=False)
    sma_20 = Column(Float, nullable=True)
    sma_50 = Column(Float, nullable=True)
    rsi_14 = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    atr_14 = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    volume_spike = Column(Boolean, default=False)
    computed_at = Column(DateTime, default=datetime.utcnow)


class TradeSignalDB(Base):
    __tablename__ = "trade_signals"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    signal = Column(String, nullable=False)   # BUY | SELL | HOLD
    final_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    entry = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow)


class BacktestResultDB(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    start_date = Column(String, nullable=False)
    end_date = Column(String, nullable=False)
    total_trades = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)
    total_return_pct = Column(Float, nullable=True)
    run_at = Column(DateTime, default=datetime.utcnow)


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables in the database (idempotent)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session and ensures it's closed."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
