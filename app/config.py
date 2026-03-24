"""
config.py
---------
Centralized configuration loaded from environment variables.
Copy .env.example to .env and fill in your keys before running.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "TradeSense AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./tradesense.db")

    # ── AWS Bedrock ───────────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # ── News ─────────────────────────────────────────────────────────────────
    # NewsAPI (https://newsapi.org) — free tier gives 100 req/day
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    NEWS_MAX_ARTICLES: int = int(os.getenv("NEWS_MAX_ARTICLES", "8"))

    # ── Market Data ──────────────────────────────────────────────────────────
    MARKET_DATA_PERIOD: str = os.getenv("MARKET_DATA_PERIOD", "3mo")  # yfinance period string
    MARKET_DATA_INTERVAL: str = os.getenv("MARKET_DATA_INTERVAL", "1d")

    # ── Signal Engine ────────────────────────────────────────────────────────
    BUY_THRESHOLD: float = float(os.getenv("BUY_THRESHOLD", "0.3"))
    SELL_THRESHOLD: float = float(os.getenv("SELL_THRESHOLD", "-0.3"))
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.4"))

    # ── Risk Manager ─────────────────────────────────────────────────────────
    ATR_STOP_MULTIPLIER: float = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))
    RISK_REWARD_RATIO: float = float(os.getenv("RISK_REWARD_RATIO", "2.0"))

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list = ["*"]  # tighten in production


settings = Settings()
