"""
routes_signals.py
-----------------
FastAPI routes for signal generation and full analysis.

GET  /signal/{ticker}    — Technical-only signal (fast, no LLM)
POST /analyze/{ticker}   — Full pipeline: market + news + LLM + signal + risk
"""

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import FullAnalysisResponse, RiskParameters, SignalResponse
from app.services.indicators import get_latest_indicators
from app.services.llm_analyzer import analyze_news
from app.services.market_data import fetch_price_data, get_market_data
from app.services.news_fetcher import get_news
from app.services.risk_manager import compute_risk
from app.services.signal_engine import generate_signal
from app.utils.logger import logger

router = APIRouter(tags=["Signals"])


@router.get("/signal/{ticker}", response_model=SignalResponse)
def get_signal(
    ticker: str,
    period: str = Query(default="3mo"),
):
    """
    Generate a trading signal using ONLY technical indicators (no LLM, fast).
    Useful for quick checks or when you want to skip news analysis.

    Example:
        GET /signal/AAPL
    """
    try:
        df = fetch_price_data(ticker.upper(), period=period)
        indicators = get_latest_indicators(ticker.upper(), df)

        # Pass empty LLM analysis so the signal uses technical score only
        from app.models.schemas import LLMAnalysisResponse
        empty_llm = LLMAnalysisResponse(
            ticker=ticker.upper(),
            analyses=[],
            aggregate_sentiment=0.0,
            aggregate_trade_bias="hold",
        )

        return generate_signal(ticker.upper(), indicators, empty_llm)

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Signal generation error for {ticker}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/analyze/{ticker}", response_model=FullAnalysisResponse)
def full_analyze(
    ticker: str,
    period: str = Query(default="3mo"),
    max_articles: int = Query(default=8, ge=1, le=20),
):
    """
    Run the full TradeSense AI pipeline for a ticker:
      1. Fetch market data
      2. Compute technical indicators
      3. Fetch recent news
      4. Analyze news with LLM
      5. Generate combined signal
      6. Compute risk parameters

    Requires OPENAI_API_KEY for LLM analysis.

    Example:
        POST /analyze/AAPL
        POST /analyze/TSLA?period=6mo&max_articles=5
    """
    ticker = ticker.upper()
    logger.info(f"Full analysis requested for {ticker}")

    # ── Step 1: Market Data ───────────────────────────────────────────────────
    try:
        df = fetch_price_data(ticker, period=period)
        market = get_market_data(ticker, period=period)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=f"Market data error: {exc}")

    # ── Step 2: Technical Indicators ──────────────────────────────────────────
    try:
        indicators = get_latest_indicators(ticker, df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indicator error: {exc}")

    # ── Step 3: News ──────────────────────────────────────────────────────────
    try:
        news = get_news(ticker, max_articles=max_articles)
    except Exception as exc:
        logger.warning(f"News fetch failed for {ticker}: {exc} — continuing without news")
        from app.models.schemas import NewsResponse
        news = NewsResponse(ticker=ticker, articles=[], count=0)

    # ── Step 4: LLM Analysis ──────────────────────────────────────────────────
    try:
        llm_analysis = analyze_news(ticker, news)
    except Exception as exc:
        logger.warning(f"LLM analysis failed for {ticker}: {exc} — using neutral")
        from app.models.schemas import LLMAnalysisResponse
        llm_analysis = LLMAnalysisResponse(
            ticker=ticker,
            analyses=[],
            aggregate_sentiment=0.0,
            aggregate_trade_bias="hold",
        )

    # ── Step 5: Signal ────────────────────────────────────────────────────────
    signal = generate_signal(ticker, indicators, llm_analysis)

    # ── Step 6: Risk ──────────────────────────────────────────────────────────
    risk = compute_risk(signal, indicators)

    return FullAnalysisResponse(
        ticker=ticker,
        market=market,
        indicators=indicators,
        news=news,
        llm_analysis=llm_analysis,
        signal=signal,
        risk=risk,
    )
