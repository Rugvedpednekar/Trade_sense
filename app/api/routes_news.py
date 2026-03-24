"""
routes_news.py
--------------
FastAPI routes for news fetching and LLM analysis.

GET  /news/{ticker}       — Fetch latest news articles
POST /news/{ticker}/analyze — Run LLM sentiment analysis on fetched news
"""

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import LLMAnalysisResponse, NewsResponse
from app.services.llm_analyzer import analyze_news
from app.services.news_fetcher import get_news
from app.utils.logger import logger

router = APIRouter(prefix="/news", tags=["News"])


@router.get("/{ticker}", response_model=NewsResponse)
def fetch_news(
    ticker: str,
    max_articles: int = Query(default=8, ge=1, le=20, description="Number of articles to fetch"),
):
    """
    Fetch recent news articles for the ticker.

    Example:
        GET /news/AAPL
        GET /news/NVDA?max_articles=5
    """
    try:
        return get_news(ticker.upper(), max_articles=max_articles)
    except Exception as exc:
        logger.error(f"News fetch error for {ticker}: {exc}")
        raise HTTPException(status_code=502, detail=f"News fetch failed: {exc}")


@router.post("/{ticker}/analyze", response_model=LLMAnalysisResponse)
def analyze_ticker_news(
    ticker: str,
    max_articles: int = Query(default=8, ge=1, le=20),
):
    """
    Fetch news for a ticker and run LLM sentiment analysis on each article.

    Requires OPENAI_API_KEY in environment.

    Example:
        POST /news/AAPL/analyze
    """
    try:
        news = get_news(ticker.upper(), max_articles=max_articles)
        return analyze_news(ticker.upper(), news)
    except Exception as exc:
        logger.error(f"LLM analysis error for {ticker}: {exc}")
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")
