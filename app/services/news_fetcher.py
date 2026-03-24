"""
news_fetcher.py
---------------
Fetches recent financial news for a ticker.

Primary:  NewsAPI (https://newsapi.org) — free tier, 100 req/day
Fallback: yfinance .news property — no key required, but less structured

The fetcher is pluggable: swap in any provider by implementing the
NewsProvider protocol and updating get_news().
"""

from datetime import datetime, timedelta
from typing import Protocol

import requests
import yfinance as yf

from app.config import settings
from app.models.schemas import NewsArticle, NewsResponse
from app.utils.logger import logger


# ── Protocol (interface for news providers) ───────────────────────────────────

class NewsProvider(Protocol):
    def fetch(self, ticker: str, max_articles: int) -> list[NewsArticle]:
        ...


# ── NewsAPI Provider ──────────────────────────────────────────────────────────

class NewsAPIProvider:
    """
    Fetches news from newsapi.org.
    Requires NEWS_API_KEY in environment.
    Free tier: 100 requests/day, articles from past month.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def fetch(self, ticker: str, max_articles: int) -> list[NewsArticle]:
        if not settings.NEWS_API_KEY:
            logger.warning("NEWS_API_KEY not set — skipping NewsAPI provider")
            return []

        from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        params = {
            "q": ticker,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_articles,
            "apiKey": settings.NEWS_API_KEY,
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error(f"NewsAPI request failed: {exc}")
            return []

        articles = []
        for item in data.get("articles", []):
            articles.append(
                NewsArticle(
                    headline=item.get("title", ""),
                    source=item.get("source", {}).get("name"),
                    published_at=item.get("publishedAt"),
                    url=item.get("url"),
                    summary=item.get("description") or item.get("content"),
                )
            )

        logger.info(f"NewsAPI returned {len(articles)} articles for {ticker}")
        return articles


# ── yfinance Fallback Provider ────────────────────────────────────────────────

class YFinanceNewsProvider:
    """
    Uses yfinance's built-in .news property as a fallback.
    No API key needed. Returns up to ~10 items, metadata-light.
    """

    def fetch(self, ticker: str, max_articles: int) -> list[NewsArticle]:
        try:
            tkr = yf.Ticker(ticker)
            raw_news = tkr.news or []
        except Exception as exc:
            logger.error(f"yfinance news fetch failed for {ticker}: {exc}")
            return []

        articles = []
        for item in raw_news[:max_articles]:
            content = item.get("content", {})
            # yfinance v0.2.x returns nested 'content' dict
            if isinstance(content, dict):
                headline = content.get("title", item.get("title", ""))
                summary = content.get("summary") or content.get("description")
                pub_date = content.get("pubDate") or content.get("displayTime")
                url = content.get("canonicalUrl", {}).get("url") if isinstance(
                    content.get("canonicalUrl"), dict
                ) else content.get("url")
                source_info = content.get("provider", {})
                source = source_info.get("displayName") if isinstance(source_info, dict) else None
            else:
                # Older yfinance flat structure
                headline = item.get("title", "")
                summary = item.get("summary")
                pub_date = item.get("providerPublishTime")
                url = item.get("link")
                source = item.get("publisher")

                # Convert unix timestamp if needed
                if isinstance(pub_date, (int, float)):
                    pub_date = datetime.utcfromtimestamp(pub_date).isoformat()

            if not headline:
                continue

            articles.append(
                NewsArticle(
                    headline=headline,
                    source=source,
                    published_at=str(pub_date) if pub_date else None,
                    url=url,
                    summary=summary,
                )
            )

        logger.info(f"yfinance news returned {len(articles)} articles for {ticker}")
        return articles


# ── Main Entry Point ──────────────────────────────────────────────────────────

def get_news(ticker: str, max_articles: int | None = None) -> NewsResponse:
    """
    Fetch recent news for a ticker using the best available provider.

    Priority:
        1. NewsAPI (if key is set)
        2. yfinance fallback

    Args:
        ticker:       Stock symbol
        max_articles: Override for max number of articles

    Returns:
        NewsResponse Pydantic model
    """
    max_articles = max_articles or settings.NEWS_MAX_ARTICLES
    ticker = ticker.upper()

    articles: list[NewsArticle] = []

    # Try NewsAPI first
    if settings.NEWS_API_KEY:
        articles = NewsAPIProvider().fetch(ticker, max_articles)

    # Fall back to yfinance if NewsAPI returned nothing
    if not articles:
        logger.info(f"Falling back to yfinance news for {ticker}")
        articles = YFinanceNewsProvider().fetch(ticker, max_articles)

    return NewsResponse(
        ticker=ticker,
        articles=articles,
        count=len(articles),
        fetched_at=datetime.utcnow().isoformat(),
    )
