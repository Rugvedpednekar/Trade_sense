"""
llm_analyzer.py
---------------
Sends each news article to an LLM and parses structured JSON analysis back.

Design decisions:
  - The LLM is ONLY asked to analyze existing text — it does NOT predict prices.
  - Output schema is strictly enforced via prompt + JSON validation.
  - The provider is pluggable: swap OpenAI for any other by implementing LLMClient.
  - On parse failure we return a neutral fallback so the pipeline never breaks.
"""

from __future__ import annotations
import os
import json
import re
from typing import Protocol
import boto3
import json

from app.config import settings
from app.models.schemas import (
    LLMArticleAnalysis,
    LLMAnalysisResponse,
    NewsArticle,
    NewsResponse,
)
from app.utils.logger import logger
from dotenv import load_dotenv

load_dotenv(override=True)


# ── Prompt Template ───────────────────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """You are a financial news analyst. Your job is to analyze a news article 
about a publicly traded company and return a structured JSON assessment.

CRITICAL RULES:
- Return ONLY valid JSON. No markdown, no explanation outside the JSON block.
- Do NOT predict the stock price or give investment advice.
- Only analyze what the text actually says.
- All fields are required.

Required JSON format:
{
  "company": "<full company name>",
  "ticker": "<TICKER>",
  "sentiment_label": "<bullish|bearish|neutral>",
  "sentiment_score": <float between -1.0 and 1.0>,
  "event_type": "<earnings|acquisition|macro|product|regulatory|management|legal|other>",
  "impact_strength": <float between 0.0 and 1.0>,
  "impact_horizon": "<intraday|short_term|medium_term|long_term>",
  "trade_bias": "<buy|sell|hold>",
  "summary": "<one sentence factual summary of what happened>",
  "explanation": "<one to two sentences explaining why this sentiment and bias>"
}

Definitions:
- sentiment_score: -1.0 = very bearish, 0.0 = neutral, 1.0 = very bullish
- impact_strength: 0.0 = no market impact expected, 1.0 = major market-moving event
- impact_horizon: intraday (<1 day), short_term (1-5 days), medium_term (1-4 weeks), long_term (>1 month)
"""


def build_user_prompt(ticker: str, article: NewsArticle) -> str:
    """Construct the per-article user message sent to the LLM."""
    text = f"Headline: {article.headline}"
    if article.summary:
        text += f"\n\nSummary/Description: {article.summary}"
    if article.source:
        text += f"\n\nSource: {article.source}"

    return (
        f"Ticker: {ticker}\n\n"
        f"{text}\n\n"
        "Analyze this article and return only the JSON object as specified."
    )


# ── LLM Client Protocol ───────────────────────────────────────────────────────

class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the raw string response from the model."""
        ...


# ── Amazon Nova ─────────────────────────────────────────────────────────────

class BedrockNovaClient:
    """
    Amazon Bedrock client for Nova models.
    Reads AWS credentials from environment variables.
    """

    def __init__(self) -> None:
        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self._model_id = settings.LLM_MODEL

    def complete(self, system: str, user: str) -> str:
        body = {
            "messages": [
                {"role": "user", "content": [{"text": user}]}
            ],
            "system": [{"text": system}],
            "inferenceConfig": {
                "temperature": settings.LLM_TEMPERATURE,
                "maxTokens": settings.LLM_MAX_TOKENS,
            }
        }

        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        return result["output"]["message"]["content"][0]["text"]

# ── JSON Parsing + Validation ─────────────────────────────────────────────────

VALID_SENTIMENT_LABELS = {"bullish", "bearish", "neutral"}
VALID_TRADE_BIASES = {"buy", "sell", "hold"}
VALID_IMPACT_HORIZONS = {"intraday", "short_term", "medium_term", "long_term"}


def _extract_json(raw: str) -> dict:
    """
    Extract JSON from the model's response.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}\nRaw response:\n{raw}") from exc


def _validate_and_coerce(data: dict, ticker: str, article: NewsArticle) -> LLMArticleAnalysis:
    """
    Validate fields and apply safe defaults/clamping for out-of-range values.
    """
    sentiment_label = str(data.get("sentiment_label", "neutral")).lower()
    if sentiment_label not in VALID_SENTIMENT_LABELS:
        sentiment_label = "neutral"

    trade_bias = str(data.get("trade_bias", "hold")).lower()
    if trade_bias not in VALID_TRADE_BIASES:
        trade_bias = "hold"

    impact_horizon = str(data.get("impact_horizon", "short_term")).lower()
    if impact_horizon not in VALID_IMPACT_HORIZONS:
        impact_horizon = "short_term"

    sentiment_score = max(-1.0, min(1.0, float(data.get("sentiment_score", 0.0))))
    impact_strength = max(0.0, min(1.0, float(data.get("impact_strength", 0.5))))

    return LLMArticleAnalysis(
        company=data.get("company", "Unknown"),
        ticker=data.get("ticker", ticker).upper(),
        sentiment_label=sentiment_label,
        sentiment_score=round(sentiment_score, 3),
        event_type=data.get("event_type", "other"),
        impact_strength=round(impact_strength, 3),
        impact_horizon=impact_horizon,
        trade_bias=trade_bias,
        summary=data.get("summary", ""),
        explanation=data.get("explanation", ""),
        headline=article.headline,
        source=article.source,
    )


def _neutral_fallback(ticker: str, article: NewsArticle, reason: str) -> LLMArticleAnalysis:
    """Return a neutral analysis when LLM call or parsing fails."""
    logger.warning(f"Using neutral fallback for '{article.headline[:60]}': {reason}")
    return LLMArticleAnalysis(
        company="Unknown",
        ticker=ticker.upper(),
        sentiment_label="neutral",
        sentiment_score=0.0,
        event_type="other",
        impact_strength=0.0,
        impact_horizon="short_term",
        trade_bias="hold",
        summary=article.headline,
        explanation=f"LLM analysis unavailable: {reason}",
        headline=article.headline,
        source=article.source,
    )


# ── Aggregate Helpers ─────────────────────────────────────────────────────────

def _aggregate_bias(analyses: list[LLMArticleAnalysis]) -> str:
    """Majority vote on trade_bias across all analyzed articles."""
    counts: dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
    for a in analyses:
        counts[a.trade_bias] = counts.get(a.trade_bias, 0) + 1
    return max(counts, key=counts.get)  # type: ignore


# ── Main Service Function ─────────────────────────────────────────────────────

def analyze_news(
    ticker: str,
    news_response: NewsResponse,
    client: LLMClient | None = None,
) -> LLMAnalysisResponse:
    """
    Run LLM analysis on each news article and return aggregate results.

    Args:
        ticker:        Stock symbol
        news_response: NewsResponse from the news service
        client:        Optional LLMClient override (for testing/swapping providers)

    Returns:
        LLMAnalysisResponse with per-article analyses and aggregated scores
    """
    ticker = ticker.upper()

    if not news_response.articles:
        logger.warning(f"No articles to analyze for {ticker}")
        return LLMAnalysisResponse(
            ticker=ticker,
            analyses=[],
            aggregate_sentiment=0.0,
            aggregate_trade_bias="hold",
        )

    # Initialize LLM client
    if client is None:
        try:
            client = BedrockNovaClient()
        except RuntimeError as exc:
            logger.error(str(exc))
            # Return neutral fallbacks for all articles
            fallbacks = [
                _neutral_fallback(ticker, article, "No LLM API key configured")
                for article in news_response.articles
            ]
            return LLMAnalysisResponse(
                ticker=ticker,
                analyses=fallbacks,
                aggregate_sentiment=0.0,
                aggregate_trade_bias="hold",
            )

    analyses: list[LLMArticleAnalysis] = []

    for article in news_response.articles:
        try:
            user_prompt = build_user_prompt(ticker, article)
            raw_response = client.complete(ANALYSIS_SYSTEM_PROMPT, user_prompt)
            parsed = _extract_json(raw_response)
            analysis = _validate_and_coerce(parsed, ticker, article)
            analyses.append(analysis)
            logger.debug(
                f"Analyzed: '{article.headline[:50]}' → "
                f"{analysis.sentiment_label} ({analysis.sentiment_score})"
            )
        except Exception as exc:
            analyses.append(_neutral_fallback(ticker, article, str(exc)))

    # Aggregate
    if analyses:
        avg_sentiment = round(
            sum(a.sentiment_score for a in analyses) / len(analyses), 3
        )
    else:
        avg_sentiment = 0.0

    aggregate_bias = _aggregate_bias(analyses) if analyses else "hold"

    logger.info(
        f"LLM analysis complete for {ticker}: "
        f"{len(analyses)} articles, avg_sentiment={avg_sentiment}, bias={aggregate_bias}"
    )

    return LLMAnalysisResponse(
        ticker=ticker,
        analyses=analyses,
        aggregate_sentiment=avg_sentiment,
        aggregate_trade_bias=aggregate_bias,
    )
