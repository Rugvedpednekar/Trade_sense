# TradeSense AI

A modular stock market **decision-support system** combining technical indicators and LLM-powered news sentiment analysis. Built for learning, portfolio projects, and resume demonstration.

> ⚠️ **Not financial advice. For educational use only.**

---

## Architecture

```
User → Streamlit Dashboard → FastAPI Backend
                                  │
                    ┌─────────────┼──────────────────┐
                    │             │                  │
               Market Data    News Fetcher       LLM Analyzer
             (yfinance)     (NewsAPI/yfinance)   (OpenAI)
                    │             │                  │
               Indicators    Signal Engine      Risk Manager
             (pandas-ta)   (weighted scoring)  (ATR-based)
                    │
                 SQLite
```

## Features

- **Market Data** — 3 months of daily OHLCV via yfinance
- **Technical Indicators** — SMA 20/50, RSI 14, MACD, ATR 14, volume spike
- **News Fetching** — NewsAPI (primary) + yfinance fallback
- **LLM Analysis** — Structured JSON sentiment per article (sentiment, event type, impact horizon, trade bias)
- **Signal Engine** — Weighted score: 60% news sentiment + 40% technical (BUY/SELL/HOLD)
- **Risk Manager** — ATR-based entry / stop-loss / target with confidence gate
- **Backtester** — Technical-only signal replay over historical data
- **FastAPI** — Clean REST API with auto-generated docs at `/docs`
- **Streamlit** — Interactive dashboard with charts and sentiment breakdown

---

## Project Structure

```
trade_sense_ai/
├── app/
│   ├── main.py                   # FastAPI app entrypoint
│   ├── config.py                 # Centralized settings (env vars)
│   ├── api/
│   │   ├── routes_market.py      # GET /market/{ticker}
│   │   ├── routes_news.py        # GET /news/{ticker}
│   │   ├── routes_signals.py     # GET /signal/{ticker}, POST /analyze/{ticker}
│   │   └── routes_backtest.py    # POST /backtest/{ticker}
│   ├── services/
│   │   ├── market_data.py        # yfinance wrapper
│   │   ├── indicators.py         # pandas-ta indicator computation
│   │   ├── news_fetcher.py       # NewsAPI + yfinance fallback
│   │   ├── llm_analyzer.py       # OpenAI structured JSON analysis
│   │   ├── signal_engine.py      # Weighted scoring + signal generation
│   │   ├── risk_manager.py       # ATR-based stop/target computation
│   │   └── backtester.py         # Historical signal replay
│   ├── models/
│   │   ├── schemas.py            # Pydantic v2 request/response models
│   │   └── database.py           # SQLAlchemy ORM + SQLite
│   └── utils/
│       ├── helpers.py            # safe_round, normalize, clamp
│       └── logger.py             # Shared structured logger
├── dashboard/
│   └── streamlit_app.py          # Streamlit UI
├── tests/
│   └── test_services.py          # Unit tests (no API calls)
├── .env.example                  # Copy to .env and fill in keys
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / download the project

```bash
cd trade_sense_ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (for LLM) | Your OpenAI API key |
| `NEWS_API_KEY` | Recommended | Free key from newsapi.org |
| `LLM_MODEL` | No | Default: `gpt-4o-mini` |

The system works **without** `OPENAI_API_KEY` — it will return neutral sentiment fallbacks.  
The system works **without** `NEWS_API_KEY` — it falls back to yfinance news.

---

## Running

### Start the FastAPI backend

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### Start the Streamlit dashboard

In a separate terminal:

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard available at: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/market/{ticker}` | OHLCV bars |
| GET | `/market/{ticker}/indicators` | Latest indicator values |
| GET | `/news/{ticker}` | Recent news articles |
| POST | `/news/{ticker}/analyze` | LLM analysis of news |
| GET | `/signal/{ticker}` | Technical-only signal (fast) |
| POST | `/analyze/{ticker}` | Full pipeline (market + LLM + signal + risk) |
| POST | `/backtest/{ticker}` | Historical backtest |

### Sample Requests

```bash
# Health check
curl http://localhost:8000/health

# Market data
curl http://localhost:8000/market/AAPL

# Technical indicators
curl http://localhost:8000/market/AAPL/indicators

# Latest news
curl http://localhost:8000/news/TSLA

# Quick technical signal (no LLM)
curl http://localhost:8000/signal/NVDA

# Full AI analysis
curl -X POST http://localhost:8000/analyze/AAPL

# Backtest
curl -X POST http://localhost:8000/backtest/AAPL \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","start_date":"2023-01-01","end_date":"2024-01-01","initial_capital":10000}'
```

---

## Sample Output

### `GET /market/AAPL/indicators`

```json
{
  "ticker": "AAPL",
  "date": "2024-05-10",
  "close": 182.40,
  "sma_20": 177.82,
  "sma_50": 173.55,
  "rsi_14": 58.3,
  "macd": 1.24,
  "macd_signal": 0.98,
  "macd_hist": 0.26,
  "atr_14": 3.12,
  "volume": 54210000.0,
  "avg_volume_20": 61430000.0,
  "volume_spike": false,
  "computed_at": "2024-05-11T10:00:00"
}
```

### `POST /analyze/AAPL` (signal section)

```json
{
  "signal": {
    "ticker": "AAPL",
    "signal": "BUY",
    "final_score": 0.62,
    "confidence": 0.62,
    "technical": {
      "close_above_sma20": true,
      "sma20_above_sma50": true,
      "rsi_healthy": true,
      "rsi_overbought": false,
      "macd_bullish": true,
      "volume_spike": false,
      "raw_score": 4,
      "normalized_score": 0.833
    },
    "news": {
      "num_articles": 6,
      "raw_score": 7.0,
      "normalized_score": 0.583
    }
  },
  "risk": {
    "signal": "BUY",
    "entry": 182.40,
    "stop_loss": 177.72,
    "target": 191.76,
    "risk_per_share": 4.68,
    "reward_per_share": 9.36,
    "risk_reward_ratio": 2.0,
    "atr_used": 3.12,
    "confidence": 0.62,
    "tradeable": true
  }
}
```

### LLM article analysis (per article)

```json
{
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "sentiment_label": "bullish",
  "sentiment_score": 0.75,
  "event_type": "earnings",
  "impact_strength": 0.8,
  "impact_horizon": "short_term",
  "trade_bias": "buy",
  "summary": "Apple beat Q2 earnings estimates with strong iPhone revenue.",
  "explanation": "Earnings beat with revenue growth signals continued consumer demand, supportive of near-term price appreciation."
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: helpers, indicator computation, signal scoring, risk manager, LLM JSON parsing/validation — all without hitting external APIs.

---

## Signal Logic Reference

### Technical Score (40% weight)

| Condition | Points |
|-----------|--------|
| Close > SMA 20 | +1 |
| SMA 20 > SMA 50 | +1 |
| RSI between 50–65 | +1 |
| MACD > MACD Signal | +1 |
| Volume spike (>1.5x avg) | +1 |
| RSI > 75 | -1 |

### News Score (60% weight)

| Condition | Points |
|-----------|--------|
| Bullish + impact > 0.6 | +2 |
| Bullish + impact ≤ 0.6 | +1 |
| Neutral | 0 |
| Bearish + impact ≤ 0.6 | -1 |
| Bearish + impact > 0.6 | -2 |

### Final Decision

```
final_score = 0.6 × news_normalized + 0.4 × tech_normalized

BUY  if final_score ≥ 0.3  AND confidence ≥ 0.4
SELL if final_score ≤ -0.3 AND confidence ≥ 0.4
HOLD otherwise
```

---

## Potential Improvements (v2+)

- Real-time WebSocket price streaming
- Per-article LLM caching to avoid re-analysis
- Full news history storage with deduplication
- Walk-forward backtesting with LLM signals
- Portfolio-level risk management
- Alternative LLM providers (Anthropic Claude, Ollama local models)
- Sector/macro overlay signals
- Email/Slack alerts on signal changes
- Dockerized deployment
