"""
streamlit_app.py
----------------
TradeSense AI — Streamlit dashboard.

Calls the FastAPI backend and displays:
  - Price chart with SMA overlays
  - Technical indicator summary
  - Latest news articles
  - LLM sentiment analysis
  - Final BUY / SELL / HOLD signal
  - Entry / stop-loss / target levels

Run with:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys

import plotly.graph_objects as go
import requests
import streamlit as st

# Allow imports from the project root when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_get(path: str, params: dict | None = None) -> dict | None:
    """GET request to the FastAPI backend. Returns JSON dict or None on error."""
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def api_post(path: str, params: dict | None = None) -> dict | None:
    """POST request to the FastAPI backend. Returns JSON dict or None on error."""
    try:
        resp = requests.post(f"{API_BASE}{path}", params=params, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TradeSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 TradeSense AI")
    st.caption("Educational trading signal system")
    st.divider()

    ticker_input = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="e.g. AAPL, TSLA, NVDA, MSFT",
    ).upper()

    period = st.selectbox(
        "Data Period",
        options=["1mo", "3mo", "6mo", "1y"],
        index=1,
    )

    max_articles = st.slider("Max News Articles", min_value=3, max_value=15, value=8)

    run_full = st.button("🚀 Run Full Analysis", use_container_width=True, type="primary")
    run_quick = st.button("⚡ Quick Signal (no LLM)", use_container_width=True)

    st.divider()
    st.caption("⚠️ Not financial advice. For learning only.")

    # Backtest section
    st.subheader("🧪 Backtest")
    bt_start = st.text_input("Start Date", "2023-01-01")
    bt_end = st.text_input("End Date", "2024-01-01")
    run_bt = st.button("Run Backtest", use_container_width=True)


# ── Helper Renderers ──────────────────────────────────────────────────────────

def signal_badge(signal: str) -> str:
    colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    return f"{colors.get(signal, '⚪')} **{signal}**"


def render_price_chart(market: dict, indicators: dict):
    """Candlestick chart with SMA overlays."""
    bars = market.get("bars", [])
    if not bars:
        st.warning("No price data to chart.")
        return

    dates = [b["date"] for b in bars]
    opens = [b["open"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    volumes = [b["volume"] for b in bars]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # SMA overlays — we only have the latest values from /indicators
    # For a real overlay we'd need the full series; here we show as reference line
    sma20 = indicators.get("sma_20")
    sma50 = indicators.get("sma_50")
    if sma20:
        fig.add_hline(y=sma20, line_dash="dot", line_color="#1E90FF",
                      annotation_text=f"SMA20 {sma20:.2f}", annotation_position="right")
    if sma50:
        fig.add_hline(y=sma50, line_dash="dash", line_color="#FFA500",
                      annotation_text=f"SMA50 {sma50:.2f}", annotation_position="right")

    fig.update_layout(
        title=f"{market.get('ticker')} Price — {period}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=450,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_volume_chart(market: dict):
    bars = market.get("bars", [])
    if not bars:
        return
    dates = [b["date"] for b in bars]
    volumes = [b["volume"] for b in bars]

    fig = go.Figure(go.Bar(x=dates, y=volumes, name="Volume", marker_color="#546e7a"))
    fig.update_layout(
        title="Volume",
        height=180,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_indicators(ind: dict):
    st.subheader("📊 Technical Indicators")
    col1, col2, col3, col4 = st.columns(4)

    def metric(col, label, value, fmt=".2f", good=None, bad=None):
        v = f"{value:{fmt}}" if value is not None else "N/A"
        delta_color = "normal"
        col.metric(label, v)

    metric(col1, "Close", ind.get("close"))
    metric(col1, "SMA 20", ind.get("sma_20"))
    metric(col2, "SMA 50", ind.get("sma_50"))
    metric(col2, "RSI 14", ind.get("rsi_14"))
    metric(col3, "MACD", ind.get("macd"))
    metric(col3, "MACD Signal", ind.get("macd_signal"))
    metric(col4, "ATR 14", ind.get("atr_14"))

    vol_spike = ind.get("volume_spike", False)
    col4.metric("Volume Spike", "✅ Yes" if vol_spike else "❌ No")


def render_news(news: dict):
    st.subheader("📰 Recent News")
    articles = news.get("articles", [])
    if not articles:
        st.info("No news articles found.")
        return

    for article in articles:
        with st.expander(f"📄 {article.get('headline', 'No headline')}", expanded=False):
            cols = st.columns([2, 1])
            cols[0].write(article.get("summary") or "_No summary available_")
            cols[1].write(f"**Source:** {article.get('source', 'Unknown')}")
            cols[1].write(f"**Date:** {article.get('published_at', 'Unknown')}")
            if article.get("url"):
                cols[1].markdown(f"[Read more →]({article['url']})")


def render_llm_analysis(llm: dict):
    st.subheader("🤖 LLM Sentiment Analysis")
    analyses = llm.get("analyses", [])

    agg_sentiment = llm.get("aggregate_sentiment", 0.0)
    agg_bias = llm.get("aggregate_trade_bias", "hold").upper()

    col1, col2 = st.columns(2)
    col1.metric("Aggregate Sentiment Score", f"{agg_sentiment:.3f}", help="-1=bearish, 0=neutral, 1=bullish")
    col2.metric("Aggregate Bias", agg_bias)

    if not analyses:
        st.info("No LLM analyses available (check OPENAI_API_KEY).")
        return

    st.write(f"**{len(analyses)} articles analyzed:**")
    for a in analyses:
        label = a.get("sentiment_label", "neutral")
        score = a.get("sentiment_score", 0)
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(label, "⚪")

        with st.expander(f"{emoji} {a.get('headline', '')[:80]}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Sentiment:** {label} ({score:+.2f})")
            c2.write(f"**Event:** {a.get('event_type', 'other')}")
            c3.write(f"**Trade Bias:** {a.get('trade_bias', 'hold').upper()}")
            st.write(f"**Summary:** {a.get('summary', '')}")
            st.write(f"**Explanation:** {a.get('explanation', '')}")
            st.write(f"Impact: strength={a.get('impact_strength'):.2f}, horizon={a.get('impact_horizon')}")


def render_signal_and_risk(signal: dict, risk: dict):
    st.subheader("🎯 Trading Signal & Risk Parameters")

    col1, col2, col3, col4 = st.columns(4)
    sig = signal.get("signal", "HOLD")
    score = signal.get("final_score", 0)
    confidence = signal.get("confidence", 0)

    col1.markdown(f"### {signal_badge(sig)}")
    col2.metric("Final Score", f"{score:+.4f}", help="-1 to +1")
    col3.metric("Confidence", f"{confidence:.1%}")
    col4.metric("Tradeable", "✅" if risk.get("tradeable") else "❌")

    st.divider()

    # Technical breakdown
    tech = signal.get("technical", {})
    news_sig = signal.get("news", {})

    col_t, col_n = st.columns(2)
    with col_t:
        st.write("**Technical Signals**")
        checks = {
            "Close > SMA 20": tech.get("close_above_sma20"),
            "SMA 20 > SMA 50": tech.get("sma20_above_sma50"),
            "RSI Healthy (50–65)": tech.get("rsi_healthy"),
            "MACD Bullish": tech.get("macd_bullish"),
            "Volume Spike": tech.get("volume_spike"),
            "RSI Overbought (>75) ⚠️": tech.get("rsi_overbought"),
        }
        for label, val in checks.items():
            icon = "✅" if val else "❌"
            st.write(f"{icon} {label}")
        st.write(f"**Tech Score:** {tech.get('raw_score', 0)} → normalized {tech.get('normalized_score', 0):.3f}")

    with col_n:
        st.write("**News Signals**")
        st.write(f"Articles analyzed: {news_sig.get('num_articles', 0)}")
        st.write(f"Raw score: {news_sig.get('raw_score', 0)}")
        st.write(f"Normalized: {news_sig.get('normalized_score', 0):.3f}")

    # Risk parameters
    if risk.get("tradeable"):
        st.divider()
        st.write("**Risk Parameters**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Entry", f"${risk.get('entry', 0):.2f}")
        r2.metric("Stop Loss", f"${risk.get('stop_loss', 0):.2f}")
        r3.metric("Target", f"${risk.get('target', 0):.2f}")
        r4.metric("Risk:Reward", f"1:{risk.get('risk_reward_ratio', 0):.1f}")
        st.caption(f"ATR used: {risk.get('atr_used', 0):.4f} | Confidence: {confidence:.1%}")
    else:
        st.info("⚠️ Signal is not tradeable (low confidence or HOLD signal).")


def render_backtest(bt: dict):
    st.subheader("🧪 Backtest Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", bt.get("total_trades", 0))
    c2.metric("Win Rate", f"{bt.get('win_rate', 0):.1f}%")
    c3.metric("Total Return", f"{bt.get('total_return_pct', 0):+.2f}%")
    c4.metric("Max Drawdown", f"{bt.get('max_drawdown_pct', 0):.2f}%")

    col_w, col_l = st.columns(2)
    col_w.metric("Winning Trades", bt.get("winning_trades", 0))
    col_l.metric("Losing Trades", bt.get("losing_trades", 0))
    st.caption(bt.get("note", ""))


# ── Main App ──────────────────────────────────────────────────────────────────

st.title("📈 TradeSense AI Dashboard")
st.caption(f"Analyzing: **{ticker_input}** | Period: **{period}**")

if run_full:
    with st.spinner(f"Running full analysis for {ticker_input}… (LLM calls may take 30–60s)"):
        data = api_post(f"/analyze/{ticker_input}", params={"period": period, "max_articles": max_articles})

    if data:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart", "📰 News & LLM", "🎯 Signal", "ℹ️ Raw"])

        with tab1:
            render_price_chart(data["market"], data["indicators"])
            render_volume_chart(data["market"])
            render_indicators(data["indicators"])

        with tab2:
            render_news(data["news"])
            st.divider()
            render_llm_analysis(data["llm_analysis"])

        with tab3:
            render_signal_and_risk(data["signal"], data["risk"])

        with tab4:
            st.json(data)

elif run_quick:
    with st.spinner(f"Fetching technical signal for {ticker_input}…"):
        market_data = api_get(f"/market/{ticker_input}", params={"period": period})
        indicators_data = api_get(f"/market/{ticker_input}/indicators", params={"period": period})
        signal_data = api_get(f"/signal/{ticker_input}", params={"period": period})

    if all([market_data, indicators_data, signal_data]):
        col1, col2 = st.columns([3, 1])
        with col1:
            render_price_chart(market_data, indicators_data)
            render_volume_chart(market_data)
        with col2:
            st.metric("Signal", signal_data.get("signal", "N/A"))
            st.metric("Score", f"{signal_data.get('final_score', 0):+.4f}")
            st.metric("Confidence", f"{signal_data.get('confidence', 0):.1%}")

        render_indicators(indicators_data)

        # Risk params for quick signal
        from app.models.schemas import RiskParameters
        if indicators_data.get("atr_14"):
            entry = indicators_data.get("close", 0)
            atr = indicators_data.get("atr_14", 0)
            sig = signal_data.get("signal", "HOLD")
            if sig == "BUY":
                sl = entry - 1.5 * atr
                tp = entry + 2 * (entry - sl)
                c1, c2, c3 = st.columns(3)
                c1.metric("Entry", f"${entry:.2f}")
                c2.metric("Stop Loss", f"${sl:.2f}")
                c3.metric("Target", f"${tp:.2f}")

elif run_bt:
    with st.spinner(f"Running backtest for {ticker_input}…"):
        bt_data = api_post(
            f"/backtest/{ticker_input}",
            params={"start_date": bt_start, "end_date": bt_end},
        )
    if bt_data:
        render_backtest(bt_data)

else:
    # Landing state
    st.info("👈 Enter a ticker and click **Run Full Analysis** or **Quick Signal** to get started.")

    st.subheader("What this system does")
    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    **📊 Market Data**
    - 3 months of daily OHLCV
    - SMA 20 & 50
    - RSI 14
    - MACD
    - ATR 14
    - Volume spikes
    """)
    col2.markdown("""
    **🤖 LLM Analysis**
    - Fetches recent news
    - Analyzes each article
    - Returns sentiment score
    - Classifies event type
    - Estimates impact horizon
    """)
    col3.markdown("""
    **🎯 Signal Engine**
    - Weighted scoring (60% news, 40% tech)
    - BUY / SELL / HOLD decision
    - ATR-based stop loss
    - Risk:Reward target
    - Confidence gate
    """)
