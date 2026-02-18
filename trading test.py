import os
import time
from datetime import datetime, timedelta

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =========================
# Config & Constants
# =========================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY_HERE")

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

# Core watchlist, condensed from your HTML universe
CORE_SYMBOLS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "ORCL", "ADBE", "CRM", "CSCO", "AMD", "INTC",

    # Financials
    "JPM", "GS", "BAC", "MS", "BLK", "XLF",

    # Healthcare
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE",

    # Industrials / Energy / ETFs
    "CAT", "BA", "GE", "XOM", "CVX", "COP",
    "SPY", "QQQ", "IWM", "XLE", "XLK", "XLV",
]

SCORE_MIN = 50  # minimum score to count as a "signal"


# =========================
# Helper functions
# =========================

def fetch_finnhub_intraday(symbol: str, resolution: str = "5"):
    """
    Fetch 1 trading day's worth of intraday candles from Finnhub.
    Returns DataFrame with columns: time, open, high, low, close, volume.
    """
    if FINNHUB_API_KEY.startswith("YOUR_"):
        return None

    now = int(time.time())
    start = now - 24 * 60 * 60

    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": start,
        "to": now,
        "token": FINNHUB_API_KEY,
    }

    try:
        r = requests.get("https://finnhub.io/api/v1/stock/candle", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    if data.get("s") != "ok" or not data.get("c"):
        return None

    df = pd.DataFrame({
        "time": pd.to_datetime(data["t"], unit="s"),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"],
    })

    return df


def fetch_yahoo_intraday(symbol: str, interval: str = "5m", range_: str = "1d"):
    """
    Fallback to Yahoo Finance if Finnhub is not configured or fails.
    """
    try:
        url = f"{YAHOO_CHART_URL}/{symbol}"
        params = {"interval": interval, "range": range_}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
    except Exception:
        return None

    try:
        result = raw["chart"]["result"][0]
        ts = result["timestamp"]
        q = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "time": pd.to_datetime(ts, unit="s"),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }).dropna()
        if df.empty:
            return None
        return df
    except Exception:
        return None


def fetch_intraday(symbol: str) -> pd.DataFrame | None:
    """
    Try Finnhub first; if unavailable/fails, fall back to Yahoo.
    """
    df = fetch_finnhub_intraday(symbol)
    if df is not None and not df.empty:
        return df

    return fetch_yahoo_intraday(symbol)


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP based on typical price (H+L+C)/3 and volume.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["volume"].cumsum()
    vwap = cum_pv / cum_vol
    return vwap


def compute_signal_for_symbol(symbol: str) -> dict | None:
    """
    Compute VWAP + EMA stack + volume spike + support/resistance + score.
    Returns dict with summary metrics.
    """
    df = fetch_intraday(symbol)
    if df is None or len(df) < 20:
        return None

    # Restrict to latest ~50 bars like your JS
    df = df.tail(50).copy()

    df["ema9"] = calculate_ema(df["close"], 9)
    df["ema20"] = calculate_ema(df["close"], 20)
    df["vwap"] = calculate_vwap(df)

    # Support / resistance from last 20 bars
    recent = df.tail(20)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    current_row = df.iloc[-1]
    current_price = float(current_row["close"])
    ema9 = float(current_row["ema9"])
    ema20 = float(current_row["ema20"])
    vwap = float(current_row["vwap"])

    # Volume spike: last volume vs avg of previous 5
    if len(df) > 6:
        avg_vol = df["volume"].iloc[-6:-1].mean()
    else:
        avg_vol = df["volume"].iloc[:-1].mean()
    vol_spike = current_row["volume"] > avg_vol * 1.5 if avg_vol > 0 else False

    ema_stack = ema9 > ema20
    above_vwap = current_price > vwap

    # Score logic mirroring your JS
    score = 50
    if above_vwap:
        score += 15
    if ema_stack:
        score += 15
    if vol_spike:
        score += 10
    if current_price > ema9 and current_price > ema20:
        score += 10

    # Risk/reward
    stop_distance = current_price - support
    if stop_distance <= 0:
        rr = 0.0
    else:
        rr = (resistance - current_price) / stop_distance

    # Very rough pseudo sentiment (placeholder instead of Finnhub sentiment API)
    sentiment_score = max(0, min(100, int(50 + (score - 50) * 0.8)))
    news_score = max(0, min(100, int(40 + (score - 50) * 0.6)))

    if score >= 85:
        analyst_rating = "Strong Buy"
    elif score >= 70:
        analyst_rating = "Buy"
    elif score >= 50:
        analyst_rating = "Hold"
    else:
        analyst_rating = "Sell"

    # Social volume proxy from relative volume
    if avg_vol == 0:
        social_volume = "Low"
    else:
        vol_ratio = current_row["volume"] / avg_vol
        if vol_ratio > 2:
            social_volume = "Very High"
        elif vol_ratio > 1.5:
            social_volume = "High"
        elif vol_ratio > 1:
            social_volume = "Medium"
        else:
            social_volume = "Low"

    return {
        "symbol": symbol,
        "price": current_price,
        "ema9": ema9,
        "ema20": ema20,
        "vwap": vwap,
        "support": support,
        "resistance": resistance,
        "volume_spike": bool(vol_spike),
        "ema_stack": bool(ema_stack),
        "above_vwap": bool(above_vwap),
        "score": int(score),
        "risk_reward": round(rr, 2),
        "sentiment": sentiment_score,
        "social_volume": social_volume,
        "news_score": news_score,
        "analyst_rating": analyst_rating,
        "df": df,
    }


def compute_trade_plan(stock: dict, account_size: float = 10_000, risk_percent: float = 1.0) -> dict:
    """
    Recreate your JS trade plan logic.
    """
    risk_amount = account_size * (risk_percent / 100.0)
    stop_distance = stock["price"] - stock["support"]
    shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0

    max_loss = shares * stop_distance
    target_profit = (stock["resistance"] - stock["price"]) * shares

    return {
        "symbol": stock["symbol"],
        "entry_price": stock["price"],
        "stop_loss": stock["support"],
        "target": stock["resistance"],
        "shares": shares,
        "max_loss": round(max_loss, 2),
        "target_profit": round(target_profit, 2),
        "risk_reward": stock["risk_reward"],
    }


def plot_candles_with_levels(stock: dict) -> go.Figure:
    """
    Plot 5-minute candles with VWAP, 9 EMA, 20 EMA, support and resistance.
    """
    df = stock["df"].copy()
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema9"],
            name="EMA 9",
            mode="lines",
            line=dict(color="#10b981", width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema20"],
            name="EMA 20",
            mode="lines",
            line=dict(color="#f97316", width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["vwap"],
            name="VWAP",
            mode="lines",
            line=dict(color="#a855f7", width=1.5, dash="dash"),
        )
    )

    fig.add_hline(
        y=stock["support"],
        line=dict(color="#22c55e", width=1.5, dash="dot"),
        annotation_text=f"Support {stock['support']:.2f}",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=stock["resistance"],
        line=dict(color="#ef4444", width=1.5, dash="dot"),
        annotation_text=f"Resistance {stock['resistance']:.2f}",
        annotation_position="top left",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        title=f"{stock['symbol']} - 5 Minute Chart (VWAP + EMA Stack)",
    )
    return fig


# =========================
# Streamlit app
# =========================

st.set_page_config(page_title="Day Trading Scanner - VWAP + EMA", layout="wide")

st.title("📈 Day Trading Scanner")
st.caption("VWAP + EMA Stack + Volume Strategy | Live Market Analysis")

with st.sidebar:
    st.header("Scan Settings")

    universe_size = st.selectbox(
        "Universe Size",
        options=[20, 50, 100, 200, "All"],
        index=4,
    )

    top_n = st.selectbox(
        "Top Picks to Show",
        options=[3, 5, 10],
        index=1,
    )

    account_size = st.number_input("Account Size ($)", value=10_000.0, min_value=1000.0, step=1000.0)
    risk_pct = st.slider("Risk per Trade (%)", min_value=0.25, max_value=2.0, value=1.0, step=0.25)

    st.markdown("---")
    st.markdown("Set `FINNHUB_API_KEY` env var to use Finnhub. Otherwise the app falls back to Yahoo Finance.")


if st.button("🔍 Scan Market", type="primary"):
    st.session_state["trigger_scan"] = True

if "trigger_scan" not in st.session_state:
    st.session_state["trigger_scan"] = False

if st.session_state["trigger_scan"]:
    # Determine symbols to scan
    if universe_size == "All":
        symbols_to_scan = CORE_SYMBOLS
    else:
        symbols_to_scan = CORE_SYMBOLS[: int(universe_size)]

    status_placeholder = st.empty()
    status_placeholder.info(f"⏳ Scanning {len(symbols_to_scan)} symbols...")

    results: list[dict] = []
    for i, sym in enumerate(symbols_to_scan, start=1):
        status_placeholder.info(f"⏳ Scanning {len(symbols_to_scan)} symbols... [{i}/{len(symbols_to_scan)}] ({sym})")
        stock = compute_signal_for_symbol(sym)
        if stock is not None:
            results.append(stock)

    # Filter by score threshold
    signals = [r for r in results if r["score"] >= SCORE_MIN]
    signals_sorted = sorted(signals, key=lambda x: x["score"], reverse=True)

    status_placeholder.success(
        f"✅ Scan complete. {len(results)} with data, {len(signals_sorted)} signals (score ≥ {SCORE_MIN})."
    )

    st.session_state["scan_results"] = results
    st.session_state["scan_signals"] = signals_sorted

else:
    st.info("Click **🔍 Scan Market** to start scanning your universe.")


results = st.session_state.get("scan_results", [])
signals_sorted = st.session_state.get("scan_signals", [])

if results:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Stocks Scanned", len(results))
    with col2:
        st.metric("Signals Found (score ≥ 50)", len(signals_sorted))
    with col3:
        avg_score = int(sum(s["score"] for s in signals_sorted) / len(signals_sorted)) if signals_sorted else 0
        st.metric("Avg Signal Score", avg_score)

    st.subheader("🎯 Top Stock Picks")

    if signals_sorted:
        top_signals = signals_sorted[: top_n]

        # Build a DataFrame for table view
        table_df = pd.DataFrame(
            [
                {
                    "Ticker": s["symbol"],
                    "Price": round(s["price"], 2),
                    "VWAP": round(s["vwap"], 2),
                    "EMA 9": round(s["ema9"], 2),
                    "EMA 20": round(s["ema20"], 2),
                    "Score": s["score"],
                    "Vol Spike": "✓" if s["volume_spike"] else "✗",
                    "Support": round(s["support"], 2),
                    "Resistance": round(s["resistance"], 2),
                    "R:R": f"{s['risk_reward']}:1",
                }
                for s in top_signals
            ]
        )

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
        )

        # Selection for detailed view
        tickers = [s["symbol"] for s in top_signals]
        selected_ticker = st.selectbox("Select a ticker for detailed analysis", tickers)

        selected_stock = next(s for s in top_signals if s["symbol"] == selected_ticker)

        st.subheader(f"📊 Detailed Analysis - {selected_stock['symbol']}")

        # Chart
        fig = plot_candles_with_levels(selected_stock)
        st.plotly_chart(fig, use_container_width=True)

        # Trade plan & sentiment
        trade_plan = compute_trade_plan(selected_stock, account_size=account_size, risk_percent=risk_pct)

        col_tp, col_sent = st.columns(2)

        with col_tp:
            st.markdown("### 📝 Trading Plan")

            st.markdown(
                f"""
- **Setup**: VWAP + EMA Stack Strategy on {trade_plan['symbol']}
- **Price vs VWAP**: {"✅ Above VWAP" if selected_stock["above_vwap"] else "❌ Below VWAP"} (VWAP {selected_stock['vwap']:.2f})
- **EMA Stack (9 > 20)**: {"✅ Confirmed" if selected_stock["ema_stack"] else "❌ Not Aligned"} (9: {selected_stock['ema9']:.2f}, 20: {selected_stock['ema20']:.2f})
- **Volume Spike**: {"✅ Spike Detected" if selected_stock["volume_spike"] else "❌ No Spike"}
- **Entry**: {trade_plan['entry_price']:.2f} (e.g., on green candle close after EMA touch)
- **Stop Loss**: {trade_plan['stop_loss']:.2f} (below 20 EMA / support)
- **Target**: {trade_plan['target']:.2f} (next resistance)
- **Position Size**: {trade_plan['shares']} shares (~{risk_pct:.2f}% of ${account_size:,.0f} at risk)
- **Max Loss**: ${trade_plan['max_loss']:.2f}
- **Target Profit**: ${trade_plan['target_profit']:.2f}
- **Risk : Reward**: {trade_plan['risk_reward']}:1
"""
            )

        with col_sent:
            st.markdown("### 💭 Market Sentiment Analysis")

            sentiment_color = (
                "🟢" if selected_stock["sentiment"] >= 70 else "🟡" if selected_stock["sentiment"] >= 50 else "🔴"
            )

            st.markdown(
                f"""
- **Overall Sentiment Score**: {sentiment_color} {selected_stock['sentiment']}/100  
- **Social Media Volume (proxy)**: {selected_stock['social_volume']}  
- **News Sentiment Score (proxy)**: {selected_stock['news_score']}/100  
- **Analyst Rating (derived from score)**: {selected_stock['analyst_rating']}  
- **Data Sources**: Finnhub (if configured) and/or Yahoo Finance intraday data.
"""
            )

        st.warning(
            "⚠️ Paper Trading Recommended: Test this strategy with paper trading for at least 30 days before risking real capital. "
            "Day trading carries significant risk and requires discipline, risk management, and quick decision-making. "
            "This tool is for educational purposes only and does not constitute financial advice."
        )

    else:
        st.info("No signals found with score ≥ 50. Consider broadening the universe or revisiting the filters.")
