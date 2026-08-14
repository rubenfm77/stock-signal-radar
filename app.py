"""
Stock Signal Radar -- personal research tool, NOT investment advice.
Compares an explainable rules-based signal against an ML signal (XGBoost,
walk-forward validated) side by side for each ticker, and gives a plain-
language verdict on whether they agree.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.fetch import fetch_ohlcv
from data.universe import UNIVERSE
from signals.rules import add_indicators, composite_signal, latest_signal
from signals.ml_model import build_labels, walk_forward_validate
from signals.ml_live import train_and_predict_latest

st.set_page_config(page_title="Stock Signal Radar", layout="wide")

st.title("📡 Stock Signal Radar")
st.warning(
    "⚠️ **Not investment advice.** This is a personal tool in active testing "
    "to compare technical and ML-based signals. Historical results do not "
    "guarantee future results. Use at your own discretion and risk.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Settings")
    market = st.selectbox("Market", list(UNIVERSE.keys()))
    ticker = st.selectbox("Ticker", UNIVERSE[market])
    period = st.selectbox("History", ["1y", "2y", "5y"], index=1)
    run_ml = st.checkbox("Run ML validation (walk-forward)", value=True,
                          help="Slower: trains XGBoost across several time windows.")

col1, col2 = st.columns([2, 1])

try:
    raw = fetch_ohlcv(ticker, period=period)
    enriched = composite_signal(add_indicators(raw))
    sig = latest_signal(ticker, enriched)
except Exception as e:
    st.error(
        f"Could not fetch enough data for {ticker}: {e}\n\n"
        "This is usually Yahoo Finance rate-limiting shared cloud IPs, not a "
        "bug in the app. Try again in a minute, or pick another ticker."
    )
    st.stop()

with col1:
    st.subheader(f"{ticker} — {sig['date'].strftime('%Y-%m-%d')}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=enriched.index, open=enriched["open"], high=enriched["high"],
        low=enriched["low"], close=enriched["close"], name="Price",
    ))
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["sma50"], name="SMA50", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["sma200"], name="SMA200", line=dict(width=1)))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Technical signal (rules)")
    color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[sig["rules_signal"]]
    st.markdown(f"### :{color}[{sig['rules_signal']}]")
    st.metric("Close price", f"{sig['close']}")
    st.metric("RSI(14)", sig["rsi14"])
    st.metric("Composite score (-4 to +4)", sig["rules_score"])
    st.caption(
        "Based on RSI, MACD, SMA50/200 crossover, and Bollinger Bands. "
        "Each indicator casts a vote; the score is the sum of votes."
    )

st.divider()

ml_signal_value = None
ml_precision_buy = None
ml_precision_sell = None

if run_ml:
    st.subheader("ML signal (XGBoost, walk-forward validated)")
    with st.spinner("Training and validating model..."):
        try:
            labeled = build_labels(enriched)
            result = walk_forward_validate(labeled)
            ml_precision_buy = result.overall_precision_buy
            ml_precision_sell = result.overall_precision_sell

            live = train_and_predict_latest(labeled)
            ml_signal_value = live["ml_signal"] if live else None

            m1, m2, m3 = st.columns(3)
            m1.metric("BUY precision (historical)", f"{result.overall_precision_buy:.0%}")
            m2.metric("SELL precision (historical)", f"{result.overall_precision_sell:.0%}")
            m3.metric("Trades taken", result.n_trades)

            st.caption(
                f"Cumulative backtest return (with {10}bps commission per side): "
                f"**{result.backtest_return_pct:.2f}** price-equivalent units. "
                "This figure is illustrative, not a projection."
            )

            st.markdown("**Per-fold detail (walk-forward, no data leakage):**")
            st.dataframe(pd.DataFrame(result.fold_metrics), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not validate the ML model for {ticker}: {e}")
else:
    st.caption("Enable ML validation in the sidebar to compare both signals.")

st.divider()

# ---- Plain-language verdict -------------------------------------------
st.subheader("🧭 Verdict")

rules_sig = sig["rules_signal"]

if not run_ml or ml_signal_value is None:
    st.info(
        f"**Rules-only read:** {ticker} is currently flagged **{rules_sig}** by the "
        "technical indicators. Enable the ML check above for a second opinion."
    )
else:
    agree = rules_sig == ml_signal_value and rules_sig in ("BUY", "SELL")
    if agree:
        confidence = ml_precision_buy if rules_sig == "BUY" else ml_precision_sell
        st.success(
            f"**Rules and ML agree: {rules_sig}.** Both the technical indicators and "
            f"the ML model independently point the same way on {ticker}. Historically, "
            f"when this model called {rules_sig}, it was right about {confidence:.0%} "
            "of the time on this ticker -- treat that as a rough calibration, not a "
            "guarantee."
        )
    elif rules_sig == "HOLD" and ml_signal_value == "HOLD":
        st.info(f"**Both signals are neutral (HOLD)** on {ticker} right now. No edge in either direction.")
    else:
        st.warning(
            f"**Mixed signal.** Rules say **{rules_sig}**, ML says **{ml_signal_value}** "
            f"for {ticker}. When the two engines disagree, that's typically a sign the "
            "trend isn't clean -- worth waiting for confirmation rather than acting on "
            "either signal alone."
        )

st.divider()
st.caption(
    "Stock Signal Radar · personal quantitative research project · "
    "does not constitute financial advice or an investment recommendation."
)
