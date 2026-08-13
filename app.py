"""
Stock Signal Radar -- personal research tool, NOT investment advice.
Compares an explainable rules-based signal against an ML signal (XGBoost,
walk-forward validated) side by side for each ticker.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.fetch import fetch_ohlcv
from data.universe import UNIVERSE, all_tickers
from signals.rules import add_indicators, composite_signal, latest_signal
from signals.ml_model import build_labels, walk_forward_validate

st.set_page_config(page_title="Stock Signal Radar", layout="wide")

st.title("📡 Stock Signal Radar")
st.warning(
    "⚠️ **No es consejo financiero.** Esto es una herramienta personal en fase de "
    "test para comparar señales técnicas y de ML. Los resultados históricos no "
    "garantizan resultados futuros. Úsalo bajo tu propio criterio y riesgo.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Configuración")
    market = st.selectbox("Mercado", list(UNIVERSE.keys()))
    ticker = st.selectbox("Ticker", UNIVERSE[market])
    period = st.selectbox("Histórico", ["1y", "2y", "5y"], index=1)
    run_ml = st.checkbox("Ejecutar validación ML (walk-forward)", value=True,
                          help="Más lento: entrena XGBoost en varias ventanas temporales.")

col1, col2 = st.columns([2, 1])

try:
    raw = fetch_ohlcv(ticker, period=period)
    enriched = composite_signal(add_indicators(raw))
    sig = latest_signal(ticker, enriched)
except Exception as e:
    st.error(f"No se pudo obtener datos para {ticker}: {e}")
    st.stop()

with col1:
    st.subheader(f"{ticker} — {sig['date'].strftime('%Y-%m-%d')}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=enriched.index, open=enriched["open"], high=enriched["high"],
        low=enriched["low"], close=enriched["close"], name="Precio",
    ))
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["sma50"], name="SMA50", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["sma200"], name="SMA200", line=dict(width=1)))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Señal técnica (reglas)")
    color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[sig["rules_signal"]]
    st.markdown(f"### :{color}[{sig['rules_signal']}]")
    st.metric("Precio cierre", f"{sig['close']}")
    st.metric("RSI(14)", sig["rsi14"])
    st.metric("Score compuesto (-4 a +4)", sig["rules_score"])
    st.caption(
        "Basado en RSI, MACD, cruce SMA50/200 y Bollinger Bands. "
        "Cada indicador vota; el score es la suma de votos."
    )

st.divider()

if run_ml:
    st.subheader("Señal ML (XGBoost, walk-forward validado)")
    with st.spinner("Entrenando y validando modelo..."):
        try:
            labeled = build_labels(enriched)
            result = walk_forward_validate(labeled)

            m1, m2, m3 = st.columns(3)
            m1.metric("Precisión BUY (histórica)", f"{result.overall_precision_buy:.0%}")
            m2.metric("Precisión SELL (histórica)", f"{result.overall_precision_sell:.0%}")
            m3.metric("Nº señales operadas", result.n_trades)

            st.caption(
                f"Retorno acumulado del backtest (con comisión {10}bps por lado): "
                f"**{result.backtest_return_pct:.2f}** unidades de precio equivalente. "
                "Esta cifra es orientativa, no una proyección."
            )

            st.markdown("**Detalle por fold (walk-forward, sin fuga de datos):**")
            st.dataframe(pd.DataFrame(result.fold_metrics), use_container_width=True, hide_index=True)

            st.info(
                "Lee la precisión como: 'de las veces que el modelo dijo BUY en el "
                "histórico, qué % subió realmente'. Una precisión cercana al 50% en "
                "una clasificación de 3 clases no es mucho mejor que el azar."
            )
        except Exception as e:
            st.error(f"No se pudo validar el modelo ML para {ticker}: {e}")
else:
    st.caption("Activa la validación ML en la barra lateral para comparar ambas señales.")

st.divider()
st.caption(
    "Stock Signal Radar · proyecto personal de análisis cuantitativo · "
    "no constituye asesoramiento financiero ni recomendación de inversión."
)
