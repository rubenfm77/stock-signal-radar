"""
Daily alert check: combines the rules-based signal and the live ML signal
per ticker, requiring BOTH to agree before flagging a BUY/SELL (conservative
by design -- see conversation notes: this forces the ML model to actually
prove out against the rules engine rather than firing alerts on its own).

Sends:
  - a Telegram + email alert immediately when a ticker's combined signal
    CHANGES from its last known state (new BUY, new SELL, or reverting to
    HOLD/mixed)
  - a daily summary of every ticker currently on BUY or SELL, regardless
    of whether it changed today

Run manually:  python -m alerts.alert_check
Run in CI:     see .github/workflows/signal-alerts.yml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow running as a script

from data.fetch import fetch_ohlcv
from data.universe import all_tickers
from signals.rules import add_indicators, composite_signal
from signals.ml_model import build_labels
from signals.ml_live import train_and_predict_latest
from alerts.notifiers import send_telegram, send_email
from alerts.state import load_state, save_state

DISCLAIMER = (
    "⚠️ Herramienta personal en test. No es consejo financiero ni "
    "recomendación de inversión."
)


def combined_signal(rules_signal: str, ml_signal: str | None) -> str:
    if ml_signal is None:
        return "N/D"  # not enough history to train yet
    if rules_signal == ml_signal and rules_signal in ("BUY", "SELL"):
        return rules_signal
    return "HOLD"


def analyze_ticker(ticker: str) -> dict | None:
    try:
        raw = fetch_ohlcv(ticker, period="2y")
        enriched = composite_signal(add_indicators(raw))
        labeled = build_labels(enriched)
        ml_result = train_and_predict_latest(labeled)
        ml_signal = ml_result["ml_signal"] if ml_result else None

        rules_row = enriched.iloc[-1]
        combined = combined_signal(rules_row["rules_signal"], ml_signal)

        return {
            "ticker": ticker,
            "date": str(rules_row.name.date()),
            "close": round(float(rules_row["close"]), 2),
            "rules_signal": rules_row["rules_signal"],
            "ml_signal": ml_signal or "N/D",
            "combined_signal": combined,
        }
    except Exception as exc:
        print(f"[alert_check] {ticker} failed: {exc}")
        return None


def format_change_alert(result: dict, previous: str) -> str:
    return (
        f"🔔 *Cambio de señal: {result['ticker']}*\n"
        f"{previous} → *{result['combined_signal']}*\n"
        f"Precio: {result['close']}  |  Reglas: {result['rules_signal']}  |  "
        f"ML: {result['ml_signal']}\n"
        f"Fecha: {result['date']}\n\n{DISCLAIMER}"
    )


def format_daily_summary(active: list[dict]) -> str:
    if not active:
        return f"📋 Resumen diario: sin señales BUY/SELL activas hoy.\n\n{DISCLAIMER}"
    lines = ["📋 *Resumen diario — señales activas*\n"]
    for r in sorted(active, key=lambda x: x["ticker"]):
        emoji = "🟢" if r["combined_signal"] == "BUY" else "🔴"
        lines.append(f"{emoji} {r['ticker']}: {r['combined_signal']} @ {r['close']}")
    lines.append(f"\n{DISCLAIMER}")
    return "\n".join(lines)


def main() -> None:
    state = load_state()
    results, active_signals, change_alerts = [], [], []

    for ticker in all_tickers():
        result = analyze_ticker(ticker)
        if result is None:
            continue
        results.append(result)

        previous = state.get(ticker, "HOLD")
        if result["combined_signal"] != previous:
            change_alerts.append(format_change_alert(result, previous))
        state[ticker] = result["combined_signal"]

        if result["combined_signal"] in ("BUY", "SELL"):
            active_signals.append(result)

    for alert_text in change_alerts:
        send_telegram(alert_text)
        send_email("📡 Stock Signal Radar — cambio de señal", alert_text)

    summary_text = format_daily_summary(active_signals)
    send_telegram(summary_text)
    send_email("📡 Stock Signal Radar — resumen diario", summary_text)

    save_state(state)
    print(f"Done. {len(results)} tickers analyzed, {len(change_alerts)} changes, "
          f"{len(active_signals)} active signals.")


if __name__ == "__main__":
    main()
