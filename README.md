# 📡 Stock Signal Radar

> ⚠️ **Educational project only. NOT investment advice, NOT a recommendation
> to buy or sell any security.** This is a personal, experimental tool built
> for learning purposes and portfolio demonstration — it is still in active
> testing (beta) and may contain errors or incomplete data. See
> [DISCLAIMER.md](./DISCLAIMER.md) for the full notice.

## What this app does

Stock Signal Radar pulls historical price data (via `yfinance`) for a curated
list of US, European, and Spanish stocks, then generates two independent
signals for each ticker so they can be compared side by side:

1. **Rule-based technical signal** — a transparent, fully explainable score
   built from RSI, MACD, the SMA50/SMA200 crossover, and Bollinger Bands.
   Every BUY/SELL/HOLD label traces back to a specific indicator value.
2. **ML signal (XGBoost)** — a gradient-boosted model trained on the same
   indicators, validated with a **walk-forward** methodology (never a random
   train/test split, to avoid leaking future data into training) and
   reported with honest metrics: precision on the BUY/SELL class and a
   commission-adjusted backtest return, not just raw accuracy.

The app is meant to help study how technical and ML-based signals agree or
diverge on real market data — not to generate trading orders. Nothing it
outputs should be acted on as financial advice.

## Why two engines instead of one

- **Rules engine** (RSI, MACD, SMA50/200 cross, Bollinger Bands): fully
  transparent, every signal traces back to a specific indicator value. No
  black box.
- **ML engine** (XGBoost): learns non-linear interactions between the same
  indicators, but only as good as its out-of-sample validation.

When both agree, that's a stronger (though still not reliable) signal. When
they diverge, the tool flags it as mixed rather than forcing a false
consensus.

## Design decisions that matter

- **Target is ATR-normalized**, not a fixed % move — makes labels comparable
  across low- and high-volatility tickers.
- **Walk-forward validation only.** Random train/test splits leak future
  information through autocorrelated rolling-window features and silently
  inflate reported accuracy. Every fold trains strictly on past data.
- **Precision on the BUY/SELL class is reported**, not just accuracy — for an
  imbalanced 3-class problem (BUY/SELL/HOLD), accuracy alone is close to
  meaningless.
- **Backtest includes commission costs** (configurable, default 10bps/side)
  so the reported return isn't a frictionless fantasy.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
stock-signal-radar/
├── data/
│   ├── universe.py    # curated ticker lists (IBEX35, EuroStoxx sample, US sample)
│   └── fetch.py        # yfinance fetch + local parquet cache
├── signals/
│   ├── rules.py         # RSI/MACD/MA-cross/Bollinger composite signal
│   └── ml_model.py       # XGBoost + walk-forward validation + cost-aware backtest
├── app.py                # Streamlit UI
├── DISCLAIMER.md
└── requirements.txt
```

## Roadmap

- [ ] Expand ticker universe beyond the curated sample
- [ ] Add fundamentals-based filters (P/E, debt/equity) as a pre-screen
- [ ] Telegram alert integration (reusing the BOE alert bot pattern)
- [ ] Model registry / provenance tracking for reproducibility
- [ ] Sector/correlation view to avoid concentrated false-consensus signals
