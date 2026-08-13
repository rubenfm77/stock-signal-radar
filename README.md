# 📡 Stock Signal Radar

Personal quant research tool comparing **rule-based technical signals** against
an **ML model (XGBoost, walk-forward validated)** across US, European, and
Spanish equity markets.

> ⚠️ **Not investment advice.** See [DISCLAIMER.md](./DISCLAIMER.md). This is
> a personal project in active testing, not a production trading system.

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
