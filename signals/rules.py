"""
Explainable rule-based signals: RSI, MACD, moving-average cross, Bollinger Bands.
Each indicator votes BUY / SELL / HOLD; the composite is a simple majority score.
This is intentionally transparent -- every signal can be traced back to a number.

Indicators are computed by hand with pandas/numpy instead of the `pandas-ta`
library. pandas-ta is effectively unmaintained and pulls in `numba`, which
routinely breaks on whatever Python version hosting platforms (e.g. Streamlit
Cloud) default to. These are the same standard formulas -- no behavior lost,
one fragile dependency removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing, equivalent to what pandas-ta / most charting tools use
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(close: pd.Series, length: int = 20, num_std: float = 2.0):
    mid = close.rolling(length).mean()
    std = close.rolling(length).std()
    return mid - num_std * std, mid + num_std * std


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach indicator columns to an OHLCV dataframe. Does not mutate input."""
    out = df.copy()
    out["rsi14"] = _rsi(out["close"], length=14)

    macd_line, signal_line, hist = _macd(out["close"], fast=12, slow=26, signal=9)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    out["sma50"] = out["close"].rolling(50).mean()
    out["sma200"] = out["close"].rolling(200).mean()

    bb_lower, bb_upper = _bollinger(out["close"], length=20, num_std=2.0)
    out["bb_lower"] = bb_lower
    out["bb_upper"] = bb_upper

    out["atr14"] = _atr(out["high"], out["low"], out["close"], length=14)
    return out


def _vote_rsi(row) -> int:
    if pd.isna(row["rsi14"]):
        return 0
    if row["rsi14"] < RSI_OVERSOLD:
        return 1
    if row["rsi14"] > RSI_OVERBOUGHT:
        return -1
    return 0


def _vote_macd(row) -> int:
    if pd.isna(row["macd_hist"]):
        return 0
    return 1 if row["macd_hist"] > 0 else -1


def _vote_ma_cross(row) -> int:
    if pd.isna(row["sma50"]) or pd.isna(row["sma200"]):
        return 0
    return 1 if row["sma50"] > row["sma200"] else -1


def _vote_bollinger(row) -> int:
    if pd.isna(row["bb_lower"]) or pd.isna(row["bb_upper"]):
        return 0
    if row["close"] <= row["bb_lower"]:
        return 1  # near/below lower band -> potential bounce
    if row["close"] >= row["bb_upper"]:
        return -1  # near/above upper band -> potential pullback
    return 0


def composite_signal(df_with_indicators: pd.DataFrame) -> pd.DataFrame:
    """Add per-indicator votes and a composite score/label for every row."""
    out = df_with_indicators.copy()
    out["vote_rsi"] = out.apply(_vote_rsi, axis=1)
    out["vote_macd"] = out.apply(_vote_macd, axis=1)
    out["vote_ma_cross"] = out.apply(_vote_ma_cross, axis=1)
    out["vote_bollinger"] = out.apply(_vote_bollinger, axis=1)

    vote_cols = ["vote_rsi", "vote_macd", "vote_ma_cross", "vote_bollinger"]
    out["rules_score"] = out[vote_cols].sum(axis=1)  # range -4..+4

    def label(score: float) -> str:
        if score >= 2:
            return "BUY"
        if score <= -2:
            return "SELL"
        return "HOLD"

    out["rules_signal"] = out["rules_score"].apply(label)
    return out


def latest_signal(ticker: str, df_with_indicators: pd.DataFrame) -> dict:
    """Convenience: return the most recent row *with a valid close* as a flat dict.

    Guards against an in-progress "today" row Yahoo sometimes includes before
    a market has opened (or after a partial/blocked fetch) -- that row can
    carry a NaN close even though the row itself isn't empty.
    """
    signalled = composite_signal(df_with_indicators)
    valid = signalled.dropna(subset=["close"])
    if valid.empty:
        raise ValueError(f"No valid close price available for {ticker}.")
    row = valid.iloc[-1]
    return {
        "ticker": ticker,
        "date": row.name,
        "close": round(float(row["close"]), 2),
        "rsi14": round(float(row["rsi14"]), 1) if pd.notna(row["rsi14"]) else None,
        "rules_score": int(row["rules_score"]),
        "rules_signal": row["rules_signal"],
    }
