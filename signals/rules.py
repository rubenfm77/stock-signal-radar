"""
Explainable rule-based signals: RSI, MACD, moving-average cross, Bollinger Bands.
Each indicator votes BUY / SELL / HOLD; the composite is a simple majority score.
This is intentionally transparent -- every signal can be traced back to a number.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach indicator columns to an OHLCV dataframe. Does not mutate input."""
    out = df.copy()
    out["rsi14"] = ta.rsi(out["close"], length=14)

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    out["macd"] = macd["MACD_12_26_9"]
    out["macd_signal"] = macd["MACDs_12_26_9"]
    out["macd_hist"] = macd["MACDh_12_26_9"]

    out["sma50"] = ta.sma(out["close"], length=50)
    out["sma200"] = ta.sma(out["close"], length=200)

    bb = ta.bbands(out["close"], length=20, std=2)
    out["bb_lower"] = bb["BBL_20_2.0"]
    out["bb_upper"] = bb["BBU_20_2.0"]

    out["atr14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
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
    """Convenience: return the most recent row's signal as a flat dict for the UI."""
    row = composite_signal(df_with_indicators).iloc[-1]
    return {
        "ticker": ticker,
        "date": row.name,
        "close": round(float(row["close"]), 2),
        "rsi14": round(float(row["rsi14"]), 1) if pd.notna(row["rsi14"]) else None,
        "rules_score": int(row["rules_score"]),
        "rules_signal": row["rules_signal"],
    }
