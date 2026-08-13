"""
Fetch OHLCV data via yfinance with a simple on-disk parquet cache.
Cache expires daily -- fine for swing-style signals, not for intraday.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_HOURS = 12


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_TTL_HOURS)


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d",
                 force_refresh: bool = False) -> pd.DataFrame:
    """Return OHLCV dataframe for a single ticker, cached on disk."""
    path = _cache_path(ticker)
    if not force_refresh and _is_fresh(path):
        return pd.read_parquet(path)

    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol/suffix.")

    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    df.to_parquet(path)
    return df


def fetch_many(tickers: list[str], period: str = "2y", interval: str = "1d",
                pause: float = 0.3) -> dict[str, pd.DataFrame]:
    """Fetch a batch of tickers, skipping failures instead of aborting the run."""
    out: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for t in tickers:
        try:
            out[t] = fetch_ohlcv(t, period=period, interval=interval)
        except Exception as exc:  # keep the pipeline alive on bad tickers
            errors[t] = str(exc)
        time.sleep(pause)  # be polite to the API
    if errors:
        print(f"[fetch_many] {len(errors)} tickers failed: {list(errors.keys())}")
    return out
