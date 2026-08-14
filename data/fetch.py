"""
Fetch OHLCV data via yfinance with a local parquet cache.

Data source history (for context if this file is revisited):
  - Started on yfinance/Yahoo Finance.
  - Tried Twelve Data as a replacement to dodge Yahoo's cloud-IP rate
    limiting -- but Twelve Data's free tier only covers US exchanges;
    Madrid, Xetra, Euronext Paris, and Milan all require a paid plan.
    That's a dead end for this app's US+Spain+Europe scope.
  - Back on yfinance. It's free and covers all the needed exchanges, but
    Yahoo does rate-limit/block shared cloud IPs (Streamlit Cloud included)
    -- and when it does, it doesn't always error, it can return a truncated
    response (a handful of rows instead of the full history). The guards
    below (MIN_ROWS_EXPECTED, retries, refusing to cache partial data) exist
    specifically to catch that failure mode instead of silently working
    with broken data.
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
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 6

# A full year of trading days is ~252. If a "2y" request comes back with far
# fewer rows than a single year, treat it as a partial/blocked response.
MIN_ROWS_EXPECTED = 150


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
        cached = pd.read_parquet(path)
        cached = cached.dropna(subset=["open", "high", "low", "close"])
        if len(cached) >= MIN_ROWS_EXPECTED:
            return cached
        # stale cache is itself a partial response -- fall through and retry

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}. Check the symbol/suffix.")

            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.index.name = "date"

            # Yahoo sometimes returns the full calendar range with most rows
            # NaN (a soft block) instead of a short-but-clean history. Row
            # *count* alone doesn't catch that -- drop empty rows first and
            # validate on what's actually usable.
            df = df.dropna(subset=["open", "high", "low", "close"])

            if len(df) < MIN_ROWS_EXPECTED:
                raise ValueError(
                    f"Partial data for {ticker}: got {len(df)} valid rows, expected at "
                    f"least {MIN_ROWS_EXPECTED}. Likely a temporary Yahoo Finance rate limit."
                )
            df.to_parquet(path)
            return df
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)  # linear backoff

    # All retries failed -- fall back to a stale (but at least complete)
    # cache if we have one, rather than crashing the whole page.
    if path.exists():
        cached = pd.read_parquet(path)
        cached = cached.dropna(subset=["open", "high", "low", "close"])
        if len(cached) >= MIN_ROWS_EXPECTED:
            return cached
    raise last_error


def fetch_many(tickers: list[str], period: str = "2y", interval: str = "1d",
                pause: float = 0.3) -> dict[str, pd.DataFrame]:
    """Fetch a batch of tickers, skipping failures instead of aborting the run."""
    out: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for t in tickers:
        try:
            out[t] = fetch_ohlcv(t, period=period, interval=interval)
        except Exception as exc:
            errors[t] = str(exc)
        time.sleep(pause)  # be polite to the API
    if errors:
        print(f"[fetch_many] {len(errors)} tickers failed: {list(errors.keys())}")
    return out
