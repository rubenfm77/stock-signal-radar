"""
Fetch OHLCV data via Twelve Data (https://twelvedata.com), with a local
parquet cache. Replaces the earlier yfinance-based fetcher: Yahoo Finance
routinely rate-limits/blocks shared cloud IPs (Streamlit Cloud included),
which was silently returning partial data. Twelve Data's free tier is more
predictable (documented rate limit instead of a silent block) and has
explicit support for European exchanges.

Requires an API key, set as:
  - Streamlit Cloud: st.secrets["TWELVEDATA_API_KEY"]  (Settings -> Secrets)
  - GitHub Actions:   env var TWELVEDATA_API_KEY (repo Secret)
  - Local dev:        env var TWELVEDATA_API_KEY

Free tier limits (as of setup): 800 requests/day, 8 requests/minute.
With ~100 tickers and a 24h cache, a full universe refresh uses ~100
requests/day -- comfortably inside the daily quota, but each fetch_many
run must pace itself to stay under 8/minute.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_HOURS = 20  # generous, to conserve the free daily quota
MIN_ROWS_EXPECTED = 150
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 8
REQUESTS_PER_MINUTE = 8
_SECONDS_BETWEEN_CALLS = 60 / REQUESTS_PER_MINUTE

BASE_URL = "https://api.twelvedata.com/time_series"

# yfinance-style suffix -> Twelve Data MIC code. Twelve Data accepts a
# `mic_code` param alongside the bare symbol for non-US exchanges.
SUFFIX_TO_MIC = {
    ".MC": "BME",     # Bolsa de Madrid
    ".DE": "XETR",    # Xetra (Germany)
    ".PA": "XPAR",    # Euronext Paris
    ".AS": "XAMS",    # Euronext Amsterdam
    ".MI": "MTAA",    # Borsa Italiana (Milan)
}

PERIOD_TO_DAYS = {"1y": 365, "2y": 730, "5y": 1825}

_last_call_time = 0.0


def _get_api_key() -> str:
    key = os.environ.get("TWELVEDATA_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets["TWELVEDATA_API_KEY"]
    except Exception:
        raise RuntimeError(
            "TWELVEDATA_API_KEY not found. Set it as a Streamlit Cloud secret "
            "or an environment variable (see data/fetch.py docstring)."
        )


def _split_symbol(ticker: str) -> tuple[str, str | None]:
    for suffix, mic in SUFFIX_TO_MIC.items():
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)], mic
    return ticker, None  # US tickers have no suffix, no mic_code needed


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _throttle() -> None:
    """Keep us under the free-tier requests/minute limit across calls."""
    global _last_call_time
    elapsed = time.monotonic() - _last_call_time
    if elapsed < _SECONDS_BETWEEN_CALLS:
        time.sleep(_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_time = time.monotonic()


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d",
                 force_refresh: bool = False) -> pd.DataFrame:
    """Return OHLCV dataframe for a single ticker, cached on disk."""
    path = _cache_path(ticker)
    if not force_refresh and _is_fresh(path):
        cached = pd.read_parquet(path)
        if len(cached) >= MIN_ROWS_EXPECTED:
            return cached

    symbol, mic_code = _split_symbol(ticker)
    days = PERIOD_TO_DAYS.get(period, 730)
    outputsize = min(int(days * 1.05), 5000)  # trading days < calendar days; pad a bit

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": outputsize,
        "apikey": _get_api_key(),
        "order": "ASC",
    }
    if mic_code:
        params["mic_code"] = mic_code

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _throttle()
            resp = requests.get(BASE_URL, params=params, timeout=20)
            data = resp.json()

            if data.get("status") == "error":
                raise ValueError(f"Twelve Data error for {ticker}: {data.get('message')}")
            values = data.get("values")
            if not values:
                raise ValueError(f"No data returned for {ticker} (symbol={symbol}, mic={mic_code}).")

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            df.index.name = "date"
            df = df.rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume",
            })[["open", "high", "low", "close", "volume"]].astype(float)

            if len(df) < MIN_ROWS_EXPECTED:
                raise ValueError(
                    f"Partial data for {ticker}: got {len(df)} rows, expected at "
                    f"least {MIN_ROWS_EXPECTED}."
                )

            df.to_parquet(path)
            return df
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if path.exists():
        cached = pd.read_parquet(path)
        if len(cached) >= MIN_ROWS_EXPECTED:
            return cached
    raise last_error


def fetch_many(tickers: list[str], period: str = "2y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Fetch a batch of tickers, skipping failures instead of aborting the run."""
    out: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for t in tickers:
        try:
            out[t] = fetch_ohlcv(t, period=period, interval=interval)
        except Exception as exc:
            errors[t] = str(exc)
    if errors:
        print(f"[fetch_many] {len(errors)} tickers failed: {list(errors.keys())}")
    return out
