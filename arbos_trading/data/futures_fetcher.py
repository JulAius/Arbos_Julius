"""
Fetch Binance Futures data: funding rate, long/short ratio, open interest.
These are derivatives context signals mentioned in the GOAL.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path


FUTURES_BASE = "https://fapi.binance.com/fapi/v1"
FUTURES_DATA = "https://fapi.binance.com/futures/data"


def fetch_funding_rate(symbol: str = "BTCUSDT", start_ms: int = None, end_ms: int = None, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical funding rates (8-hour settlement, interpolated to 15m).
    Returns DataFrame with columns: open_time, funding_rate
    """
    url = f"{FUTURES_BASE}/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame(columns=["open_time", "funding_rate"])

    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"])
    return df[["open_time", "funding_rate"]].sort_values("open_time").reset_index(drop=True)


def fetch_all_funding_rates(symbol: str = "BTCUSDT", start_date: str = "2024-01-01") -> pd.DataFrame:
    """Fetch all funding rates since start_date via pagination."""
    start_dt = pd.to_datetime(start_date)
    all_data = []
    current_start = int(start_dt.timestamp() * 1000)
    end_ms = int(datetime.utcnow().timestamp() * 1000)

    while current_start < end_ms:
        df_chunk = fetch_funding_rate(symbol, start_ms=current_start, end_ms=end_ms, limit=1000)
        if df_chunk.empty:
            break
        all_data.append(df_chunk)
        last_ts = int(df_chunk["open_time"].max().timestamp() * 1000)
        if last_ts <= current_start:
            break
        current_start = last_ts + 1
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame(columns=["open_time", "funding_rate"])

    result = pd.concat(all_data).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    print(f"Fetched {len(result)} funding rate snapshots from {result['open_time'].iloc[0]} to {result['open_time'].iloc[-1]}")
    return result


def fetch_ls_ratio(symbol: str = "BTCUSDT", period: str = "15m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch global long/short account ratio (sentiment signal).
    period: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
    Returns DataFrame with columns: open_time, long_short_ratio, long_account, short_account
    """
    url = f"{FUTURES_DATA}/globalLongShortAccountRatio"
    params = {"symbol": symbol, "period": period, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame(columns=["open_time", "long_short_ratio"])

    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["long_short_ratio"] = pd.to_numeric(df["longShortRatio"])
    df["long_account"] = pd.to_numeric(df["longAccount"])
    df["short_account"] = pd.to_numeric(df["shortAccount"])
    return df[["open_time", "long_short_ratio", "long_account", "short_account"]].sort_values("open_time").reset_index(drop=True)


def fetch_open_interest(symbol: str = "BTCUSDT", period: str = "15m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch open interest history.
    Returns DataFrame with columns: open_time, open_interest
    """
    url = f"{FUTURES_DATA}/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame(columns=["open_time", "open_interest"])

    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"])
    return df[["open_time", "open_interest"]].sort_values("open_time").reset_index(drop=True)


def load_or_fetch_futures_features(data_dir: str = "data", symbol: str = "BTCUSDT", start_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Load cached futures features or fetch from API.
    Returns a DataFrame indexed on open_time with futures-derived features.
    """
    cache_path = Path(data_dir) / "btcusdt_futures_features.csv"

    if cache_path.exists():
        df_cached = pd.read_csv(cache_path, parse_dates=["open_time"])
        last_ts = df_cached["open_time"].max()
        now_utc = datetime.utcnow()
        age_hours = (now_utc - last_ts).total_seconds() / 3600
        if age_hours < 1:
            print(f"[futures] Loaded cached futures features ({len(df_cached)} rows, age={age_hours:.1f}h)")
            return df_cached
        print(f"[futures] Cache stale ({age_hours:.1f}h) — refreshing...")

    # Fetch funding rates (8h intervals, need to interpolate to 15m)
    print("[futures] Fetching funding rates...")
    df_funding = fetch_all_funding_rates(symbol, start_date)

    # Fetch long/short ratio (15m available directly)
    print("[futures] Fetching L/S ratio (500 most recent 15m periods)...")
    try:
        df_ls = fetch_ls_ratio(symbol, period="15m", limit=500)
    except Exception as e:
        print(f"[futures] L/S ratio fetch failed: {e}")
        df_ls = pd.DataFrame(columns=["open_time", "long_short_ratio", "long_account", "short_account"])

    # Fetch open interest history
    print("[futures] Fetching open interest (500 most recent 15m periods)...")
    try:
        df_oi = fetch_open_interest(symbol, period="15m", limit=500)
    except Exception as e:
        print(f"[futures] OI fetch failed: {e}")
        df_oi = pd.DataFrame(columns=["open_time", "open_interest"])

    # Build a 15m timestamp grid from start_date to now
    start_ts = pd.to_datetime(start_date)
    end_ts = datetime.utcnow().replace(second=0, microsecond=0)
    # Round end_ts down to nearest 15m
    end_ts = end_ts - timedelta(minutes=end_ts.minute % 15, seconds=end_ts.second)
    ts_grid = pd.date_range(start=start_ts, end=end_ts, freq="15min")
    df_grid = pd.DataFrame({"open_time": ts_grid})

    # Merge funding rate (forward-fill from 8h snapshots)
    if not df_funding.empty:
        df_grid = df_grid.merge(df_funding, on="open_time", how="left")
        df_grid["funding_rate"] = df_grid["funding_rate"].ffill()
        # funding_rate features
        df_grid["funding_rate_ma3"] = df_grid["funding_rate"].rolling(3 * 8).mean()  # 3-day MA (3 × 8h)
        df_grid["funding_rate_dev"] = df_grid["funding_rate"] - df_grid["funding_rate_ma3"]
        df_grid["funding_rate_abs"] = df_grid["funding_rate"].abs()
        df_grid["funding_positive"] = (df_grid["funding_rate"] > 0).astype(float)
        df_grid["funding_high"] = (df_grid["funding_rate"].abs() > 0.0005).astype(float)  # extreme funding

    # Merge L/S ratio (15m, only recent 500 bars)
    if not df_ls.empty:
        df_grid = df_grid.merge(df_ls, on="open_time", how="left")
        df_grid["long_short_ratio"] = df_grid["long_short_ratio"].ffill()
        df_grid["long_account"] = df_grid["long_account"].ffill()
        df_grid["short_account"] = df_grid["short_account"].ffill()
        # L/S features
        df_grid["ls_ratio_ma10"] = df_grid["long_short_ratio"].rolling(10).mean()
        df_grid["ls_ratio_dev"] = df_grid["long_short_ratio"] - df_grid["ls_ratio_ma10"]
        df_grid["ls_ratio_z"] = df_grid["ls_ratio_dev"] / (df_grid["long_short_ratio"].rolling(10).std() + 1e-10)

    # Merge open interest (15m, only recent 500 bars)
    if not df_oi.empty:
        df_grid = df_grid.merge(df_oi, on="open_time", how="left")
        df_grid["open_interest"] = df_grid["open_interest"].ffill()
        # OI features
        df_grid["oi_change"] = df_grid["open_interest"].pct_change()
        df_grid["oi_ma20"] = df_grid["open_interest"].rolling(20).mean()
        df_grid["oi_ratio"] = df_grid["open_interest"] / (df_grid["oi_ma20"] + 1e-10)

    # Save cache
    df_grid.to_csv(cache_path, index=False)
    print(f"[futures] Saved {len(df_grid)} rows of futures features to {cache_path}")
    return df_grid
