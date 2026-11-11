"""BTCUSDT 5-minute data acquisition and feature construction.

Source: Binance public market data. The primary path is the REST endpoint
``/api/v3/klines``; when that endpoint is unreachable (some ISPs serve a block
page, which shows up as a TLS certificate mismatch) the loader falls back to the
official public data bucket at ``data.binance.vision``, which publishes the same
klines as monthly and daily archives. Both paths return identical schemas.

Features are point-in-time: every column at bar ``t`` is computed from bars
``<= t``. The label is ``r_{t+1}``, so a decision made at the close of ``t``
earns the next bar's return and pays its cost at ``t``.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Iterable

import numpy as np
import pandas as pd

__all__ = ["Panel", "Standardizer", "load_panel", "build_features", "BARS_PER_YEAR"]

INTERVAL_MINUTES: Final[dict[str, int]] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
BARS_PER_YEAR: Final[int] = 365 * 24 * 12  # 5-minute bars, crypto trades 24/7

_KLINE_COLUMNS: Final[list[str]] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

_REST_HOSTS: Final[tuple[str, ...]] = (
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
)

_VISION: Final[str] = "https://data.binance.vision/data/spot"


@dataclass(frozen=True)
class Panel:
    """Aligned, point-in-time feature matrix and next-bar returns."""

    timestamps: pd.DatetimeIndex
    features: np.ndarray  # (N, F) float32, unscaled
    returns: np.ndarray  # (N,) float32, r_{t+1} aligned to features at t
    feature_names: tuple[str, ...]
    symbol: str
    interval: str

    def __len__(self) -> int:
        return self.features.shape[0]


class Standardizer:
    """Per-feature z-score fitted on a training slice only."""

    def __init__(self, clip: float = 5.0) -> None:
        self.clip = clip
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer.fit must be called before transform")
        z = (x - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "deadband-ssm/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_klines(raw: bytes, with_header: bool) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=0 if with_header else None)
    if not with_header:
        df.columns = _KLINE_COLUMNS[: df.shape[1]]
    else:
        df = df.iloc[:, : len(_KLINE_COLUMNS)]
        df.columns = _KLINE_COLUMNS[: df.shape[1]]
    numeric = [c for c in df.columns if c not in ("ignore",)]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Binance switched several archives to microsecond timestamps in 2025.
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df["timestamp"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    return df.dropna(subset=["close"]).reset_index(drop=True)


def _months(start: date, end: date) -> Iterable[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _download_vision(symbol: str, interval: str, start: date, end: date, cache: Path) -> pd.DataFrame:
    """Monthly archives for complete months, daily archives for the tail."""
    frames: list[pd.DataFrame] = []
    today = datetime.now(timezone.utc).date()
    last_complete = (today.replace(day=1) - timedelta(days=1))
    for year, month in _months(start, end):
        if date(year, month, 1) > last_complete:
            break
        name = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
        path = cache / name
        if not path.exists():
            url = f"{_VISION}/monthly/klines/{symbol}/{interval}/{name}"
            try:
                path.write_bytes(_http_get(url))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            time.sleep(0.05)
        with zipfile.ZipFile(path) as zf:
            inner = zf.namelist()[0]
            raw = zf.read(inner)
        frames.append(_parse_klines(raw, with_header=not raw.startswith(b"1")))

    # Daily archives cover the current, still-incomplete month.
    cursor = max(start, last_complete + timedelta(days=1))
    while cursor <= end and cursor < today:
        name = f"{symbol}-{interval}-{cursor.isoformat()}.zip"
        path = cache / name
        if not path.exists():
            url = f"{_VISION}/daily/klines/{symbol}/{interval}/{name}"
            try:
                path.write_bytes(_http_get(url, timeout=30))
            except urllib.error.HTTPError:
                cursor += timedelta(days=1)
                continue
        with zipfile.ZipFile(path) as zf:
            raw = zf.read(zf.namelist()[0])
        frames.append(_parse_klines(raw, with_header=not raw.startswith(b"1")))
        cursor += timedelta(days=1)

    if not frames:
        raise RuntimeError("no archives could be downloaded from data.binance.vision")
    return pd.concat(frames, ignore_index=True)


def _download_rest(symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
    step = INTERVAL_MINUTES[interval] * 60_000
    t0 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    t1 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    last_err: Exception | None = None
    for host in _REST_HOSTS:
        try:
            cursor = t0
            frames.clear()
            while cursor < t1:
                url = (
                    f"{host}/api/v3/klines?symbol={symbol}&interval={interval}"
                    f"&startTime={cursor}&limit=1000"
                )
                rows = pd.read_json(io.BytesIO(_http_get(url, timeout=30)))
                if rows.empty:
                    break
                rows = rows.iloc[:, : len(_KLINE_COLUMNS)]
                rows.columns = _KLINE_COLUMNS[: rows.shape[1]]
                frames.append(rows)
                cursor = int(rows["open_time"].iloc[-1]) + step
                time.sleep(0.12)
            if frames:
                df = pd.concat(frames, ignore_index=True)
                for col in _KLINE_COLUMNS[:11]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
                df["timestamp"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
                return df
        except Exception as exc:  # noqa: BLE001 - try the next host
            last_err = exc
            continue
    raise RuntimeError(f"all REST hosts failed; last error: {last_err}")


def load_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start: str = "2020-01-01",
    end: str | None = None,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Download (or load cached) raw klines, deduplicated and time-sorted."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    end_date = date.fromisoformat(end) if end else datetime.now(timezone.utc).date()
    start_date = date.fromisoformat(start)
    merged = cache / f"{symbol}-{interval}-{start_date}-{end_date}.csv.gz"
    if merged.exists():
        df = pd.read_csv(merged, parse_dates=["timestamp"])
        return df

    try:
        df = _download_rest(symbol, interval, start_date, end_date)
        source = "rest"
    except Exception as exc:  # noqa: BLE001 - documented fallback
        print(f"[data] REST unavailable ({type(exc).__name__}: {exc}); using data.binance.vision")
        df = _download_vision(symbol, interval, start_date, end_date, cache)
        source = "vision"

    df = (
        df.drop_duplicates(subset="open_time")
        .sort_values("timestamp")
        .loc[lambda d: (d["timestamp"] >= pd.Timestamp(start_date, tz="UTC")) & (d["timestamp"] <= pd.Timestamp(end_date, tz="UTC"))]
        .reset_index(drop=True)
    )
    df.to_csv(merged, index=False, compression="gzip")
    print(f"[data] {symbol} {interval}: {len(df):,} bars {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]} (source={source})")
    return df


def build_features(
    df: pd.DataFrame, window: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build the four stationary features and the next-bar label.

    Columns (all strictly causal, ``t`` uses bars ``<= t``):

    1. ``logret``  -- ``ln(C_t / C_{t-1})``
    2. ``volz``    -- z-score of the 20-bar realised volatility of returns
    3. ``vimb``    -- z-score of ``2 * taker_buy_base / volume - 1``
    4. ``hlspread``-- ``(H_t - L_t) / C_t``

    Returns ``(X, y, mask, names)`` where ``y[t] = logret[t+1]`` and ``mask``
    marks the rows that survived the warm-up period.
    """
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    taker = df["taker_buy_base"].to_numpy(dtype=np.float64)

    logret = np.zeros_like(close)
    logret[1:] = np.log(close[1:] / close[:-1])

    # Realised volatility of the last `window` returns, then z-scored over the
    # same window (both computed with past data only).
    s = pd.Series(logret)
    sigma = s.rolling(window, min_periods=window).std(ddof=0)
    volz = (sigma - sigma.rolling(window, min_periods=window).mean()) / (
        sigma.rolling(window, min_periods=window).std(ddof=0) + 1e-6
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(volume > 0, 2.0 * taker / volume - 1.0, 0.0)
    imb_s = pd.Series(imb)
    vimb = (imb_s - imb_s.rolling(window, min_periods=window).mean()) / (
        imb_s.rolling(window, min_periods=window).std(ddof=0) + 1e-6
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        hlspread = np.where(close > 0, (high - low) / close, 0.0)

    X = np.column_stack([logret, volz.to_numpy(), vimb.to_numpy(), hlspread]).astype(np.float32)
    y = np.full(len(close), np.nan, dtype=np.float32)
    y[:-1] = logret[1:]

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    names = ("logret", "volz", "vimb", "hlspread")
    return X, y, valid, names


def load_panel(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start: str = "2020-01-01",
    end: str | None = None,
    cache_dir: str | Path = "data/cache",
    window: int = 20,
) -> Panel:
    """Full pipeline: raw klines -> features -> :class:`Panel`."""
    df = load_klines(symbol, interval, start, end, cache_dir)
    X, y, mask, names = build_features(df, window=window)
    ts = pd.DatetimeIndex(df["timestamp"].to_numpy()[mask])
    return Panel(
        timestamps=ts,
        features=X[mask],
        returns=y[mask],
        feature_names=names,
        symbol=symbol,
        interval=interval,
    )
