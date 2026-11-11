"""Download daily OHLCV from Stooq into data/raw/<SYM>.csv (cached, idempotent)."""
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "XLE", "XLK", "XLF", "XLV"]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def fetch(sym: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            raw = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")
            if not raw.startswith("Date,"):
                print(f"  {sym}: unexpected payload {raw[:80]!r}")
                return None
            df = pd.read_csv(io.StringIO(raw))
            df["Date"] = pd.to_datetime(df["Date"])
            return df.sort_values("Date").reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001 - network retry
            print(f"  {sym}: attempt {attempt + 1} failed ({exc})")
            time.sleep(2 + 2 * attempt)
    return None


def main() -> int:
    ok = 0
    for sym in TICKERS:
        out = RAW / f"{sym}.csv"
        if out.exists() and out.stat().st_size > 1000:
            df = pd.read_csv(out)
            print(f"{sym}: cached {len(df)} rows {df.Date.iloc[0]}..{df.Date.iloc[-1]}")
            ok += 1
            continue
        df = fetch(sym)
        if df is None or len(df) < 500:
            print(f"{sym}: FAILED")
            continue
        df.to_csv(out, index=False)
        print(f"{sym}: {len(df)} rows {df.Date.iloc[0].date()}..{df.Date.iloc[-1].date()}")
        ok += 1
        time.sleep(0.6)
    print(f"\n{ok}/{len(TICKERS)} series available")
    return 0 if ok >= 8 else 1


if __name__ == "__main__":
    sys.exit(main())
