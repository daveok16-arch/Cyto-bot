"""Dukascopy tick ingestion for spot XAUUSD.

Dukascopy publishes hourly tick files (LZMA of 20-byte records: ms offset,
ask, bid, ask volume, bid volume). Prices are integers with 1e-3 scaling for
XAUUSD. This gives genuine bid/ask and trade volume, which is what order-flow
features actually require -- 1-minute OHLCV cannot provide it.

Files are cached per UTC day so the download is resumable.
"""
from __future__ import annotations

import lzma
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data" / "ticks"
HOURS = CACHE / "hours"
BASE = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m0:02d}/{d:02d}/{h:02d}h_ticks.bi5"
SCALE = 1e3  # XAUUSD point value
FIELDS = ["ms", "ask", "bid", "avol", "bvol"]


def _hpath(d: date, h: int) -> Path:
    return HOURS / f"XAUUSD_{d:%Y%m%d}_{h:02d}.csv.gz"


def on_disk(d: date, h: int) -> bool:
    return _hpath(d, h).exists()


def cached_count(d: date) -> int:
    files = HOURS.glob(f"XAUUSD_{d:%Y%m%d}_*.csv.gz") if HOURS.exists() else []
    return sum(1 for f in files if f.stat().st_size > 0)


def _url(d: date, h: int) -> str:
    # Dukascopy months are 0-indexed
    return BASE.format(y=d.year, m0=d.month - 1, d=d.day, h=h)


def _fetch_hour(d: date, h: int, retries: int = 6):
    """Returns a DataFrame, or None only when the file is genuinely absent
    (weekend/holiday). Raises on repeated transient failures so the caller does
    not mistake a 503 for 'no data' and silently produce a truncated dataset."""
    url = _url(d, h)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            if not raw:
                return None
            dec = lzma.decompress(raw)
            n = len(dec) // 20
            if n == 0:
                return None
            buf = dec[: n * 20]
            ints = np.frombuffer(buf, dtype=">u4").reshape(n, 5)
            floats = np.frombuffer(
                ints[:, 3:].astype(">u4").tobytes(), dtype=">f4"
            ).reshape(n, 2)
            base = datetime(d.year, d.month, d.day, h)
            return pd.DataFrame({
                "ts": base + pd.to_timedelta(ints[:, 0], unit="ms"),
                "ask": ints[:, 1] / SCALE,
                "bid": ints[:, 2] / SCALE,
                "avol": floats[:, 0].astype(np.float64),
                "bvol": floats[:, 1].astype(np.float64),
            })
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # lzma truncation, timeouts, connection resets
            last = e
        time.sleep(2.0 * (attempt + 1) ** 2)  # quadratic backoff, up to ~98s
    raise RuntimeError(f"failed {url}: {last}")


def download_day(d: date, workers: int = 4):
    """Download hours for one UTC day, caching each hour separately so a single
    transient failure never discards successful work. Returns total ticks."""
    HOURS.mkdir(parents=True, exist_ok=True)
    missing = [h for h in range(24) if not on_disk(d, h)]
    if not missing:
        return cached_count(d)

    fails = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_hour, d, h): h for h in missing}
        for f in as_completed(futs):
            h = futs[f]
            try:
                r = f.result()
            except Exception as e:
                fails.append((h, str(e)[:60]))
                continue
            if r is not None and len(r):
                r.to_csv(_hpath(d, h), index=False, compression="gzip")
            else:
                _hpath(d, h).write_bytes(b"")  # genuinely absent hour
    return cached_count(d), fails


def fill_gaps(days: list, workers: int = 4, rounds: int = 4, log=print) -> int:
    """Retry missing hours across `days` until stable. Transient 503s are common,
    so a second and third pass materially improves completeness."""
    total = 0
    for r in range(rounds):
        remaining = 0
        for d in days:
            missing = [h for h in range(24) if not on_disk(d, h)]
            if not missing:
                continue
            remaining += len(missing)
            try:
                download_day(d, workers)
            except Exception:
                pass
        log(f"  [round {r+1}] hours still missing before pass: {remaining}")
        if remaining == 0:
            break
    return total


def download_range(start: date, end: date, workers: int = 4, log=print) -> int:
    total = 0
    d, i = start, 0
    nd = (end - start).days + 1
    days = []
    while d <= end:
        i += 1
        if d.weekday() == 5:
            d += timedelta(days=1)
            continue
        days.append(d)
        try:
            n, fails = download_day(d, workers)
            total += n
            if fails:
                log(f"  WARN {d}: {len(fails)} hours failed: {[h for h,_ in fails]}")
        except Exception as e:
            log(f"  WARN {d}: {e}")
        log(f"  {d} [{i}/{nd}] hours={cached_count(d)} cumulative ticks={total}")
        d += timedelta(days=1)
    return total


def load_day(d: date) -> pd.DataFrame:
    if not HOURS.exists():
        return pd.DataFrame()
    files = sorted(HOURS.glob(f"XAUUSD_{d:%Y%m%d}_*.csv.gz"))
    frames = []
    for f in files:
        if f.stat().st_size == 0:
            continue
        x = pd.read_csv(f)
        if x.empty or "ts" not in x.columns:
            continue
        x["ts"] = pd.to_datetime(x["ts"], utc=True, format="ISO8601")
        frames.append(x)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df


def load_range(start: date, end: date) -> pd.DataFrame:
    frames = []
    d = start
    while d <= end:
        x = load_day(d)
        if len(x):
            frames.append(x)
        d += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values("ts").reset_index(drop=True)