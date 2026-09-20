"""Download a range of Dukascopy XAUUSD ticks. Resumable: cached days are skipped.
Usage: python -m src.download_ticks 2026-07-05 2026-09-18
"""
from __future__ import annotations

import datetime as dt
import sys

from . import ticks


def main():
    start = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date(2026, 7, 5)
    end = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date(2026, 9, 18)
    print(f"downloading XAUUSD ticks {start} -> {end}", flush=True)
    n = ticks.download_range(start, end, workers=4)
    days = [d for d in (start + dt.timedelta(days=i) for i in range((end-start).days+1)) if d.weekday() != 5]
    ticks.fill_gaps(days, workers=4)
    print(f"done: {n} ticks", flush=True)


if __name__ == "__main__":
    main()