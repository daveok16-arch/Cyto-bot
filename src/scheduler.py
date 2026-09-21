"""Long-running scheduler for always-on hosts (Cloud Run, a VM, Docker).

GitHub Actions cron is best-effort: runs can be delayed, and a scheduled workflow
is disabled after 60 days of repository inactivity. For a bot that must fire near
a bar close, an always-on process is more reliable.

This loop keeps its own clock. It:
  * aligns to the next bar boundary (plus a small settle delay so the bar is closed)
  * skips weekends, when the gold market is shut
  * retries a failed cycle with backoff instead of dying
  * stops cleanly on SIGTERM/SIGINT so a platform restart is not an error

Usage:
  python -m src.scheduler                    # run forever
  python -m src.scheduler --once             # single cycle (same as cloud_run)
  SIGNAL_INTERVAL_SECONDS=300 python -m src.scheduler   # fixed interval
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import sys
import time

from .cloud_run import main as run_once

STOP = False


def _handle(signum, _frame):
    global STOP
    STOP = True
    print(f"received signal {signum}; finishing current cycle then exiting",
          file=sys.stderr, flush=True)


def next_boundary(now: dt.datetime, bar_minutes: int, settle: int) -> dt.datetime:
    """Next bar close (plus settle seconds) strictly after `now`, UTC."""
    minute = (now.minute // bar_minutes + 1) * bar_minutes
    base = now.replace(second=0, microsecond=0)
    if minute >= 60:
        base = base.replace(minute=0) + dt.timedelta(hours=1)
    else:
        base = base.replace(minute=minute)
    return base + dt.timedelta(seconds=settle)


def market_open(t: dt.datetime) -> bool:
    """Gold trades ~23h/day Sun 22:00 UTC to Fri 21:00 UTC. Saturday is shut."""
    if t.weekday() == 5:                       # Saturday
        return False
    if t.weekday() == 6 and t.hour < 22:       # Sunday before the open
        return False
    if t.weekday() == 4 and t.hour >= 21:      # Friday after the close
        return False
    if t.hour == 21:                           # the daily 21:00 UTC break
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scheduled signal loop")
    ap.add_argument("--once", action="store_true", help="one cycle then exit")
    ap.add_argument("--bar-minutes", type=int, default=5)
    ap.add_argument("--settle", type=int,
                    default=int(os.environ.get("SIGNAL_SETTLE_SECONDS", "20")),
                    help="seconds after the boundary before fetching")
    ap.add_argument("--interval", type=int,
                    default=int(os.environ.get("SIGNAL_INTERVAL_SECONDS", "0")),
                    help="fixed seconds between cycles; 0 = align to bars")
    args = ap.parse_args(argv)

    if args.once:
        return run_once([])

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    size = (os.environ.get("SIGNAL_SYMBOL", "GC=F"),
            os.environ.get("SIGNAL_BAR", "5min"))
    print(f"scheduler started for {size[0]} {size[1]}; "
          f"{'fixed ' + str(args.interval) + 's' if args.interval else 'bar-aligned'}",
          flush=True)

    backoff = 30
    while not STOP:
        now = dt.datetime.now(dt.timezone.utc)
        if args.interval:
            target = now + dt.timedelta(seconds=args.interval)
        else:
            target = next_boundary(now, args.bar_minutes, args.settle)

        while not STOP and dt.datetime.now(dt.timezone.utc) < target:
            time.sleep(min(5, max(1, (target - dt.datetime.now(
                dt.timezone.utc)).total_seconds())))

        now = dt.datetime.now(dt.timezone.utc)
        if not market_open(now):
            print(f"{now:%Y-%m-%d %H:%M UTC} market closed; skipping", flush=True)
            if args.interval:
                continue
            time.sleep(60)
            continue

        try:
            rc = run_once([])
            backoff = 30
            if rc != 0:
                print(f"cycle returned {rc}", file=sys.stderr, flush=True)
        except Exception as e:                      # noqa: BLE001
            print(f"cycle crashed: {e}", file=sys.stderr, flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)

    print("scheduler stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())