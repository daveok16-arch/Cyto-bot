"""Cloud runner: one scheduled cycle, then exit.

Designed for a stateless scheduler (GitHub Actions cron, Cloud Run job, cron).
Each invocation:
  1. fetches recent bars
  2. retrains on that same source (no persisted state, no train/serve mismatch)
  3. predicts and decides
  4. reports to Telegram
  5. logs the signal to disk if a path is writable

Never raises on a market-data failure in a way that spams Telegram: a single
error notice is sent and the process exits non-zero so the scheduler flags it.

Usage:
  python -m src.cloud_run                  # one cycle
  python -m src.cloud_run --heartbeat      # also send a short alive message
  python -m src.cloud_run --dry-run        # print instead of sending
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from . import live, notify

LOG = Path(__file__).resolve().parent.parent / "results" / "signals.jsonl"


def _log(sig: dict) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(sig, default=str) + "\n")
    except OSError:
        pass  # read-only container filesystem is expected on some platforms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one signal cycle")
    ap.add_argument("--symbol", default=os.environ.get("SIGNAL_SYMBOL", "GC=F"))
    ap.add_argument("--bar", default=os.environ.get("SIGNAL_BAR", "5min"))
    ap.add_argument("--horizon", type=int,
                    default=int(os.environ.get("SIGNAL_HORIZON", "1")))
    ap.add_argument("--payout", type=float,
                    default=float(os.environ.get("SIGNAL_PAYOUT", "0.80")))
    ap.add_argument("--history", default=os.environ.get("SIGNAL_HISTORY", "5d"))
    ap.add_argument("--heartbeat", action="store_true",
                    help="also send a short alive message")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the message instead of sending it")
    args = ap.parse_args(argv)

    try:
        sig = live.run_signal(symbol=args.symbol, bar=args.bar,
                              horizon=args.horizon, payout=args.payout,
                              history_range=args.history)
    except Exception as e:                       # noqa: BLE001 - report anything
        tb = traceback.format_exc(limit=3)
        msg = notify.format_error(f"{e}\n{tb}", context=f"{args.symbol} {args.bar}")
        print(msg, file=sys.stderr)
        if not args.dry_run:
            try:
                notify.send(msg)
            except notify.TelegramError as te:
                print(f"telegram failed too: {te}", file=sys.stderr)
        return 1

    _log(sig)
    text = notify.format_signal(sig)
    if args.heartbeat:
        text = notify.format_heartbeat(sig) + "\n\n" + text

    if args.dry_run:
        print(text)
    else:
        try:
            notify.send(text)
            print(f"sent: {sig['decision']} p_up={sig['p_up']:.4f}")
        except notify.TelegramError as e:
            print(f"telegram error: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())