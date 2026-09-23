"""Telegram delivery for bot signals.

Design intent: the message must be as honest as the research. A signal bot that
truncates "NO_TRADE, edge -0.0198" into "signal!" would be worse than useless, so
the formatter always states the decision, the margin over the breakeven bar, and
the measured evidence base. The default output is NO_TRADE and it is presented as
the correct answer rather than an absence of one.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

API = "https://api.telegram.org/bot{token}/{method}"


class TelegramError(RuntimeError):
    pass


def _call(method: str, payload: dict, token: str | None = None,
          timeout: int = 30) -> dict:
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise TelegramError("TELEGRAM_BOT_TOKEN is not set")
    url = API.format(token=token, method=method)
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise TelegramError(f"HTTP {e.code}: {body}") from e
    if not out.get("ok"):
        raise TelegramError(f"telegram rejected: {out}")
    return out


def send(text: str, chat_id: str | None = None, token: str | None = None,
         disable_preview: bool = True) -> dict:
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not chat_id:
        raise TelegramError("TELEGRAM_CHAT_ID is not set")
    return _call("sendMessage", {
        "chat_id": chat_id, "text": text, "parse_mode": "HTML",
        "disable_web_page_preview": str(disable_preview).lower(),
    }, token=token)


# ------------------------------------------------------------------ formatting
DECISION_LABEL = {
    "UP": "UP", "DOWN": "DOWN", "NO_TRADE": "NO TRADE",
}


def format_signal(sig: dict) -> str:
    """Render a run_signal() dict as an honest Telegram message."""
    d = sig.get("decision", "NO_TRADE")
    label = DECISION_LABEL.get(d, d)
    p = sig.get("p_up", float("nan"))
    be = sig.get("breakeven", 0.80)
    edge = sig.get("edge", float("nan"))
    m = sig.get("metrics", {})

    if d == "NO_TRADE":
        head = "<b>NO TRADE</b> — no edge over the payout bar"
    else:
        head = f"<b>{label}</b> — model favours the {label.lower()} side"

    lines = [
        head,
        "",
        f"<b>Instrument</b>: {sig.get('symbol','?')} ({sig.get('bar','?')} bars)",
        f"<b>As of</b>: {sig.get('as_of','?')}",
        f"<b>Last price</b>: {sig.get('last_close', float('nan')):,.2f}",
        "",
        f"<b>P(next bar up)</b>: {p:.4f}",
        f"<b>Breakeven at {(be and 1/be - 1)*100:.0f}% payout</b>: {be:.4f}",
        f"<b>Edge over breakeven</b>: {edge:+.4f}",
    ]
    if d == "NO_TRADE":
        lines += ["",
                  "The model's best side does not clear the cost of the bet. "
                  "Declining is the correct action, not a failure."]

    lines += [
        "",
        "<b>Model (this run)</b>",
        f"bars: {m.get('n_bars','?')} "
        f"(train {m.get('n_train','?')} / holdout {m.get('n_holdout','?')})",
        f"holdout accuracy: {m.get('holdout_accuracy', float('nan')):.4f} "
        f"(a coin is 0.5000)",
        f"holdout Brier: {m.get('holdout_brier', float('nan')):.5f}",
    ]

    sg = sig.get("skill_gate")
    if sg:
        verdict = "PASSED" if sg.get("passed") else "FAILED"
        lines.append(
            f"skill gate: <b>{verdict}</b> "
            f"({sg.get('holdout_accuracy', float('nan')):.4f} vs "
            f"{sg.get('threshold', float('nan')):.2f})")
        if not sg.get("passed"):
            lines.append(
                "→ the model has no demonstrated skill this run, so no "
                "directional call is issued even if the probability looked "
                "favourable")

    lines += [
        "",
        "<i>Evidence base: across 7 studies on 16M ticks and 36 years of index "
        "data, no directional model cleared its costs; a perfect direction oracle "
        "still lost money because the median 1-minute move (1.69bp) is smaller "
        "than the round trip (1.70bp). Holdout accuracy above is a single split "
        "on a few hundred bars and is NOT a validated edge.</i>",
        "",
        "Not financial advice. Paper/research only.",
    ]
    return "\n".join(lines)


def format_heartbeat(sig: dict) -> str:
    return (f"<b>Bot alive</b>\n"
            f"{sig.get('symbol','?')} {sig.get('bar','?')} @ "
            f"{sig.get('last_close', float('nan')):,.2f}\n"
            f"decision: <b>{DECISION_LABEL.get(sig.get('decision'),'?')}</b> "
            f"(p={sig.get('p_up', float('nan')):.4f})")


def format_error(err: str, context: str = "") -> str:
    return (f"<b>Bot error</b>\n"
            f"{context}\n<code>{err[:500]}</code>")