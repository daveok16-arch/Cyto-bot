"""Tests for the live signal pipeline, notifier, and cloud runner.

Network calls are never made here: the HTTP layer is tested via its formatting
and error behaviour, and the live pipeline is tested on synthetic bars. The point
is to lock in the properties that matter -- causal features, honest messaging, and
that the decision gate still refuses bad bets.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import live, notify


def _bars(n=1200, seed=0, freq="5min"):
    rng = np.random.default_rng(seed)
    c = 4400 * np.exp(np.cumsum(rng.normal(0, 0.0002, n)))
    ts = pd.date_range("2026-09-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "ts": ts, "open": c * (1 - 0.0001), "high": c * 1.0003,
        "low": c * 0.9997, "close": c, "volume": rng.integers(50, 500, n),
    })


# ------------------------------------------------------------- features
def test_features_are_causal():
    """A change in the far future must not alter past features."""
    b = _bars(600)
    f1 = live.price_features(b)
    b2 = b.copy()
    b2.loc[500:, "close"] *= 1.05
    f2 = live.price_features(b2)
    common = [c for c in live.FEATURE_COLS if c in f1.columns]
    assert np.allclose(f1[common].iloc[:480], f2[common].iloc[:480],
                       equal_nan=True)


def test_features_have_expected_columns():
    f = live.price_features(_bars(300))
    for c in live.FEATURE_COLS:
        assert c in f.columns, c
    assert len(f) == 300


def test_feature_frame_accepts_datetime_index():
    """The training path may pass an indexed frame rather than a 'ts' column."""
    b = _bars(300).set_index("ts")
    f = live.price_features(b)
    assert "tod_sin" in f.columns and f["tod_sin"].notna().all()


def test_labels_look_forward_only():
    b = _bars(50)
    y = live.make_labels(b["close"], horizon=1)
    assert pd.isna(y.iloc[-1])
    up = (b["close"].shift(-1) > b["close"]).astype(int)
    assert (y.iloc[:-1].to_numpy() == up.iloc[:-1].to_numpy()).all()


# ------------------------------------------------------------- messaging
def _sig(decision="NO_TRADE", p=0.46, edge=-0.01):
    return {"symbol": "GC=F", "bar": "5min", "as_of": "2026-09-21 00:00:00+00:00",
            "last_close": 4406.6, "p_up": p, "decision": decision,
            "breakeven": 0.5556, "edge": edge,
            "metrics": {"n_bars": 995, "n_train": 746, "n_holdout": 249,
                        "holdout_accuracy": 0.5542, "holdout_brier": 0.2395}}


def test_message_states_no_trade_plainly():
    msg = notify.format_signal(_sig())
    assert "NO TRADE" in msg
    assert "Declining is the correct action" in msg


def test_message_always_includes_the_evidence_caveat():
    """Every message must carry the honest framing, including a trade signal."""
    for d in ("NO_TRADE", "UP", "DOWN"):
        msg = notify.format_signal(_sig(decision=d, p=0.99, edge=0.4))
        assert "NOT a validated edge" in msg
        assert "Not financial advice" in msg
        assert "perfect direction oracle still lost money" in msg


def test_message_shows_breakeven_and_edge():
    msg = notify.format_signal(_sig())
    assert "Breakeven" in msg and "55" in msg
    assert "Edge over breakeven" in msg


def test_up_signal_is_labelled():
    msg = notify.format_signal(_sig(decision="UP", p=0.60, edge=0.045))
    assert "UP" in msg and "up side" in msg


def test_error_message_truncates_long_traces():
    msg = notify.format_error("x" * 2000, context="ctx")
    assert len(msg) < 900


def test_heartbeat_is_short():
    msg = notify.format_heartbeat(_sig())
    assert "Bot alive" in msg
    assert len(msg) < 250


def test_skill_gate_blocks_a_directional_call_from_a_useless_model(monkeypatch):
    """The guard that matters: a model with no skill must not emit UP/DOWN.

    Verified earlier that this can happen for real: a 249-bar holdout scored
    0.5703 by chance and the bot emitted a DOWN signal. The gate exists so that
    noise cannot be presented as a signal.
    """
    import numpy as np
    import pandas as pd
    from src import live

    # force a flat, skill-free model: always ~0.30, so max(p,1-p) < breakeven
    class FakeBot:
        contenders_, weights_, calibrator_, fitted_ = [], None, None, True

        def predict_proba(self, X):
            return np.full(len(X), 0.30)

    monkeypatch.setattr(live, "fetch_ohlcv",
                        lambda *a, **k: _bars(1200))
    monkeypatch.setattr(live, "train_model",
                        lambda *a, **k: (FakeBot(), {
                            "n_bars": 1200, "n_train": 900, "n_holdout": 300,
                            "holdout_accuracy": 0.50, "holdout_brier": 0.25,
                            "up_rate": 0.5, "weights": {}, "contender_brier": {},
                        }, None))
    sig = live.run_signal(min_holdout_accuracy=0.54)
    assert sig["decision"] == "NO_TRADE"
    assert sig["skill_gate"]["passed"] is False
    assert "skill gate" in sig["reason"]


def test_skill_gate_message_explains_a_failed_gate():
    s = _sig(decision="NO_TRADE")
    s["skill_gate"] = {"threshold": 0.54, "passed": False,
                       "holdout_accuracy": 0.5012}
    msg = notify.format_signal(s)
    assert "FAILED" in msg
    assert "no demonstrated skill" in msg


def test_skill_gate_message_reports_a_pass():
    s = _sig(decision="NO_TRADE")
    s["skill_gate"] = {"threshold": 0.54, "passed": True,
                       "holdout_accuracy": 0.5703}
    msg = notify.format_signal(s)
    assert "PASSED" in msg
    assert "no demonstrated skill" not in msg


# ------------------------------------------------------------- sender
def test_send_requires_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(notify.TelegramError):
        notify.send("hi")


def test_send_requires_chat_id_when_token_present(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(notify.TelegramError):
        notify.send("hi")


# ------------------------------------------------------------- runner
def test_cloud_run_dry_run_never_sends(monkeypatch, capsys, tmp_path):
    """dry-run must not touch Telegram even if a network is available."""
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("send() must not be called in dry-run")

    monkeypatch.setattr(notify, "send", boom)
    monkeypatch.setattr(live, "run_signal", lambda **k: _sig())
    from src import cloud_run
    rc = cloud_run.main(["--dry-run"])
    assert rc == 0
    assert called["n"] == 0
    out = capsys.readouterr().out
    assert "NO TRADE" in out


def test_cloud_run_reports_error_and_returns_nonzero(monkeypatch):
    sent = {}

    def fake_send(text, **k):
        sent["text"] = text
        return {"ok": True}

    def fail(**k):
        raise RuntimeError("feed unavailable")

    monkeypatch.setattr(live, "run_signal", fail)
    monkeypatch.setattr(notify, "send", fake_send)
    from src import cloud_run
    rc = cloud_run.main([])
    assert rc == 1
    assert "feed unavailable" in sent["text"]


def test_cloud_run_logs_signal(monkeypatch, tmp_path, capsys):
    logfile = tmp_path / "signals.jsonl"
    monkeypatch.setattr(live, "run_signal", lambda **k: _sig())
    monkeypatch.setattr(notify, "send", lambda *a, **k: {"ok": True})
    from src import cloud_run
    monkeypatch.setattr(cloud_run, "LOG", logfile)
    rc = cloud_run.main([])
    assert rc == 0
    row = json.loads(logfile.read_text().strip())
    assert row["decision"] == "NO_TRADE"