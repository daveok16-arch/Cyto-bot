"""Tests for the scheduler: bar alignment, market hours, and clean shutdown."""
from __future__ import annotations

import datetime as dt

import pytest

from src import scheduler


def _t(y, m, d, hh, mm, ss=0):
    return dt.datetime(y, m, d, hh, mm, ss, tzinfo=dt.timezone.utc)


# --------------------------------------------------------- bar alignment
def test_next_boundary_aligns_to_next_bar():
    got = scheduler.next_boundary(_t(2026, 9, 21, 10, 1), 5, settle=20)
    assert got == _t(2026, 9, 21, 10, 5, 20)


def test_next_boundary_is_strictly_after_now():
    """On an exact boundary we must wait for the NEXT one, not fire immediately."""
    got = scheduler.next_boundary(_t(2026, 9, 21, 10, 5), 5, settle=0)
    assert got > _t(2026, 9, 21, 10, 5)
    assert got == _t(2026, 9, 21, 10, 10)


def test_next_boundary_rolls_over_the_hour():
    got = scheduler.next_boundary(_t(2026, 9, 21, 10, 58), 5, settle=0)
    assert got == _t(2026, 9, 21, 11, 0)


def test_next_boundary_rolls_over_midnight():
    got = scheduler.next_boundary(_t(2026, 9, 21, 23, 58), 5, settle=0)
    assert got == _t(2026, 9, 22, 0, 0)


@pytest.mark.parametrize("minutes,expected", [
    (1, _t(2026, 9, 21, 10, 7)), (15, _t(2026, 9, 21, 10, 15)),
    (30, _t(2026, 9, 21, 10, 30)), (60, _t(2026, 9, 21, 11, 0)),
])
def test_next_boundary_supports_other_bar_sizes(minutes, expected):
    assert scheduler.next_boundary(_t(2026, 9, 21, 10, 6), minutes, 0) == expected


# --------------------------------------------------------- market hours
def test_saturday_is_closed():
    assert not scheduler.market_open(_t(2026, 9, 26, 12, 0))   # Saturday


def test_friday_evening_is_closed():
    assert not scheduler.market_open(_t(2026, 9, 25, 21, 30))  # Friday 21:30


def test_sunday_before_open_is_closed():
    assert not scheduler.market_open(_t(2026, 9, 27, 12, 0))   # Sunday noon


def test_daily_break_hour_is_closed():
    assert not scheduler.market_open(_t(2026, 9, 21, 21, 15))


def test_midweek_daytime_is_open():
    assert scheduler.market_open(_t(2026, 9, 21, 10, 0))       # Monday 10:00
    assert scheduler.market_open(_t(2026, 9, 24, 3, 0))        # Thursday 03:00


def test_sunday_after_open_is_open():
    assert scheduler.market_open(_t(2026, 9, 27, 23, 0))       # Sunday 23:00


# --------------------------------------------------------- shutdown
def test_stop_flag_is_respected(monkeypatch):
    called = {"n": 0}

    def fake_once(_):
        called["n"] += 1
        # set the stop flag so the loop exits after the first cycle
        scheduler.STOP = True
        return 0

    monkeypatch.setattr(scheduler, "run_once", fake_once)
    monkeypatch.setattr(scheduler, "market_open", lambda t: True)
    scheduler.STOP = False
    rc = scheduler.main(["--interval", "1"])
    assert rc == 0
    assert called["n"] == 1
    scheduler.STOP = False


def test_signals_are_installed_in_loop_mode(monkeypatch):
    installed = {}

    def rec(sig, fn):
        installed[sig] = fn

    monkeypatch.setattr(scheduler.signal, "signal", rec)
    monkeypatch.setattr(scheduler, "run_once", lambda _: 0)
    monkeypatch.setattr(scheduler, "market_open", lambda t: False)

    def stop_soon(seconds):
        scheduler.STOP = True

    monkeypatch.setattr(scheduler.time, "sleep", stop_soon)
    scheduler.STOP = False
    scheduler.main(["--interval", "1"])
    assert signal.SIGTERM in installed
    assert signal.SIGINT in installed
    scheduler.STOP = False


import signal  # noqa: E402  (used by the test above)