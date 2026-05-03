from __future__ import annotations

from datetime import datetime, timedelta

from common.events import MarketSnapshot
from execution.paper import PaperExecutionEngine


def test_update_market_carries_forward_last_known_prices():
    exe = PaperExecutionEngine(initial_cash=100_000)
    t0 = datetime(2024, 1, 1, 9, 15)

    exe.update_market(MarketSnapshot(ts=t0, prices={"AAA": 100.0, "BBB": 200.0}))
    assert exe.last_prices == {"AAA": 100.0, "BBB": 200.0}

    # Next minute BBB is missing from the sparse snapshot. Its last known price should survive.
    exe.update_market(MarketSnapshot(ts=t0 + timedelta(minutes=1), prices={"AAA": 101.0}))
    assert exe.last_prices == {"AAA": 101.0, "BBB": 200.0}


def test_nav_does_not_drop_when_held_symbol_is_missing_for_a_minute():
    exe = PaperExecutionEngine(initial_cash=0.0)
    exe.portfolio.set_position("AAA", 10)

    t0 = datetime(2024, 1, 1, 9, 15)
    exe.update_market(MarketSnapshot(ts=t0, prices={"AAA": 100.0}))
    assert exe.snapshot(t0).nav == 1000.0

    # Sparse snapshot with no AAA quote should not mark the position to zero.
    exe.update_market(MarketSnapshot(ts=t0 + timedelta(minutes=1), prices={}))
    assert exe.snapshot(t0 + timedelta(minutes=1)).nav == 1000.0
