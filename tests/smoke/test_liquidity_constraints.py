from datetime import datetime

from common.events import MarketSnapshot, PositionSnapshot
from runner.engine import LiquidityConstraints, LiquidityTracker, _rebalance


def test_rebalance_respects_bar_participation_and_adv_position_cap():
    liq = LiquidityConstraints(
        enabled=True,
        max_bar_participation=0.10,
        max_position_adv_participation=0.10,
        adv_lookback_days=20,
        min_adv_history_days=1,
    )
    tracker = LiquidityTracker(lookback_days=20)

    # One completed historical day with 100k shares traded creates an ADV cap
    # of 10k shares when max_position_adv_participation=10%.
    tracker.update(datetime(2020, 1, 1, 15, 30), {"ABC": 100_000})
    tracker.update(datetime(2020, 1, 2, 9, 15), {"ABC": 1_000})

    snap = MarketSnapshot(
        ts=datetime(2020, 1, 2, 9, 15),
        prices={"ABC": 1.0},
        volumes={"ABC": 1_000},
    )
    port = PositionSnapshot(
        ts=snap.ts,
        cash=1_000_000,
        positions={},
        mtm_prices={},
        nav=1_000_000,
    )

    orders = _rebalance(snap.ts, {"ABC": 1.0}, snap, port, liq, tracker)
    assert len(orders) == 1
    assert orders[0].side == "BUY"
    assert orders[0].qty == 100  # 10% of the current minute volume, not 1M shares

    near_cap = PositionSnapshot(
        ts=snap.ts,
        cash=1_000_000,
        positions={"ABC": 9_990},
        mtm_prices={"ABC": 1.0},
        nav=1_009_990,
    )
    orders = _rebalance(snap.ts, {"ABC": 1.0}, snap, near_cap, liq, tracker)
    assert len(orders) == 1
    assert orders[0].qty == 10  # position cap reaches 10k shares
