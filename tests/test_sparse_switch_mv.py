
from datetime import datetime, timedelta
import random

from common.events import MarketSnapshot, PositionSnapshot
from strategy.sparse_switch_mv import SparseSwitchMVStrategy


def make_port(ts):
    return PositionSnapshot(ts=ts, cash=1_000_000.0, positions={}, mtm_prices={}, nav=1_000_000.0)


def test_sparse_switch_mv_smoke():
    random.seed(7)
    strat = SparseSwitchMVStrategy(
        rebalance_minutes=375,
        estimation_frequency="1d",
        lookback_bars=30,
        mean_lookback_periods=5,
        cov_lookback_periods=10,
        support_k=3,
        min_history_periods=4,
        max_weight_per_asset=0.7,
    )

    ts = datetime(2024, 1, 1, 9, 15)
    prices = {"AAA": 100.0, "BBB": 120.0, "CCC": 80.0, "DDD": 150.0}
    for minute in range(375 * 8):
        for i, sym in enumerate(list(prices.keys())):
            drift = 0.0002 * (i + 1)
            noise = random.gauss(0.0, 0.0015)
            prices[sym] *= max(0.5, 1.0 + drift + noise)
        snap = MarketSnapshot(ts=ts, prices=dict(prices))
        strat.on_snapshot(snap, make_port(ts))
        ts += timedelta(minutes=1)

    tw = getattr(strat, "_last_target_weights", {})
    assert isinstance(tw, dict)
    assert len(tw) <= 3
    if tw:
        s = sum(tw.values())
        assert abs(s - 1.0) < 1e-6
        assert all(v >= 0.0 for v in tw.values())
