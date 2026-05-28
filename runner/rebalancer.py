from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional

from common.events import MarketSnapshot, OrderRequest, PositionSnapshot
from runner.liquidity import (
    LiquidityConstraints,
    LiquidityTracker,
    _bar_qty_cap,
    _clean_weight,
    _diag_add_symbol,
    _has_missing_or_zero_bar_volume,
    _new_rebalance_diag,
    _position_qty_cap,
)


def _rebalance_with_diagnostics(
    ts,
    target_w: Dict[str, float],
    snap: MarketSnapshot,
    port: PositionSnapshot,
    liquidity: Optional[LiquidityConstraints] = None,
    liquidity_tracker: Optional[LiquidityTracker] = None,
) -> tuple[List[OrderRequest], dict]:
    prices = snap.prices
    nav = float(port.nav)
    cash = float(port.cash)
    cur_pos = dict(port.positions)
    liq = liquidity or LiquidityConstraints(enabled=False)
    diag = _new_rebalance_diag(ts, liq, liquidity_tracker)

    desired_notional = {sym: _clean_weight(w) * nav for sym, w in target_w.items()}
    orders: List[OrderRequest] = []

    # Sell logic. Liquidity caps can force a gradual sell-down when the current
    # position exceeds the ADV-based capacity, even if the optimizer still wants
    # a larger theoretical target.
    for sym, qty in cur_pos.items():
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue
        cur_qty = int(qty)
        tgt_val = desired_notional.get(sym, 0.0)
        desired_qty_uncapped = int(math.floor(max(0.0, tgt_val) / px))
        desired_qty = desired_qty_uncapped

        pos_cap = _position_qty_cap(sym, liq, liquidity_tracker)
        if pos_cap is not None and desired_qty > pos_cap:
            desired_qty = min(desired_qty, pos_cap)
            diag["adv_cap_hits"] += 1
            _diag_add_symbol(diag, "symbols_adv_capped", sym)

        requested_sell_qty = int(cur_qty - desired_qty)
        if requested_sell_qty <= 0:
            continue

        diag["desired_sell_qty"] += requested_sell_qty
        diag["desired_sell_notional"] += requested_sell_qty * px

        sell_qty = requested_sell_qty
        bar_cap = _bar_qty_cap(sym, "SELL", snap, liq)
        if bar_cap is not None and sell_qty > bar_cap:
            diag["bar_cap_hits"] += 1
            _diag_add_symbol(diag, "symbols_bar_capped", sym)
            if bar_cap == 0 and _has_missing_or_zero_bar_volume(sym, snap):
                diag["missing_volume_blocks"] += 1
                _diag_add_symbol(diag, "symbols_missing_volume", sym)
            sell_qty = min(sell_qty, bar_cap)

        deferred = max(0, requested_sell_qty - sell_qty)
        diag["deferred_sell_qty"] += deferred
        diag["deferred_sell_notional"] += deferred * px

        if sell_qty > 0:
            orders.append(OrderRequest(ts=ts, order_id=str(uuid.uuid4()), symbol=sym, side="SELL", qty=sell_qty))
            diag["sell_orders"] += 1
            diag["submitted_sell_qty"] += sell_qty
            diag["submitted_sell_notional"] += sell_qty * px

    # Buy logic. The optimizer may ask for a target weight, but the execution
    # layer only moves toward that target at a realistic participation rate and
    # refuses to accumulate more than a configurable fraction of rolling ADV.
    remaining = cash
    for sym, w in sorted(target_w.items(), key=lambda kv: _clean_weight(kv[1]), reverse=True):
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue
        qty = int(cur_pos.get(sym, 0))
        tgt_val = desired_notional.get(sym, 0.0)
        desired_qty_uncapped = int(math.floor(max(0.0, tgt_val) / px))
        desired_qty = desired_qty_uncapped

        pos_cap = _position_qty_cap(sym, liq, liquidity_tracker)
        if pos_cap is not None and desired_qty > pos_cap:
            desired_qty = min(desired_qty, pos_cap)
            diag["adv_cap_hits"] += 1
            _diag_add_symbol(diag, "symbols_adv_capped", sym)

        requested_buy_qty = int(desired_qty - qty)
        if requested_buy_qty <= 0:
            continue

        diag["desired_buy_qty"] += requested_buy_qty
        diag["desired_buy_notional"] += requested_buy_qty * px

        buy_qty = requested_buy_qty
        bar_cap = _bar_qty_cap(sym, "BUY", snap, liq)
        if bar_cap is not None and buy_qty > bar_cap:
            diag["bar_cap_hits"] += 1
            _diag_add_symbol(diag, "symbols_bar_capped", sym)
            if bar_cap == 0 and _has_missing_or_zero_bar_volume(sym, snap):
                diag["missing_volume_blocks"] += 1
                _diag_add_symbol(diag, "symbols_missing_volume", sym)
            buy_qty = min(buy_qty, bar_cap)

        affordable_qty = int(math.floor(remaining / px))
        if buy_qty > affordable_qty:
            diag["cash_cap_hits"] += 1
            _diag_add_symbol(diag, "symbols_cash_capped", sym)
            buy_qty = min(buy_qty, affordable_qty)

        deferred = max(0, requested_buy_qty - buy_qty)
        diag["deferred_buy_qty"] += deferred
        diag["deferred_buy_notional"] += deferred * px

        if buy_qty > 0:
            orders.append(OrderRequest(ts=ts, order_id=str(uuid.uuid4()), symbol=sym, side="BUY", qty=buy_qty))
            diag["buy_orders"] += 1
            diag["submitted_buy_qty"] += buy_qty
            diag["submitted_buy_notional"] += buy_qty * px
            remaining -= buy_qty * px

    diag["orders"] = len(orders)
    return orders, diag


def _rebalance(
    ts,
    target_w: Dict[str, float],
    snap: MarketSnapshot,
    port: PositionSnapshot,
    liquidity: Optional[LiquidityConstraints] = None,
    liquidity_tracker: Optional[LiquidityTracker] = None,
) -> List[OrderRequest]:
    orders, _ = _rebalance_with_diagnostics(
        ts,
        target_w,
        snap,
        port,
        liquidity=liquidity,
        liquidity_tracker=liquidity_tracker,
    )
    return orders


rebalance_with_diagnostics = _rebalance_with_diagnostics
rebalance = _rebalance
