from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from common.events import MarketSnapshot
from runner.config import Config
from runner.telemetry import _safe_float_for_ui


@dataclass
class LiquidityConstraints:
    """Data-driven execution capacity controls.

    These are not fixed share-count limits. They scale with the actual traded
    volume of each symbol, so liquid large-cap names can absorb larger orders
    while illiquid/penny names are naturally throttled.
    """

    enabled: bool = False
    max_bar_participation: float = 0.10
    max_position_adv_participation: float = 0.10
    adv_lookback_days: int = 20
    min_adv_history_days: int = 5
    require_volume_for_buys: bool = True
    apply_to_sells: bool = True


class LiquidityTracker:
    """Online rolling ADV estimator from per-minute volume snapshots."""

    def __init__(self, lookback_days: int = 20):
        self.lookback_days = max(1, int(lookback_days))
        self.current_date = None
        self.current_day_volume: Dict[str, float] = defaultdict(float)
        self.daily_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.lookback_days))
        self.last_bar_volume: Dict[str, float] = {}
        self.seen_volume_snapshots: int = 0
        self.seen_volume_symbols: set[str] = set()

    def update(self, ts: datetime, volumes: Optional[Dict[str, float]]) -> None:
        d = ts.date()
        if self.current_date is None:
            self.current_date = d
        elif d != self.current_date:
            self._finalize_current_day()
            self.current_date = d
            self.current_day_volume = defaultdict(float)

        clean: Dict[str, float] = {}
        for sym, v in (volumes or {}).items():
            try:
                vv = float(v)
            except Exception:
                continue
            if math.isfinite(vv) and vv > 0:
                clean[str(sym)] = vv
                self.current_day_volume[str(sym)] += vv
        if clean:
            self.seen_volume_snapshots += 1
            self.seen_volume_symbols.update(clean.keys())
        self.last_bar_volume = clean

    def _finalize_current_day(self) -> None:
        for sym, vol in self.current_day_volume.items():
            if math.isfinite(float(vol)) and float(vol) > 0:
                self.daily_history[sym].append(float(vol))

    def adv_shares(self, sym: str, min_history_days: int = 1) -> Optional[float]:
        hist = self.daily_history.get(sym)
        if not hist or len(hist) < int(min_history_days):
            return None
        return float(sum(hist) / len(hist))


def _make_liquidity_constraints(cfg: Config) -> LiquidityConstraints:
    return LiquidityConstraints(
        enabled=bool(cfg.get("execution", "liquidity", "enabled", default=False)),
        max_bar_participation=float(cfg.get("execution", "liquidity", "max_bar_participation", default=0.10)),
        max_position_adv_participation=float(cfg.get("execution", "liquidity", "max_position_adv_participation", default=0.10)),
        adv_lookback_days=int(cfg.get("execution", "liquidity", "adv_lookback_days", default=20)),
        min_adv_history_days=int(cfg.get("execution", "liquidity", "min_adv_history_days", default=5)),
        require_volume_for_buys=bool(cfg.get("execution", "liquidity", "require_volume_for_buys", default=True)),
        apply_to_sells=bool(cfg.get("execution", "liquidity", "apply_to_sells", default=True)),
    )


def _clean_weight(w: float) -> float:
    try:
        v = float(w)
    except Exception:
        return 0.0
    if not math.isfinite(v) or v <= 0:
        return 0.0
    return v


def _bar_qty_cap(sym: str, side: str, snap: MarketSnapshot, liq: LiquidityConstraints) -> Optional[int]:
    if not liq.enabled:
        return None
    if side == "SELL" and not liq.apply_to_sells:
        return None

    vol = getattr(snap, "volumes", {}) or {}
    try:
        bar_vol = float(vol.get(sym, 0.0))
    except Exception:
        bar_vol = 0.0

    if not math.isfinite(bar_vol) or bar_vol <= 0:
        # Unknown volume means no realistic way to estimate participation.
        # For buys, block by default; for sells, follow apply_to_sells.
        if side == "BUY" and liq.require_volume_for_buys:
            return 0
        if side == "SELL" and liq.apply_to_sells:
            return 0
        return None

    cap = int(math.floor(max(0.0, liq.max_bar_participation) * bar_vol))
    return max(0, cap)


def _has_missing_or_zero_bar_volume(sym: str, snap: MarketSnapshot) -> bool:
    vol = getattr(snap, "volumes", {}) or {}
    try:
        bar_vol = float(vol.get(sym, 0.0))
    except Exception:
        bar_vol = 0.0
    return (not math.isfinite(bar_vol)) or bar_vol <= 0.0


def _position_qty_cap(sym: str, liq: LiquidityConstraints, tracker: Optional[LiquidityTracker]) -> Optional[int]:
    if not liq.enabled or tracker is None:
        return None
    adv = tracker.adv_shares(sym, min_history_days=liq.min_adv_history_days)
    if adv is None or not math.isfinite(adv) or adv <= 0:
        return None
    cap = int(math.floor(max(0.0, liq.max_position_adv_participation) * adv))
    return max(0, cap)


def _new_rebalance_diag(ts, liquidity: LiquidityConstraints, tracker: Optional[LiquidityTracker]) -> dict:
    return {
        "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        "liquidity_enabled": bool(liquidity.enabled),
        "orders": 0,
        "buy_orders": 0,
        "sell_orders": 0,
        "desired_buy_qty": 0,
        "desired_sell_qty": 0,
        "submitted_buy_qty": 0,
        "submitted_sell_qty": 0,
        "deferred_buy_qty": 0,
        "deferred_sell_qty": 0,
        "desired_buy_notional": 0.0,
        "desired_sell_notional": 0.0,
        "submitted_buy_notional": 0.0,
        "submitted_sell_notional": 0.0,
        "deferred_buy_notional": 0.0,
        "deferred_sell_notional": 0.0,
        "bar_cap_hits": 0,
        "adv_cap_hits": 0,
        "cash_cap_hits": 0,
        "missing_volume_blocks": 0,
        "symbols_bar_capped": [],
        "symbols_adv_capped": [],
        "symbols_cash_capped": [],
        "symbols_missing_volume": [],
        "volume_snapshots_seen": int(getattr(tracker, "seen_volume_snapshots", 0) or 0) if tracker is not None else 0,
        "volume_symbols_seen": int(len(getattr(tracker, "seen_volume_symbols", set()) or set())) if tracker is not None else 0,
    }


def _diag_add_symbol(diag: dict, key: str, sym: str, *, limit: int = 25) -> None:
    vals = diag.setdefault(key, [])
    sym = str(sym)
    if sym not in vals and len(vals) < int(limit):
        vals.append(sym)


def _accumulate_rebalance_diag(cum: dict, diag: Optional[dict]) -> None:
    if not isinstance(diag, dict):
        return
    cum["rebalance_checks"] = int(cum.get("rebalance_checks", 0)) + 1
    for key in (
        "orders", "buy_orders", "sell_orders",
        "desired_buy_qty", "desired_sell_qty", "submitted_buy_qty", "submitted_sell_qty",
        "deferred_buy_qty", "deferred_sell_qty",
        "bar_cap_hits", "adv_cap_hits", "cash_cap_hits", "missing_volume_blocks",
    ):
        cum[key] = int(cum.get(key, 0)) + int(diag.get(key, 0) or 0)
    for key in (
        "desired_buy_notional", "desired_sell_notional",
        "submitted_buy_notional", "submitted_sell_notional",
        "deferred_buy_notional", "deferred_sell_notional",
    ):
        cum[key] = float(cum.get(key, 0.0)) + _safe_float_for_ui(diag.get(key, 0.0))
    for src_key, dst_key in (
        ("symbols_bar_capped", "symbols_bar_capped"),
        ("symbols_adv_capped", "symbols_adv_capped"),
        ("symbols_cash_capped", "symbols_cash_capped"),
        ("symbols_missing_volume", "symbols_missing_volume"),
    ):
        cur = list(cum.get(dst_key, []))
        for sym in diag.get(src_key, []) or []:
            if sym not in cur and len(cur) < 50:
                cur.append(sym)
        cum[dst_key] = cur


def _friction_config_for_ui(cfg: Config, liquidity: LiquidityConstraints) -> dict:
    return {
        "initial_cash": float(cfg.get("execution", "initial_cash", default=1_000_000)),
        "slippage_model": str(cfg.get("execution", "slippage", "model", default="fixed_bps")),
        "slippage_bps": float(cfg.get("execution", "slippage", "bps", default=0.0)),
        "fees_model": str(cfg.get("execution", "fees", "model", default="fixed_bps")),
        "fees_bps": float(cfg.get("execution", "fees", "bps", default=0.0)),
        "liquidity_enabled": bool(liquidity.enabled),
        "max_bar_participation": float(liquidity.max_bar_participation),
        "max_position_adv_participation": float(liquidity.max_position_adv_participation),
        "adv_lookback_days": int(liquidity.adv_lookback_days),
        "min_adv_history_days": int(liquidity.min_adv_history_days),
        "require_volume_for_buys": bool(liquidity.require_volume_for_buys),
        "apply_to_sells": bool(liquidity.apply_to_sells),
    }


def _frictions_payload_for_ui(
    cfg: Config,
    liquidity: LiquidityConstraints,
    tracker: Optional[LiquidityTracker],
    cumulative_diag: dict,
    last_diag: dict,
) -> dict:
    return {
        "config": _friction_config_for_ui(cfg, liquidity),
        "liquidity": {
            "enabled": bool(liquidity.enabled),
            "volume_snapshots_seen": int(getattr(tracker, "seen_volume_snapshots", 0) or 0) if tracker is not None else 0,
            "volume_symbols_seen": int(len(getattr(tracker, "seen_volume_symbols", set()) or set())) if tracker is not None else 0,
        },
        "last_rebalance": last_diag or {},
        "cumulative_rebalance": cumulative_diag or {},
    }


make_liquidity_constraints = _make_liquidity_constraints
accumulate_rebalance_diag = _accumulate_rebalance_diag
frictions_payload_for_ui = _frictions_payload_for_ui
