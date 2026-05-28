from __future__ import annotations

import math
from typing import Any

from common.events import MarketSnapshot, PositionSnapshot


def _safe_float_for_ui(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _extract_strategy_telemetry(strat, snap: MarketSnapshot, port: PositionSnapshot, tick_count: int) -> dict:
    payload = {
        "ts": snap.ts.isoformat(),
        "tick": int(tick_count),
        "strategy": strat.__class__.__name__,
    }

    metrics = None
    for name in ("get_dashboard_metrics", "get_ui_metrics", "get_telemetry"):
        fn = getattr(strat, name, None)
        if callable(fn):
            try:
                metrics = fn(snap=snap, portfolio=port)
            except TypeError:
                metrics = fn()
            except Exception:
                metrics = None
            if metrics is not None:
                break

    if not isinstance(metrics, dict):
        target_w = getattr(strat, "_last_target_weights", None)
        weights = {}
        if isinstance(target_w, dict):
            weights = {
                str(sym): _safe_float_for_ui(wt)
                for sym, wt in sorted(target_w.items(), key=lambda kv: abs(_safe_float_for_ui(kv[1])), reverse=True)[:10]
                if abs(_safe_float_for_ui(wt)) > 1e-12
            }

        metrics = {
            "mode": "generic",
            "status": f"tick={tick_count}",
            "scalars": {
                "tick": int(tick_count),
                "nav": _safe_float_for_ui(getattr(port, "nav", 0.0)),
                "cash": _safe_float_for_ui(getattr(port, "cash", 0.0)),
                "target_count": int(len(weights)),
            },
            "weights": {"target": weights},
            "lists": {},
            "latest_update": {},
        }

        for attr in (
            "_decision_steps",
            "_active_turnover",
            "_seg_cum_lr",
            "_last_reward",
            "rebalance_every_minutes",
            "max_assets",
            "lookback",
            "lookback_short",
            "lookback_long",
        ):
            if hasattr(strat, attr):
                metrics.setdefault("scalars", {})[attr] = _safe_float_for_ui(getattr(strat, attr))

        learner = getattr(strat, "_learner", None)
        if learner is not None and hasattr(learner, "updates"):
            metrics.setdefault("scalars", {})["learner_updates"] = int(getattr(learner, "updates", 0))

        buffer = getattr(strat, "_buffer", None)
        if buffer is not None:
            try:
                metrics.setdefault("scalars", {})["buffer_size"] = int(len(buffer))
            except Exception:
                pass

    payload.update(metrics)
    return payload


extract_strategy_telemetry = _extract_strategy_telemetry
safe_float_for_ui = _safe_float_for_ui
