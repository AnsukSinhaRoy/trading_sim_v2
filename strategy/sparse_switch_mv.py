from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.events import MarketSnapshot, OrderRequest, PositionSnapshot
from strategy.sparse_mv.history import SparseHistoryMixin
from strategy.sparse_mv.math_utils import _condition_number, _safe_float
from strategy.sparse_mv.optimizer import SparseOptimizerMixin
from strategy.sparse_mv.regret import SparseRegretMixin


@dataclass
class SparseSwitchMVStrategy(SparseRegretMixin, SparseOptimizerMixin, SparseHistoryMixin):
    """
    Sparse switching mean-variance allocator for minute-driven event simulators.

    Design:
    - consumes minute snapshots,
    - internally aggregates to lower-frequency closes (default: daily),
    - estimates mean/covariance from aggregated return history,
    - selects a top-k support by a proximal-gradient style score,
    - solves a restricted long-only simplex allocation on that support,
    - exposes target weights through `_last_target_weights` so the engine can route
      orders via the existing rebalance path.
    """

    rebalance_minutes: int = 375
    lookback_bars: int = 260
    estimation_frequency: str = "1d"
    mean_lookback_periods: int = 60
    cov_lookback_periods: int = 120
    support_k: int = 15
    lambda_risk: float = 10.0
    kappa_switch: float = 1.0
    step_size: float = 0.5
    cov_shrinkage: float = 0.10
    cov_epsilon: float = 1e-5
    persistence_bonus: float = 0.0
    min_history_periods: int = 40
    max_weight_per_asset: Optional[float] = None
    warm_start: bool = True
    verbose_debug: bool = False
    optimization_max_iters: int = 200
    optimization_tol: float = 1e-8
    min_target_weight: float = 1e-6
    allow_same_period_rebalance: bool = False

    # Optional telemetry-only regret diagnostic.
    # This does not change support selection, optimization, or execution.
    # Default mode is a one-step hindsight return gap:
    #   regret_t = max(0, r_t^T w_seen_t - r_t^T w_est_t)
    # where w_est_t is the portfolio produced before the next return is known,
    # and w_seen_t is produced by rerunning the same optimizer after that return is known.
    track_regret: bool = False
    publish_regret: bool = False
    regret_mode: str = "hindsight_return_gap"
    regret_publish_on: str = "every_publish"
    regret_publish_fields: Optional[Sequence[str]] = None

    _ticks: int = 0
    _last_target_weights: Dict[str, float] = field(default_factory=dict)
    _period_closes: Deque[Tuple[object, Dict[str, float]]] = field(default_factory=deque)
    _current_period_key: object = None
    _current_period_close: Dict[str, float] = field(default_factory=dict)
    _last_rebalance_period_key: object = None
    _last_diag: Dict[str, object] = field(default_factory=dict)

    _pending_regret_decision: Optional[Dict[str, object]] = None
    _last_return_regret: Optional[float] = None
    _last_raw_return_gap: Optional[float] = None
    _last_actual_portfolio_return: Optional[float] = None
    _last_hindsight_portfolio_return: Optional[float] = None
    _last_hindsight_objective: Optional[float] = None
    _last_actual_objective: Optional[float] = None
    _cumulative_return_regret: float = 0.0
    _sum_abs_return_gap: float = 0.0
    _regret_count: int = 0
    _last_regret_ts: Optional[str] = None
    _last_hindsight_support: List[str] = field(default_factory=list)
    _last_published_regret_count: int = 0

    def on_snapshot(self, snap: MarketSnapshot, portfolio: PositionSnapshot) -> List[OrderRequest]:
        self._ticks += 1
        self._update_aggregated_history(snap)

        if self._ticks % max(1, int(self.rebalance_minutes)) != 0:
            return []

        if self.estimation_frequency != "1d":
            self._set_skip_diag(snap.ts, f"unsupported estimation_frequency={self.estimation_frequency}")
            return []

        current_period = self._period_key(snap.ts)
        if (not self.allow_same_period_rebalance) and self._last_rebalance_period_key == current_period:
            self._set_skip_diag(snap.ts, "already rebalanced in current aggregated period")
            return []

        snapshots = self._snapshot_history_with_current()
        self._maybe_update_hindsight_return_regret(snapshots, snap.ts)
        if len(snapshots) < max(3, self.min_history_periods + 1):
            self._set_skip_diag(snap.ts, f"insufficient aggregated history: periods={len(snapshots)}")
            return []

        symbols, returns = self._build_return_matrix(snapshots)
        if returns.shape[0] < max(2, self.min_history_periods):
            self._set_skip_diag(snap.ts, f"insufficient return rows: rows={returns.shape[0]}")
            return []
        if len(symbols) == 0:
            self._set_skip_diag(snap.ts, "no symbols with valid aggregated return history")
            return []

        prev_w = self._extract_prev_weights(symbols)
        mu, sigma_raw, valid_counts = self._estimate_statistics(returns)
        eligible = valid_counts >= max(2, int(self.min_history_periods))
        if not np.any(eligible):
            self._set_skip_diag(snap.ts, "no symbols passed min_history_periods filter")
            return []

        symbols_e = [sym for sym, keep in zip(symbols, eligible) if keep]
        mu_e = mu[eligible]
        prev_w_e = prev_w[eligible]
        returns_e = returns[:, eligible]
        sigma_e = self._compute_covariance(returns_e)
        sigma_e = self._stabilize_covariance(sigma_e)

        support = self._select_support(symbols_e, mu_e, sigma_e, prev_w_e)
        if not support:
            self._set_skip_diag(snap.ts, "support selection produced empty support")
            return []

        target_full, objective_value, support_prev_mass = self._solve_full_target(symbols_e, mu_e, sigma_e, prev_w_e, support)
        target = self._weights_dict(symbols_e, target_full)

        turnover = self._turnover(self._last_target_weights, target)
        self._last_target_weights = target
        self._last_rebalance_period_key = current_period

        support_syms = [symbols_e[i] for i in support]
        prev_support_syms = [sym for sym, w in self._last_diag.get("target_weights", {}).items() if _safe_float(w) > 0.0]
        self._last_diag = {
            "mode": "sparse_switch_mv",
            "status": "rebalanced",
            "ts": snap.ts.isoformat(),
            "period_key": str(current_period),
            "support": support_syms,
            "previous_support": prev_support_syms,
            "support_size": len(support_syms),
            "turnover": float(turnover),
            "lambda_risk": float(self.lambda_risk),
            "kappa_switch": float(self.kappa_switch),
            "mu_mean": float(np.mean(mu_e)) if mu_e.size else 0.0,
            "mu_max": float(np.max(mu_e)) if mu_e.size else 0.0,
            "mu_min": float(np.min(mu_e)) if mu_e.size else 0.0,
            "cov_condition": float(_condition_number(sigma_e)),
            "cov_shrinkage": float(self.cov_shrinkage),
            "cov_epsilon": float(self.cov_epsilon),
            "objective_value": float(objective_value),
            "support_prev_mass": float(support_prev_mass),
            "target_weights": dict(sorted(target.items(), key=lambda kv: kv[1], reverse=True)),
            "latest_update": {
                "selected_support": support_syms,
                "target_weights": dict(sorted(target.items(), key=lambda kv: kv[1], reverse=True)[:10]),
            },
            "scalars": {
                "support_size": len(support_syms),
                "turnover": float(turnover),
                "lambda_risk": float(self.lambda_risk),
                "kappa_switch": float(self.kappa_switch),
                "mu_mean": float(np.mean(mu_e)) if mu_e.size else 0.0,
                "mu_max": float(np.max(mu_e)) if mu_e.size else 0.0,
                "mu_min": float(np.min(mu_e)) if mu_e.size else 0.0,
                "cov_condition": float(_condition_number(sigma_e)),
                "objective_value": float(objective_value),
                **self._regret_scalars_for_telemetry(snap.ts),
            },
            "weights": {
                "target": dict(sorted(target.items(), key=lambda kv: kv[1], reverse=True)[:10]),
            },
            "lists": {
                "support": support_syms,
                "previous_support": prev_support_syms,
            },
        }

        self._store_pending_regret_decision(
            period_key=current_period,
            symbols=symbols_e,
            mu=mu_e,
            sigma=sigma_e,
            weights=target_full,
            prev_weights=prev_w_e,
            estimated_objective=objective_value,
        )

        if self.verbose_debug:
            print(
                f"[SparseSwitchMV] ts={snap.ts} support={support_syms} turnover={turnover:.4f} "
                f"obj={objective_value:.6f} cond={_condition_number(sigma_e):.2e}"
            )

        return []

    def get_telemetry(self, snap: Optional[MarketSnapshot] = None, portfolio: Optional[PositionSnapshot] = None) -> Dict[str, object]:
        if self._last_diag:
            return {
                "mode": self._last_diag.get("mode", "sparse_switch_mv"),
                "status": self._last_diag.get("status", "idle"),
                "scalars": self._filter_regret_scalars_for_snapshot(
                    dict(self._last_diag.get("scalars", {})), snap
                ),
                "weights": dict(self._last_diag.get("weights", {"target": {}})),
                "lists": dict(self._last_diag.get("lists", {})),
                "latest_update": dict(self._last_diag.get("latest_update", {})),
            }
        return {
            "mode": "sparse_switch_mv",
            "status": "warming_up",
            "scalars": {
                "support_size": 0,
                "turnover": 0.0,
                "lambda_risk": float(self.lambda_risk),
                "kappa_switch": float(self.kappa_switch),
            },
            "weights": {"target": dict(self._last_target_weights)},
            "lists": {},
            "latest_update": {},
        }

    def _set_skip_diag(self, ts: datetime, reason: str) -> None:
        self._last_diag = {
            "mode": "sparse_switch_mv",
            "status": reason,
            "ts": ts.isoformat(),
            "scalars": {
                "support_size": int(len(self._last_target_weights)),
                "turnover": 0.0,
                "lambda_risk": float(self.lambda_risk),
                "kappa_switch": float(self.kappa_switch),
                **self._regret_scalars_for_telemetry(ts),
            },
            "weights": {"target": dict(sorted(self._last_target_weights.items(), key=lambda kv: kv[1], reverse=True)[:10])},
            "lists": {},
            "latest_update": {"reason": reason},
        }
        if self.verbose_debug:
            print(f"[SparseSwitchMV] skip @ {ts}: {reason}")


STRATEGY_CLASS = SparseSwitchMVStrategy
