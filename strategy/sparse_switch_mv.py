
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple
from collections import deque
from datetime import datetime
import math

import numpy as np

from common.events import MarketSnapshot, OrderRequest, PositionSnapshot


def _is_finite_pos(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _project_to_capped_simplex(v: np.ndarray, z: float = 1.0, cap: Optional[float] = None) -> np.ndarray:
    """
    Euclidean projection onto {w >= 0, sum(w)=z, w_i <= cap if cap is not None}.

    Uses bisection on the dual variable. Stable and dependency-light.
    """
    x = np.asarray(v, dtype=float)
    n = int(x.size)
    if n == 0:
        return x.copy()

    if z <= 0.0:
        return np.zeros_like(x)

    if cap is not None:
        cap = float(cap)
        if cap <= 0.0:
            return np.zeros_like(x)
        if cap * n < z - 1e-12:
            # Infeasible cap; relax to uniform feasible allocation.
            return np.full(n, z / n, dtype=float)

    lo = float(np.min(x) - (cap if cap is not None else z) - 1.0)
    hi = float(np.max(x) + 1.0)

    for _ in range(80):
        tau = 0.5 * (lo + hi)
        w = x - tau
        if cap is None:
            w = np.maximum(w, 0.0)
        else:
            w = np.clip(w, 0.0, cap)
        s = float(w.sum())
        if s > z:
            lo = tau
        else:
            hi = tau

    w = x - hi
    if cap is None:
        w = np.maximum(w, 0.0)
    else:
        w = np.clip(w, 0.0, cap)

    s = float(w.sum())
    if s <= 0.0:
        if cap is None:
            return np.full(n, z / n, dtype=float)
        out = np.zeros(n, dtype=float)
        remaining = z
        for i in np.argsort(-x):
            take = min(cap, remaining)
            out[i] = take
            remaining -= take
            if remaining <= 1e-12:
                break
        return out
    return w * (z / s)


def _condition_number(mat: np.ndarray) -> float:
    try:
        eigvals = np.linalg.eigvalsh(mat)
    except np.linalg.LinAlgError:
        return float("inf")
    eigvals = np.asarray(eigvals, dtype=float)
    if eigvals.size == 0:
        return 0.0
    mn = float(np.min(np.abs(eigvals)))
    mx = float(np.max(np.abs(eigvals)))
    if mn <= 1e-18:
        return float("inf")
    return mx / mn


@dataclass
class SparseSwitchMVStrategy:
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
    regret_publish_on: str = "rebalance"
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

    def _regret_enabled(self) -> bool:
        return bool(self.track_regret or self.publish_regret) and str(self.regret_mode).lower() in {
            "hindsight_return_gap",
            "one_step_hindsight",
            "return_gap",
            # Backward-compatible aliases from the earlier patch. They now use
            # the corrected hindsight-return formulation rather than the signed
            # realized-minus-estimated loss gap.
            "estimation_gap",
            "estimation_regret",
            "realized_loss_gap",
        }

    def _store_pending_regret_decision(
        self,
        period_key: object,
        symbols: Sequence[str],
        mu: np.ndarray,
        sigma: np.ndarray,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        estimated_objective: float,
    ) -> None:
        if not self._regret_enabled():
            return
        self._pending_regret_decision = {
            "period_key": period_key,
            "symbols": list(symbols),
            "mu_est": np.asarray(mu, dtype=float).copy(),
            "sigma": np.asarray(sigma, dtype=float).copy(),
            "estimated_weights": np.asarray(weights, dtype=float).copy(),
            "prev_weights": np.asarray(prev_weights, dtype=float).copy(),
            "estimated_objective": float(estimated_objective),
        }

    def _maybe_update_hindsight_return_regret(
        self,
        snapshots: Sequence[Tuple[object, Dict[str, float]]],
        ts: datetime,
    ) -> None:
        """
        Compute a one-step hindsight return regret after the next aggregated
        return becomes available.

        At decision time, the live strategy has already chosen w_est using only
        past/estimated data. Once the next daily return r is known, this method
        reruns the same support-selection + restricted optimizer with r used as
        the now-seen mean vector. It then compares realized portfolio returns:

            raw_gap_t = r^T w_seen - r^T w_est
            regret_t  = max(raw_gap_t, 0)

        The max keeps cumulative regret non-decreasing. The signed raw gap is
        also published separately as raw_return_gap. This is telemetry only; it
        deliberately does not feed back into support selection, optimization,
        target weights, or order generation.
        """
        if not self._regret_enabled() or self._pending_regret_decision is None:
            return
        if len(snapshots) < 2:
            return

        pending = self._pending_regret_decision
        decision_period = pending.get("period_key")
        idx = None
        for i, (pkey, _) in enumerate(snapshots[:-1]):
            if pkey == decision_period:
                idx = i
                break
        if idx is None or idx + 1 >= len(snapshots):
            return

        prev_close = snapshots[idx][1]
        next_close = snapshots[idx + 1][1]
        symbols = list(pending.get("symbols", []))
        mu_est = np.asarray(pending.get("mu_est", []), dtype=float)
        sigma = np.asarray(pending.get("sigma", []), dtype=float)
        w_est = np.asarray(pending.get("estimated_weights", []), dtype=float)
        prev_w = np.asarray(pending.get("prev_weights", []), dtype=float)

        n = len(symbols)
        if n == 0 or mu_est.shape != (n,) or w_est.shape != (n,) or prev_w.shape != (n,) or sigma.shape != (n, n):
            self._pending_regret_decision = None
            return

        realized = np.array(mu_est, dtype=float, copy=True)
        valid_mask = np.zeros(n, dtype=bool)
        for j, sym in enumerate(symbols):
            p0 = _safe_float(prev_close.get(sym, float("nan")), default=float("nan"))
            p1 = _safe_float(next_close.get(sym, float("nan")), default=float("nan"))
            if _is_finite_pos(p0) and _is_finite_pos(p1):
                r = (p1 / p0) - 1.0
                if math.isfinite(r):
                    realized[j] = float(r)
                    valid_mask[j] = True

        if not bool(np.any(valid_mask)):
            return

        # Do not reward/punish missing prices. Keep their realized signal at the
        # estimate and compare both portfolios only on assets with known returns.
        realized_for_return = np.zeros(n, dtype=float)
        realized_for_return[valid_mask] = realized[valid_mask]

        hindsight_support = self._select_support(symbols, realized, sigma, prev_w)
        if not hindsight_support:
            return

        w_seen, hindsight_objective, _ = self._solve_full_target(
            symbols, realized, sigma, prev_w, hindsight_support
        )

        actual_return = float(realized_for_return @ w_est)
        hindsight_return = float(realized_for_return @ w_seen)
        raw_gap = float(hindsight_return - actual_return)
        regret = float(max(raw_gap, 0.0))

        self._last_return_regret = regret
        self._last_raw_return_gap = raw_gap
        self._last_actual_portfolio_return = actual_return
        self._last_hindsight_portfolio_return = hindsight_return
        self._last_hindsight_objective = float(hindsight_objective)
        self._last_actual_objective = float(_safe_float(pending.get("estimated_objective", 0.0)))
        self._cumulative_return_regret += regret
        self._sum_abs_return_gap += abs(raw_gap)
        self._regret_count += 1
        self._last_regret_ts = ts.isoformat()
        self._last_hindsight_support = [symbols[int(i)] for i in hindsight_support]
        self._pending_regret_decision = None

    def _regret_scalars_for_telemetry(self, ts: Optional[datetime] = None) -> Dict[str, float]:
        if not self.publish_regret or self._last_return_regret is None:
            return {}
        mean_abs_gap = self._sum_abs_return_gap / max(1, self._regret_count)
        scalars = {
            # Dashboard aliases. These now represent one-step hindsight return
            # regret, not the older signed estimation-loss gap.
            "regret": float(self._last_return_regret),
            "cum_regret": float(self._cumulative_return_regret),
            "return_regret": float(self._last_return_regret),
            "cumulative_return_regret": float(self._cumulative_return_regret),
            "raw_return_gap": float(self._last_raw_return_gap if self._last_raw_return_gap is not None else 0.0),
            "actual_portfolio_return": float(
                self._last_actual_portfolio_return if self._last_actual_portfolio_return is not None else 0.0
            ),
            "hindsight_portfolio_return": float(
                self._last_hindsight_portfolio_return if self._last_hindsight_portfolio_return is not None else 0.0
            ),
            "mean_abs_return_gap": float(mean_abs_gap),
            "regret_count": int(self._regret_count),
        }
        if self._last_actual_objective is not None:
            scalars["estimated_objective"] = float(self._last_actual_objective)
        if self._last_hindsight_objective is not None:
            scalars["hindsight_objective"] = float(self._last_hindsight_objective)

        fields = self.regret_publish_fields
        if fields:
            wanted = {str(x) for x in fields}
            # Keep the dashboard aliases if the explicit field list uses the
            # descriptive names.
            if "return_regret" in wanted:
                wanted.add("regret")
            if "cumulative_return_regret" in wanted:
                wanted.add("cum_regret")
            scalars = {k: v for k, v in scalars.items() if k in wanted}
        return scalars

    def _filter_regret_scalars_for_snapshot(
        self,
        scalars: Dict[str, object],
        snap: Optional[MarketSnapshot],
    ) -> Dict[str, object]:
        """
        Decide when regret scalars are emitted to the dashboard.

        The engine publishes the `learn` topic only every `ui.publish_every_ticks`.
        That usually does not coincide with `rebalance_minutes` (for example,
        376 vs 375 in the cube config). A strict same-tick rebalance gate therefore
        computes regret but silently drops it before the UI ever sees it.

        Default behavior: emit each newly computed regret value exactly once on
        the next telemetry publish. Use regret_publish_on=always/every_publish if
        you want the last regret value repeated on every learn payload.
        """
        if not self.publish_regret:
            return scalars

        mode = str(self.regret_publish_on).lower().strip()
        if mode in {"always", "every", "every_publish", "snapshot", "all"}:
            return scalars

        regret_keys = {
            "regret",
            "cum_regret",
            "return_regret",
            "cumulative_return_regret",
            "raw_return_gap",
            "actual_portfolio_return",
            "hindsight_portfolio_return",
            "mean_abs_return_gap",
            "regret_count",
            "estimated_objective",
            "hindsight_objective",
            # Backward-compatible names from older telemetry payloads.
            "last_estimation_regret",
            "cumulative_estimation_regret",
            "positive_estimation_regret",
            "mean_abs_estimation_gap",
            "estimated_loss",
            "observed_loss",
        }

        # For historical compatibility, `rebalance` now means: emit once at the
        # next dashboard publish after the rebalance/regret computation.
        # This is the only reliable behavior under throttled ZMQ publishing.
        if self._last_return_regret is None or self._regret_count <= self._last_published_regret_count:
            return {k: v for k, v in scalars.items() if k not in regret_keys}

        self._last_published_regret_count = int(self._regret_count)
        return scalars

    def _period_key(self, ts: datetime) -> object:
        if self.estimation_frequency == "1d":
            return ts.date()
        raise ValueError(f"Unsupported estimation_frequency={self.estimation_frequency}")

    def _update_aggregated_history(self, snap: MarketSnapshot) -> None:
        pkey = self._period_key(snap.ts)
        if self._current_period_key is None:
            self._current_period_key = pkey
        elif pkey != self._current_period_key:
            self._finalize_current_period()
            self._current_period_key = pkey
            self._current_period_close = {}

        for sym, px in snap.prices.items():
            v = _safe_float(px, default=float("nan"))
            if _is_finite_pos(v):
                self._current_period_close[str(sym)] = v

    def _finalize_current_period(self) -> None:
        if self._current_period_key is None or not self._current_period_close:
            return
        self._period_closes.append((self._current_period_key, dict(self._current_period_close)))
        while len(self._period_closes) > max(8, int(self.lookback_bars)):
            self._period_closes.popleft()

    def _snapshot_history_with_current(self) -> List[Tuple[object, Dict[str, float]]]:
        items = list(self._period_closes)
        if self._current_period_key is not None and self._current_period_close:
            items.append((self._current_period_key, dict(self._current_period_close)))
        if len(items) > max(8, int(self.lookback_bars)):
            items = items[-int(self.lookback_bars):]
        return items

    def _build_return_matrix(self, snapshots: Sequence[Tuple[object, Dict[str, float]]]) -> Tuple[List[str], np.ndarray]:
        symbol_set = set()
        for _, closes in snapshots:
            symbol_set.update(closes.keys())
        symbols = sorted(symbol_set)
        if not symbols or len(snapshots) < 2:
            return [], np.zeros((0, 0), dtype=float)

        rows: List[List[float]] = []
        for i in range(1, len(snapshots)):
            prev_close = snapshots[i - 1][1]
            cur_close = snapshots[i][1]
            row = []
            any_valid = False
            for sym in symbols:
                p0 = _safe_float(prev_close.get(sym, float("nan")), default=float("nan"))
                p1 = _safe_float(cur_close.get(sym, float("nan")), default=float("nan"))
                if _is_finite_pos(p0) and _is_finite_pos(p1):
                    r = (p1 / p0) - 1.0
                    if math.isfinite(r):
                        row.append(float(r))
                        any_valid = True
                        continue
                row.append(float("nan"))
            if any_valid:
                rows.append(row)

        if not rows:
            return symbols, np.zeros((0, len(symbols)), dtype=float)
        return symbols, np.asarray(rows, dtype=float)

    def _extract_prev_weights(self, symbols: Sequence[str]) -> np.ndarray:
        raw = np.array([_safe_float(self._last_target_weights.get(sym, 0.0)) for sym in symbols], dtype=float)
        raw[raw < 0.0] = 0.0
        s = float(raw.sum())
        if s <= 0.0:
            return np.zeros_like(raw)
        return raw / s

    def _estimate_statistics(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_rows, n_cols = returns.shape
        mu = np.zeros(n_cols, dtype=float)
        valid_counts = np.zeros(n_cols, dtype=int)
        mean_lb = max(1, min(int(self.mean_lookback_periods), n_rows))

        for j in range(n_cols):
            col = returns[:, j]
            valid = col[np.isfinite(col)]
            valid_counts[j] = int(valid.size)
            if valid.size > 0:
                mu[j] = float(np.mean(valid[-mean_lb:]))
            else:
                mu[j] = 0.0

        cov = self._compute_covariance(returns)
        return mu, cov, valid_counts

    def _compute_covariance(self, returns: np.ndarray) -> np.ndarray:
        n_rows, n_cols = returns.shape
        if n_cols == 0:
            return np.zeros((0, 0), dtype=float)
        if n_rows == 0:
            return np.eye(n_cols, dtype=float) * float(self.cov_epsilon)

        arr = np.array(returns, dtype=float, copy=True)
        col_means = np.nanmean(arr, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(~np.isfinite(arr))
        if inds[0].size > 0:
            arr[inds] = col_means[inds[1]]

        if arr.shape[0] == 1:
            cov = np.zeros((n_cols, n_cols), dtype=float)
        else:
            cov = np.cov(arr, rowvar=False, ddof=1)
            cov = np.asarray(cov, dtype=float)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]], dtype=float)
        if cov.shape != (n_cols, n_cols):
            cov = np.eye(n_cols, dtype=float) * float(self.cov_epsilon)
        return cov

    def _stabilize_covariance(self, sigma: np.ndarray) -> np.ndarray:
        n = sigma.shape[0]
        if n == 0:
            return sigma
        diag = np.diag(np.diag(sigma))
        out = (1.0 - float(self.cov_shrinkage)) * sigma + float(self.cov_shrinkage) * diag
        out = out + np.eye(n, dtype=float) * float(self.cov_epsilon)
        out = 0.5 * (out + out.T)
        return out

    def _select_support(self, symbols: Sequence[str], mu: np.ndarray, sigma: np.ndarray, prev_w: np.ndarray) -> List[int]:
        if len(symbols) == 0:
            return []
        k = max(1, min(int(self.support_k), len(symbols)))
        grad = -mu + float(self.lambda_risk) * (sigma @ prev_w)
        u = prev_w - float(self.step_size) * grad

        if self.persistence_bonus != 0.0:
            u = u + float(self.persistence_bonus) * (prev_w > 0.0).astype(float)

        ranked = np.argsort(-u)
        support = sorted(int(i) for i in ranked[:k])
        return support

    def _solve_full_target(
        self,
        symbols: Sequence[str],
        mu: np.ndarray,
        sigma: np.ndarray,
        prev_w: np.ndarray,
        support: Sequence[int],
    ) -> Tuple[np.ndarray, float, float]:
        support = list(sorted(int(i) for i in support))
        mu_s = mu[support]
        sigma_s = sigma[np.ix_(support, support)]
        prev_s = prev_w[support]
        support_prev_mass = float(np.sum(prev_s))

        init = self._initial_support_weights(prev_s, len(support))
        w_s = self._projected_gradient_restricted(mu_s, sigma_s, prev_s, init)

        full = np.zeros(len(symbols), dtype=float)
        full[np.asarray(support, dtype=int)] = w_s
        full[full < float(self.min_target_weight)] = 0.0
        s = float(full.sum())
        if s > 0.0:
            full /= s

        obj = self._objective(full, mu, sigma, prev_w)
        return full, obj, support_prev_mass

    def _initial_support_weights(self, prev_s: np.ndarray, m: int) -> np.ndarray:
        prev_s = np.asarray(prev_s, dtype=float)
        if self.warm_start and prev_s.size == m and np.sum(prev_s) > 0.0:
            return _project_to_capped_simplex(prev_s, z=1.0, cap=self.max_weight_per_asset)
        return _project_to_capped_simplex(np.ones(m, dtype=float) / max(1, m), z=1.0, cap=self.max_weight_per_asset)

    def _projected_gradient_restricted(self, mu_s: np.ndarray, sigma_s: np.ndarray, prev_s: np.ndarray, w0: np.ndarray) -> np.ndarray:
        w = _project_to_capped_simplex(np.asarray(w0, dtype=float), z=1.0, cap=self.max_weight_per_asset)
        lipschitz = float(self.lambda_risk) * self._largest_eigenvalue(sigma_s) + float(self.kappa_switch)
        step = 1.0 / max(1e-8, lipschitz + 1e-8)
        step = min(step, 1.0)

        prev_obj = self._restricted_objective(w, mu_s, sigma_s, prev_s)
        for _ in range(max(10, int(self.optimization_max_iters))):
            grad = -mu_s + float(self.lambda_risk) * (sigma_s @ w) + float(self.kappa_switch) * (w - prev_s)
            cand = _project_to_capped_simplex(w - step * grad, z=1.0, cap=self.max_weight_per_asset)
            obj = self._restricted_objective(cand, mu_s, sigma_s, prev_s)
            if obj <= prev_obj + 1e-14:
                if np.linalg.norm(cand - w, ord=2) <= float(self.optimization_tol):
                    w = cand
                    break
                w = cand
                prev_obj = obj
                continue

            local_step = step * 0.5
            improved = False
            for _ in range(12):
                cand = _project_to_capped_simplex(w - local_step * grad, z=1.0, cap=self.max_weight_per_asset)
                obj = self._restricted_objective(cand, mu_s, sigma_s, prev_s)
                if obj <= prev_obj + 1e-14:
                    if np.linalg.norm(cand - w, ord=2) <= float(self.optimization_tol):
                        w = cand
                        improved = True
                        prev_obj = obj
                        break
                    w = cand
                    prev_obj = obj
                    step = local_step
                    improved = True
                    break
                local_step *= 0.5
            if not improved:
                break
        return w

    def _largest_eigenvalue(self, mat: np.ndarray) -> float:
        if mat.size == 0:
            return 0.0
        try:
            vals = np.linalg.eigvalsh(mat)
            return float(np.max(np.abs(vals)))
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(mat, ord=2))

    def _restricted_objective(self, w: np.ndarray, mu_s: np.ndarray, sigma_s: np.ndarray, prev_s: np.ndarray) -> float:
        return float(
            -mu_s @ w
            + 0.5 * float(self.lambda_risk) * (w @ sigma_s @ w)
            + 0.5 * float(self.kappa_switch) * np.sum((w - prev_s) ** 2)
        )

    def _objective(self, w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, prev_w: np.ndarray) -> float:
        return float(
            -mu @ w
            + 0.5 * float(self.lambda_risk) * (w @ sigma @ w)
            + 0.5 * float(self.kappa_switch) * np.sum((w - prev_w) ** 2)
        )

    def _weights_dict(self, symbols: Sequence[str], weights: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sym, w in zip(symbols, weights):
            wf = float(w)
            if wf > float(self.min_target_weight):
                out[str(sym)] = wf
        return out

    def _turnover(self, old: Dict[str, float], new: Dict[str, float]) -> float:
        syms = set(old) | set(new)
        return float(sum(abs(_safe_float(old.get(sym, 0.0)) - _safe_float(new.get(sym, 0.0))) for sym in syms))


STRATEGY_CLASS = SparseSwitchMVStrategy
