
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

    _ticks: int = 0
    _last_target_weights: Dict[str, float] = field(default_factory=dict)
    _period_closes: Deque[Tuple[object, Dict[str, float]]] = field(default_factory=deque)
    _current_period_key: object = None
    _current_period_close: Dict[str, float] = field(default_factory=dict)
    _last_rebalance_period_key: object = None
    _last_diag: Dict[str, object] = field(default_factory=dict)

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
            },
            "weights": {
                "target": dict(sorted(target.items(), key=lambda kv: kv[1], reverse=True)[:10]),
            },
            "lists": {
                "support": support_syms,
                "previous_support": prev_support_syms,
            },
        }

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
                "scalars": dict(self._last_diag.get("scalars", {})),
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
            },
            "weights": {"target": dict(sorted(self._last_target_weights.items(), key=lambda kv: kv[1], reverse=True)[:10])},
            "lists": {},
            "latest_update": {"reason": reason},
        }
        if self.verbose_debug:
            print(f"[SparseSwitchMV] skip @ {ts}: {reason}")

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
