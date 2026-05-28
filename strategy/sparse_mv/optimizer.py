from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .math_utils import _project_to_capped_simplex, _safe_float


class SparseOptimizerMixin:
    """Support selection and restricted simplex optimization."""

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
