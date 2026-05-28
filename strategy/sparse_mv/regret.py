from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from common.events import MarketSnapshot
from .math_utils import _is_finite_pos, _safe_float


class SparseRegretMixin:
    """Telemetry-only hindsight-return regret diagnostics."""

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
