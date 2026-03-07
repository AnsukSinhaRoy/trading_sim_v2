from __future__ import annotations

import numpy as np


def shaped_reward(
    cum_logret: float,
    step_rets: list[float] | None = None,
    turnover: float = 0.0,
    tc_penalty: float = 0.0,
    vol_penalty: float = 0.0,
    dd_penalty: float = 0.0,
) -> float:
    """Reward that encourages stable returns.

    - cum_logret: cumulative log-return over holding period
    - step_rets: per-snapshot portfolio log-returns for volatility / drawdown
    - turnover: L1 turnover applied at rebalance

    Returns a scalar reward.
    """
    r = float(cum_logret)

    if step_rets:
        x = np.asarray(step_rets, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size >= 5:
            r -= float(vol_penalty) * float(x.std())
            # drawdown on cumulative curve (log space)
            curve = np.cumsum(x)
            peak = np.maximum.accumulate(curve)
            dd = curve - peak
            max_dd = float(dd.min())  # negative
            r += float(dd_penalty) * max_dd  # penalize by adding negative number

    r -= float(tc_penalty) * float(turnover)
    return float(r)
