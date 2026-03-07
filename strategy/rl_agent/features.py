from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 5:
        return 0.0
    am = a.mean()
    bm = b.mean()
    av = a - am
    bv = b - bm
    denom = np.sqrt((av * av).sum()) * np.sqrt((bv * bv).sum())
    if denom <= 1e-12:
        return 0.0
    return float((av * bv).sum() / denom)


def _max_drawdown(log_returns: np.ndarray) -> float:
    """Max drawdown measured on cumulative log-return curve."""
    if log_returns.size == 0:
        return 0.0
    curve = np.cumsum(log_returns)
    peak = np.maximum.accumulate(curve)
    dd = curve - peak
    return float(dd.min())  # negative or 0


@dataclass
class FeatureConfig:
    lookback_short: int = 1950
    lookback_long: int = 23400
    corr_short: int = 1950
    corr_long: int = 23400


class FeatureEncoder:
    """Builds per-symbol features from price history.

    Designed for 1-min bars but works for any fixed-frequency stream.
    Stores only a rolling window of prices per symbol.
    """

    def __init__(
        self,
        cfg: FeatureConfig,
    ):
        self.cfg = cfg
        self._prices: Dict[str, deque] = {}

        self._maxlen = int(max(cfg.lookback_long, cfg.corr_long, cfg.lookback_short, cfg.corr_short) + 1)

    def update(self, prices: Dict[str, float]):
        for sym, px in prices.items():
            if px is None or not np.isfinite(px) or px <= 0:
                continue
            dq = self._prices.setdefault(sym, deque(maxlen=self._maxlen))
            dq.append(float(px))

    def ready_symbols(self, min_len: int) -> List[str]:
        out = []
        for s, dq in self._prices.items():
            if len(dq) >= int(min_len):
                out.append(s)
        return out

    def _returns(self, sym: str, window: int) -> np.ndarray:
        dq = self._prices.get(sym)
        if dq is None or len(dq) < 2:
            return np.zeros(0, dtype=np.float64)
        p = np.asarray(dq, dtype=np.float64)
        # use last window+1 prices
        w = min(int(window) + 1, p.size)
        p = p[-w:]
        r = np.diff(np.log(p))
        # cleanup non-finite
        r = r[np.isfinite(r)]
        return r

    def encode(self, symbols: List[str]) -> torch.Tensor:
        cfg = self.cfg
        n = len(symbols)
        if n == 0:
            return torch.zeros((0, 10), dtype=torch.float32)

        # Prepare returns matrices for market proxy
        # We'll compute market return as the cross-sectional mean of log-returns.
        # Align by using the minimum available length per symbol in the window.
        def build_market(window: int) -> np.ndarray:
            mats = []
            for s in symbols:
                r = self._returns(s, window)
                if r.size >= 2:
                    mats.append(r)
            if not mats:
                return np.zeros(0, dtype=np.float64)
            L = min(m.size for m in mats)
            if L < 2:
                return np.zeros(0, dtype=np.float64)
            stack = np.vstack([m[-L:] for m in mats])
            mkt = np.nanmean(stack, axis=0)
            mkt = mkt[np.isfinite(mkt)]
            return mkt

        mkt_s = build_market(cfg.corr_short)
        mkt_l = build_market(cfg.corr_long)

        feats = np.zeros((n, 10), dtype=np.float64)

        for i, s in enumerate(symbols):
            r_s = self._returns(s, cfg.lookback_short)
            r_l = self._returns(s, cfg.lookback_long)

            mom_s = float(r_s.sum()) if r_s.size else 0.0
            mom_l = float(r_l.sum()) if r_l.size else 0.0

            vol_s = float(r_s.std()) if r_s.size >= 5 else 0.0
            vol_l = float(r_l.std()) if r_l.size >= 5 else 0.0

            dd_s = _max_drawdown(r_s)
            dd_l = _max_drawdown(r_l)

            corr_s = _safe_corr(r_s[-min(r_s.size, mkt_s.size):], mkt_s[-min(r_s.size, mkt_s.size):]) if (r_s.size and mkt_s.size) else 0.0
            corr_l = _safe_corr(r_l[-min(r_l.size, mkt_l.size):], mkt_l[-min(r_l.size, mkt_l.size):]) if (r_l.size and mkt_l.size) else 0.0

            # Trend slope (simple)
            dq = self._prices.get(s)
            if dq is not None and len(dq) >= 20:
                p = np.asarray(dq, dtype=np.float64)
                w = min(p.size, 200)
                lp = np.log(p[-w:])
                x = np.linspace(0.0, 1.0, w)
                # slope via cov/var
                xv = x - x.mean()
                denom = float((xv * xv).sum())
                slope = float(((lp - lp.mean()) * xv).sum() / (denom + 1e-12))
            else:
                slope = 0.0

            # Pack
            feats[i] = [
                mom_s,
                mom_l,
                vol_s,
                vol_l,
                dd_s,
                dd_l,
                corr_s,
                corr_l,
                slope,
                1.0,  # bias feature
            ]

        # Cross-sectional normalization for stability
        for j in range(feats.shape[1] - 1):
            col = feats[:, j]
            mu = np.nanmean(col)
            sd = np.nanstd(col)
            if sd > 1e-8:
                feats[:, j] = (col - mu) / sd
            else:
                feats[:, j] = col - mu

        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(feats, dtype=torch.float32)
