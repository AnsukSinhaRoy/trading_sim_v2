from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import math

from common.events import MarketSnapshot, OrderRequest, PositionSnapshot


@dataclass
class ToyRebalanceStrategy:
    rebalance_every_minutes: int = 60
    target_count: int = 3
    max_gross_leverage: float = 1.0

    _history: Dict[str, List[float]] = field(default_factory=dict)
    _ticks: int = 0
    _last_target_weights: Dict[str, float] = field(default_factory=dict)

    def on_snapshot(self, snap: MarketSnapshot, portfolio: PositionSnapshot) -> List[OrderRequest]:
        self._ticks += 1

        # Collect price history, but ignore bad ticks (NaN/inf/non-positive)
        for sym, px in snap.prices.items():
            try:
                v = float(px)
            except (TypeError, ValueError):
                continue

            if (not math.isfinite(v)) or v <= 0.0:
                # Skip vendor "0" bars / missing bars / bad ticks
                continue

            self._history.setdefault(sym, []).append(v)
            if len(self._history[sym]) > 120:
                self._history[sym] = self._history[sym][-120:]

        # Only rebalance on schedule
        if self._ticks % max(1, self.rebalance_every_minutes) != 0:
            return []

        scores = []
        for sym, series in self._history.items():
            if len(series) < 2:
                continue

            window = series[-60:] if len(series) >= 60 else series
            base = window[0]
            last = window[-1]

            # Guard against divide-by-zero / bad values
            if (not math.isfinite(base)) or (not math.isfinite(last)) or base <= 0.0:
                continue

            mom = (last / base) - 1.0
            scores.append((sym, mom))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in scores[: self.target_count]] if scores else []

        self._last_target_weights = (
            {sym: (self.max_gross_leverage / len(top)) for sym in top} if top else {}
        )

        # Toy strategy currently doesn't place orders (only computes targets)
        return []
