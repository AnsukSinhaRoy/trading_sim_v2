from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)

    def position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def set_position(self, symbol: str, qty: int) -> None:
        self.positions[symbol] = int(qty)

    def update_cash(self, delta: float) -> None:
        self.cash = float(self.cash + delta)

    def nav(self, mtm_prices: Dict[str, float]) -> float:
        total = float(self.cash)
        for sym, qty in self.positions.items():
            total += float(qty) * float(mtm_prices.get(sym, 0.0))
        return total
