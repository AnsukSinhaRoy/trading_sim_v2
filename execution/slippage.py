from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class FixedBpsSlippage:
    bps: float
    def apply(self, side: Literal["BUY","SELL"], ref_price: float) -> float:
        s = self.bps / 10000.0
        return ref_price * (1.0 + s) if side == "BUY" else ref_price * (1.0 - s)

@dataclass
class FixedBpsFees:
    bps: float
    def fee(self, notional: float) -> float:
        return abs(notional) * (self.bps / 10000.0)
