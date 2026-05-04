from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, AsyncIterator
import random, math
from common.events import MarketSnapshot

@dataclass
class SyntheticMinuteFeed:
    symbols: List[str]
    start: datetime
    minutes: int
    init_prices: Dict[str, float]
    vol_bps: float = 15.0
    drift_bps: float = 0.0
    speed: str = "fast"

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        prices = {s: float(self.init_prices.get(s, 100.0)) for s in self.symbols}
        ts = self.start
        for _ in range(self.minutes):
            for s in self.symbols:
                vol = self.vol_bps / 10000.0
                drift = self.drift_bps / 10000.0
                prices[s] *= math.exp(drift + random.gauss(0.0, vol))
                prices[s] = max(0.01, prices[s])

            # Synthetic volume is only a rough stand-in so liquidity-aware
            # execution can be exercised in demos. Real backtests should use
            # volume.parquet from the matrix store.
            volumes = {s: float(max(1, int(random.lognormvariate(13.0, 0.5)))) for s in self.symbols}
            yield MarketSnapshot(ts=ts, prices=dict(prices), volumes=volumes)
            ts = ts + timedelta(minutes=1)

            if self.speed == "realtime":
                await asyncio.sleep(1.0)
            else:
                await asyncio.sleep(0)
