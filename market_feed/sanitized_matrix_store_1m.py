from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Literal

import pandas as pd

from common.events import MarketSnapshot


@dataclass
class SanitizedMatrixStoreMinuteFeed:
    """Matrix-store feed with a runtime guardrail against absurd minute jumps.

    Why this exists:
    - offline repair should handle persistent split/bonus-like regime shifts
    - but a live backtest can still encounter stray bad prints or rogue bars
    - this feed rejects those bars instead of letting the strategy trade on them

    Policy:
    - keep the sparse snapshot behavior (NaN values are omitted)
    - maintain the last *accepted* good price per symbol
    - if the next bar moves by more than `max_abs_return`, reject it and keep the
      previously accepted price alive inside the execution engine
    """

    store_dir: str
    start: datetime
    end: datetime
    symbols: Optional[List[str]] = None
    speed: Literal['fast', 'realtime'] = 'fast'
    max_abs_return: float = 0.35
    min_price: float = 1e-9
    stats_every_rows: int = 0

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        root = Path(self.store_dir)
        date_dirs = sorted([p for p in root.glob('date=*') if p.is_dir()])
        if not date_dirs:
            raise FileNotFoundError(f'No date=* partitions under {root}')

        last_good: Dict[str, float] = {}
        seen_rows = 0
        rejected = 0

        for dd in date_dirs:
            day_path = dd / 'close.parquet'
            if not day_path.exists():
                continue

            df = pd.read_parquet(day_path)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.loc[(df['ts'] >= self.start) & (df['ts'] <= self.end)]
            if df.empty:
                continue

            cols = [c for c in df.columns if c != 'ts']
            if self.symbols:
                keep = [s for s in self.symbols if s in cols]
                df = df[['ts'] + keep]
                cols = keep

            for _, row in df.iterrows():
                seen_rows += 1
                ts = row['ts'].to_pydatetime()
                prices: Dict[str, float] = {}
                for sym in cols:
                    v = row[sym]
                    if pd.isna(v):
                        continue
                    px = float(v)
                    if not (px > self.min_price):
                        rejected += 1
                        continue

                    prev = last_good.get(sym)
                    if prev is not None:
                        abs_return = abs(px / prev - 1.0)
                        if abs_return > self.max_abs_return:
                            rejected += 1
                            continue
                    last_good[sym] = px
                    prices[sym] = px

                yield MarketSnapshot(ts=ts, prices=prices)

                if self.stats_every_rows and seen_rows % int(self.stats_every_rows) == 0:
                    print(f'[SanitizedMatrixStoreMinuteFeed] rows={seen_rows} rejected_quotes={rejected}')

                if self.speed == 'realtime':
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(0)
