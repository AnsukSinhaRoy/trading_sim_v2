from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Literal
import json

import pandas as pd

from common.events import MarketSnapshot

@dataclass
class MatrixStoreMinuteFeed:
    """Stream multi-asset 1-minute snapshots from a prebuilt matrix store.

    Expected layout:
      <store_dir>/date=YYYY-MM-DD/close.parquet
    where close.parquet has columns: ts, <symbol1>, <symbol2>, ...

    We load one date file at a time -> low peak memory, fast startup.
    """
    store_dir: str
    start: datetime
    end: datetime
    symbols: Optional[List[str]] = None   # optional subset
    speed: Literal["fast", "realtime"] = "fast"

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        root = Path(self.store_dir)
        # Find date partitions
        date_dirs = sorted([p for p in root.glob("date=*") if p.is_dir()])
        if not date_dirs:
            raise FileNotFoundError(f"No date=* partitions under {root}")

        for dd in date_dirs:
            date_str = dd.name.split("=", 1)[1]
            day_path = dd / "close.parquet"
            if not day_path.exists():
                continue

            df = pd.read_parquet(day_path)
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.loc[(df["ts"] >= self.start) & (df["ts"] <= self.end)]
            if df.empty:
                continue

            cols = [c for c in df.columns if c != "ts"]
            if self.symbols:
                keep = [s for s in self.symbols if s in cols]
                df = df[["ts"] + keep]
                cols = keep

            # Stream rows
            for _, row in df.iterrows():
                ts = row["ts"].to_pydatetime()
                prices: Dict[str, float] = {}
                for sym in cols:
                    v = row[sym]
                    if pd.isna(v):
                        continue
                    prices[sym] = float(v)
                yield MarketSnapshot(ts=ts, prices=prices)

                if self.speed == "realtime":
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(0)
