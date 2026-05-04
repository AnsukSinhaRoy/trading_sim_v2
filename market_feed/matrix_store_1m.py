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
    # Optional separate source for volume matrices. Useful when the repaired
    # price store contains only close.parquet but the original cube store still
    # contains volume.parquet. If omitted, the feed also tries a conservative
    # auto-fallback for common names like 1m_cube_store_repaired_v2 -> 1m_cube_store.
    volume_store_dir: Optional[str] = None

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        root = Path(self.store_dir)
        volume_roots = self._volume_roots(root)
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

            # Optional volume matrix. If present, emit traded share volume along
            # with prices so the execution layer can enforce liquidity-aware
            # participation constraints. Missing volume simply means "unknown"
            # for that symbol/minute; it is not filled or invented here.
            vol_by_ts = None
            volume_path = self._resolve_volume_path(dd, volume_roots)
            if volume_path is not None and volume_path.exists():
                vol_df = pd.read_parquet(volume_path)
                vol_df["ts"] = pd.to_datetime(vol_df["ts"])
                vol_df = vol_df.loc[(vol_df["ts"] >= self.start) & (vol_df["ts"] <= self.end)]
                vol_cols = [c for c in cols if c in vol_df.columns]
                if vol_cols:
                    vol_by_ts = vol_df[["ts"] + vol_cols].set_index("ts")

            # Stream rows
            for _, row in df.iterrows():
                ts_pd = row["ts"]
                ts = ts_pd.to_pydatetime()
                prices: Dict[str, float] = {}
                volumes: Dict[str, float] = {}
                vol_row = None
                if vol_by_ts is not None:
                    try:
                        vol_row = vol_by_ts.loc[ts_pd]
                        if isinstance(vol_row, pd.DataFrame):
                            vol_row = vol_row.iloc[-1]
                    except KeyError:
                        vol_row = None
                for sym in cols:
                    v = row[sym]
                    if pd.isna(v):
                        continue
                    prices[sym] = float(v)
                    if vol_row is not None and sym in vol_by_ts.columns:
                        vv = vol_row[sym]
                        if not pd.isna(vv):
                            volumes[sym] = float(vv)
                yield MarketSnapshot(ts=ts, prices=prices, volumes=volumes)

                if self.speed == "realtime":
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(0)

    def _volume_roots(self, price_root: Path) -> List[Path]:
        roots: List[Path] = []
        if self.volume_store_dir:
            roots.append(Path(self.volume_store_dir))

        # Common local workflow: repair scripts write close.parquet into a sibling
        # store named e.g. 1m_cube_store_repaired_v2, while volume.parquet remains
        # in the original 1m_cube_store. Try that sibling automatically.
        name = price_root.name
        if "_repaired" in name:
            roots.append(price_root.with_name(name.split("_repaired", 1)[0]))

        # Keep deterministic order and drop duplicates.
        out: List[Path] = []
        seen = set()
        for r in roots:
            key = str(r)
            if key not in seen and r != price_root:
                out.append(r)
                seen.add(key)
        return out

    def _resolve_volume_path(self, price_day_dir: Path, volume_roots: List[Path]) -> Optional[Path]:
        local = price_day_dir / "volume.parquet"
        if local.exists():
            return local
        for root in volume_roots:
            candidate = root / price_day_dir.name / "volume.parquet"
            if candidate.exists():
                return candidate
        return None
