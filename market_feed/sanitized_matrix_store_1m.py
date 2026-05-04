from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Literal, Tuple

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
    # If an apparent jump persists for several consecutive, internally-consistent
    # bars, treat it as a new valid price regime instead of suppressing the
    # symbol forever. This is deliberately conservative: one-off bad ticks still
    # get rejected, while split/adjustment-like level shifts can re-enter the
    # stream.
    rebase_after_rejections: int = 3
    rebase_continuity_max_abs_return: float = 0.10
    rebase_max_abs_return: float = 0.95
    # Optional separate source for volume matrices. Useful when the repaired
    # price store contains only close.parquet but the original cube store still
    # contains volume.parquet. If omitted, the feed also tries a conservative
    # auto-fallback for common names like 1m_cube_store_repaired_v2 -> 1m_cube_store.
    volume_store_dir: Optional[str] = None

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        root = Path(self.store_dir)
        volume_roots = self._volume_roots(root)
        date_dirs = sorted([p for p in root.glob('date=*') if p.is_dir()])
        if not date_dirs:
            raise FileNotFoundError(f'No date=* partitions under {root}')

        last_good: Dict[str, float] = {}
        seen_rows = 0
        rejected = 0
        rebased = 0
        pending_rebase: Dict[str, Tuple[float, int]] = {}

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

            vol_by_ts = None
            volume_path = self._resolve_volume_path(dd, volume_roots)
            if volume_path is not None and volume_path.exists():
                vol_df = pd.read_parquet(volume_path)
                vol_df['ts'] = pd.to_datetime(vol_df['ts'])
                vol_df = vol_df.loc[(vol_df['ts'] >= self.start) & (vol_df['ts'] <= self.end)]
                vol_cols = [c for c in cols if c in vol_df.columns]
                if vol_cols:
                    vol_by_ts = vol_df[['ts'] + vol_cols].set_index('ts')

            for _, row in df.iterrows():
                seen_rows += 1
                ts_pd = row['ts']
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
                    px = float(v)
                    if not (px > self.min_price):
                        rejected += 1
                        continue

                    prev = last_good.get(sym)
                    if prev is not None:
                        abs_return = abs(px / prev - 1.0)
                        if abs_return > self.max_abs_return:
                            if self._should_rebase_symbol(sym, px, prev, pending_rebase):
                                last_good[sym] = px
                                prices[sym] = px
                                if vol_row is not None and sym in vol_by_ts.columns:
                                    vv = vol_row[sym]
                                    if not pd.isna(vv):
                                        volumes[sym] = float(vv)
                                pending_rebase.pop(sym, None)
                                rebased += 1
                            else:
                                rejected += 1
                            continue

                    pending_rebase.pop(sym, None)
                    last_good[sym] = px
                    prices[sym] = px
                    if vol_row is not None and sym in vol_by_ts.columns:
                        vv = vol_row[sym]
                        if not pd.isna(vv):
                            volumes[sym] = float(vv)

                yield MarketSnapshot(ts=ts, prices=prices, volumes=volumes)

                if self.stats_every_rows and seen_rows % int(self.stats_every_rows) == 0:
                    print(
                        '[SanitizedMatrixStoreMinuteFeed] '
                        f'rows={seen_rows} rejected_quotes={rejected} rebased_symbols={rebased}'
                    )

                if self.speed == 'realtime':
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(0)

    def _should_rebase_symbol(
        self,
        sym: str,
        px: float,
        previous_good: float,
        pending_rebase: Dict[str, Tuple[float, int]],
    ) -> bool:
        """Return True when a rejected jump has become a stable new price regime.

        Without this, a persistent level shift makes the feed compare every future
        quote against a stale old price, so the symbol disappears permanently.
        We only rebase after multiple consecutive rejected quotes that are close
        to each other and not absurdly far from the old level.
        """
        required = int(self.rebase_after_rejections)
        if required <= 0:
            return False
        if previous_good <= 0:
            return False

        jump_from_old = abs(px / previous_good - 1.0)
        if jump_from_old > float(self.rebase_max_abs_return):
            pending_rebase.pop(sym, None)
            return False

        last_candidate, count = pending_rebase.get(sym, (float('nan'), 0))
        if count > 0 and last_candidate > 0:
            candidate_move = abs(px / last_candidate - 1.0)
            if candidate_move <= float(self.rebase_continuity_max_abs_return):
                count += 1
            else:
                count = 1
        else:
            count = 1

        pending_rebase[sym] = (px, count)
        return count >= required


# Volume fallback helpers are attached below the class body for readability.
def _sanitized_volume_roots(self, price_root: Path) -> List[Path]:
    roots: List[Path] = []
    if self.volume_store_dir:
        roots.append(Path(self.volume_store_dir))
    name = price_root.name
    if "_repaired" in name:
        roots.append(price_root.with_name(name.split("_repaired", 1)[0]))

    out: List[Path] = []
    seen = set()
    for r in roots:
        key = str(r)
        if key not in seen and r != price_root:
            out.append(r)
            seen.add(key)
    return out


def _sanitized_resolve_volume_path(self, price_day_dir: Path, volume_roots: List[Path]) -> Optional[Path]:
    local = price_day_dir / "volume.parquet"
    if local.exists():
        return local
    for root in volume_roots:
        candidate = root / price_day_dir.name / "volume.parquet"
        if candidate.exists():
            return candidate
    return None


SanitizedMatrixStoreMinuteFeed._volume_roots = _sanitized_volume_roots
SanitizedMatrixStoreMinuteFeed._resolve_volume_path = _sanitized_resolve_volume_path
