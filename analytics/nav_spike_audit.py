from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from common.eventlog import EventLogger


@dataclass
class SpikeAuditResult:
    spikes: pd.DataFrame
    contributions: pd.DataFrame


class MatrixStorePriceReader:
    """Small helper that lazily loads daily matrix-store partitions and serves as-of prices.

    The matrix store is expected to look like:
        <store_dir>/date=YYYY-MM-DD/close.parquet

    We cache partitions by date because a spike audit usually revisits the same few days
    many times while explaining different NAV jumps.
    """

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load_day(self, day: datetime) -> pd.DataFrame:
        key = day.strftime('%Y-%m-%d')
        if key in self._cache:
            return self._cache[key]

        path = self.store_dir / f'date={key}' / 'close.parquet'
        if not path.exists():
            df = pd.DataFrame(columns=['ts'])
        else:
            df = pd.read_parquet(path)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts').reset_index(drop=True)
        self._cache[key] = df
        return df

    def _load_days_for_window(self, ts0: datetime, ts1: datetime) -> pd.DataFrame:
        start_day = (ts0 - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = ts1.replace(hour=0, minute=0, second=0, microsecond=0)
        days: List[pd.DataFrame] = []
        cur = start_day
        while cur <= end_day:
            days.append(self._load_day(cur))
            cur += timedelta(days=1)
        days = [d for d in days if not d.empty]
        if not days:
            return pd.DataFrame(columns=['ts'])
        return pd.concat(days, ignore_index=True).sort_values('ts').reset_index(drop=True)

    def asof_prices(self, ts: datetime, symbols: Iterable[str], lookback_days: int = 2) -> Dict[str, Optional[float]]:
        start = ts - timedelta(days=max(1, int(lookback_days)))
        frame = self._load_days_for_window(start, ts)
        out: Dict[str, Optional[float]] = {}
        if frame.empty:
            return {sym: None for sym in symbols}

        frame = frame.loc[frame['ts'] <= ts]
        for sym in symbols:
            if sym not in frame.columns:
                out[sym] = None
                continue
            s = frame[['ts', sym]].dropna()
            if s.empty:
                out[sym] = None
                continue
            out[sym] = float(s.iloc[-1][sym])
        return out


def _read_position_snapshots(run_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = EventLogger.read(Path(run_dir) / 'events.jsonl')
    nav_rows: List[dict] = []
    pos_rows: List[dict] = []
    for r in rows:
        if r.get('kind') != 'position_snapshot':
            continue
        ts = r.get('ts')
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        nav_rows.append({
            'ts': ts,
            'nav': float(r.get('nav', 0.0)),
            'cash': float(r.get('cash', 0.0)),
        })
        pos_rows.append({
            'ts': ts,
            'positions': dict(r.get('positions', {})),
        })

    nav_df = pd.DataFrame(nav_rows).drop_duplicates(subset=['ts']).sort_values('ts').reset_index(drop=True)
    pos_df = pd.DataFrame(pos_rows).drop_duplicates(subset=['ts']).sort_values('ts').reset_index(drop=True)
    return nav_df, pos_df


def _pct_change_safe(prev: float, cur: float) -> Optional[float]:
    if abs(prev) < 1e-12:
        return None
    return (cur / prev) - 1.0


def audit_nav_spikes(
    run_dir: str | Path,
    store_dir: str | Path,
    abs_nav_change: Optional[float] = None,
    pct_nav_change: float = 0.05,
    top_k_symbols: int = 10,
) -> SpikeAuditResult:
    """Explain large NAV jumps by attributing them to held symbols.

    The audit uses consecutive PositionSnapshot events from a finished run. For every large
    NAV move, it reconstructs *as-of* prices from the matrix store and estimates:

        contribution(symbol) ~= position_before_jump * (price_t - price_t_minus_1)

    This is not a perfect PnL decomposition when orders were also filled on the same minute,
    but it is usually enough to identify the symbols that caused the cliff or spike.
    """

    nav_df, pos_df = _read_position_snapshots(run_dir)
    if nav_df.empty:
        raise ValueError(f'No position_snapshot events found under {run_dir}')

    nav_df['prev_nav'] = nav_df['nav'].shift(1)
    nav_df['prev_ts'] = nav_df['ts'].shift(1)
    nav_df['nav_change'] = nav_df['nav'] - nav_df['prev_nav']
    nav_df['nav_change_pct'] = [
        _pct_change_safe(p, c) if pd.notna(p) else None
        for p, c in zip(nav_df['prev_nav'], nav_df['nav'])
    ]

    mask = nav_df['prev_ts'].notna()
    if abs_nav_change is not None:
        mask &= nav_df['nav_change'].abs() >= float(abs_nav_change)
    else:
        mask &= nav_df['nav_change_pct'].abs() >= float(pct_nav_change)

    spikes = nav_df.loc[mask, ['prev_ts', 'ts', 'prev_nav', 'nav', 'nav_change', 'nav_change_pct']].copy()
    spikes = spikes.rename(columns={'prev_ts': 'ts_prev', 'ts': 'ts_now'}).reset_index(drop=True)
    if spikes.empty:
        return SpikeAuditResult(spikes=spikes, contributions=pd.DataFrame())

    pos_lookup = pos_df.set_index('ts')['positions'].to_dict()
    price_reader = MatrixStorePriceReader(store_dir)

    contributions_rows: List[dict] = []
    enriched_rows: List[dict] = []

    for idx, row in spikes.iterrows():
        ts_prev = pd.Timestamp(row['ts_prev']).to_pydatetime()
        ts_now = pd.Timestamp(row['ts_now']).to_pydatetime()
        positions = dict(pos_lookup.get(pd.Timestamp(row['ts_prev']), {}))
        held = [sym for sym, qty in positions.items() if int(qty) != 0]

        prev_px = price_reader.asof_prices(ts_prev, held)
        now_px = price_reader.asof_prices(ts_now, held)

        spike_total_est = 0.0
        missing = 0
        rows_here: List[dict] = []
        for sym in held:
            qty = int(positions.get(sym, 0))
            p0 = prev_px.get(sym)
            p1 = now_px.get(sym)
            if p0 is None or p1 is None:
                missing += 1
                continue
            contrib = qty * (float(p1) - float(p0))
            spike_total_est += contrib
            rows_here.append({
                'spike_id': idx,
                'ts_prev': ts_prev,
                'ts_now': ts_now,
                'symbol': sym,
                'qty_before': qty,
                'price_prev': float(p0),
                'price_now': float(p1),
                'price_return': (float(p1) / float(p0) - 1.0) if abs(float(p0)) > 1e-12 else None,
                'estimated_nav_contribution': float(contrib),
                'abs_estimated_nav_contribution': abs(float(contrib)),
            })

        rows_here = sorted(rows_here, key=lambda x: x['abs_estimated_nav_contribution'], reverse=True)
        rows_here = rows_here[:max(1, int(top_k_symbols))]
        contributions_rows.extend(rows_here)
        enriched_rows.append({
            **row.to_dict(),
            'held_symbols_before': len(held),
            'symbols_with_missing_prices': missing,
            'estimated_explained_change': spike_total_est,
            'top_symbol': rows_here[0]['symbol'] if rows_here else None,
            'top_symbol_estimated_contribution': rows_here[0]['estimated_nav_contribution'] if rows_here else None,
        })

    spikes_out = pd.DataFrame(enriched_rows)
    contrib_out = pd.DataFrame(contributions_rows)
    if not contrib_out.empty:
        contrib_out = contrib_out.sort_values(['spike_id', 'abs_estimated_nav_contribution'], ascending=[True, False]).reset_index(drop=True)
    return SpikeAuditResult(spikes=spikes_out, contributions=contrib_out)


def save_nav_spike_audit(result: SpikeAuditResult, output_dir: str | Path) -> Tuple[Path, Optional[Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spikes_path = output_dir / 'nav_spikes.csv'
    result.spikes.to_csv(spikes_path, index=False)
    contrib_path: Optional[Path] = None
    if not result.contributions.empty:
        contrib_path = output_dir / 'nav_spike_contributions.csv'
        result.contributions.to_csv(contrib_path, index=False)
    return spikes_path, contrib_path
