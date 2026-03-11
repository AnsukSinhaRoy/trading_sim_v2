from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import shutil

import numpy as np
import pandas as pd


CANONICAL_FACTORS: Tuple[float, ...] = (
    1.25, 1.3333333333, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0,
    1/1.25, 1/1.3333333333, 1/1.5, 1/2.0, 1/2.5, 1/3.0, 1/4.0, 1/5.0, 1/10.0,
)


@dataclass
class IntradayAnomaly:
    symbol: str
    date: str
    ts: str
    raw_prev_median: float
    raw_next_median: float
    ratio_next_over_prev: float
    snapped_ratio: float
    correction_applied_to_suffix: float
    valid_prev_points: int
    valid_next_points: int
    rule: str


@dataclass
class RepairSummary:
    output_store: str
    scanned_dates: int
    anomalies_found: int
    symbols_touched: int
    manifest_path: str
    anomalies_csv: str


class IntradayRepairer:
    """Repair persistent intraday level shifts in a matrix-store feed.

    Key idea:
    - walk the store date by date in chronological order
    - maintain a cumulative multiplier per symbol
    - when a split-like shift is detected intraday, multiply the *suffix* of the series
      by a correction factor so the path becomes continuous
    - carry that correction forward to all future days for the same symbol

    This is the right direction for a backtest/simulation that does NOT adjust position
    quantities at the split timestamp. We keep the synthetic price path continuous so a
    held position does not suffer a fake 50%/80% collapse purely from data encoding.
    """

    def __init__(
        self,
        input_store: str | Path,
        output_store: str | Path,
        symbols: Optional[Sequence[str]] = None,
        lookback_bars: int = 20,
        lookahead_bars: int = 20,
        min_valid_window: int = 8,
        min_jump_abs_return: float = 0.35,
        factor_tolerance: float = 0.12,
        copy_non_price_files: bool = True,
    ):
        self.input_store = Path(input_store)
        self.output_store = Path(output_store)
        self.symbols = list(symbols) if symbols is not None else None
        self.lookback_bars = int(lookback_bars)
        self.lookahead_bars = int(lookahead_bars)
        self.min_valid_window = int(min_valid_window)
        self.min_jump_abs_return = float(min_jump_abs_return)
        self.factor_tolerance = float(factor_tolerance)
        self.copy_non_price_files = bool(copy_non_price_files)
        self.cumulative_multiplier: Dict[str, float] = {}
        self.anomalies: List[IntradayAnomaly] = []

    def run(self) -> RepairSummary:
        if not self.input_store.exists():
            raise FileNotFoundError(f'Input store not found: {self.input_store}')
        if self.output_store.exists():
            shutil.rmtree(self.output_store)
        self.output_store.mkdir(parents=True, exist_ok=True)

        date_dirs = sorted([p for p in self.input_store.glob('date=*') if p.is_dir()])
        scanned_dates = 0

        for dd in date_dirs:
            scanned_dates += 1
            in_path = dd / 'close.parquet'
            if not in_path.exists():
                continue
            out_dir = self.output_store / dd.name
            out_dir.mkdir(parents=True, exist_ok=True)

            df = pd.read_parquet(in_path)
            if 'ts' not in df.columns:
                raise ValueError(f"Expected 'ts' column in {in_path}")
            df['ts'] = pd.to_datetime(df['ts'])

            cols = [c for c in df.columns if c != 'ts']
            if self.symbols is not None:
                cols = [c for c in cols if c in self.symbols]

            for sym in cols:
                mult = float(self.cumulative_multiplier.get(sym, 1.0))
                if mult != 1.0:
                    df[sym] = df[sym] * mult
                df[sym] = self._repair_symbol_series(df['ts'], df[sym], sym=sym, date_str=dd.name.split('=', 1)[1])

            df.to_parquet(out_dir / 'close.parquet', index=False)

        if self.copy_non_price_files:
            for child in self.input_store.iterdir():
                if child.name.startswith('date='):
                    continue
                target = self.output_store / child.name
                if child.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)

        manifest_path = self.output_store / 'intraday_repair_manifest.json'
        anomalies_csv = self.output_store / 'intraday_anomalies.csv'
        pd.DataFrame([asdict(a) for a in self.anomalies]).to_csv(anomalies_csv, index=False)
        manifest = {
            'input_store': str(self.input_store),
            'output_store': str(self.output_store),
            'parameters': {
                'lookback_bars': self.lookback_bars,
                'lookahead_bars': self.lookahead_bars,
                'min_valid_window': self.min_valid_window,
                'min_jump_abs_return': self.min_jump_abs_return,
                'factor_tolerance': self.factor_tolerance,
            },
            'cumulative_multiplier': self.cumulative_multiplier,
            'anomalies_found': len(self.anomalies),
            'anomaly_examples': [asdict(a) for a in self.anomalies[:50]],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

        return RepairSummary(
            output_store=str(self.output_store),
            scanned_dates=scanned_dates,
            anomalies_found=len(self.anomalies),
            symbols_touched=len({a.symbol for a in self.anomalies}),
            manifest_path=str(manifest_path),
            anomalies_csv=str(anomalies_csv),
        )

    def _repair_symbol_series(self, ts: pd.Series, values: pd.Series, sym: str, date_str: str) -> pd.Series:
        s = values.astype('float64').copy()
        if s.dropna().shape[0] < (self.min_valid_window * 2 + 1):
            return s

        i = self.lookback_bars
        n = len(s)
        while i < n - self.lookahead_bars:
            prev_valid = s.iloc[max(0, i - self.lookback_bars):i].dropna()
            next_valid = s.iloc[i:min(n, i + self.lookahead_bars)].dropna()
            if len(prev_valid) < self.min_valid_window or len(next_valid) < self.min_valid_window:
                i += 1
                continue

            prev_median = float(prev_valid.median())
            next_median = float(next_valid.median())
            if prev_median <= 0 or next_median <= 0:
                i += 1
                continue

            ratio = next_median / prev_median
            snapped = self._snap_factor(ratio)
            if snapped is None:
                i += 1
                continue

            first_next = next_valid.iloc[0]
            raw_move = abs(float(first_next) / prev_median - 1.0)
            if raw_move < self.min_jump_abs_return:
                i += 1
                continue

            correction = prev_median / next_median
            s.iloc[i:] = s.iloc[i:] * correction
            self.cumulative_multiplier[sym] = float(self.cumulative_multiplier.get(sym, 1.0) * correction)
            self.anomalies.append(IntradayAnomaly(
                symbol=sym,
                date=date_str,
                ts=pd.Timestamp(ts.iloc[i]).isoformat(),
                raw_prev_median=prev_median,
                raw_next_median=next_median,
                ratio_next_over_prev=ratio,
                snapped_ratio=snapped,
                correction_applied_to_suffix=correction,
                valid_prev_points=len(prev_valid),
                valid_next_points=len(next_valid),
                rule='persistent_intraday_regime_shift',
            ))
            i += self.lookahead_bars
        return s

    def _snap_factor(self, ratio: float) -> Optional[float]:
        best = None
        best_err = float('inf')
        for f in CANONICAL_FACTORS:
            err = abs(ratio - f) / max(abs(f), 1e-12)
            if err < best_err:
                best = f
                best_err = err
        if best is None or best_err > self.factor_tolerance:
            return None
        return float(best)


def repair_intraday_anomalies(
    input_store: str | Path,
    output_store: str | Path,
    symbols: Optional[Sequence[str]] = None,
    **kwargs,
) -> RepairSummary:
    repairer = IntradayRepairer(input_store=input_store, output_store=output_store, symbols=symbols, **kwargs)
    return repairer.run()
