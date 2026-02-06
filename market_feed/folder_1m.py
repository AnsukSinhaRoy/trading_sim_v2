from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Literal
import re
import json
import logging

import pandas as pd

from common.events import MarketSnapshot

Format = Literal["csv", "parquet"]
log = logging.getLogger("levitate")

def _read_symbol_frame(path: Path, fmt: Format, timestamp_col: str, price_col: str) -> pd.DataFrame:
    if fmt == "csv":
        df = pd.read_csv(path, usecols=[timestamp_col, price_col], parse_dates=[timestamp_col])
    elif fmt == "parquet":
        df = pd.read_parquet(path, columns=[timestamp_col, price_col])
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    df = df.dropna(subset=[timestamp_col, price_col]).sort_values(timestamp_col)
    df = df.set_index(timestamp_col).rename(columns={price_col: "price"})
    return df[["price"]]

def _build_calendar(start: datetime, end: datetime, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=freq)

def _pattern_to_glob(file_pattern: str, ext: str) -> str:
    return file_pattern.replace("{symbol}", "*").replace("{ext}", ext)

def _pattern_to_regex(file_pattern: str, ext: str) -> re.Pattern:
    escaped = re.escape(file_pattern)
    escaped = escaped.replace(re.escape("{symbol}"), r"(?P<symbol>.+)")
    escaped = escaped.replace(re.escape("{ext}"), re.escape(ext))
    return re.compile(r"^" + escaped + r"$")

def _discover_symbols(data_dir: Path, file_pattern: str, ext: str, recursive: bool = True) -> List[str]:
    glob_pat = _pattern_to_glob(file_pattern, ext)
    rx = _pattern_to_regex(file_pattern, ext)
    paths = list(data_dir.rglob(glob_pat)) if recursive else list(data_dir.glob(glob_pat))
    symbols: List[str] = []
    for p in paths:
        m = rx.match(p.name)
        if not m:
            continue
        sym = m.group("symbol")
        if sym and sym not in symbols:
            symbols.append(sym)
    symbols.sort()
    return symbols

def _load_universe_file(path: str) -> List[str]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"universe_file must be a JSON list of symbols: {path}")
    return [str(x) for x in data]

@dataclass
class FolderMinuteFeed:
    data_dir: str
    symbols: Optional[List[str]]
    start: datetime
    end: datetime
    fmt: Format = "csv"
    file_pattern: str = "{symbol}_minute.{ext}"
    timestamp_col: str = "date"
    price_col: str = "close"
    freq: str = "1min"
    fill: Literal["ffill", "none"] = "ffill"
    speed: Literal["fast", "realtime"] = "fast"

    discover_symbols: bool = True
    discover_recursive: bool = True

    universe_file: Optional[str] = None
    universe_mode: Literal["intersect", "order_only"] = "intersect"

    progress_every: int = 25  # log every N symbols loaded

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        data_dir = Path(self.data_dir)
        ext = "csv" if self.fmt == "csv" else "parquet"

        symbols = list(self.symbols) if self.symbols else []
        if (not symbols) and self.discover_symbols:
            symbols = _discover_symbols(data_dir, self.file_pattern, ext, recursive=self.discover_recursive)
            if not symbols:
                raise FileNotFoundError(f"No files found in {data_dir} matching pattern {self.file_pattern} (ext={ext})")
            log.info("MarketFeed(folder_1m): autodiscovered %d symbols", len(symbols))

        if self.universe_file:
            uni = _load_universe_file(self.universe_file)
            discovered_set = set(symbols)
            if self.universe_mode == "intersect":
                symbols = [s for s in uni if s in discovered_set]
            elif self.universe_mode == "order_only":
                ordered = [s for s in uni if s in discovered_set]
                remaining = sorted([s for s in symbols if s not in set(ordered)])
                symbols = ordered + remaining
            log.info("MarketFeed(folder_1m): universe applied mode=%s -> %d symbols", self.universe_mode, len(symbols))

        # Preload & align (this is the phase where you'll see a pause for huge universes)
        frames: Dict[str, pd.DataFrame] = {}
        total = len(symbols)
        for i, sym in enumerate(symbols, start=1):
            path = data_dir / self.file_pattern.format(symbol=sym, ext=ext)
            if not path.exists():
                continue
            df = _read_symbol_frame(path, self.fmt, self.timestamp_col, self.price_col)
            df = df.loc[(df.index >= self.start) & (df.index <= self.end)]
            frames[sym] = df
            if self.progress_every and (i % self.progress_every == 0 or i == total):
                log.info("MarketFeed(folder_1m): loaded %d/%d symbols", i, total)

        if not frames:
            raise FileNotFoundError("No symbol data loaded (check data_dir, pattern, and universe filtering).")

        cal = _build_calendar(self.start, self.end, self.freq)
        aligned: Dict[str, pd.Series] = {}
        for sym, df in frames.items():
            s = df["price"].reindex(cal)
            if self.fill == "ffill":
                s = s.ffill()
            aligned[sym] = s

        # Stream snapshots
        for ts in cal:
            prices = {}
            for sym, s in aligned.items():
                v = s.loc[ts]
                if pd.isna(v):
                    continue
                prices[sym] = float(v)

            yield MarketSnapshot(ts=ts.to_pydatetime(), prices=prices)

            if self.speed == "realtime":
                await asyncio.sleep(1.0)
            else:
                await asyncio.sleep(0)
