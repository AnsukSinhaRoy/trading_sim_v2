from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Literal

import pandas as pd

from runner.config import Config, parse_dt
from runner.logging_utils import setup_logging

log = logging.getLogger("levitate")

# ---------------- helpers ----------------

def _parse_hhmmss(s: str) -> time:
    hh, mm, ss = [int(x) for x in s.split(":")]
    return time(hh, mm, ss)

def _discover_files(raw_dir: Path, glob_pat: str, recursive: bool) -> List[Path]:
    return sorted(raw_dir.rglob(glob_pat) if recursive else raw_dir.glob(glob_pat))

def _extract_symbol(fname: str, symbol_regex: str) -> Optional[str]:
    m = re.match(symbol_regex, fname)
    return m.group("symbol") if m else None

def _load_universe(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"universe_file must be a JSON list of symbols: {path}")
    return [str(x) for x in data]

def _fields_to_build(cfg: Config) -> List[str]:
    fields = cfg.get("preprocess", "fields", default=["close"])
    if isinstance(fields, str):
        fields = [fields]
    fields = [str(x).lower() for x in fields]
    for f in fields:
        if f not in ("open", "high", "low", "close", "volume"):
            raise ValueError(f"Unsupported field in preprocess.fields: {f}")
    return fields

def _field_cols(cfg: Config) -> Dict[str, str]:
    # Canonical -> raw column mapping
    return {
        "open":   cfg.get("preprocess", "ohlcv_cols", "open",   default="open"),
        "high":   cfg.get("preprocess", "ohlcv_cols", "high",   default="high"),
        "low":    cfg.get("preprocess", "ohlcv_cols", "low",    default="low"),
        "close":  cfg.get("preprocess", "ohlcv_cols", "close",  default="close"),
        "volume": cfg.get("preprocess", "ohlcv_cols", "volume", default="volume"),
    }

def _iter_csv_chunks(path: Path, usecols: List[str], date_col: str, chunksize: int) -> Iterable[pd.DataFrame]:
    it = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    for chunk in it:
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        yield chunk

def _apply_filters(
    df: pd.DataFrame,
    date_col: str,
    start: Optional[datetime],
    end: Optional[datetime],
    market_start: time,
    market_end: time,
) -> pd.DataFrame:
    df = df.dropna(subset=[date_col]).copy()
    ts = df[date_col]
    if start is not None:
        df = df.loc[ts >= start]
        ts = df[date_col]
    if end is not None:
        df = df.loc[ts <= end]
        ts = df[date_col]
    if df.empty:
        return df
    t = ts.dt.time
    return df.loc[(t >= market_start) & (t <= market_end)]

# ---------------- Stage A: long store (optional intermediate) ----------------

def build_long_store(cfg: Config, run_dir: Path) -> Path:
    """Convert raw CSVs to a long parquet dataset partitioned by date + symbol.

    Output layout:
      <out_dir>/<dataset_name>/1m_long_store/date=YYYY-MM-DD/symbol=XYZ/*.parquet

    Columns: ts, symbol, <fields...>
    """
    raw_dir = Path(cfg.get("preprocess", "raw_dir"))
    out_dir = Path(cfg.get("preprocess", "out_dir", default="processed_data"))
    dataset = cfg.get("preprocess", "dataset_name", default="dataset")
    long_store = out_dir / dataset / "1m_long_store"
    long_store.mkdir(parents=True, exist_ok=True)

    glob_pat = cfg.get("preprocess", "glob", default="*.csv")
    recursive = bool(cfg.get("preprocess", "recursive", default=True))
    symbol_regex = cfg.get("preprocess", "symbol_regex", default=r"^(?P<symbol>.+)_minute\.csv$")
    date_col = cfg.get("preprocess", "timestamp_col", default="date")
    chunksize = int(cfg.get("preprocess", "chunksize", default=750_000))

    market_start = _parse_hhmmss(cfg.get("preprocess", "market_start", default="09:15:00"))
    market_end = _parse_hhmmss(cfg.get("preprocess", "market_end", default="15:30:00"))

    start_dt = cfg.get("preprocess", "start", default=None)
    end_dt = cfg.get("preprocess", "end", default=None)
    start = parse_dt(start_dt) if start_dt else None
    end = parse_dt(end_dt) if end_dt else None

    fields = _fields_to_build(cfg)
    cols = _field_cols(cfg)
    strict = bool(cfg.get("preprocess", "strict_fields", default=True))

    uni = _load_universe(cfg.get("preprocess", "universe_file", default=None))
    uni_set = set(uni) if uni else None

    files = _discover_files(raw_dir, glob_pat, recursive)
    log.info("Preprocess(long): discovered %d files under %s", len(files), raw_dir)

    failures: Dict[str, str] = {}
    n_symbols = 0
    n_parts = 0

    raw_field_cols = [cols[f] for f in fields]
    usecols = [date_col] + raw_field_cols

    for f in files:
        sym = _extract_symbol(f.name, symbol_regex)
        if not sym:
            continue
        if uni_set and sym not in uni_set:
            continue

        n_symbols += 1
        log.info("Preprocess(long): symbol=%s file=%s", sym, f.name)

        try:
            for chunk in _iter_csv_chunks(f, usecols=usecols, date_col=date_col, chunksize=chunksize):
                missing = [c for c in usecols if c not in chunk.columns]
                if missing:
                    msg = f"missing columns {missing} in {f.name}"
                    if strict:
                        raise ValueError(msg)
                    log.warning("Preprocess(long): %s (skipping chunk)", msg)
                    continue

                chunk = _apply_filters(chunk, date_col, start, end, market_start, market_end)
                if chunk.empty:
                    continue

                out = pd.DataFrame({"ts": pd.to_datetime(chunk[date_col]), "symbol": sym})

                for fld in fields:
                    rawc = cols[fld]
                    if fld == "volume":
                        out[fld] = pd.to_numeric(chunk[rawc], errors="coerce").fillna(0.0)
                    else:
                        out[fld] = pd.to_numeric(chunk[rawc], errors="coerce")

                out = out.dropna(subset=["ts"])
                price_fields = [f for f in fields if f != "volume"]
                if price_fields:
                    out = out.dropna(subset=price_fields, how="all")
                if out.empty:
                    continue

                out["date"] = out["ts"].dt.date.astype(str)

                out.to_parquet(
                    long_store,
                    engine="pyarrow",
                    partition_cols=["date", "symbol"],
                    index=False,
                )
                n_parts += 1

        except Exception as e:
            failures[sym] = str(e)
            log.exception("Preprocess(long): FAILED symbol=%s (%s)", sym, e)

    meta = {
        "raw_dir": str(raw_dir),
        "long_store_dir": str(long_store),
        "files_discovered": len(files),
        "symbols_attempted": n_symbols,
        "parts_written": n_parts,
        "fields": fields,
        "ohlcv_cols": cols,
        "config": cfg.raw.get("preprocess", {}),
        "failures": failures,
    }
    (run_dir / "preprocess_long_store_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Preprocess(long): complete -> %s", long_store)
    return long_store

# ---------------- Stage B: cube store (daily matrices per field) ----------------

FillMode = Literal["ffill", "from_close", "zero", "none"]

def _fill_wide(wide: pd.DataFrame, close_ref: Optional[pd.DataFrame], mode: FillMode) -> pd.DataFrame:
    if mode == "none":
        return wide
    if mode == "ffill":
        return wide.ffill()
    if mode == "zero":
        return wide.fillna(0.0)
    if mode == "from_close":
        wide2 = wide.ffill()
        if close_ref is not None:
            wide2 = wide2.fillna(close_ref)
        return wide2
    raise ValueError(f"Unknown fill mode: {mode}")

def build_cube_store(cfg: Config, run_dir: Path, long_store_dir: Path) -> Path:
    """Build daily OHLCV matrices (cube store).

    Output layout:
      <out_dir>/<dataset_name>/1m_cube_store/date=YYYY-MM-DD/<field>.parquet

    Each file is a 2D matrix: rows=minutes, cols=symbols (+ a leading ts column).
    """
    out_dir = Path(cfg.get("preprocess", "out_dir", default="processed_data"))
    dataset = cfg.get("preprocess", "dataset_name", default="dataset")
    cube_store = out_dir / dataset / "1m_cube_store"
    cube_store.mkdir(parents=True, exist_ok=True)

    long_store_dir = Path(long_store_dir)

    market_start = _parse_hhmmss(cfg.get("preprocess", "market_start", default="09:15:00"))
    market_end = _parse_hhmmss(cfg.get("preprocess", "market_end", default="15:30:00"))
    freq = cfg.get("preprocess", "freq", default="1min")

    fields = _fields_to_build(cfg)
    fields_internal = fields[:] if "close" in fields else fields + ["close"]

    fill = {
        "close":  cfg.get("preprocess", "fill_rules", "close",  default="ffill"),
        "open":   cfg.get("preprocess", "fill_rules", "open",   default="from_close"),
        "high":   cfg.get("preprocess", "fill_rules", "high",   default="from_close"),
        "low":    cfg.get("preprocess", "fill_rules", "low",    default="from_close"),
        "volume": cfg.get("preprocess", "fill_rules", "volume", default="zero"),
    }

    uni = _load_universe(cfg.get("preprocess", "universe_file", default=None))
    uni_set = set(uni) if uni else None

    date_parts = sorted([p for p in long_store_dir.glob("date=*") if p.is_dir()])
    if not date_parts:
        raise FileNotFoundError(f"No date partitions found under {long_store_dir}")

    start_dt = cfg.get("preprocess", "start", default=None)
    end_dt = cfg.get("preprocess", "end", default=None)
    start_date = parse_dt(start_dt).date() if start_dt else None
    end_date = parse_dt(end_dt).date() if end_dt else None

    progress_every_days = int(cfg.get("preprocess", "progress_every_days", default=10))
    written_days = 0
    failures: Dict[str, str] = {}

    for dp in date_parts:
        date_str = dp.name.split("=", 1)[1]
        d = datetime.fromisoformat(date_str).date()
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue

        rows = []
        for sp in dp.glob("symbol=*"):
            sym = sp.name.split("=", 1)[1]
            if uni_set and sym not in uni_set:
                continue
            for part in sp.glob("*.parquet"):
                try:
                    df = pd.read_parquet(part)  
                    if "symbol" not in df.columns:
                        df["symbol"] = sym
# contains ts, symbol, <fields...>
                    rows.append(df)
                except Exception:
                    continue

        if not rows:
            continue

        day = pd.concat(rows, ignore_index=True)
        day["ts"] = pd.to_datetime(day["ts"])

        start_ts = datetime.combine(d, market_start)
        end_ts = datetime.combine(d, market_end)
        cal = pd.date_range(start=start_ts, end=end_ts, freq=freq)

        close_ref = None
        if "close" in fields_internal and "close" in day.columns:
            close_w = day.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")
            close_w = close_w.reindex(cal).sort_index()
            close_ref = _fill_wide(close_w, None, fill.get("close", "ffill"))

        out_day_dir = cube_store / f"date={date_str}"
        out_day_dir.mkdir(parents=True, exist_ok=True)

        for fld in fields:
            if fld not in day.columns:
                failures[f"{date_str}:{fld}"] = f"missing field column '{fld}' in long store"
                continue

            w = day.pivot_table(index="ts", columns="symbol", values=fld, aggfunc="last")
            w = w.reindex(cal).sort_index()

            w = _fill_wide(w, close_ref, fill.get(fld, "none"))

            if fld == "volume":
                w = w.fillna(0.0).astype("float32")
            else:
                w = w.astype("float32")

            if uni:
                cols = [s for s in uni if s in w.columns]
                rest = sorted([c for c in w.columns if c not in set(cols)])
                w = w[cols + rest]

            w.reset_index(names="ts").to_parquet(out_day_dir / f"{fld}.parquet", index=False)

        written_days += 1
        if progress_every_days and (written_days % progress_every_days == 0):
            log.info("Preprocess(cube): wrote %d days (latest=%s)", written_days, date_str)

    manifest = {
        "cube_store_dir": str(cube_store),
        "long_store_dir": str(long_store_dir),
        "days_written": written_days,
        "fields": fields,
        "fill_rules": fill,
        "market_hours": {"start": str(market_start), "end": str(market_end)},
        "freq": freq,
        "failures": failures,
        "config": cfg.raw.get("preprocess", {}),
    }
    (run_dir / "preprocess_cube_store_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Preprocess(cube): complete -> %s (days=%d)", cube_store, written_days)
    return cube_store

# ---------------- Orchestrator / CLI ----------------

def run_preprocess(config_path: str) -> None:
    cfg = Config.load(config_path)

    base_out = Path(cfg.get("preprocess", "out_dir", default="processed_data"))
    run_id = f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = base_out / "preprocess_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir, level=str(cfg.get("preprocess", "log_level", default="INFO")))
    log.info("Preprocess run created: %s", run_dir)
    (run_dir / "effective_config.json").write_text(json.dumps(cfg.raw, indent=2), encoding="utf-8")

    mode = str(cfg.get("preprocess", "mode", default="cube")).lower()

    long_dir = cfg.get("preprocess", "long_store_dir", default=None)
    if mode in ("long", "cube", "both"):
        if not long_dir:
            long_dir = str(build_long_store(cfg, run_dir))
        else:
            long_dir = str(long_dir)

    if mode in ("cube", "both"):
        build_cube_store(cfg, run_dir, long_store_dir=Path(long_dir))

    cleanup = bool(cfg.get("preprocess", "cleanup_long_store", default=False))
    if cleanup and long_dir:
        try:
            import shutil
            shutil.rmtree(long_dir)
            log.info("Preprocess: cleaned up long store: %s", long_dir)
        except Exception:
            log.warning("Preprocess: failed to cleanup long store: %s", long_dir)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_preprocess(args.config)

if __name__ == "__main__":
    main()
