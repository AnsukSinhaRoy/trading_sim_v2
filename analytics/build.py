from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
from common.eventlog import EventLogger

def _save(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)

def build_derived_from_events(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    rows = EventLogger.read(run_dir / "events.jsonl")

    nav_rows, fills_rows, pos_rows, open_rows, close_rows = [], [], [], [], []
    for r in rows:
        if "ts" in r and isinstance(r["ts"], str):
            r["ts"] = datetime.fromisoformat(r["ts"])
        kind = r.get("kind")
        if kind == "position_snapshot":
            nav_rows.append({"ts": r["ts"], "nav": float(r["nav"]), "cash": float(r["cash"])})
            pos_rows.append({"ts": r["ts"], "positions": r.get("positions", {})})
        elif kind == "fill":
            fills_rows.append({k: r[k] for k in ["ts","order_id","symbol","side","qty","price","ref_price","fees"]})
        elif kind == "trade_open":
            open_rows.append({k: r[k] for k in ["ts","trade_id","symbol","side","qty","entry_price"]})
        elif kind == "trade_close":
            close_rows.append({k: r[k] for k in ["ts","trade_id","symbol","side","qty","entry_price","exit_price","pnl"]})

    derived = run_dir / "derived"
    derived.mkdir(exist_ok=True)

    if nav_rows:
        nav_df = pd.DataFrame(nav_rows).drop_duplicates(subset=["ts"]).sort_values("ts")
        _save(nav_df, derived / "nav.parquet")
    if fills_rows:
        _save(pd.DataFrame(fills_rows).sort_values("ts"), derived / "fills.parquet")
    if pos_rows:
        pos_df = pd.DataFrame(pos_rows).drop_duplicates(subset=["ts"]).sort_values("ts")
        expanded = pos_df["positions"].apply(lambda d: pd.Series(d, dtype="int64")).fillna(0).astype("int64")
        expanded.insert(0, "ts", pos_df["ts"].values)
        _save(expanded, derived / "positions.parquet")
    if open_rows:
        _save(pd.DataFrame(open_rows).sort_values("ts"), derived / "trades_open.parquet")
    if close_rows:
        _save(pd.DataFrame(close_rows).sort_values("ts"), derived / "trades_closed.parquet")
