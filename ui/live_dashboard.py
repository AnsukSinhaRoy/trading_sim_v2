from __future__ import annotations

# Make top-level packages importable when running `streamlit run ui/live_dashboard.py`
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path as _Path

import pandas as pd
import plotly.express as px
import streamlit as st

from common.eventlog import EventLogger

def _tables(rows):
    nav, fills, open_tr, close_tr = [], [], [], []
    for r in rows:
        if "ts" in r and isinstance(r["ts"], str):
            r["ts"] = datetime.fromisoformat(r["ts"])
        k = r.get("kind")
        if k == "position_snapshot":
            nav.append({"ts": r["ts"], "nav": float(r["nav"]), "cash": float(r["cash"])})
        elif k == "fill":
            fills.append({kk: r[kk] for kk in ["ts","symbol","side","qty","price","ref_price","fees","order_id"]})
        elif k == "trade_open":
            open_tr.append({kk: r[kk] for kk in ["ts","trade_id","symbol","side","qty","entry_price"]})
        elif k == "trade_close":
            close_tr.append({kk: r[kk] for kk in ["ts","trade_id","symbol","side","qty","entry_price","exit_price","pnl"]})

    nav_df = pd.DataFrame(nav).drop_duplicates(subset=["ts"]).sort_values("ts") if nav else pd.DataFrame(columns=["ts","nav","cash"])
    fills_df = pd.DataFrame(fills).sort_values("ts") if fills else pd.DataFrame(columns=["ts","symbol","side","qty","price","ref_price","fees","order_id"])
    open_df = pd.DataFrame(open_tr).sort_values("ts") if open_tr else pd.DataFrame(columns=["ts","trade_id","symbol","side","qty","entry_price"])
    close_df = pd.DataFrame(close_tr).sort_values("ts") if close_tr else pd.DataFrame(columns=["ts","trade_id","symbol","side","qty","entry_price","exit_price","pnl"])
    return nav_df, fills_df, open_df, close_df

def _rerun():
    # Streamlit compat across versions
    if hasattr(st, "rerun"):
        st.rerun()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args, _ = ap.parse_known_args()

    run_dir = _Path(args.run)
    events = run_dir / "events.jsonl"

    st.set_page_config(page_title="Live NAV Dashboard", layout="wide")
    st.title("Live NAV Dashboard (event-log driven)")

    if not events.exists():
        st.error(f"Missing: {events}")
        st.stop()

    auto = st.checkbox("Auto-refresh (2s)", value=True)
    if auto:
        import time
        time.sleep(2)
        try:
            _rerun()
        except Exception:
            st.warning("Auto-refresh not supported; please refresh manually.")

    rows = EventLogger.read(events)
    nav, fills, open_tr, close_tr = _tables(rows)

    if len(nav):
        last = nav.iloc[-1]
        c1,c2,c3 = st.columns(3)
        c1.metric("NAV", f"{last['nav']:.2f}")
        c2.metric("Cash", f"{last['cash']:.2f}")
        c3.metric("Closed PnL", f"{(close_tr['pnl'].sum() if len(close_tr) else 0.0):.2f}")

    st.subheader("NAV")
    if len(nav) >= 2:
        st.plotly_chart(px.line(nav, x="ts", y="nav"), use_container_width=True)
    else:
        st.write("Waiting for data...")

    a,b = st.columns(2)
    with a:
        st.subheader("Recent fills")
        st.dataframe(fills.tail(200), use_container_width=True)
    with b:
        st.subheader("Closed trades")
        st.dataframe(close_tr.tail(200), use_container_width=True)

    st.subheader("Open trades (toy tracker)")
    st.dataframe(open_tr.tail(200), use_container_width=True)

if __name__ == "__main__":
    main()
