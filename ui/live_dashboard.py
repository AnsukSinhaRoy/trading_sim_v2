from __future__ import annotations

# Make top-level packages importable when running `streamlit run ui/live_dashboard.py`
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path as _Path
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from common.eventlog import EventLogger

def _load_trades(events_path):
    # We still read the event log for trades, but it is now much smaller (no market data)
    rows = EventLogger.read(events_path)
    fills, open_tr, close_tr = [], [], []
    for r in rows:
        k = r.get("kind")
        if k == "fill":
            r["ts"] = datetime.fromisoformat(r["ts"])
            fills.append({kk: r[kk] for kk in ["ts","symbol","side","qty","price","ref_price","fees","order_id"]})
        elif k == "trade_open":
            r["ts"] = datetime.fromisoformat(r["ts"])
            open_tr.append({kk: r[kk] for kk in ["ts","trade_id","symbol","side","qty","entry_price"]})
        elif k == "trade_close":
            r["ts"] = datetime.fromisoformat(r["ts"])
            close_tr.append({kk: r[kk] for kk in ["ts","trade_id","symbol","side","qty","entry_price","exit_price","pnl"]})

    fills_df = pd.DataFrame(fills).sort_values("ts") if fills else pd.DataFrame(columns=["ts","symbol","side","qty","price","ref_price","fees","order_id"])
    open_df = pd.DataFrame(open_tr).sort_values("ts") if open_tr else pd.DataFrame(columns=["ts","trade_id","symbol","side","qty","entry_price"])
    close_df = pd.DataFrame(close_tr).sort_values("ts") if close_tr else pd.DataFrame(columns=["ts","trade_id","symbol","side","qty","entry_price","exit_price","pnl"])
    return fills_df, open_df, close_df

def _rerun():
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
    nav_csv = run_dir / "nav.csv"

    st.set_page_config(page_title="Live NAV Dashboard", layout="wide")
    st.title("Live NAV Dashboard (Optimized)")

    if not events.exists():
        st.error(f"Waiting for start... {events} not found.")
        time.sleep(2)
        _rerun()

    auto = st.checkbox("Auto-refresh (2s)", value=True)
    if auto:
        time.sleep(2)
        try:
            _rerun()
        except Exception:
            pass

    # OPTIMIZATION: Read NAV from the dedicated CSV (Fast)
    if nav_csv.exists():
        try:
            nav_df = pd.read_csv(nav_csv)
            nav_df["ts"] = pd.to_datetime(nav_df["ts"])
        except Exception:
            nav_df = pd.DataFrame(columns=["ts", "nav", "cash"])
    else:
        nav_df = pd.DataFrame(columns=["ts", "nav", "cash"])

    # Load trades from event log (now smaller, so manageable)
    fills, open_tr, close_tr = _load_trades(events)

    if len(nav_df):
        last = nav_df.iloc[-1]
        c1,c2,c3 = st.columns(3)
        c1.metric("NAV", f"{last['nav']:.2f}")
        c2.metric("Cash", f"{last['cash']:.2f}")
        c3.metric("Closed PnL", f"{(close_tr['pnl'].sum() if len(close_tr) else 0.0):.2f}")

    st.subheader("NAV")
    if len(nav_df) >= 2:
        # Downsample for chart speed if huge
        chart_data = nav_df if len(nav_df) < 5000 else nav_df.iloc[::5]
        # IMPORTANT: treat timestamps as categorical to avoid showing non-trading gaps (weekends/holidays)
        chart_data = chart_data.copy()
        chart_data["ts_str"] = chart_data["ts"].dt.strftime("%Y-%m-%d %H:%M")
        fig = px.line(chart_data, x="ts_str", y="nav")
        fig.update_xaxes(type="category", title_text="Time")
        fig.update_yaxes(title_text="NAV")
        st.plotly_chart(fig, use_container_width=True)
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