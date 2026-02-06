import sys
import argparse
import zmq
import json
from collections import deque
from typing import Dict, Any, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QSplitter
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
import pyqtgraph as pg
from datetime import datetime


# --- Background Listener Thread ---
class ZmqListener(QThread):
    """Receives ZMQ messages on a background thread.

    Design goals:
    - Never block/freeze the Qt GUI thread.
    - Avoid backlog in FAST backtests by draining messages quickly and keeping only the latest NAV.
    """
    data_signal = pyqtSignal(str, dict)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        ctx = zmq.Context.instance()

        # Use ONE SUB socket subscribed to both topics.
        # This avoids platform-specific quirks with CONFLATE and guarantees we always
        # receive NAV if we receive fills (same transport, same socket).
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 10000)
        sock.connect(self.url)
        sock.setsockopt_string(zmq.SUBSCRIBE, "nav")
        sock.setsockopt_string(zmq.SUBSCRIBE, "fill")

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        def _safe_emit(topic_b: bytes, msg_b: bytes):
            try:
                data = json.loads(msg_b.decode("utf-8"))
                self.data_signal.emit(topic_b.decode("utf-8"), data)
            except Exception as e:
                # Keep listener alive.
                print(f"ZMQ decode error: {e}")

        try:
            while self._running:
                events = dict(poller.poll(100))  # 100ms tick
                if sock not in events:
                    continue

                # Drain everything available quickly.
                # For NAV: GUI keeps only the latest message.
                # For FILL: GUI buffers and displays all fills.
                while True:
                    try:
                        topic, msg = sock.recv_multipart(flags=zmq.NOBLOCK)
                        _safe_emit(topic, msg)
                    except zmq.Again:
                        break
        finally:
            try:
                sock.close(0)
            except Exception:
                pass


# --- Main Dashboard Window ---
class RealTimeDashboard(QMainWindow):
    def __init__(self, url: str):
        super().__init__()
        self.url = url

        self.setWindowTitle("Levitate Real-Time Monitor")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc;")

        # Data buffers
        # Keep the latest NAV message and a bounded time-series for plotting.
        self.nav_data: List[float] = []
        self.nav_ts: List[float] = []  # epoch seconds for DateAxisItem
        self._latest_nav = None
        self._fills_buffer = deque()
        self._recent_fills = deque(maxlen=50)  # for the Overview "moving rows" effect

        # Trade blotter state (derived from fills)
        # We assume long-only behavior (engine rejects sells beyond holdings), so
        # a "trade" is defined as: first BUY when position=0 -> final SELL when position returns to 0.
        self._pos_from_fills: Dict[str, int] = {}
        self._open_trade_by_symbol: Dict[str, dict] = {}

        # Persist closed trades for the Trade Inspector. Row index == trade index.
        self._trades: List[dict] = []

        # Per-symbol running PnL state (from fills + latest marks).
        # For each symbol: {qty, avg_cost, realized}
        self._pnl_state: Dict[str, dict] = {}
        self._latest_marks: Dict[str, float] = {}
        self._latest_positions: Dict[str, int] = {}

        self._initial_nav = None

        self.setup_ui()

        # Start listener thread
        self.listener = ZmqListener(self.url)
        self.listener.data_signal.connect(self.handle_update)
        self.listener.start()

        # UI flush timer (smooth updates even with bursty data)
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(100)  # 10 FPS UI updates
        self.ui_timer.timeout.connect(self.flush_ui)
        self.ui_timer.start()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 1. Header Stats
        stats_layout = QHBoxLayout()
        self.lbl_nav = self.create_stat_label("NAV: -")
        self.lbl_cash = self.create_stat_label("Cash: -")
        self.lbl_pnl = self.create_stat_label("PnL: -")
        self.lbl_ts = self.create_stat_label("TS: -")

        stats_layout.addWidget(self.lbl_nav)
        stats_layout.addWidget(self.lbl_cash)
        stats_layout.addWidget(self.lbl_pnl)
        stats_layout.addWidget(self.lbl_ts)
        layout.addLayout(stats_layout)

        # 2. Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #333; }"
            "QTabBar::tab { background: #2a2a2a; color: #dcdcdc; padding: 8px 14px; }"
            "QTabBar::tab:selected { background: #3a3a3a; }"
        )

        # --- Overview Tab (NAV chart) ---
        overview = QWidget()
        ov_layout = QVBoxLayout(overview)

        # Plot with a timestamp x-axis (DateAxisItem) instead of implicit "steps".
        date_axis = pg.DateAxisItem(orientation="bottom")
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": date_axis})
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.setTitle("Live NAV", color="#dcdcdc")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.nav_curve = self.plot_widget.plot(pen=pg.mkPen(color="#00d4ff", width=2))
        ov_layout.addWidget(self.plot_widget)

        # Keep a small "moving" fills table on Overview (looks great + helps debugging).
        self.ov_fills_table = self.create_table(["Time", "Symbol", "Side", "Qty", "Price", "Fees"])
        self.ov_fills_table.setMaximumHeight(240)
        ov_layout.addWidget(self.ov_fills_table)
        self.tabs.addTab(overview, "Overview")

        # --- Positions Tab ---
        positions = QWidget()
        pos_layout = QVBoxLayout(positions)
        self.pos_table = self.create_table(["Symbol", "Qty", "Value"])
        pos_layout.addWidget(self.pos_table)
        self.tabs.addTab(positions, "Positions")

        # --- Fills Tab ---
        fills = QWidget()
        fills_layout = QVBoxLayout(fills)
        self.fills_table = self.create_table(["Time", "Symbol", "Side", "Qty", "Price", "Fees"])
        fills_layout.addWidget(self.fills_table)
        self.tabs.addTab(fills, "Fills")

        # --- PnL Tab (per-symbol realized/unrealized) ---
        pnl = QWidget()
        pnl_layout = QVBoxLayout(pnl)
        self.pnl_table = self.create_table([
            "Symbol", "Qty", "Avg Cost", "Mark", "Unrealized", "Realized", "Total"
        ])
        pnl_layout.addWidget(self.pnl_table)
        self.tabs.addTab(pnl, "PnL")

        # --- Trades Tab (round-trip blotter) ---
        trades = QWidget()
        trades_layout = QVBoxLayout(trades)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Vertical)

        self.trades_table = self.create_table([
            "Entry Time", "Symbol", "Entry Qty", "Entry VWAP",
            "Exit Time", "Exit VWAP", "PnL", "Duration", "Max Pos"
        ])
        splitter.addWidget(self.trades_table)

        inspector = QWidget()
        insp_layout = QVBoxLayout(inspector)
        self.trade_inspector_lbl = QLabel("Trade Inspector: select a trade row")
        self.trade_inspector_lbl.setStyleSheet("font-size: 14px; padding: 6px; border: 1px solid #333;")
        insp_layout.addWidget(self.trade_inspector_lbl)
        self.trade_fills_table = self.create_table(["Time", "Side", "Qty", "Price", "Fees"])
        insp_layout.addWidget(self.trade_fills_table)
        splitter.addWidget(inspector)

        trades_layout.addWidget(splitter)
        self.tabs.addTab(trades, "Trades")

        # Hook selection -> inspector
        self.trades_table.itemSelectionChanged.connect(self._on_trade_selected)

        layout.addWidget(self.tabs, stretch=1)

    def create_stat_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; border: 1px solid #333;")
        return lbl

    def create_table(self, headers):
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet("QHeaderView::section { background-color: #333; color: white; }")
        return table

    def handle_update(self, topic, data):
        """Runs on GUI thread (queued signal). Keep it VERY light."""
        if topic == "nav":
            self._latest_nav = data
        elif topic == "fill":
            self._fills_buffer.append(data)

    def flush_ui(self):
        """Runs at a fixed UI rate to keep the GUI responsive."""
        if self._latest_nav:
            nav = float(self._latest_nav.get("nav", 0.0))
            cash = float(self._latest_nav.get("cash", 0.0))

            if self._initial_nav is None:
                self._initial_nav = nav

            pnl = (nav - self._initial_nav) if self._initial_nav is not None else 0.0

            self.lbl_nav.setText(f"NAV: {nav:,.2f}")
            self.lbl_cash.setText(f"Cash: {cash:,.2f}")
            self.lbl_pnl.setText(f"PnL: {pnl:,.2f}")

            ts = str(self._latest_nav.get("ts", ""))
            self.lbl_ts.setText(f"TS: {self._fmt_time(ts)}")

            # Plot NAV over time with an actual timestamp x-axis
            dt = self._safe_parse_iso(ts)
            x = dt.timestamp() if dt else (self.nav_ts[-1] + 1 if self.nav_ts else 0)
            self.nav_ts.append(float(x))
            self.nav_data.append(nav)
            if len(self.nav_data) > 2000:
                self.nav_data = self.nav_data[-2000:]
                self.nav_ts = self.nav_ts[-2000:]
            self.nav_curve.setData(self.nav_ts, self.nav_data)

            # Update Symbol / Qty / Value table from the same NAV packet
            positions = self._latest_nav.get("positions", {}) if isinstance(self._latest_nav, dict) else {}
            pos_values = self._latest_nav.get("pos_values", {}) if isinstance(self._latest_nav, dict) else {}
            self._latest_positions = {str(k): int(v) for k, v in (positions or {}).items()}
            self._latest_marks = {}
            for sym, qty in self._latest_positions.items():
                if qty:
                    try:
                        self._latest_marks[sym] = float((pos_values or {}).get(sym, 0.0)) / float(qty)
                    except Exception:
                        self._latest_marks[sym] = 0.0
            self._render_positions(positions, pos_values)
            self._render_pnl_table()

        # Render a limited number of fills per UI tick
        max_rows = 25
        for _ in range(max_rows):
            if not self._fills_buffer:
                break
            data = self._fills_buffer.popleft()

            # Update running per-symbol PnL + trade blotter from fills
            self._update_pnl_from_fill(data)
            self._update_trade_blotter_from_fill(data)

            # Also append to the Overview "Recent Fills" table
            self._recent_fills.append(data)

            ts = str(data.get("ts", ""))
            # If ts is ISO, show HH:MM:SS
            time_str = ts[11:19] if len(ts) >= 19 else ts

            row = self.fills_table.rowCount()
            self.fills_table.insertRow(row)
            self.fills_table.setItem(row, 0, QTableWidgetItem(time_str))
            self.fills_table.setItem(row, 1, QTableWidgetItem(str(data.get("symbol", ""))))
            self.fills_table.setItem(row, 2, QTableWidgetItem(str(data.get("side", ""))))
            self.fills_table.setItem(row, 3, QTableWidgetItem(str(data.get("qty", ""))))
            self.fills_table.setItem(row, 4, QTableWidgetItem(str(data.get("price", ""))))
            self.fills_table.setItem(row, 5, QTableWidgetItem(str(data.get("fees", ""))))
            self.fills_table.scrollToBottom()

        # Refresh Overview fills table from the bounded recent buffer
        self._render_recent_fills()

        # Keep table from growing without bound
        if self.fills_table.rowCount() > 1000:
            # delete oldest rows
            remove_count = self.fills_table.rowCount() - 1000
            for _ in range(remove_count):
                self.fills_table.removeRow(0)

        # Keep trades table bounded
        if hasattr(self, "trades_table") and self.trades_table.rowCount() > 2000:
            remove_count = self.trades_table.rowCount() - 2000
            for _ in range(remove_count):
                self.trades_table.removeRow(0)
            # Keep inspector backing list aligned
            for _ in range(min(remove_count, len(self._trades))):
                self._trades.pop(0)


    def _render_positions(self, positions: dict, pos_values: dict):
        # Rebuild the table from the latest snapshot (small universe => fast enough)
        self.pos_table.setRowCount(0)
        # Sort by value (descending)
        rows = []
        for sym, qty in (positions or {}).items():
            try:
                v = float((pos_values or {}).get(sym, 0.0))
            except Exception:
                v = 0.0
            rows.append((sym, int(qty), v))
        rows.sort(key=lambda x: x[2], reverse=True)

        for sym, qty, v in rows:
            r = self.pos_table.rowCount()
            self.pos_table.insertRow(r)
            self.pos_table.setItem(r, 0, QTableWidgetItem(str(sym)))
            self.pos_table.setItem(r, 1, QTableWidgetItem(str(qty)))
            self.pos_table.setItem(r, 2, QTableWidgetItem(f"{v:,.2f}"))

    def _render_recent_fills(self) -> None:
        """Render the small "moving rows" fills table on the Overview tab."""
        if not hasattr(self, "ov_fills_table"):
            return
        self.ov_fills_table.setRowCount(0)
        # Show newest at the bottom.
        for f in list(self._recent_fills)[-50:]:
            ts = str(f.get("ts", ""))
            time_str = ts[11:19] if len(ts) >= 19 else ts
            r = self.ov_fills_table.rowCount()
            self.ov_fills_table.insertRow(r)
            self.ov_fills_table.setItem(r, 0, QTableWidgetItem(time_str))
            self.ov_fills_table.setItem(r, 1, QTableWidgetItem(str(f.get("symbol", ""))))
            self.ov_fills_table.setItem(r, 2, QTableWidgetItem(str(f.get("side", ""))))
            self.ov_fills_table.setItem(r, 3, QTableWidgetItem(str(f.get("qty", ""))))
            self.ov_fills_table.setItem(r, 4, QTableWidgetItem(str(f.get("price", ""))))
            self.ov_fills_table.setItem(r, 5, QTableWidgetItem(str(f.get("fees", ""))))
        self.ov_fills_table.scrollToBottom()

    def _update_pnl_from_fill(self, fill: dict) -> None:
        """Update per-symbol realized PnL + cost basis from fills (long-only)."""
        sym = str(fill.get("symbol", ""))
        side = str(fill.get("side", "")).upper()
        if not sym or side not in {"BUY", "SELL"}:
            return

        try:
            qty = int(fill.get("qty", 0))
        except Exception:
            qty = 0
        if qty <= 0:
            return

        try:
            price = float(fill.get("price", 0.0))
        except Exception:
            price = 0.0
        try:
            fees = float(fill.get("fees", 0.0))
        except Exception:
            fees = 0.0

        st = self._pnl_state.get(sym)
        if st is None:
            st = {"qty": 0, "avg_cost": 0.0, "realized": 0.0}
            self._pnl_state[sym] = st

        cur_qty = int(st.get("qty", 0))
        cur_avg = float(st.get("avg_cost", 0.0))

        if side == "BUY":
            # Treat buy fees as part of the cost basis.
            old_cost = cur_qty * cur_avg
            new_cost = qty * price + fees
            new_qty = cur_qty + qty
            st["qty"] = new_qty
            st["avg_cost"] = (old_cost + new_cost) / float(new_qty) if new_qty else 0.0
        else:
            sell_qty = min(qty, cur_qty)
            # Realized PnL: (sell - cost) - sell_fees
            st["realized"] = float(st.get("realized", 0.0)) + (sell_qty * (price - cur_avg) - fees)
            st["qty"] = max(0, cur_qty - sell_qty)
            if st["qty"] == 0:
                st["avg_cost"] = 0.0

    def _render_pnl_table(self) -> None:
        if not hasattr(self, "pnl_table"):
            return

        rows = []
        for sym, st in (self._pnl_state or {}).items():
            qty = int(st.get("qty", 0))
            avg_cost = float(st.get("avg_cost", 0.0))
            realized = float(st.get("realized", 0.0))
            mark = float(self._latest_marks.get(sym, 0.0))
            unreal = (qty * (mark - avg_cost)) if qty else 0.0
            total = realized + unreal

            # Hide completely empty symbols to reduce noise.
            if qty == 0 and abs(total) < 1e-9 and abs(realized) < 1e-9:
                continue

            rows.append((sym, qty, avg_cost, mark, unreal, realized, total))

        rows.sort(key=lambda x: abs(x[6]), reverse=True)

        self.pnl_table.setRowCount(0)
        for sym, qty, avg_cost, mark, unreal, realized, total in rows:
            r = self.pnl_table.rowCount()
            self.pnl_table.insertRow(r)
            self.pnl_table.setItem(r, 0, QTableWidgetItem(str(sym)))
            self.pnl_table.setItem(r, 1, QTableWidgetItem(str(qty)))
            self.pnl_table.setItem(r, 2, QTableWidgetItem(f"{avg_cost:,.4f}"))
            self.pnl_table.setItem(r, 3, QTableWidgetItem(f"{mark:,.4f}"))
            self.pnl_table.setItem(r, 4, QTableWidgetItem(f"{unreal:,.2f}"))
            self.pnl_table.setItem(r, 5, QTableWidgetItem(f"{realized:,.2f}"))
            self.pnl_table.setItem(r, 6, QTableWidgetItem(f"{total:,.2f}"))

    def _fmt_time(self, ts: str) -> str:
        """Format ISO datetime string to something readable."""
        if not ts:
            return ""
        # Most of your engine emits ISO like 2026-02-06T12:34:56
        if len(ts) >= 19 and "T" in ts:
            return ts.replace("T", " ")[:19]
        return ts

    def _safe_parse_iso(self, ts: str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    def _update_trade_blotter_from_fill(self, fill: dict) -> None:
        """Create a per-trade (round-trip) record from raw fills.

        Definition (long-only):
        - Trade opens on first BUY when position was 0.
        - Trade closes when position returns to 0 after SELL fills.

        We compute:
        - Entry VWAP from all BUY fills during the trade
        - Exit VWAP from all SELL fills during the trade
        - PnL = (sell_notional - sell_fees) - (buy_notional + buy_fees)
        """
        sym = str(fill.get("symbol", ""))
        side = str(fill.get("side", "")).upper()
        if not sym or side not in {"BUY", "SELL"}:
            return

        try:
            qty = int(fill.get("qty", 0))
        except Exception:
            qty = 0
        if qty <= 0:
            return

        try:
            price = float(fill.get("price", 0.0))
        except Exception:
            price = 0.0
        try:
            fees = float(fill.get("fees", 0.0))
        except Exception:
            fees = 0.0

        ts = str(fill.get("ts", ""))
        pos = int(self._pos_from_fills.get(sym, 0))

        # Minimal fill record for the Trade Inspector.
        fill_rec = {
            "ts": ts,
            "side": side,
            "qty": qty,
            "price": price,
            "fees": fees,
        }

        if side == "BUY":
            # Open a trade if we were flat
            if pos == 0 and sym not in self._open_trade_by_symbol:
                self._open_trade_by_symbol[sym] = {
                    "entry_ts": ts,
                    "buy_qty": 0,
                    "buy_value": 0.0,
                    "buy_fees": 0.0,
                    "sell_qty": 0,
                    "sell_value": 0.0,
                    "sell_fees": 0.0,
                    "exit_ts": "",
                    "max_pos": 0,
                    "fills": [],
                }

            # If we somehow missed the open (e.g., UI started mid-run), still create one.
            if sym not in self._open_trade_by_symbol:
                self._open_trade_by_symbol[sym] = {
                    "entry_ts": ts,
                    "buy_qty": 0,
                    "buy_value": 0.0,
                    "buy_fees": 0.0,
                    "sell_qty": 0,
                    "sell_value": 0.0,
                    "sell_fees": 0.0,
                    "exit_ts": "",
                    "max_pos": 0,
                    "fills": [],
                }

            t = self._open_trade_by_symbol[sym]
            t["fills"].append(fill_rec)
            t["buy_qty"] += qty
            t["buy_value"] += qty * price
            t["buy_fees"] += fees
            pos += qty
            t["max_pos"] = max(int(t.get("max_pos", 0)), int(pos))

        else:  # SELL
            t = self._open_trade_by_symbol.get(sym)
            if t is None:
                # Sell without an open trade (should not happen in long-only), just update pos defensively.
                pos = max(0, pos - qty)
                self._pos_from_fills[sym] = pos
                return

            t["fills"].append(fill_rec)
            t["sell_qty"] += qty
            t["sell_value"] += qty * price
            t["sell_fees"] += fees
            t["exit_ts"] = ts
            pos = pos - qty
            t["max_pos"] = max(int(t.get("max_pos", 0)), int(max(pos, 0)))
            if pos <= 0:
                # Close the trade (round trip complete)
                entry_qty = int(t["buy_qty"]) or 0
                exit_qty = int(t["sell_qty"]) or 0
                entry_vwap = (float(t["buy_value"]) / entry_qty) if entry_qty else 0.0
                exit_vwap = (float(t["sell_value"]) / exit_qty) if exit_qty else 0.0

                pnl = (float(t["sell_value"]) - float(t["sell_fees"])) - (float(t["buy_value"]) + float(t["buy_fees"]))

                # Duration (best-effort)
                d0 = self._safe_parse_iso(str(t.get("entry_ts", "")))
                d1 = self._safe_parse_iso(str(t.get("exit_ts", "")))
                duration_s = int((d1 - d0).total_seconds()) if (d0 and d1) else 0
                duration_str = self._fmt_duration(duration_s) if duration_s else ""

                max_pos = int(t.get("max_pos", 0))

                # Persist trade for the inspector (row index == trade index)
                trade_obj = {
                    "symbol": sym,
                    "entry_ts": str(t.get("entry_ts", "")),
                    "exit_ts": str(t.get("exit_ts", "")),
                    "entry_qty": entry_qty,
                    "entry_vwap": entry_vwap,
                    "exit_vwap": exit_vwap,
                    "pnl": pnl,
                    "duration_s": duration_s,
                    "duration": duration_str,
                    "max_pos": max_pos,
                    "fills": list(t.get("fills", [])),
                }
                self._trades.append(trade_obj)

                self._append_trade_row(
                    entry_ts=str(t.get("entry_ts", "")),
                    symbol=sym,
                    entry_qty=entry_qty,
                    entry_vwap=entry_vwap,
                    exit_ts=str(t.get("exit_ts", "")),
                    exit_vwap=exit_vwap,
                    pnl=pnl,
                    duration=duration_str,
                    max_pos=max_pos,
                )

                # Reset state
                self._open_trade_by_symbol.pop(sym, None)
                pos = 0

        self._pos_from_fills[sym] = max(0, int(pos))

    def _append_trade_row(self, entry_ts: str, symbol: str, entry_qty: int, entry_vwap: float,
                          exit_ts: str, exit_vwap: float, pnl: float,
                          duration: str, max_pos: int) -> None:
        r = self.trades_table.rowCount()
        self.trades_table.insertRow(r)
        self.trades_table.setItem(r, 0, QTableWidgetItem(self._fmt_time(entry_ts)))
        self.trades_table.setItem(r, 1, QTableWidgetItem(str(symbol)))
        self.trades_table.setItem(r, 2, QTableWidgetItem(str(entry_qty)))
        self.trades_table.setItem(r, 3, QTableWidgetItem(f"{entry_vwap:,.4f}"))
        self.trades_table.setItem(r, 4, QTableWidgetItem(self._fmt_time(exit_ts)))
        self.trades_table.setItem(r, 5, QTableWidgetItem(f"{exit_vwap:,.4f}"))
        self.trades_table.setItem(r, 6, QTableWidgetItem(f"{pnl:,.2f}"))
        self.trades_table.setItem(r, 7, QTableWidgetItem(str(duration)))
        self.trades_table.setItem(r, 8, QTableWidgetItem(str(max_pos)))
        self.trades_table.scrollToBottom()

    def _fmt_duration(self, seconds: int) -> str:
        seconds = max(0, int(seconds))
        days, rem = divmod(seconds, 86400)
        hh, rem = divmod(rem, 3600)
        mm, ss = divmod(rem, 60)
        if days:
            return f"{days}d {hh:02d}:{mm:02d}:{ss:02d}"
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _on_trade_selected(self) -> None:
        """Populate the Trade Inspector from the selected trade row."""
        if not hasattr(self, "trade_fills_table") or not hasattr(self, "trade_inspector_lbl"):
            return
        items = self.trades_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        if row < 0 or row >= len(self._trades):
            return

        t = self._trades[row]
        sym = str(t.get("symbol", ""))
        entry_ts = self._fmt_time(str(t.get("entry_ts", "")))
        exit_ts = self._fmt_time(str(t.get("exit_ts", "")))
        pnl = float(t.get("pnl", 0.0))
        duration = str(t.get("duration", ""))
        max_pos = int(t.get("max_pos", 0))

        self.trade_inspector_lbl.setText(
            f"Trade Inspector: {sym} | Entry: {entry_ts} | Exit: {exit_ts} | "
            f"Dur: {duration} | MaxPos: {max_pos} | PnL: {pnl:,.2f}"
        )

        fills = t.get("fills", []) or []
        self.trade_fills_table.setRowCount(0)
        for f in fills:
            r = self.trade_fills_table.rowCount()
            self.trade_fills_table.insertRow(r)
            self.trade_fills_table.setItem(r, 0, QTableWidgetItem(self._fmt_time(str(f.get("ts", "")))))
            self.trade_fills_table.setItem(r, 1, QTableWidgetItem(str(f.get("side", ""))))
            self.trade_fills_table.setItem(r, 2, QTableWidgetItem(str(f.get("qty", ""))))
            self.trade_fills_table.setItem(r, 3, QTableWidgetItem(str(f.get("price", ""))))
            self.trade_fills_table.setItem(r, 4, QTableWidgetItem(str(f.get("fees", ""))))
        self.trade_fills_table.scrollToBottom()

    def closeEvent(self, event):
        try:
            self.listener.stop()
            self.listener.wait(1000)
        except Exception:
            pass
        super().closeEvent(event)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="tcp://127.0.0.1:5555", help="ZMQ PUB url (default: tcp://127.0.0.1:5555)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    window = RealTimeDashboard(url=args.url)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
