import sys
import argparse
import zmq
import json
import time
import math
from collections import deque
from typing import Dict, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QSplitter
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
import pyqtgraph as pg
from datetime import datetime


# --- Custom axis: dense (no gaps for missing days) ---
class DenseTimeAxis(pg.AxisItem):
    """Axis that treats x as an integer index and renders the corresponding
    timestamp label from a backing list. This avoids weekend/holiday gaps
    when plotting sparse trading timestamps.
    """

    def __init__(self, get_dt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_dt = get_dt

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            i = int(round(v))
            dt = None
            try:
                dt = self._get_dt(i)
            except Exception:
                dt = None
            if dt is None:
                out.append('')
            else:
                out.append(dt.strftime('%Y-%m-%d\n%H:%M'))
        return out


# --- Background Listener Thread ---
class ZmqListener(QThread):
    """Receives ZMQ messages on a background thread.

    Key constraints for the dashboard:
    - Qt GUI thread must stay responsive.
    - In fast backtests, per-message Qt signals can overwhelm the event queue.

    Strategy:
    - Drain the SUB socket quickly.
    - Keep only the latest NAV per drain cycle.
    - Batch FILL messages and emit them in chunks.
    """

    nav_signal = pyqtSignal(dict)
    fills_signal = pyqtSignal(list)
    learn_signal = pyqtSignal(dict)

    def __init__(self, url: str, *, max_fills_emit: int = 2000, poll_ms: int = 50, max_drain_ms: int = 15):
        super().__init__()
        self.url = url
        self._running = True
        self._max_fills_emit = int(max(1, max_fills_emit))
        self._poll_ms = int(max(1, poll_ms))
        self._max_drain_s = float(max(1, max_drain_ms)) / 1000.0

    def stop(self):
        self._running = False

    def run(self):
        ctx = zmq.Context.instance()

        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 10000)
        sock.connect(self.url)
        sock.setsockopt_string(zmq.SUBSCRIBE, "nav")
        sock.setsockopt_string(zmq.SUBSCRIBE, "fill")
        sock.setsockopt_string(zmq.SUBSCRIBE, "learn")

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        latest_nav = None
        latest_learn = None
        fills_batch: List[dict] = []

        def _decode(msg_b: bytes):
            try:
                return json.loads(msg_b.decode("utf-8"))
            except Exception as e:
                print(f"ZMQ decode error: {e}")
                return None

        def _flush():
            nonlocal latest_nav, latest_learn, fills_batch
            if latest_nav is not None:
                self.nav_signal.emit(latest_nav)
                latest_nav = None
            if latest_learn is not None:
                self.learn_signal.emit(latest_learn)
                latest_learn = None
            if fills_batch:
                # Emit in chunks to avoid huge cross-thread payloads.
                n = len(fills_batch)
                if n <= self._max_fills_emit:
                    self.fills_signal.emit(fills_batch)
                else:
                    for i in range(0, n, self._max_fills_emit):
                        self.fills_signal.emit(fills_batch[i:i + self._max_fills_emit])
                fills_batch = []

        try:
            while self._running:
                events = dict(poller.poll(self._poll_ms))

                # Even if no new events, flush pending batches (rare).
                if sock not in events:
                    _flush()
                    continue

                # Drain all available messages quickly, but time-bound.
                start = time.perf_counter()
                while True:
                    if (time.perf_counter() - start) >= self._max_drain_s:
                        break
                    try:
                        topic_b, msg_b = sock.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break

                    data = _decode(msg_b)
                    if data is None:
                        continue

                    topic = topic_b.decode("utf-8", errors="ignore")
                    if topic == "nav":
                        latest_nav = data
                    elif topic == "fill":
                        fills_batch.append(data)
                    elif topic == "learn":
                        latest_learn = data

                _flush()
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

        # NAV history (full horizon) + a throttled/downsampled plot.
        self.nav_data: List[float] = []
        self.nav_x: List[float] = []   # dense integer index for plotting
        self.nav_dt: List[datetime] = []  # timestamp labels for DenseTimeAxis
        self._latest_nav = None
        self._latest_learning = None
        self._initial_nav = None

        # Live backtest analytics state. Metrics are computed from received NAV
        # snapshots, so the numbers update online and do not depend on post-run
        # analytics files. Risk-free rate is fixed at 4% annualized as requested.
        self._risk_free_rate_annual = 0.04
        self._nav_returns: List[float] = []
        self._drawdown_x: List[float] = []
        self._drawdown_y: List[float] = []
        self._last_metrics_update = 0.0
        self._metrics_fps = 2.0

        # Online-learning plot state. True regret needs strategy/oracle telemetry.
        # The dashboard records it when scalars such as regret/cum_regret are
        # published on the `learn` topic; otherwise the plot stays explicit about
        # the missing oracle signal instead of faking it.
        self._ol_x: List[float] = []
        self._ol_regret_y: List[float] = []
        self._ol_cum_regret_y: List[float] = []
        self._ol_aux_x: List[float] = []
        self._ol_reward_y: List[float] = []
        self._ol_loss_y: List[float] = []
        self._last_learning_key = None
        self._last_ol_plot_update = 0.0
        self._ol_plot_fps = 2.0

        # Fill pipeline
        self._fills_buffer = deque()            # raw fills waiting to be processed
        self._recent_fills = deque(maxlen=50)   # for Overview moving rows

        # Fills table rendering (decoupled from fill processing)
        self._fills_display = deque(maxlen=5000)   # what the Fills tab displays
        self._fills_pending_render = deque()       # newly processed fills awaiting UI insertion
        self._fills_table_max_rows = 5000
        self._fills_table_needs_rebuild = True

        # Trade blotter state (derived from fills)
        self._pos_from_fills: Dict[str, int] = {}
        self._open_trade_by_symbol: Dict[str, dict] = {}
        self._trades: List[dict] = []

        # Per-symbol running PnL state (from fills + latest marks)
        self._pnl_state: Dict[str, dict] = {}
        self._latest_marks: Dict[str, float] = {}
        self._latest_positions: Dict[str, int] = {}
        self._latest_visible_symbols: List[str] = []
        self._latest_prices: Dict[str, float] = {}
        self._latest_target_weights: Dict[str, float] = {}

        # Throttles
        self._plot_fps = 2.0
        self._max_plot_points = 20000
        self._last_plot_update = 0.0

        self._fills_table_fps = 4.0
        self._last_fills_table_update = 0.0

        self._pnl_fps = 5.0
        self._last_pnl_update = 0.0

        self.setup_ui()

        # Start listener thread
        self.listener = ZmqListener(self.url)
        self.listener.nav_signal.connect(self.handle_nav_update)
        self.listener.fills_signal.connect(self.handle_fills_update)
        self.listener.learn_signal.connect(self.handle_learning_update)
        self.listener.start()

        # UI flush timer (smooth updates even with bursty data)
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(100)  # 10 FPS UI tick
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
        self.lbl_learning = self.create_stat_label("Learning: -")

        stats_layout.addWidget(self.lbl_nav)
        stats_layout.addWidget(self.lbl_cash)
        stats_layout.addWidget(self.lbl_pnl)
        stats_layout.addWidget(self.lbl_ts)
        stats_layout.addWidget(self.lbl_learning)
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

        dense_axis = DenseTimeAxis(self._get_nav_dt, orientation="bottom")
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": dense_axis})
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.setTitle("Live NAV", color="#dcdcdc")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.nav_curve = self.plot_widget.plot(pen=pg.mkPen(color="#00d4ff", width=2))
        ov_layout.addWidget(self.plot_widget)

        self.ov_fills_table = self.create_table(["Time", "Symbol", "Side", "Qty", "Price", "Fees"])
        self.ov_fills_table.setMaximumHeight(240)
        ov_layout.addWidget(self.ov_fills_table)
        self._overview_tab_index = self.tabs.addTab(overview, "Overview")

        # --- Backtest Metrics Tab ---
        metrics = QWidget()
        metrics_layout = QVBoxLayout(metrics)

        self.metrics_summary_lbl = QLabel(
            "Backtest metrics will appear after at least two NAV snapshots. "
            "Sharpe/Sortino use a fixed 4% annual risk-free rate."
        )
        self.metrics_summary_lbl.setWordWrap(True)
        self.metrics_summary_lbl.setStyleSheet("font-size: 14px; padding: 8px; border: 1px solid #333;")
        metrics_layout.addWidget(self.metrics_summary_lbl)

        metrics_splitter = QSplitter()
        metrics_splitter.setOrientation(Qt.Orientation.Vertical)

        self.metrics_table = self.create_table(["Metric", "Value", "Notes"])
        metrics_splitter.addWidget(self.metrics_table)

        dd_axis = DenseTimeAxis(self._get_nav_dt, orientation="bottom")
        self.drawdown_plot = pg.PlotWidget(axisItems={"bottom": dd_axis})
        self.drawdown_plot.setBackground("#1e1e1e")
        self.drawdown_plot.setTitle("Drawdown", color="#dcdcdc")
        self.drawdown_plot.showGrid(x=True, y=True, alpha=0.3)
        self.drawdown_curve = self.drawdown_plot.plot(pen=pg.mkPen(color="#ff6b6b", width=2))
        metrics_splitter.addWidget(self.drawdown_plot)

        metrics_layout.addWidget(metrics_splitter)
        self._metrics_tab_index = self.tabs.addTab(metrics, "Backtest Metrics")

        # --- Return Distribution Tab ---
        returns_tab = QWidget()
        returns_layout = QVBoxLayout(returns_tab)
        self.returns_summary_lbl = QLabel(
            "Return distribution will appear after enough NAV snapshots. "
            "VaR and CVaR are computed from sampled NAV-to-NAV returns."
        )
        self.returns_summary_lbl.setWordWrap(True)
        self.returns_summary_lbl.setStyleSheet("font-size: 14px; padding: 8px; border: 1px solid #333;")
        returns_layout.addWidget(self.returns_summary_lbl)

        returns_splitter = QSplitter()
        returns_splitter.setOrientation(Qt.Orientation.Vertical)

        self.return_dist_plot = pg.PlotWidget()
        self.return_dist_plot.setBackground("#1e1e1e")
        self.return_dist_plot.setTitle("Sample Return Distribution", color="#dcdcdc")
        self.return_dist_plot.setLabel("bottom", "Sample return (%)")
        self.return_dist_plot.setLabel("left", "Frequency")
        self.return_dist_plot.showGrid(x=True, y=True, alpha=0.3)
        returns_splitter.addWidget(self.return_dist_plot)
        self.return_dist_bars = None

        self.return_risk_table = self.create_table(["Metric", "Value", "Notes"])
        returns_splitter.addWidget(self.return_risk_table)

        returns_layout.addWidget(returns_splitter)
        self._returns_tab_index = self.tabs.addTab(returns_tab, "Return Distribution")

        # --- Positions Tab ---
        positions = QWidget()
        pos_layout = QVBoxLayout(positions)
        self.positions_summary_lbl = QLabel(
            "Positions table shows the current algorithm-visible universe. "
            "Stocks with zero quantity are visible to the algorithm but not currently held."
        )
        self.positions_summary_lbl.setWordWrap(True)
        self.positions_summary_lbl.setStyleSheet("font-size: 14px; padding: 8px; border: 1px solid #333;")
        pos_layout.addWidget(self.positions_summary_lbl)
        self.pos_table = self.create_table(["Symbol", "Visible", "Qty", "Price", "Value", "Target W"])
        pos_layout.addWidget(self.pos_table)
        self._positions_tab_index = self.tabs.addTab(positions, "Positions")

        # --- Fills Tab ---
        fills = QWidget()
        fills_layout = QVBoxLayout(fills)
        self.fills_table = self.create_table(["Time", "Symbol", "Side", "Qty", "Price", "Fees"])
        fills_layout.addWidget(self.fills_table)
        self._fills_tab_index = self.tabs.addTab(fills, "Fills")

        # --- Online Parameters Tab ---
        learning_params = QWidget()
        learning_params_layout = QVBoxLayout(learning_params)

        self.learn_summary_lbl = QLabel(
            "No learning telemetry received yet. Parameters, support, target weights, "
            "and scalar diagnostics will appear here."
        )
        self.learn_summary_lbl.setWordWrap(True)
        self.learn_summary_lbl.setStyleSheet("font-size: 14px; padding: 8px; border: 1px solid #333;")
        learning_params_layout.addWidget(self.learn_summary_lbl)

        params_splitter = QSplitter()
        params_splitter.setOrientation(Qt.Orientation.Vertical)

        self.learn_scalars_table = self.create_table(["Metric", "Value"])
        params_splitter.addWidget(self.learn_scalars_table)

        self.learn_weights_table = self.create_table(["Bucket", "Symbol", "Weight"])
        params_splitter.addWidget(self.learn_weights_table)

        self.learn_lists_table = self.create_table(["Name", "Value"])
        params_splitter.addWidget(self.learn_lists_table)

        learning_params_layout.addWidget(params_splitter)
        self._learning_params_tab_index = self.tabs.addTab(learning_params, "Online Parameters")

        # --- Online Regret Tab ---
        learning_regret = QWidget()
        learning_regret_layout = QVBoxLayout(learning_regret)

        self.regret_summary_lbl = QLabel(
            "No regret telemetry received yet. Regret plots require the strategy to publish "
            "regret/cum_regret or loss/oracle_loss in learn.scalars."
        )
        self.regret_summary_lbl.setWordWrap(True)
        self.regret_summary_lbl.setStyleSheet("font-size: 14px; padding: 8px; border: 1px solid #333;")
        learning_regret_layout.addWidget(self.regret_summary_lbl)

        self.ol_regret_plot = pg.PlotWidget()
        self.ol_regret_plot.setBackground("#1e1e1e")
        self.ol_regret_plot.setTitle("Online Regret", color="#dcdcdc")
        self.ol_regret_plot.setLabel("bottom", "Tick")
        self.ol_regret_plot.setLabel("left", "Regret")
        self.ol_regret_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ol_regret_curve = self.ol_regret_plot.plot(
            name="regret", pen=pg.mkPen(color="#ffd166", width=2)
        )
        self.ol_cum_regret_curve = self.ol_regret_plot.plot(
            name="cum_regret", pen=pg.mkPen(color="#06d6a0", width=2)
        )
        self.ol_regret_plot.addLegend()
        learning_regret_layout.addWidget(self.ol_regret_plot)
        self._learning_regret_tab_index = self.tabs.addTab(learning_regret, "Online Regret")

        # --- PnL Tab ---
        pnl = QWidget()
        pnl_layout = QVBoxLayout(pnl)
        self.pnl_table = self.create_table([
            "Symbol", "Qty", "Avg Cost", "Mark", "Unrealized", "Realized", "Total"
        ])
        pnl_layout.addWidget(self.pnl_table)
        self._pnl_tab_index = self.tabs.addTab(pnl, "PnL")

        # --- Trades Tab ---
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
        self._trades_tab_index = self.tabs.addTab(trades, "Trades")

        self.trades_table.itemSelectionChanged.connect(self._on_trade_selected)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs, stretch=1)

    def _on_tab_changed(self, idx: int) -> None:
        # If user opens Fills tab after a while, rebuild from the bounded deque.
        if idx == getattr(self, "_fills_tab_index", -1):
            self._fills_table_needs_rebuild = True

    def _get_nav_dt(self, idx: int):
        """Return the datetime label for a given dense x index."""
        if 0 <= idx < len(self.nav_dt):
            return self.nav_dt[idx]
        return None

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

    # --- Listener callbacks (Qt GUI thread; keep very light) ---
    def handle_nav_update(self, data: dict):
        self._latest_nav = data

    def handle_learning_update(self, data: dict):
        self._latest_learning = data

    def handle_fills_update(self, fills: list):
        # Append to processing queue.
        for f in fills:
            self._fills_buffer.append(f)
        # Safety: prevent unbounded memory if UI can't keep up for a long time.
        if len(self._fills_buffer) > 1_000_000:
            drop = len(self._fills_buffer) - 1_000_000
            for _ in range(drop):
                self._fills_buffer.popleft()

    def _nav_plot_series(self):
        n = len(self.nav_data)
        if n <= self._max_plot_points:
            return self.nav_x, self.nav_data
        stride = int(math.ceil(n / float(self._max_plot_points)))
        return self.nav_x[::stride], self.nav_data[::stride]

    def flush_ui(self):
        """Runs at a fixed UI rate to keep the GUI responsive."""
        now = time.perf_counter()

        # 1) Apply latest NAV snapshot (keep only latest).
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

            # Append to full-horizon NAV history.
            dt = self._safe_parse_iso(ts)
            # Use a dense x index so the chart does not include gaps for non-trading days.
            x = float(len(self.nav_data))
            self.nav_x.append(x)
            # Keep the timestamp for x-axis label rendering.
            if dt is None:
                dt = datetime.now()
            self.nav_dt.append(dt)
            if self.nav_data:
                prev_nav = float(self.nav_data[-1])
                if prev_nav > 0.0 and math.isfinite(prev_nav) and math.isfinite(nav):
                    self._nav_returns.append((nav / prev_nav) - 1.0)
            self.nav_data.append(nav)
            self._update_drawdown_series()

            # Throttled + downsampled redraw.
            if (now - self._last_plot_update) >= (1.0 / self._plot_fps):
                self._last_plot_update = now
                xs, ys = self._nav_plot_series()
                self.nav_curve.setData(xs, ys)

            if (now - self._last_metrics_update) >= (1.0 / self._metrics_fps):
                self._last_metrics_update = now
                self._render_backtest_metrics()

            # Positions, visible universe, prices & marks from NAV packet.
            positions = self._latest_nav.get("positions", {}) if isinstance(self._latest_nav, dict) else {}
            pos_values = self._latest_nav.get("pos_values", {}) if isinstance(self._latest_nav, dict) else {}
            visible_symbols = self._latest_nav.get("visible_symbols", []) if isinstance(self._latest_nav, dict) else []
            prices = self._latest_nav.get("prices", {}) if isinstance(self._latest_nav, dict) else {}
            target_weights = self._latest_nav.get("target_weights", {}) if isinstance(self._latest_nav, dict) else {}

            self._latest_positions = {str(k): int(v) for k, v in (positions or {}).items()}
            self._latest_visible_symbols = [str(x) for x in (visible_symbols or [])]
            self._latest_prices = {
                str(k): float(v) for k, v in (prices or {}).items()
                if self._finite_float(v) is not None
            }
            self._latest_target_weights = {
                str(k): float(v) for k, v in (target_weights or {}).items()
                if self._finite_float(v) is not None
            }

            self._latest_marks = {}
            for sym, qty in self._latest_positions.items():
                if qty:
                    mark = self._latest_prices.get(sym)
                    if mark is None:
                        try:
                            mark = float((pos_values or {}).get(sym, 0.0)) / float(qty)
                        except Exception:
                            mark = 0.0
                    self._latest_marks[sym] = float(mark)

            self._render_positions(positions, pos_values)

            # Mark this NAV packet consumed. Without this, the UI timer would
            # duplicate the same NAV snapshot and corrupt online metrics.
            self._latest_nav = None

        # 1b) Apply latest learning telemetry.
        if self._latest_learning:
            self._render_learning_panel(self._latest_learning)
            self._latest_learning = None

        # 2) Process fills (state updates) with a time budget so we can catch up.
        backlog = len(self._fills_buffer)
        budget = 0.02
        if backlog > 2000:
            budget = 0.04
        if backlog > 10000:
            budget = 0.08

        start = time.perf_counter()
        processed = 0
        max_per_tick = 50000

        while self._fills_buffer and (time.perf_counter() - start) < budget and processed < max_per_tick:
            f = self._fills_buffer.popleft()

            # Update state
            self._update_pnl_from_fill(f)
            self._update_trade_blotter_from_fill(f)

            # Buffers for display
            self._recent_fills.append(f)
            self._fills_display.append(f)

            # Only queue for rendering if the Fills tab is active; otherwise rebuild on demand.
            if self.tabs.currentIndex() == self._fills_tab_index:
                self._fills_pending_render.append(f)
            else:
                self._fills_table_needs_rebuild = True

            processed += 1

        # 3) Lightweight UI updates
        self._render_recent_fills()
        self._flush_fills_table(now)

        # PnL table render (throttled). We render regardless of tab to keep it fresh,
        # but it's still throttled to avoid excessive work.
        if (now - self._last_pnl_update) >= (1.0 / self._pnl_fps):
            self._last_pnl_update = now
            self._render_pnl_table()

    def _finite_float(self, value, default=None):
        try:
            v = float(value)
        except Exception:
            return default
        if not math.isfinite(v):
            return default
        return v

    def _estimate_periods_per_year(self) -> float:
        """Estimate annualization factor from received NAV timestamps.

        For 1-minute Indian cash-market data this settles near 252*375=94,500
        periods/year. If timestamps are sparse, the median positive intraday
        spacing makes the estimate robust to overnight/weekend gaps.
        """
        if len(self.nav_dt) < 2:
            return 252.0 * 375.0

        diffs = []
        for a, b in zip(self.nav_dt[:-1], self.nav_dt[1:]):
            try:
                sec = (b - a).total_seconds()
            except Exception:
                continue
            if sec > 0:
                diffs.append(float(sec))
        if not diffs:
            return 252.0 * 375.0

        intraday = [d for d in diffs if d <= 6.5 * 3600.0]
        use = intraday if intraday else diffs
        use = sorted(use)
        med = use[len(use) // 2]
        if med <= 0:
            return 252.0 * 375.0

        trading_seconds_per_year = 252.0 * 375.0 * 60.0
        ppy = trading_seconds_per_year / med
        return max(1.0, min(1_000_000.0, ppy))

    def _elapsed_years(self) -> float:
        if len(self.nav_dt) >= 2:
            try:
                seconds = (self.nav_dt[-1] - self.nav_dt[0]).total_seconds()
                if seconds > 0:
                    return max(seconds / (365.25 * 24.0 * 3600.0), 1e-9)
            except Exception:
                pass
        ppy = self._estimate_periods_per_year()
        return max(float(len(self._nav_returns)) / ppy, 1e-9)

    def _std(self, values: List[float], sample: bool = True) -> float:
        vals = [float(x) for x in values if math.isfinite(float(x))]
        n = len(vals)
        if n <= (1 if sample else 0):
            return 0.0
        mean = sum(vals) / n
        denom = (n - 1) if sample and n > 1 else n
        return math.sqrt(sum((x - mean) ** 2 for x in vals) / float(max(1, denom)))

    def _quantile(self, values: List[float], q: float):
        vals = sorted(float(x) for x in values if math.isfinite(float(x)))
        if not vals:
            return None
        q = max(0.0, min(1.0, float(q)))
        pos = (len(vals) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    def _fmt_pct(self, value) -> str:
        v = self._finite_float(value)
        if v is None:
            return "-"
        return f"{100.0 * v:,.2f}%"

    def _fmt_num(self, value, digits: int = 4) -> str:
        v = self._finite_float(value)
        if v is None:
            return "-"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:,.{digits}f}"

    def _set_three_col_rows(self, table: QTableWidget, rows):
        table.setRowCount(0)
        for key, value, note in rows:
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(str(key)))
            table.setItem(r, 1, QTableWidgetItem(str(value)))
            table.setItem(r, 2, QTableWidgetItem(str(note)))

    def _update_drawdown_series(self) -> None:
        if not self.nav_data:
            return
        peak = -float("inf")
        xs = []
        ys = []
        for i, nav in enumerate(self.nav_data):
            if not math.isfinite(float(nav)) or float(nav) <= 0.0:
                continue
            peak = max(peak, float(nav))
            dd = (float(nav) / peak) - 1.0 if peak > 0.0 else 0.0
            xs.append(float(i))
            ys.append(float(dd))
        self._drawdown_x = xs
        self._drawdown_y = ys

    def _tail_risk(self, returns: List[float], confidence: float):
        vals = sorted(float(x) for x in returns if math.isfinite(float(x)))
        if not vals:
            return None, None
        alpha = 1.0 - float(confidence)
        q = self._quantile(vals, alpha)
        if q is None:
            return None, None
        tail_count = max(1, int(math.ceil(alpha * len(vals))))
        tail = vals[:tail_count]
        cvar_ret = sum(tail) / float(len(tail)) if tail else q
        # Report VaR/CVaR as positive loss fractions. A negative value would mean
        # the observed lower tail was still profitable, so clamp at zero.
        return max(0.0, -float(q)), max(0.0, -float(cvar_ret))

    def _render_return_distribution(self) -> None:
        if not hasattr(self, "return_dist_plot"):
            return
        returns = [float(r) for r in self._nav_returns if math.isfinite(float(r))]
        if len(returns) < 2:
            if hasattr(self, "returns_summary_lbl"):
                self.returns_summary_lbl.setText("Return distribution needs at least two NAV-to-NAV return samples.")
            return

        var90, cvar90 = self._tail_risk(returns, 0.90)
        var95, cvar95 = self._tail_risk(returns, 0.95)
        var99, cvar99 = self._tail_risk(returns, 0.99)
        mean_ret = sum(returns) / float(len(returns))
        std_ret = self._std(returns, sample=True)
        best_ret = max(returns)
        worst_ret = min(returns)

        # Lightweight histogram without numpy dependency in the UI module.
        mn, mx = min(returns), max(returns)
        if mx <= mn:
            mn -= 1e-12
            mx += 1e-12
        bin_count = min(60, max(10, int(math.sqrt(len(returns)))))
        width = (mx - mn) / float(bin_count)
        counts = [0] * bin_count
        for r in returns:
            idx = int((r - mn) / width)
            if idx >= bin_count:
                idx = bin_count - 1
            if idx < 0:
                idx = 0
            counts[idx] += 1
        centers_pct = [(mn + (i + 0.5) * width) * 100.0 for i in range(bin_count)]
        bar_width_pct = width * 100.0 * 0.90

        if self.return_dist_bars is not None:
            try:
                self.return_dist_plot.removeItem(self.return_dist_bars)
            except Exception:
                pass
        self.return_dist_bars = pg.BarGraphItem(
            x=centers_pct,
            height=counts,
            width=bar_width_pct,
            brush=pg.mkBrush(80, 160, 220, 180),
        )
        self.return_dist_plot.addItem(self.return_dist_bars)

        rows = [
            ("Samples", f"{len(returns):,}", "NAV-to-NAV sampled returns"),
            ("Mean Sample Return", self._fmt_pct(mean_ret), "Arithmetic mean"),
            ("Sample Std Dev", self._fmt_pct(std_ret), "Non-annualized sample volatility"),
            ("Best Sample Return", self._fmt_pct(best_ret), "Right tail"),
            ("Worst Sample Return", self._fmt_pct(worst_ret), "Left tail"),
            ("VaR 90%", self._fmt_pct(var90), "Loss threshold exceeded in worst 10% samples"),
            ("CVaR 90%", self._fmt_pct(cvar90), "Average loss in worst 10% samples"),
            ("VaR 95%", self._fmt_pct(var95), "Loss threshold exceeded in worst 5% samples"),
            ("CVaR 95%", self._fmt_pct(cvar95), "Average loss in worst 5% samples"),
            ("VaR 99%", self._fmt_pct(var99), "Loss threshold exceeded in worst 1% samples"),
            ("CVaR 99%", self._fmt_pct(cvar99), "Average loss in worst 1% samples"),
        ]
        self._set_three_col_rows(self.return_risk_table, rows)
        if hasattr(self, "returns_summary_lbl"):
            self.returns_summary_lbl.setText(
                f"Samples: {len(returns):,} | Mean: {self._fmt_pct(mean_ret)} | "
                f"Std: {self._fmt_pct(std_ret)} | VaR95: {self._fmt_pct(var95)} | "
                f"CVaR95: {self._fmt_pct(cvar95)}"
            )

    def _render_backtest_metrics(self) -> None:
        if not hasattr(self, "metrics_table"):
            return
        if len(self.nav_data) < 2:
            return

        initial_nav = float(self.nav_data[0])
        latest_nav = float(self.nav_data[-1])
        returns = [r for r in self._nav_returns if math.isfinite(float(r))]
        ppy = self._estimate_periods_per_year()
        years = self._elapsed_years()
        rf_period = (1.0 + self._risk_free_rate_annual) ** (1.0 / ppy) - 1.0 if ppy > 1 else self._risk_free_rate_annual

        total_return = (latest_nav / initial_nav - 1.0) if initial_nav > 0 else 0.0
        cagr = ((latest_nav / initial_nav) ** (1.0 / years) - 1.0) if initial_nav > 0 and years > 0 else 0.0

        avg_ret = sum(returns) / float(len(returns)) if returns else 0.0
        excess = [r - rf_period for r in returns]
        avg_excess = sum(excess) / float(len(excess)) if excess else 0.0
        vol = self._std(returns, sample=True)
        ann_vol = vol * math.sqrt(ppy) if vol > 0 else 0.0
        sharpe = (math.sqrt(ppy) * avg_excess / vol) if vol > 0 else None

        downside = [min(0.0, r - rf_period) for r in returns]
        downside_dev = math.sqrt(sum(x * x for x in downside) / float(len(downside))) if downside else 0.0
        ann_downside = downside_dev * math.sqrt(ppy) if downside_dev > 0 else 0.0
        sortino = (math.sqrt(ppy) * avg_excess / downside_dev) if downside_dev > 0 else None

        max_dd = min(self._drawdown_y) if self._drawdown_y else 0.0
        current_dd = self._drawdown_y[-1] if self._drawdown_y else 0.0
        calmar = (cagr / abs(max_dd)) if max_dd < 0 else None

        hit_rate = (sum(1 for r in returns if r > 0.0) / float(len(returns))) if returns else 0.0
        best_ret = max(returns) if returns else 0.0
        worst_ret = min(returns) if returns else 0.0
        ann_mean_return = avg_ret * ppy

        var90, cvar90 = self._tail_risk(returns, 0.90)
        var95, cvar95 = self._tail_risk(returns, 0.95)
        var99, cvar99 = self._tail_risk(returns, 0.99)

        gross_exposure = 0.0
        latest_pos_count = 0
        visible_count = 0
        if isinstance(self._latest_nav, dict):
            pos_values = self._latest_nav.get("pos_values", {}) or {}
            positions = self._latest_nav.get("positions", {}) or {}
            visible_symbols = self._latest_nav.get("visible_symbols", []) or []
            gross_exposure = sum(abs(float(v)) for v in pos_values.values() if self._finite_float(v) is not None)
            latest_pos_count = sum(1 for q in positions.values() if self._finite_float(q, 0.0))
            visible_count = len(set(str(x) for x in visible_symbols))
        cash = 0.0
        if isinstance(self._latest_nav, dict):
            cash = self._finite_float(self._latest_nav.get("cash", 0.0), 0.0) or 0.0
        gross_exposure_pct = gross_exposure / latest_nav if latest_nav > 0.0 else 0.0
        cash_pct = cash / latest_nav if latest_nav > 0.0 else 0.0

        rows = [
            ("NAV observations", f"{len(self.nav_data):,}", "Received dashboard samples"),
            ("Start NAV", f"{initial_nav:,.2f}", self._fmt_time(self.nav_dt[0].isoformat()) if self.nav_dt else ""),
            ("Latest NAV", f"{latest_nav:,.2f}", self._fmt_time(self.nav_dt[-1].isoformat()) if self.nav_dt else ""),
            ("Total Return", self._fmt_pct(total_return), "Latest NAV / Start NAV - 1"),
            ("CAGR / Annualized Return", self._fmt_pct(cagr), "Calendar-time compounded growth"),
            ("Annualized Mean Return", self._fmt_pct(ann_mean_return), "Arithmetic mean of sampled returns × periods/year"),
            ("Annualized Volatility", self._fmt_pct(ann_vol), "Std(sample returns) × sqrt(periods/year)"),
            ("Sharpe", self._fmt_num(sharpe), "Annual risk-free rate = 4%"),
            ("Sortino", self._fmt_num(sortino), "Downside deviation vs 4% risk-free"),
            ("Max Drawdown", self._fmt_pct(max_dd), "Worst peak-to-trough NAV fall"),
            ("Current Drawdown", self._fmt_pct(current_dd), "Latest NAV below running peak"),
            ("Calmar", self._fmt_num(calmar), "CAGR / |Max Drawdown|"),
            ("Hit Rate", self._fmt_pct(hit_rate), "Fraction of positive sampled returns"),
            ("Best Sample Return", self._fmt_pct(best_ret), "Best NAV-to-NAV sample"),
            ("Worst Sample Return", self._fmt_pct(worst_ret), "Worst NAV-to-NAV sample"),
            ("Average Sample Return", self._fmt_pct(avg_ret), "Mean NAV-to-NAV sample"),
            ("VaR 90%", self._fmt_pct(var90), "Observed periodic loss quantile"),
            ("CVaR 90%", self._fmt_pct(cvar90), "Average loss beyond VaR 90%"),
            ("VaR 95%", self._fmt_pct(var95), "Observed periodic loss quantile"),
            ("CVaR 95%", self._fmt_pct(cvar95), "Average loss beyond VaR 95%"),
            ("VaR 99%", self._fmt_pct(var99), "Observed periodic loss quantile"),
            ("CVaR 99%", self._fmt_pct(cvar99), "Average loss beyond VaR 99%"),
            ("Gross Exposure", self._fmt_pct(gross_exposure_pct), "Sum(|position values|) / NAV"),
            ("Cash %", self._fmt_pct(cash_pct), "Cash / NAV"),
            ("Open Positions", f"{latest_pos_count:,}", "Non-zero positions in latest NAV packet"),
            ("Algorithm-visible symbols", f"{visible_count:,}", "Symbols present in latest market snapshot"),
            ("Closed Trades", f"{len(self._trades):,}", "Completed round trips inferred from fills"),
            ("Displayed Fills", f"{len(self._fills_display):,}", "Bounded UI fill history"),
            ("Estimated periods/year", f"{ppy:,.0f}", "Inferred from median NAV timestamp spacing"),
        ]
        self._set_three_col_rows(self.metrics_table, rows)
        self.metrics_summary_lbl.setText(
            f"Risk-free rate: {self._fmt_pct(self._risk_free_rate_annual)} annual | "
            f"Return: {self._fmt_pct(total_return)} | CAGR: {self._fmt_pct(cagr)} | "
            f"Sharpe: {self._fmt_num(sharpe)} | Sortino: {self._fmt_num(sortino)} | "
            f"Max DD: {self._fmt_pct(max_dd)}"
        )
        if hasattr(self, "drawdown_curve"):
            self.drawdown_curve.setData(self._drawdown_x, self._drawdown_y)
        self._render_return_distribution()

    def _first_scalar(self, scalars: dict, names):
        for name in names:
            if name in scalars:
                v = self._finite_float(scalars.get(name))
                if v is not None:
                    return v
        return None

    def _update_online_learning_series(self, payload: dict, scalars: dict, latest_update: dict) -> None:
        if not hasattr(self, "ol_regret_curve"):
            return

        tick = payload.get("tick", None)
        x = self._finite_float(tick)
        if x is None:
            x = float(len(self._ol_x))

        key = (str(payload.get("ts", "")), int(x) if math.isfinite(float(x)) else len(self._ol_x))
        if key == self._last_learning_key:
            return
        self._last_learning_key = key

        merged = {}
        if isinstance(scalars, dict):
            merged.update(scalars)
        if isinstance(latest_update, dict):
            for k, v in latest_update.items():
                merged.setdefault(k, v)

        regret = self._first_scalar(merged, (
            "regret", "instant_regret", "step_regret", "round_regret",
            "static_regret", "dynamic_regret", "portfolio_regret",
        ))
        cum_regret = self._first_scalar(merged, (
            "cum_regret", "cumm_regret", "cumulative_regret", "total_regret",
            "cumulative_static_regret", "cumulative_dynamic_regret",
            "cum_static_regret", "cum_dynamic_regret",
        ))

        loss = self._first_scalar(merged, ("loss", "round_loss", "portfolio_loss", "learner_loss", "total_loss"))
        oracle_loss = self._first_scalar(merged, ("oracle_loss", "benchmark_loss", "best_loss", "comparator_loss"))
        if regret is None and loss is not None and oracle_loss is not None:
            regret = float(loss) - float(oracle_loss)
        if cum_regret is None and regret is not None:
            prev = self._ol_cum_regret_y[-1] if self._ol_cum_regret_y else 0.0
            cum_regret = prev + float(regret)

        if regret is not None or cum_regret is not None:
            self._ol_x.append(float(x))
            self._ol_regret_y.append(float(regret) if regret is not None else 0.0)
            self._ol_cum_regret_y.append(float(cum_regret) if cum_regret is not None else (self._ol_cum_regret_y[-1] if self._ol_cum_regret_y else 0.0))

        reward = self._first_scalar(merged, ("reward", "last_reward", "step_reward", "episode_reward"))
        if loss is None:
            loss = self._first_scalar(merged, ("policy_loss", "value_loss"))
        if reward is not None or loss is not None:
            # Keep the auxiliary history for telemetry counts/debugging, but do not plot
            # reward/loss anymore. The UI now has a dedicated regret tab.
            self._ol_aux_x.append(float(x))
            self._ol_reward_y.append(float(reward) if reward is not None else (self._ol_reward_y[-1] if self._ol_reward_y else 0.0))
            self._ol_loss_y.append(float(loss) if loss is not None else (self._ol_loss_y[-1] if self._ol_loss_y else 0.0))

        now = time.perf_counter()
        if (now - self._last_ol_plot_update) >= (1.0 / self._ol_plot_fps):
            self._last_ol_plot_update = now
            self.ol_regret_curve.setData(self._ol_x, self._ol_regret_y)
            self.ol_cum_regret_curve.setData(self._ol_x, self._ol_cum_regret_y)

    def _fmt_metric_value(self, value) -> str:
        if isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:,.2f}"
            if abs(value) >= 1:
                return f"{value:,.4f}"
            return f"{value:.6f}"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(x) for x in value)
        if isinstance(value, dict):
            return ", ".join(f"{k}={self._fmt_metric_value(v)}" for k, v in value.items())
        return str(value)

    def _set_two_col_rows(self, table: QTableWidget, rows):
        table.setRowCount(0)
        for key, value in rows:
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(str(key)))
            table.setItem(r, 1, QTableWidgetItem(self._fmt_metric_value(value)))

    def _render_learning_panel(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return

        strategy = str(payload.get("strategy", "-"))
        status = str(payload.get("status", "-"))
        ts = self._fmt_time(str(payload.get("ts", "")))
        scalars = payload.get("scalars", {}) if isinstance(payload.get("scalars", {}), dict) else {}
        weights = payload.get("weights", {}) if isinstance(payload.get("weights", {}), dict) else {}
        lists = payload.get("lists", {}) if isinstance(payload.get("lists", {}), dict) else {}
        latest_update = payload.get("latest_update", {}) if isinstance(payload.get("latest_update", {}), dict) else {}
        blocked_until = payload.get("blocked_until", {}) if isinstance(payload.get("blocked_until", {}), dict) else {}

        header_parts = [strategy]
        if status and status != "-":
            header_parts.append(status)
        if ts:
            header_parts.append(ts)
        self.lbl_learning.setText("Learning: " + " | ".join(header_parts[:3]))

        self._update_online_learning_series(payload, scalars, latest_update)

        regret_points = len(self._ol_x)
        self.learn_summary_lbl.setText(
            f"Strategy: {strategy}\n"
            f"Status: {status}\n"
            f"Timestamp: {ts or '-'}\n"
            f"Scalar diagnostics: {len(scalars)} | Weight buckets: {len(weights)} | Lists: {len(lists)}"
        )
        if hasattr(self, "regret_summary_lbl"):
            self.regret_summary_lbl.setText(
                f"Strategy: {strategy}\n"
                f"Status: {status}\n"
                f"Timestamp: {ts or '-'}\n"
                f"Regret points: {regret_points} | Cumulative regret points: {len(self._ol_cum_regret_y)}\n"
                "Note: regret is only meaningful if the strategy publishes an oracle/comparator loss or regret scalar."
            )

        scalar_rows = sorted(scalars.items(), key=lambda kv: str(kv[0]))
        if latest_update:
            scalar_rows.extend((f"update::{k}", v) for k, v in sorted(latest_update.items(), key=lambda kv: str(kv[0])))
        self._set_two_col_rows(self.learn_scalars_table, scalar_rows)

        self.learn_weights_table.setRowCount(0)
        for bucket_name, bucket in weights.items():
            if not isinstance(bucket, dict):
                continue
            for sym, weight in sorted(bucket.items(), key=lambda kv: abs(float(kv[1])) if isinstance(kv[1], (int, float)) else 0.0, reverse=True):
                r = self.learn_weights_table.rowCount()
                self.learn_weights_table.insertRow(r)
                self.learn_weights_table.setItem(r, 0, QTableWidgetItem(str(bucket_name)))
                self.learn_weights_table.setItem(r, 1, QTableWidgetItem(str(sym)))
                self.learn_weights_table.setItem(r, 2, QTableWidgetItem(self._fmt_metric_value(weight)))

        list_rows = []
        for key, value in sorted(lists.items(), key=lambda kv: str(kv[0])):
            list_rows.append((key, value))
        if blocked_until:
            list_rows.append(("blocked_until", blocked_until))
        self._set_two_col_rows(self.learn_lists_table, list_rows)

    def _render_positions(self, positions: dict, pos_values: dict):
        """Render the algorithm-visible universe, not just non-zero holdings.

        The engine publishes visible_symbols from the latest MarketSnapshot. That is
        the set of stocks the strategy can currently see. Positions/values/targets
        are overlaid on top of that universe.
        """
        self.pos_table.setRowCount(0)

        visible = set(str(x) for x in (self._latest_visible_symbols or []))
        held = set(str(x) for x in (positions or {}).keys())
        targeted = set(str(x) for x in (self._latest_target_weights or {}).keys())
        all_symbols = visible | held | targeted

        rows = []
        active_count = 0
        targeted_count = 0
        for sym in all_symbols:
            qty = 0
            try:
                qty = int((positions or {}).get(sym, 0))
            except Exception:
                qty = 0
            if qty != 0:
                active_count += 1

            price = self._latest_prices.get(sym)
            if price is None and qty:
                try:
                    price = float((pos_values or {}).get(sym, 0.0)) / float(qty)
                except Exception:
                    price = 0.0
            if price is None:
                price = 0.0

            try:
                value = float((pos_values or {}).get(sym, 0.0))
            except Exception:
                value = float(qty) * float(price)

            target_w = float(self._latest_target_weights.get(sym, 0.0))
            if abs(target_w) > 1e-12:
                targeted_count += 1

            is_visible = sym in visible
            # Sort held symbols first, then target symbols, then visible-only names.
            rows.append((
                0 if qty != 0 else (1 if abs(target_w) > 1e-12 else 2),
                -abs(value),
                sym,
                is_visible,
                qty,
                float(price),
                float(value),
                target_w,
            ))

        rows.sort(key=lambda x: (x[0], x[1], x[2]))

        if hasattr(self, "positions_summary_lbl"):
            self.positions_summary_lbl.setText(
                f"Algorithm-visible symbols: {len(visible):,} | "
                f"Current non-zero holdings: {active_count:,} | "
                f"Non-zero target weights: {targeted_count:,}. "
                "Rows with Qty=0 are visible/targeted names that are not currently held."
            )

        self.pos_table.setUpdatesEnabled(False)
        self.pos_table.blockSignals(True)
        try:
            for _, _, sym, is_visible, qty, price, value, target_w in rows:
                r = self.pos_table.rowCount()
                self.pos_table.insertRow(r)
                self.pos_table.setItem(r, 0, QTableWidgetItem(str(sym)))
                self.pos_table.setItem(r, 1, QTableWidgetItem("yes" if is_visible else "no"))
                self.pos_table.setItem(r, 2, QTableWidgetItem(str(qty)))
                self.pos_table.setItem(r, 3, QTableWidgetItem(f"{price:,.4f}" if price else "-"))
                self.pos_table.setItem(r, 4, QTableWidgetItem(f"{value:,.2f}"))
                self.pos_table.setItem(r, 5, QTableWidgetItem(self._fmt_pct(target_w) if abs(target_w) > 1e-12 else "-"))
        finally:
            self.pos_table.blockSignals(False)
            self.pos_table.setUpdatesEnabled(True)

    def _render_recent_fills(self) -> None:
        if not hasattr(self, "ov_fills_table"):
            return
        self.ov_fills_table.setRowCount(0)
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

    def _flush_fills_table(self, now: float) -> None:
        # Only touch the big fills table at a limited rate.
        if (now - self._last_fills_table_update) < (1.0 / self._fills_table_fps):
            return
        self._last_fills_table_update = now

        if self.tabs.currentIndex() != self._fills_tab_index:
            # Avoid growing pending queue when tab is not active.
            self._fills_pending_render.clear()
            return

        if self._fills_table_needs_rebuild:
            self._rebuild_fills_table_from_display()
            self._fills_table_needs_rebuild = False
            self._fills_pending_render.clear()
            return

        if not self._fills_pending_render:
            return

        # Append a bounded number of rows per refresh to avoid UI stalls.
        batch_max = 500
        n = min(batch_max, len(self._fills_pending_render))

        self.fills_table.setUpdatesEnabled(False)
        self.fills_table.blockSignals(True)
        try:
            for _ in range(n):
                f = self._fills_pending_render.popleft()
                self._append_fill_row_to_table(f)

            # Keep table from growing without bound (display-only cap).
            overflow = self.fills_table.rowCount() - self._fills_table_max_rows
            for _ in range(max(0, overflow)):
                self.fills_table.removeRow(0)
        finally:
            self.fills_table.blockSignals(False)
            self.fills_table.setUpdatesEnabled(True)

        self.fills_table.scrollToBottom()

    def _rebuild_fills_table_from_display(self) -> None:
        if not hasattr(self, "fills_table"):
            return
        self.fills_table.setUpdatesEnabled(False)
        self.fills_table.blockSignals(True)
        try:
            self.fills_table.setRowCount(0)
            for f in list(self._fills_display)[-self._fills_table_max_rows:]:
                self._append_fill_row_to_table(f)
        finally:
            self.fills_table.blockSignals(False)
            self.fills_table.setUpdatesEnabled(True)
        self.fills_table.scrollToBottom()

    def _append_fill_row_to_table(self, data: dict) -> None:
        ts = str(data.get("ts", ""))
        # Show full date+time in Fills tab to match your earlier request.
        time_str = self._fmt_time(ts)
        row = self.fills_table.rowCount()
        self.fills_table.insertRow(row)
        self.fills_table.setItem(row, 0, QTableWidgetItem(time_str))
        self.fills_table.setItem(row, 1, QTableWidgetItem(str(data.get("symbol", ""))))
        self.fills_table.setItem(row, 2, QTableWidgetItem(str(data.get("side", ""))))
        self.fills_table.setItem(row, 3, QTableWidgetItem(str(data.get("qty", ""))))
        self.fills_table.setItem(row, 4, QTableWidgetItem(str(data.get("price", ""))))
        self.fills_table.setItem(row, 5, QTableWidgetItem(str(data.get("fees", ""))))

    def _update_pnl_from_fill(self, fill: dict) -> None:
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
            old_cost = cur_qty * cur_avg
            new_cost = qty * price + fees
            new_qty = cur_qty + qty
            st["qty"] = new_qty
            st["avg_cost"] = (old_cost + new_cost) / float(new_qty) if new_qty else 0.0
        else:
            sell_qty = min(qty, cur_qty)
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
        if not ts:
            return ""
        if len(ts) >= 19 and "T" in ts:
            return ts.replace("T", " ")[:19]
        return ts

    def _safe_parse_iso(self, ts: str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    def _update_trade_blotter_from_fill(self, fill: dict) -> None:
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

        fill_rec = {
            "ts": ts,
            "side": side,
            "qty": qty,
            "price": price,
            "fees": fees,
        }

        if side == "BUY":
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

        else:
            t = self._open_trade_by_symbol.get(sym)
            if t is None:
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
                entry_qty = int(t["buy_qty"]) or 0
                exit_qty = int(t["sell_qty"]) or 0
                entry_vwap = (float(t["buy_value"]) / entry_qty) if entry_qty else 0.0
                exit_vwap = (float(t["sell_value"]) / exit_qty) if exit_qty else 0.0

                pnl = (float(t["sell_value"]) - float(t["sell_fees"])) - (float(t["buy_value"]) + float(t["buy_fees"]))

                d0 = self._safe_parse_iso(str(t.get("entry_ts", "")))
                d1 = self._safe_parse_iso(str(t.get("exit_ts", "")))
                duration_s = int((d1 - d0).total_seconds()) if (d0 and d1) else 0
                duration_str = self._fmt_duration(duration_s) if duration_s else ""

                max_pos = int(t.get("max_pos", 0))

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

                self._open_trade_by_symbol.pop(sym, None)
                pos = 0

        self._pos_from_fills[sym] = max(0, int(pos))

    def _append_trade_row(self, entry_ts: str, symbol: str, entry_qty: int, entry_vwap: float,
                          exit_ts: str, exit_vwap: float, pnl: float,
                          duration: str, max_pos: int) -> None:
        # NOTE: no artificial cap (previously 2000) — user explicitly requested unlimited trades.
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
