import sys
import argparse
from pathlib import Path
import time
import math
import bisect
from collections import deque
from typing import Dict, List


# Allow both `python ui/qt_dashboard.py` and `python -m ui.qt_dashboard`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.axis import DenseTimeAxis
from ui.listener import ZmqListener
from ui.widgets import FullWidthTabBar, GuardedViewBox, SmartTableItem

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QTabBar, QSplitter,
    QLineEdit, QPushButton, QAbstractItemView, QComboBox
)
from PyQt6.QtCore import QTimer, Qt, QSize
import pyqtgraph as pg
from datetime import datetime


# --- Custom axis: dense (no gaps for missing days) ---







# --- Background Listener Thread ---


# --- Main Dashboard Window ---
class RealTimeDashboard(QMainWindow):
    def __init__(self, url: str):
        super().__init__()
        self.url = url

        self.setWindowTitle("Levitate Real-Time Monitor")
        self.resize(1200, 800)
        self._render_backend = self._configure_render_backend()
        self.setStyleSheet(self._app_stylesheet())
        self.statusBar().showMessage(
            f"Charts: Ctrl + mouse-wheel to zoom; double-click/reset button restores view. Render: {self._render_backend}"
        )

        # NAV history (full horizon) + a throttled/downsampled plot.
        self.nav_data: List[float] = []
        self.nav_x: List[float] = []   # dense integer index for plotting
        self.nav_dt: List[datetime] = []  # timestamp labels for DenseTimeAxis
        self.nav_time_s: List[float] = []  # epoch seconds for fast window slicing
        self._latest_nav = None
        self._last_nav_packet = None
        self._latest_learning = None
        self._initial_nav = None

        # Live backtest analytics state. Metrics are computed from received NAV
        # snapshots, so the numbers update online and do not depend on post-run
        # analytics files. Risk-free rate is fixed at 4% annualized as requested.
        self._risk_free_rate_annual = 0.04
        self._nav_returns: List[float] = []
        self._drawdown_x: List[float] = []
        self._drawdown_y: List[float] = []
        self._running_peak_nav = None
        self._last_metrics_update = 0.0
        self._metrics_fps = 1.0

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
        self._trades_table_needs_rebuild = False

        # Friction state (derived from fills). Kept incremental so the tab is cheap.
        self._friction_total_turnover = 0.0
        self._friction_total_fees = 0.0
        self._friction_total_slippage = 0.0
        self._friction_recent = deque(maxlen=300)

        # Per-symbol running PnL state (from fills + latest marks)
        self._pnl_state: Dict[str, dict] = {}
        self._latest_marks: Dict[str, float] = {}
        self._latest_positions: Dict[str, int] = {}
        self._latest_visible_symbols: List[str] = []
        self._latest_prices: Dict[str, float] = {}
        self._latest_target_weights: Dict[str, float] = {}

        # Asset Analyser state. This is intentionally selected-asset only;
        # collecting every symbol's history in the GUI would steal CPU/RAM from
        # the backtest. The selected asset keeps a bounded live session cache.
        self._asset_history: Dict[str, deque] = {}
        self._asset_plot_dt: List[datetime] = []
        self._asset_combo_symbols: List[str] = []
        self._asset_last_sample_key = None
        self._asset_flow_by_symbol: Dict[str, dict] = {}
        self._asset_plot_fps = 2.0
        self._last_asset_plot_update = 0.0
        self._asset_max_points = 12000

        # Throttles
        self._plot_fps = 2.0
        self._max_plot_points = 20000
        self._last_plot_update = 0.0

        self._fills_table_fps = 4.0
        self._last_fills_table_update = 0.0

        self._overview_fills_fps = 2.0
        self._last_overview_fills_update = 0.0

        self._positions_fps = 1.0
        self._last_positions_update = 0.0

        self._pnl_fps = 2.0
        self._last_pnl_update = 0.0

        self._frictions_fps = 1.0
        self._last_frictions_update = 0.0

        self.setup_ui()
        # The first tab-bar paint can happen before QTabWidget has its final
        # width. Force one geometry sync now and another after the event loop
        # has completed the initial layout pass.
        QTimer.singleShot(0, self._sync_tab_bar_width)
        QTimer.singleShot(120, self._sync_tab_bar_width)

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
        # Native QTabBar expanding is inconsistent across styles. A custom tab
        # bar gives deterministic equal-width tabs that span the full window.
        self.tabs.setTabBar(FullWidthTabBar(self.tabs))
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setDocumentMode(False)

        # --- Overview Tab (NAV chart) ---
        overview = QWidget()
        ov_layout = QVBoxLayout(overview)

        dense_axis = DenseTimeAxis(self._get_nav_dt, orientation="bottom")
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": dense_axis}, viewBox=GuardedViewBox())
        self._style_plot(self.plot_widget, "Live NAV")
        self.nav_curve = self.plot_widget.plot(pen=pg.mkPen(color="#1a73e8", width=2))
        ov_layout.addLayout(self._nav_plot_header())
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
        self.metrics_summary_lbl.setStyleSheet("font-size: 14px; padding: 10px; border: 1px solid #e0e3eb; border-radius: 10px; background: #ffffff; color: #3c4043;")
        metrics_layout.addWidget(self.metrics_summary_lbl)

        metrics_splitter = QSplitter()
        metrics_splitter.setOrientation(Qt.Orientation.Vertical)

        self.metrics_table = self.create_table(["Metric", "Value", "Notes"])
        metrics_splitter.addWidget(self.metrics_table)

        dd_axis = DenseTimeAxis(self._get_nav_dt, orientation="bottom")
        self.drawdown_plot = pg.PlotWidget(axisItems={"bottom": dd_axis}, viewBox=GuardedViewBox())
        self._style_plot(self.drawdown_plot, "Drawdown")
        self.drawdown_plot.setMaximumHeight(190)
        self.drawdown_plot.setMinimumHeight(120)
        self.drawdown_curve = self.drawdown_plot.plot(pen=pg.mkPen(color="#d93025", width=2))
        drawdown_panel = QWidget()
        drawdown_layout = QVBoxLayout(drawdown_panel)
        drawdown_layout.setContentsMargins(0, 0, 0, 0)
        drawdown_layout.addLayout(self._plot_header("Drawdown", self.drawdown_plot))
        drawdown_layout.addWidget(self.drawdown_plot)
        metrics_splitter.addWidget(drawdown_panel)

        metrics_splitter.setStretchFactor(0, 4)
        metrics_splitter.setStretchFactor(1, 1)
        metrics_splitter.setSizes([520, 170])
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
        self.returns_summary_lbl.setStyleSheet("font-size: 14px; padding: 10px; border: 1px solid #e0e3eb; border-radius: 10px; background: #ffffff; color: #3c4043;")
        returns_layout.addWidget(self.returns_summary_lbl)

        returns_splitter = QSplitter()
        returns_splitter.setOrientation(Qt.Orientation.Vertical)

        self.return_dist_plot = pg.PlotWidget(viewBox=GuardedViewBox())
        self._style_plot(self.return_dist_plot, "Sample Return Distribution")
        self.return_dist_plot.setLabel("bottom", "Sample return (%)")
        self.return_dist_plot.setLabel("left", "Frequency")
        self.return_dist_plot.setMaximumHeight(210)
        self.return_dist_plot.setMinimumHeight(130)
        returns_splitter.addWidget(self.return_dist_plot)
        self.return_dist_bars = None

        self.return_risk_table = self.create_table(["Metric", "Value", "Notes"])
        returns_splitter.addWidget(self.return_risk_table)

        returns_splitter.setStretchFactor(0, 1)
        returns_splitter.setStretchFactor(1, 3)
        returns_splitter.setSizes([180, 420])
        returns_layout.addLayout(self._plot_header("Return Distribution", self.return_dist_plot))
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
        self.positions_summary_lbl.setStyleSheet("font-size: 14px; padding: 10px; border: 1px solid #e0e3eb; border-radius: 10px; background: #ffffff; color: #3c4043;")
        pos_layout.addWidget(self.positions_summary_lbl)
        pos_search_layout = QHBoxLayout()
        self.pos_search = QLineEdit()
        self.pos_search.setPlaceholderText("Search symbol in visible universe...")
        self.pos_search.textChanged.connect(self._apply_positions_filter)
        pos_search_layout.addWidget(self.pos_search)
        pos_layout.addLayout(pos_search_layout)

        self.pos_table = self.create_table(["Symbol", "Visible", "Qty", "Price", "Value", "Target W"])
        self.pos_table.setSortingEnabled(True)
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
        self.learn_summary_lbl.setStyleSheet("font-size: 14px; padding: 10px; border: 1px solid #e0e3eb; border-radius: 10px; background: #ffffff; color: #3c4043;")
        learning_params_layout.addWidget(self.learn_summary_lbl)

        params_splitter = QSplitter()
        params_splitter.setOrientation(Qt.Orientation.Vertical)

        self.learn_scalars_table = self.create_table(["Metric", "Value"])
        self.learn_scalars_table.setMinimumHeight(260)
        params_splitter.addWidget(self.learn_scalars_table)

        self.learn_weights_table = self.create_table(["Bucket", "Symbol", "Weight"])
        self.learn_weights_table.setMinimumHeight(170)
        params_splitter.addWidget(self.learn_weights_table)

        self.learn_lists_table = self.create_table(["Name", "Value"])
        self.learn_lists_table.setMaximumHeight(125)
        params_splitter.addWidget(self.learn_lists_table)

        params_splitter.setStretchFactor(0, 5)
        params_splitter.setStretchFactor(1, 3)
        params_splitter.setStretchFactor(2, 1)
        params_splitter.setSizes([390, 240, 95])
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
        self.regret_summary_lbl.setStyleSheet("font-size: 14px; padding: 10px; border: 1px solid #e0e3eb; border-radius: 10px; background: #ffffff; color: #3c4043;")
        learning_regret_layout.addWidget(self.regret_summary_lbl)

        self.ol_regret_plot = pg.PlotWidget(viewBox=GuardedViewBox())
        self._style_plot(self.ol_regret_plot, "Online Regret")
        self.ol_regret_plot.setLabel("bottom", "Tick")
        self.ol_regret_plot.setLabel("left", "Regret")
        self.ol_regret_curve = self.ol_regret_plot.plot(
            name="regret", pen=pg.mkPen(color="#f9ab00", width=2)
        )
        self.ol_cum_regret_curve = self.ol_regret_plot.plot(
            name="cum_regret", pen=pg.mkPen(color="#188038", width=2)
        )
        self.ol_regret_plot.addLegend()
        learning_regret_layout.addLayout(self._plot_header("Online Regret", self.ol_regret_plot))
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
        # --- Frictions Tab ---
        frictions = QWidget()
        fr_layout = QVBoxLayout(frictions)
        self.frictions_summary_lbl = QLabel(
            "Frictions are computed incrementally from fills. Fees are exact when published; "
            "slippage is shown only when the fill payload contains a slippage/impact field or a reference price."
        )
        self.frictions_summary_lbl.setWordWrap(True)
        self.frictions_summary_lbl.setStyleSheet("font-size: 14px; padding: 10px; border: 1px solid #e0e3eb; border-radius: 10px; background: #ffffff; color: #3c4043;")
        fr_layout.addWidget(self.frictions_summary_lbl)

        fr_splitter = QSplitter()
        fr_splitter.setOrientation(Qt.Orientation.Vertical)
        self.frictions_table = self.create_table(["Metric", "Value", "Notes"])
        fr_splitter.addWidget(self.frictions_table)
        self.frictions_recent_table = self.create_table([
            "Time", "Symbol", "Side", "Qty", "Price", "Turnover", "Fees", "Slippage", "Total Cost"
        ])
        fr_splitter.addWidget(self.frictions_recent_table)
        fr_layout.addWidget(fr_splitter)
        self._frictions_tab_index = self.tabs.addTab(frictions, "Frictions")

        # --- Asset Analyser Tab ---
        asset_tab = QWidget()
        asset_layout = QVBoxLayout(asset_tab)

        asset_controls = QHBoxLayout()
        self.asset_combo = QComboBox()
        self.asset_combo.setEditable(True)
        self.asset_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.asset_combo.setMinimumWidth(220)
        self.asset_combo.currentTextChanged.connect(self._on_asset_selection_changed)

        self.asset_window_combo = QComboBox()
        self.asset_window_combo.addItems(["1D", "1W", "1M", "6M", "1Y", "3Y", "All"] )
        self.asset_window_combo.setCurrentText("1D")
        self.asset_window_combo.currentTextChanged.connect(lambda _=None: self._render_asset_analyser())

        self.asset_chart_combo = QComboBox()
        self.asset_chart_combo.addItems(["Close line"])
        self.asset_chart_combo.setToolTip("Candlestick mode needs OHLC packets; this first version uses live close prices.")

        asset_controls.addWidget(QLabel("Asset"))
        asset_controls.addWidget(self.asset_combo, stretch=2)
        asset_controls.addWidget(QLabel("Window"))
        asset_controls.addWidget(self.asset_window_combo)
        asset_controls.addWidget(QLabel("Chart"))
        asset_controls.addWidget(self.asset_chart_combo)
        asset_controls.addStretch(1)
        asset_layout.addLayout(asset_controls)

        self.asset_summary_lbl = QLabel(
            "Select an asset from the visible universe. The chart uses a bounded live cache for the selected asset only."
        )
        self.asset_summary_lbl.setWordWrap(True)
        self.asset_summary_lbl.setStyleSheet("font-size: 14px; padding: 10px; border: 1px solid #e0e3eb; border-radius: 10px; background: #ffffff; color: #3c4043;")
        asset_layout.addWidget(self.asset_summary_lbl)

        asset_splitter = QSplitter()
        asset_splitter.setOrientation(Qt.Orientation.Vertical)

        asset_price_axis = DenseTimeAxis(self._get_asset_dt, orientation="bottom")
        self.asset_price_plot = pg.PlotWidget(axisItems={"bottom": asset_price_axis}, viewBox=GuardedViewBox())
        self._style_plot(self.asset_price_plot, "Selected Asset Close Price")
        self.asset_price_plot.setLabel("left", "Close")
        self.asset_price_curve = self.asset_price_plot.plot(pen=pg.mkPen(color="#1a73e8", width=2))
        asset_price_panel = QWidget()
        asset_price_layout = QVBoxLayout(asset_price_panel)
        asset_price_layout.setContentsMargins(0, 0, 0, 0)
        asset_price_layout.addLayout(self._plot_header("Price", self.asset_price_plot))
        asset_price_layout.addWidget(self.asset_price_plot)
        asset_splitter.addWidget(asset_price_panel)

        asset_volume_axis = DenseTimeAxis(self._get_asset_dt, orientation="bottom")
        self.asset_volume_plot = pg.PlotWidget(axisItems={"bottom": asset_volume_axis}, viewBox=GuardedViewBox())
        self._style_plot(self.asset_volume_plot, "Selected Asset Volume")
        self.asset_volume_plot.setLabel("left", "Volume")
        self.asset_volume_curve = self.asset_volume_plot.plot(pen=pg.mkPen(color="#188038", width=1))
        self.asset_volume_plot.setMaximumHeight(170)
        self.asset_volume_plot.setMinimumHeight(100)
        asset_volume_panel = QWidget()
        asset_volume_layout = QVBoxLayout(asset_volume_panel)
        asset_volume_layout.setContentsMargins(0, 0, 0, 0)
        asset_volume_layout.addLayout(self._plot_header("Volume", self.asset_volume_plot))
        asset_volume_layout.addWidget(self.asset_volume_plot)
        asset_splitter.addWidget(asset_volume_panel)

        asset_tables_panel = QWidget()
        asset_tables_layout = QHBoxLayout(asset_tables_panel)
        self.asset_info_table = self.create_table(["Metric", "Value"])
        self.asset_flow_table = self.create_table(["Metric", "Value"])
        asset_tables_layout.addWidget(self.asset_info_table, stretch=1)
        asset_tables_layout.addWidget(self.asset_flow_table, stretch=1)
        asset_splitter.addWidget(asset_tables_panel)

        asset_splitter.setStretchFactor(0, 5)
        asset_splitter.setStretchFactor(1, 2)
        asset_splitter.setStretchFactor(2, 2)
        asset_splitter.setSizes([420, 130, 180])
        asset_layout.addWidget(asset_splitter)
        self._asset_tab_index = self.tabs.addTab(asset_tab, "Asset Analyser")

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
        self.trade_inspector_lbl.setStyleSheet("font-size: 14px; padding: 8px; border: 1px solid #e0e3eb; border-radius: 10px; background: #ffffff; color: #3c4043;")
        insp_layout.addWidget(self.trade_inspector_lbl)
        self.trade_fills_table = self.create_table(["Time", "Side", "Qty", "Price", "Fees"])
        insp_layout.addWidget(self.trade_fills_table)
        splitter.addWidget(inspector)

        trades_layout.addWidget(splitter)
        self._trades_tab_index = self.tabs.addTab(trades, "Trades")

        self.trades_table.itemSelectionChanged.connect(self._on_trade_selected)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs, stretch=1)
        self._sync_tab_bar_width()


    def _configure_render_backend(self) -> str:
        """Prefer OpenGL for pyqtgraph curves; fall back without failing startup.

        This affects chart rendering. Qt tables and labels are still normal Qt
        widgets, so they cannot be honestly advertised as fully GPU-rendered.
        """
        try:
            pg.setConfigOptions(useOpenGL=True, antialias=False)
            return "OpenGL requested"
        except Exception:
            try:
                pg.setConfigOptions(antialias=False)
            except Exception:
                pass
            return "CPU fallback"

    def _app_stylesheet(self) -> str:
        """Static Material-ish stylesheet.

        This is intentionally CSS-only: no runtime table resizing, no cell scanning,
        no animation timers. It improves polish without stealing CPU from the
        simulation.
        """
        return """
        QMainWindow, QWidget {
            background: #f8fafc;
            color: #202124;
            font-family: Segoe UI, Arial, sans-serif;
            font-size: 13px;
        }
        QStatusBar {
            background: #f8fafc;
            color: #5f6368;
            border-top: 1px solid #e8eaed;
        }
        QLabel {
            color: #202124;
        }
        QTabWidget::pane {
            border: none;
            margin-top: 0px;
            background: #f8fafc;
        }
        QTabBar {
            background: #f8fafc;
            qproperty-drawBase: 0;
        }
        QTabBar::tab {
            background: #eef2f7;
            color: #3c4043;
            padding: 10px 6px;
            margin: 0px 1px 0px 0px;
            border: 1px solid transparent;
            border-radius: 10px;
            font-weight: 600;
            min-height: 22px;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            color: #1a73e8;
            border: 1px solid #d8dee8;
        }
        QTabBar::tab:hover {
            background: #e8f0fe;
            color: #174ea6;
        }
        QTableWidget {
            background: #ffffff;
            alternate-background-color: #f8fbff;
            color: #202124;
            gridline-color: #edf0f5;
            border: 1px solid #e0e3eb;
            border-radius: 10px;
            selection-background-color: #d2e3fc;
            selection-color: #202124;
        }
        QHeaderView::section {
            background: #f1f3f4;
            color: #3c4043;
            border: none;
            border-right: 1px solid #e0e3eb;
            border-bottom: 1px solid #e0e3eb;
            padding: 7px 8px;
            font-weight: 700;
        }
        QTableCornerButton::section {
            background: #f1f3f4;
            border: none;
            border-bottom: 1px solid #e0e3eb;
            border-right: 1px solid #e0e3eb;
        }
        QLineEdit {
            background: #ffffff;
            border: 1px solid #dadce0;
            border-radius: 10px;
            padding: 8px 11px;
            color: #202124;
        }
        QLineEdit:focus {
            border: 1px solid #1a73e8;
        }
        QComboBox {
            background: #ffffff;
            border: 1px solid #dadce0;
            border-radius: 9px;
            padding: 6px 10px;
            color: #202124;
            min-height: 20px;
        }
        QComboBox:hover {
            border: 1px solid #c6dafc;
        }
        QComboBox::drop-down {
            border: none;
            width: 22px;
        }
        QPushButton {
            background: #ffffff;
            color: #1a73e8;
            border: 1px solid #dadce0;
            border-radius: 9px;
            padding: 6px 12px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #f1f6ff;
            border: 1px solid #c6dafc;
        }
        QPushButton:pressed {
            background: #e8f0fe;
        }
        QSplitter::handle {
            background: #eef2f7;
            height: 5px;
        }
        QScrollBar:vertical {
            background: transparent;
            width: 12px;
            margin: 2px;
        }
        QScrollBar::handle:vertical {
            background: #c9d3e1;
            min-height: 32px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background: #9aa7b8;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            background: transparent;
            height: 12px;
            margin: 2px;
        }
        QScrollBar::handle:horizontal {
            background: #c9d3e1;
            min-width: 32px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #9aa7b8;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        """

    def _style_plot(self, plot: pg.PlotWidget, title: str = "") -> None:
        plot.setBackground("#ffffff")
        if title:
            plot.setTitle(title, color="#202124", size="11pt")
        plot.showGrid(x=True, y=True, alpha=0.18)
        try:
            plot.getAxis("bottom").setPen(pg.mkPen("#dadce0"))
            plot.getAxis("left").setPen(pg.mkPen("#dadce0"))
            plot.getAxis("bottom").setTextPen(pg.mkPen("#5f6368"))
            plot.getAxis("left").setTextPen(pg.mkPen("#5f6368"))
        except Exception:
            pass

    def _plot_header(self, title: str, plot: pg.PlotWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: 700; font-size: 14px; padding-left: 2px; color: #202124;")
        btn = QPushButton("Reset view")
        btn.clicked.connect(lambda _=False, p=plot: self._reset_plot_view(p))
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(btn)
        return row

    def _nav_plot_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel("Live NAV")
        lbl.setStyleSheet("font-weight: 700; font-size: 14px; padding-left: 2px; color: #202124;")

        self.nav_window_combo = QComboBox()
        self.nav_window_combo.addItems(["1D", "1W", "1M", "6M", "1Y", "3Y", "All"])
        self.nav_window_combo.setCurrentText("All")
        self.nav_window_combo.setToolTip("Display window for the NAV chart. This does not discard stored NAV history.")
        self.nav_window_combo.currentTextChanged.connect(lambda _=None: self._refresh_nav_plot())

        self.nav_chart_combo = QComboBox()
        self.nav_chart_combo.addItems(["NAV", "Return %", "Log NAV"])
        self.nav_chart_combo.setCurrentText("NAV")
        self.nav_chart_combo.setToolTip("NAV shows absolute equity, Return % rebases the visible window, Log NAV shows ln(NAV).")
        self.nav_chart_combo.currentTextChanged.connect(lambda _=None: self._refresh_nav_plot())

        btn = QPushButton("Reset view")
        btn.clicked.connect(lambda _=False: self._reset_plot_view(self.plot_widget))

        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(QLabel("Window"))
        row.addWidget(self.nav_window_combo)
        row.addWidget(QLabel("View"))
        row.addWidget(self.nav_chart_combo)
        row.addWidget(btn)
        return row

    def _sync_tab_bar_width(self) -> None:
        if not hasattr(self, "tabs"):
            return
        try:
            bar = self.tabs.tabBar()
            width = max(1, self.tabs.width())
            if hasattr(bar, "set_available_width"):
                bar.set_available_width(width)
            else:
                bar.setFixedWidth(width)
                bar.updateGeometry()
            self.tabs.updateGeometry()
            self.tabs.update()
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_tab_bar_width)
        QTimer.singleShot(120, self._sync_tab_bar_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_tab_bar_width)

    def _reset_plot_view(self, plot: pg.PlotWidget) -> None:
        try:
            plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            plot.autoRange()
        except Exception:
            pass

    def _on_tab_changed(self, idx: int) -> None:
        # If user opens Fills tab after a while, rebuild from the bounded deque.
        if idx == getattr(self, "_fills_tab_index", -1):
            self._fills_table_needs_rebuild = True
        elif idx == getattr(self, "_positions_tab_index", -1) and isinstance(self._last_nav_packet, dict):
            self._render_positions(self._last_nav_packet.get("positions", {}) or {}, self._last_nav_packet.get("pos_values", {}) or {})
        elif idx == getattr(self, "_metrics_tab_index", -1):
            self._render_backtest_metrics()
        elif idx == getattr(self, "_returns_tab_index", -1):
            self._render_return_distribution()
        elif idx == getattr(self, "_pnl_tab_index", -1):
            self._render_pnl_table()
        elif idx == getattr(self, "_frictions_tab_index", -1):
            self._render_frictions_panel()
        elif idx == getattr(self, "_asset_tab_index", -1):
            self._refresh_asset_combo()
            self._capture_asset_sample_from_latest_nav()
            self._render_asset_analyser()
        elif idx == getattr(self, "_trades_tab_index", -1):
            if self._trades_table_needs_rebuild:
                self._rebuild_trades_table()
                self._trades_table_needs_rebuild = False

    def _get_nav_dt(self, idx: int):
        """Return the datetime label for a given dense x index."""
        if 0 <= idx < len(self.nav_dt):
            return self.nav_dt[idx]
        return None

    def _selected_asset(self) -> str:
        if not hasattr(self, "asset_combo"):
            return ""
        return self.asset_combo.currentText().strip()

    def _get_asset_history(self, symbol: str):
        if not symbol:
            return deque(maxlen=self._asset_max_points)
        if symbol not in self._asset_history:
            self._asset_history[symbol] = deque(maxlen=self._asset_max_points)
        return self._asset_history[symbol]

    def _get_asset_dt(self, idx: int):
        if 0 <= idx < len(self._asset_plot_dt):
            return self._asset_plot_dt[idx]
        return None

    def _asset_window_seconds(self):
        if not hasattr(self, "asset_window_combo"):
            return 6.5 * 3600.0
        label = self.asset_window_combo.currentText().strip()
        return {
            "1D": 6.5 * 3600.0,
            "1W": 7.0 * 24.0 * 3600.0,
            "1M": 30.0 * 24.0 * 3600.0,
            "6M": 182.0 * 24.0 * 3600.0,
            "1Y": 365.0 * 24.0 * 3600.0,
            "3Y": 3.0 * 365.0 * 24.0 * 3600.0,
            "All": None,
        }.get(label, 6.5 * 3600.0)

    def _asset_universe(self):
        symbols = set(self._latest_visible_symbols or [])
        symbols.update(self._latest_prices.keys())
        symbols.update(self._latest_positions.keys())
        symbols.update(self._latest_target_weights.keys())
        return sorted(str(s) for s in symbols if str(s))

    def _refresh_asset_combo(self) -> None:
        if not hasattr(self, "asset_combo"):
            return
        symbols = self._asset_universe()
        if symbols == self._asset_combo_symbols:
            return

        previous = self.asset_combo.currentText().strip()
        self._asset_combo_symbols = symbols
        self.asset_combo.blockSignals(True)
        try:
            self.asset_combo.clear()
            self.asset_combo.addItems(symbols)
            if previous and previous in symbols:
                self.asset_combo.setCurrentText(previous)
            elif symbols:
                self.asset_combo.setCurrentIndex(0)
        finally:
            self.asset_combo.blockSignals(False)

    def _on_asset_selection_changed(self, _text=None) -> None:
        self._asset_last_sample_key = None
        self._capture_asset_sample_from_latest_nav()
        self._render_asset_analyser()

    def _extract_volume_for_symbol(self, packet: dict, symbol: str):
        if not isinstance(packet, dict) or not symbol:
            return None
        # Accept several likely payload names without forcing a backend change.
        for key in ("volumes", "volume", "latest_volumes", "last_volumes", "trade_volumes"):
            obj = packet.get(key)
            if isinstance(obj, dict) and symbol in obj:
                v = self._finite_float(obj.get(symbol))
                if v is not None:
                    return float(v)
        return None

    def _capture_asset_sample_from_latest_nav(self) -> None:
        if not isinstance(self._last_nav_packet, dict):
            return
        if not hasattr(self, "asset_combo"):
            return

        symbol = self._selected_asset()
        if not symbol:
            return

        price = self._latest_prices.get(symbol)
        if price is None:
            prices = self._last_nav_packet.get("prices", {}) or {}
            if isinstance(prices, dict):
                price = self._finite_float(prices.get(symbol))
        if price is None:
            return

        ts = str(self._last_nav_packet.get("ts", ""))
        dt = self._safe_parse_iso(ts) or (self.nav_dt[-1] if self.nav_dt else datetime.now())
        key = (symbol, ts, len(self.nav_data))
        if key == self._asset_last_sample_key:
            return
        self._asset_last_sample_key = key

        volume = self._extract_volume_for_symbol(self._last_nav_packet, symbol)
        hist = self._get_asset_history(symbol)
        # One sample per timestamp. If the same timestamp arrives again, replace
        # instead of appending, which keeps the plot stable during bursty updates.
        if hist and hist[-1][1] == dt:
            hist[-1] = (float(len(hist) - 1), dt, float(price), volume)
        else:
            hist.append((float(len(hist)), dt, float(price), volume))

    def _asset_series_for_plot(self, symbol: str):
        hist = list(self._asset_history.get(symbol, []))
        if not hist:
            self._asset_plot_dt = []
            return [], [], []

        window_s = self._asset_window_seconds()
        if window_s is not None and hist:
            end_dt = hist[-1][1]
            try:
                cutoff = end_dt.timestamp() - float(window_s)
                hist = [row for row in hist if row[1].timestamp() >= cutoff]
            except Exception:
                pass

        if len(hist) > self._asset_max_points:
            stride = int(math.ceil(len(hist) / float(self._asset_max_points)))
            hist = hist[::stride]

        self._asset_plot_dt = [row[1] for row in hist]
        xs = [float(i) for i in range(len(hist))]
        prices = [float(row[2]) for row in hist]
        vols = [float(row[3]) if row[3] is not None and math.isfinite(float(row[3])) else 0.0 for row in hist]
        return xs, prices, vols

    def _render_asset_analyser(self) -> None:
        if not hasattr(self, "asset_price_curve"):
            return
        symbol = self._selected_asset()
        if not symbol:
            if hasattr(self, "asset_summary_lbl"):
                self.asset_summary_lbl.setText("No asset universe has been received yet.")
            return

        xs, prices, vols = self._asset_series_for_plot(symbol)
        self.asset_price_curve.setData(xs, prices)
        self.asset_volume_curve.setData(xs, vols)

        latest_price = self._latest_prices.get(symbol)
        qty = int(self._latest_positions.get(symbol, 0) or 0)
        latest_nav = float(self.nav_data[-1]) if self.nav_data else 0.0
        value = float(qty) * float(latest_price or 0.0)
        current_w = (value / latest_nav) if latest_nav > 0.0 else 0.0
        target_w = float(self._latest_target_weights.get(symbol, 0.0) or 0.0)
        visible = "yes" if symbol in set(self._latest_visible_symbols or []) else "no"
        flow = self._asset_flow_by_symbol.get(symbol, {})

        info_rows = [
            ("Symbol", symbol),
            ("Visible", visible),
            ("Latest Price", f"{latest_price:,.4f}" if latest_price is not None else "-"),
            ("Quantity Held", f"{qty:,}"),
            ("Position Value", f"{value:,.2f}"),
            ("Current Weight", self._fmt_pct(current_w)),
            ("Target Weight", self._fmt_pct(target_w) if abs(target_w) > 1e-12 else "-"),
            ("Cached Samples", f"{len(self._asset_history.get(symbol, [])):,}"),
        ]
        self._set_two_col_rows(self.asset_info_table, info_rows)

        flow_rows = [
            ("Last Action", str(flow.get("last_side", "-"))),
            ("Last Qty", f"{int(flow.get('last_qty', 0) or 0):,}"),
            ("Last Price", self._fmt_num(flow.get("last_price"), 4)),
            ("Last Time", self._fmt_time(str(flow.get("last_ts", "")))),
            ("Cumulative Bought", f"{int(flow.get('buy_qty', 0) or 0):,}"),
            ("Cumulative Sold", f"{int(flow.get('sell_qty', 0) or 0):,}"),
            ("Buy Turnover", f"{float(flow.get('buy_value', 0.0) or 0.0):,.2f}"),
            ("Sell Turnover", f"{float(flow.get('sell_value', 0.0) or 0.0):,.2f}"),
        ]
        self._set_two_col_rows(self.asset_flow_table, flow_rows)

        if hasattr(self, "asset_summary_lbl"):
            vol_note = "Volume is shown when the NAV/market packet contains per-symbol volume."
            if prices:
                self.asset_summary_lbl.setText(
                    f"{symbol}: {len(prices):,} displayed samples | latest close: "
                    f"{prices[-1]:,.4f} | qty: {qty:,} | weight: {self._fmt_pct(current_w)}. {vol_note}"
                )
            else:
                self.asset_summary_lbl.setText(
                    f"{symbol}: waiting for live price samples. {vol_note}"
                )

    def create_stat_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size: 15px; font-weight: 700; padding: 10px 12px; "
            "border: 1px solid #e0e3eb; border-radius: 12px; background: #ffffff; color: #202124;"
        )
        return lbl

    def create_table(self, headers):
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        # Speed-first table policy: keep stable Stretch mode. Do not resize
        # columns to contents during live updates.
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionsClickable(True)
        table.horizontalHeader().setHighlightSections(False)
        table.verticalHeader().setDefaultSectionSize(25)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setShowGrid(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        return table

    def _item(self, text, sort_value=None):
        item = SmartTableItem(str(text), sort_value=sort_value)
        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return item

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

    def _nav_window_seconds(self):
        if not hasattr(self, "nav_window_combo"):
            return None
        label = self.nav_window_combo.currentText().strip()
        return {
            "1D": 6.5 * 3600.0,
            "1W": 7.0 * 24.0 * 3600.0,
            "1M": 30.0 * 24.0 * 3600.0,
            "6M": 182.0 * 24.0 * 3600.0,
            "1Y": 365.0 * 24.0 * 3600.0,
            "3Y": 3.0 * 365.0 * 24.0 * 3600.0,
            "All": None,
        }.get(label, None)

    def _nav_plot_series(self):
        n = len(self.nav_data)
        if n <= 0:
            return [], []

        start = 0
        window_s = self._nav_window_seconds()
        if window_s is not None and self.nav_time_s:
            try:
                cutoff = float(self.nav_time_s[-1]) - float(window_s)
                # Binary search keeps window changes cheap even after years of
                # minute-level NAV samples. Avoid scanning the whole history.
                start = bisect.bisect_left(self.nav_time_s, cutoff)
                start = max(0, min(start, n - 1))
            except Exception:
                start = 0

        xs = self.nav_x[start:]
        values = self.nav_data[start:]
        if not xs or not values:
            return [], []

        view = self.nav_chart_combo.currentText().strip() if hasattr(self, "nav_chart_combo") else "NAV"
        if view == "Return %":
            base = float(values[0]) if values and float(values[0]) != 0.0 else 1.0
            ys = [((float(v) / base) - 1.0) * 100.0 for v in values]
            try:
                self.plot_widget.setLabel("left", "Return (%)")
            except Exception:
                pass
        elif view == "Log NAV":
            ys = [math.log(max(float(v), 1e-12)) for v in values]
            try:
                self.plot_widget.setLabel("left", "ln(NAV)")
            except Exception:
                pass
        else:
            ys = [float(v) for v in values]
            try:
                self.plot_widget.setLabel("left", "NAV")
            except Exception:
                pass

        if len(xs) > self._max_plot_points:
            stride = int(math.ceil(len(xs) / float(self._max_plot_points)))
            xs = xs[::stride]
            ys = ys[::stride]
        return xs, ys

    def _refresh_nav_plot(self) -> None:
        if not hasattr(self, "nav_curve"):
            return
        try:
            xs, ys = self._nav_plot_series()
            self.nav_curve.setData(xs, ys)
            self._reset_plot_view(self.plot_widget)
        except Exception:
            pass

    def flush_ui(self):
        """Runs at a fixed UI rate to keep the GUI responsive."""
        now = time.perf_counter()

        # 1) Apply latest NAV snapshot (keep only latest).
        if self._latest_nav:
            self._last_nav_packet = dict(self._latest_nav) if isinstance(self._latest_nav, dict) else self._latest_nav
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
            try:
                self.nav_time_s.append(float(dt.timestamp()))
            except Exception:
                self.nav_time_s.append(float(len(self.nav_time_s)))
            if self.nav_data:
                prev_nav = float(self.nav_data[-1])
                if prev_nav > 0.0 and math.isfinite(prev_nav) and math.isfinite(nav):
                    self._nav_returns.append((nav / prev_nav) - 1.0)
            self.nav_data.append(nav)
            self._append_drawdown_point(x, nav)

            # Throttled + downsampled redraw.
            if (now - self._last_plot_update) >= (1.0 / self._plot_fps):
                self._last_plot_update = now
                xs, ys = self._nav_plot_series()
                self.nav_curve.setData(xs, ys)

            if (now - self._last_metrics_update) >= (1.0 / self._metrics_fps):
                self._last_metrics_update = now
                active = self.tabs.currentIndex()
                if active == self._metrics_tab_index:
                    self._render_backtest_metrics()
                elif active == self._returns_tab_index:
                    self._render_return_distribution()

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

            if self.tabs.currentIndex() == getattr(self, "_asset_tab_index", -1):
                self._refresh_asset_combo()
                self._capture_asset_sample_from_latest_nav()
                if (now - self._last_asset_plot_update) >= (1.0 / self._asset_plot_fps):
                    self._last_asset_plot_update = now
                    self._render_asset_analyser()

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

            if (
                self.tabs.currentIndex() == self._positions_tab_index
                and (now - self._last_positions_update) >= (1.0 / self._positions_fps)
            ):
                self._last_positions_update = now
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
            self._update_friction_from_fill(f)
            self._update_asset_flow_from_fill(f)

            # Buffers for display
            self._recent_fills.append(f)
            self._fills_display.append(f)

            # Only queue for rendering if the Fills tab is active; otherwise rebuild on demand.
            if self.tabs.currentIndex() == self._fills_tab_index:
                self._fills_pending_render.append(f)
            else:
                self._fills_table_needs_rebuild = True

            processed += 1

        # 3) Lightweight UI updates. Touch heavy widgets only when visible.
        if (
            self.tabs.currentIndex() == self._overview_tab_index
            and (now - self._last_overview_fills_update) >= (1.0 / self._overview_fills_fps)
        ):
            self._last_overview_fills_update = now
            self._render_recent_fills()

        self._flush_fills_table(now)

        if (
            self.tabs.currentIndex() == self._pnl_tab_index
            and (now - self._last_pnl_update) >= (1.0 / self._pnl_fps)
        ):
            self._last_pnl_update = now
            self._render_pnl_table()

        if (
            self.tabs.currentIndex() == getattr(self, "_frictions_tab_index", -1)
            and (now - self._last_frictions_update) >= (1.0 / self._frictions_fps)
        ):
            self._last_frictions_update = now
            self._render_frictions_panel()

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

    def _skewness(self, values: List[float]):
        vals = [float(x) for x in values if math.isfinite(float(x))]
        n = len(vals)
        if n < 3:
            return None
        mean = sum(vals) / n
        m2 = sum((x - mean) ** 2 for x in vals) / n
        if m2 <= 0.0:
            return None
        m3 = sum((x - mean) ** 3 for x in vals) / n
        return m3 / (m2 ** 1.5)

    def _excess_kurtosis(self, values: List[float]):
        vals = [float(x) for x in values if math.isfinite(float(x))]
        n = len(vals)
        if n < 4:
            return None
        mean = sum(vals) / n
        m2 = sum((x - mean) ** 2 for x in vals) / n
        if m2 <= 0.0:
            return None
        m4 = sum((x - mean) ** 4 for x in vals) / n
        return (m4 / (m2 ** 2)) - 3.0

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
            table.setItem(r, 0, self._item(str(key)))
            table.setItem(r, 1, self._item(str(value)))
            table.setItem(r, 2, self._item(str(note)))


    def _append_drawdown_point(self, x: float, nav: float) -> None:
        try:
            nav_f = float(nav)
        except Exception:
            return
        if not math.isfinite(nav_f) or nav_f <= 0.0:
            return
        if self._running_peak_nav is None or nav_f > float(self._running_peak_nav):
            self._running_peak_nav = nav_f
        peak = float(self._running_peak_nav or nav_f)
        dd = (nav_f / peak) - 1.0 if peak > 0.0 else 0.0
        self._drawdown_x.append(float(x))
        self._drawdown_y.append(float(dd))

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
        self._running_peak_nav = peak if peak != -float("inf") else None

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
        median_ret = self._quantile(returns, 0.50)
        q25_ret = self._quantile(returns, 0.25)
        q75_ret = self._quantile(returns, 0.75)
        iqr_ret = (q75_ret - q25_ret) if q25_ret is not None and q75_ret is not None else None
        skew_ret = self._skewness(returns)
        kurt_ret = self._excess_kurtosis(returns)
        positive_rate = sum(1 for r in returns if r > 0.0) / float(len(returns))
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
            ("Median Sample Return", self._fmt_pct(median_ret), "50th percentile"),
            ("Sample Std Dev", self._fmt_pct(std_ret), "Non-annualized sample volatility"),
            ("Interquartile Range", self._fmt_pct(iqr_ret), "75th percentile - 25th percentile"),
            ("Skewness", self._fmt_num(skew_ret), "Negative = heavier left tail"),
            ("Excess Kurtosis", self._fmt_num(kurt_ret), "Positive = fatter tails than Gaussian"),
            ("Positive Sample Rate", self._fmt_pct(positive_rate), "Fraction of positive sampled returns"),
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
                f"Std: {self._fmt_pct(std_ret)} | Skew: {self._fmt_num(skew_ret)} | "
                f"Excess Kurtosis: {self._fmt_num(kurt_ret)} | VaR95: {self._fmt_pct(var95)} | "
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
        nav_packet = self._last_nav_packet if isinstance(self._last_nav_packet, dict) else self._latest_nav
        if isinstance(nav_packet, dict):
            pos_values = nav_packet.get("pos_values", {}) or {}
            positions = nav_packet.get("positions", {}) or {}
            visible_symbols = nav_packet.get("visible_symbols", []) or []
            gross_exposure = sum(abs(float(v)) for v in pos_values.values() if self._finite_float(v) is not None)
            latest_pos_count = sum(1 for q in positions.values() if self._finite_float(q, 0.0))
            visible_count = len(set(str(x) for x in visible_symbols))
        cash = 0.0
        if isinstance(nav_packet, dict):
            cash = self._finite_float(nav_packet.get("cash", 0.0), 0.0) or 0.0
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
            table.setItem(r, 0, self._item(str(key)))
            table.setItem(r, 1, self._item(self._fmt_metric_value(value)))

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
                self.learn_weights_table.setItem(r, 0, self._item(str(bucket_name)))
                self.learn_weights_table.setItem(r, 1, self._item(str(sym)))
                self.learn_weights_table.setItem(r, 2, self._item(self._fmt_metric_value(weight)))

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
        was_sorting = self.pos_table.isSortingEnabled()
        self.pos_table.setSortingEnabled(False)
        try:
            for _, _, sym, is_visible, qty, price, value, target_w in rows:
                r = self.pos_table.rowCount()
                self.pos_table.insertRow(r)
                self.pos_table.setItem(r, 0, self._item(str(sym), sort_value=str(sym).lower()))
                self.pos_table.setItem(r, 1, self._item("yes" if is_visible else "no", sort_value=1 if is_visible else 0))
                self.pos_table.setItem(r, 2, self._item(str(qty), sort_value=int(qty)))
                self.pos_table.setItem(r, 3, self._item(f"{price:,.4f}" if price else "-", sort_value=float(price)))
                self.pos_table.setItem(r, 4, self._item(f"{value:,.2f}", sort_value=float(value)))
                self.pos_table.setItem(r, 5, self._item(self._fmt_pct(target_w) if abs(target_w) > 1e-12 else "-", sort_value=float(target_w)))
        finally:
            self.pos_table.setSortingEnabled(was_sorting)
            self.pos_table.blockSignals(False)
            self.pos_table.setUpdatesEnabled(True)
        self._apply_positions_filter()

    def _apply_positions_filter(self) -> None:
        if not hasattr(self, "pos_table") or not hasattr(self, "pos_search"):
            return
        needle = self.pos_search.text().strip().lower()
        for row in range(self.pos_table.rowCount()):
            item = self.pos_table.item(row, 0)
            sym = item.text().lower() if item is not None else ""
            self.pos_table.setRowHidden(row, bool(needle and needle not in sym))

    def _render_recent_fills(self) -> None:
        if not hasattr(self, "ov_fills_table"):
            return
        self.ov_fills_table.setRowCount(0)
        for f in list(self._recent_fills)[-50:]:
            ts = str(f.get("ts", ""))
            time_str = ts[11:19] if len(ts) >= 19 else ts
            r = self.ov_fills_table.rowCount()
            self.ov_fills_table.insertRow(r)
            self.ov_fills_table.setItem(r, 0, self._item(time_str))
            self.ov_fills_table.setItem(r, 1, self._item(str(f.get("symbol", ""))))
            self.ov_fills_table.setItem(r, 2, self._item(str(f.get("side", ""))))
            self.ov_fills_table.setItem(r, 3, self._item(str(f.get("qty", ""))))
            self.ov_fills_table.setItem(r, 4, self._item(str(f.get("price", ""))))
            self.ov_fills_table.setItem(r, 5, self._item(str(f.get("fees", ""))))
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
        self.fills_table.setItem(row, 0, self._item(time_str))
        self.fills_table.setItem(row, 1, self._item(str(data.get("symbol", ""))))
        self.fills_table.setItem(row, 2, self._item(str(data.get("side", ""))))
        self.fills_table.setItem(row, 3, self._item(str(data.get("qty", ""))))
        self.fills_table.setItem(row, 4, self._item(str(data.get("price", ""))))
        self.fills_table.setItem(row, 5, self._item(str(data.get("fees", ""))))

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
            self.pnl_table.setItem(r, 0, self._item(str(sym)))
            self.pnl_table.setItem(r, 1, self._item(str(qty)))
            self.pnl_table.setItem(r, 2, self._item(f"{avg_cost:,.4f}"))
            self.pnl_table.setItem(r, 3, self._item(f"{mark:,.4f}"))
            self.pnl_table.setItem(r, 4, self._item(f"{unreal:,.2f}"))
            self.pnl_table.setItem(r, 5, self._item(f"{realized:,.2f}"))
            self.pnl_table.setItem(r, 6, self._item(f"{total:,.2f}"))

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


    def _first_float_from(self, data: dict, names, default=0.0):
        for name in names:
            if name in data:
                v = self._finite_float(data.get(name))
                if v is not None:
                    return float(v)
        return default

    def _estimate_fill_slippage_cost(self, fill: dict, qty: int, price: float) -> float:
        explicit = self._first_float_from(
            fill,
            [
                "slippage_cost", "slippage", "impact_cost", "market_impact",
                "estimated_slippage_cost", "exec_slippage_cost",
            ],
            None,
        )
        if explicit is not None:
            return abs(float(explicit))

        ref = self._first_float_from(
            fill,
            ["requested_price", "arrival_price", "decision_price", "mid_price", "ref_price"],
            None,
        )
        if ref is not None and ref > 0 and price > 0 and qty > 0:
            return abs(float(price) - float(ref)) * float(qty)
        return 0.0

    def _update_friction_from_fill(self, fill: dict) -> None:
        try:
            qty = int(fill.get("qty", 0))
        except Exception:
            qty = 0
        price = self._finite_float(fill.get("price", 0.0), 0.0) or 0.0
        fees = abs(self._finite_float(fill.get("fees", 0.0), 0.0) or 0.0)
        turnover = abs(float(qty) * float(price))
        slippage = self._estimate_fill_slippage_cost(fill, qty, price)
        total = fees + slippage

        self._friction_total_turnover += turnover
        self._friction_total_fees += fees
        self._friction_total_slippage += slippage
        self._friction_recent.append({
            "ts": str(fill.get("ts", "")),
            "symbol": str(fill.get("symbol", "")),
            "side": str(fill.get("side", "")),
            "qty": qty,
            "price": price,
            "turnover": turnover,
            "fees": fees,
            "slippage": slippage,
            "total": total,
        })

    def _update_asset_flow_from_fill(self, fill: dict) -> None:
        sym = str(fill.get("symbol", ""))
        if not sym:
            return
        side = str(fill.get("side", "")).lower()
        try:
            qty = abs(int(fill.get("qty", 0) or 0))
        except Exception:
            qty = 0
        price = self._finite_float(fill.get("price", 0.0), 0.0) or 0.0
        turnover = float(qty) * float(price)

        obj = self._asset_flow_by_symbol.setdefault(sym, {
            "buy_qty": 0,
            "sell_qty": 0,
            "buy_value": 0.0,
            "sell_value": 0.0,
            "last_side": "-",
            "last_qty": 0,
            "last_price": 0.0,
            "last_ts": "",
        })
        if side in ("buy", "b"):
            obj["buy_qty"] += qty
            obj["buy_value"] += turnover
        elif side in ("sell", "s"):
            obj["sell_qty"] += qty
            obj["sell_value"] += turnover
        obj["last_side"] = side.upper() if side else "-"
        obj["last_qty"] = qty
        obj["last_price"] = price
        obj["last_ts"] = str(fill.get("ts", ""))

    def _tracking_gap_summary(self):
        latest_nav = float(self.nav_data[-1]) if self.nav_data else 0.0
        if latest_nav <= 0.0:
            return 0.0, 0.0, 0, 0
        symbols = set(self._latest_positions.keys()) | set(self._latest_target_weights.keys())
        l1_gap = 0.0
        max_gap = 0.0
        active_targets = 0
        for sym in symbols:
            qty = int(self._latest_positions.get(sym, 0) or 0)
            price = float(self._latest_prices.get(sym, 0.0) or 0.0)
            current_w = (qty * price) / latest_nav if latest_nav > 0 else 0.0
            target_w = float(self._latest_target_weights.get(sym, 0.0) or 0.0)
            if abs(target_w) > 1e-12:
                active_targets += 1
            gap = abs(current_w - target_w)
            l1_gap += gap
            max_gap = max(max_gap, gap)
        return l1_gap, max_gap, active_targets, len(symbols)

    def _render_frictions_panel(self) -> None:
        if not hasattr(self, "frictions_table"):
            return
        turnover = float(self._friction_total_turnover)
        fees = float(self._friction_total_fees)
        slippage = float(self._friction_total_slippage)
        total = fees + slippage
        latest_nav = float(self.nav_data[-1]) if self.nav_data else 0.0
        bps_turnover = (10000.0 * total / turnover) if turnover > 0.0 else 0.0
        nav_drag = (total / latest_nav) if latest_nav > 0.0 else 0.0
        l1_gap, max_gap, active_targets, tracked_symbols = self._tracking_gap_summary()

        rows = [
            ("Gross turnover", f"{turnover:,.2f}", "Sum of absolute traded notional from received fills"),
            ("Cumulative fees", f"{fees:,.2f}", "Exact only if execution publishes fees in fill payload"),
            ("Estimated slippage", f"{slippage:,.2f}", "Uses slippage/impact fields or reference-price difference when available"),
            ("Total friction cost", f"{total:,.2f}", "Fees + estimated slippage"),
            ("Friction bps on turnover", f"{bps_turnover:,.2f}", "10000 × friction cost / gross turnover"),
            ("Friction drag vs NAV", self._fmt_pct(nav_drag), "Friction cost / latest NAV"),
            ("Target-tracking L1 gap", self._fmt_pct(l1_gap), "Sum |current weight - target weight|"),
            ("Max single-symbol gap", self._fmt_pct(max_gap), "Worst absolute current-vs-target weight gap"),
            ("Active target names", f"{active_targets:,}", "Non-zero target weights in latest packet"),
            ("Tracked names", f"{tracked_symbols:,}", "Union of latest positions and target weights"),
        ]
        self._set_three_col_rows(self.frictions_table, rows)
        self.frictions_summary_lbl.setText(
            f"Turnover: {turnover:,.2f} | Fees: {fees:,.2f} | "
            f"Est. slippage: {slippage:,.2f} | Total friction: {total:,.2f} | "
            f"Cost/turnover: {bps_turnover:,.2f} bps | NAV drag: {self._fmt_pct(nav_drag)}"
        )

        self.frictions_recent_table.setRowCount(0)
        for rec in list(self._friction_recent)[-120:]:
            r = self.frictions_recent_table.rowCount()
            self.frictions_recent_table.insertRow(r)
            self.frictions_recent_table.setItem(r, 0, self._item(self._fmt_time(str(rec.get("ts", "")))))
            self.frictions_recent_table.setItem(r, 1, self._item(str(rec.get("symbol", ""))))
            self.frictions_recent_table.setItem(r, 2, self._item(str(rec.get("side", ""))))
            self.frictions_recent_table.setItem(r, 3, self._item(str(rec.get("qty", ""))))
            self.frictions_recent_table.setItem(r, 4, self._item(f"{float(rec.get('price', 0.0)):,.4f}"))
            self.frictions_recent_table.setItem(r, 5, self._item(f"{float(rec.get('turnover', 0.0)):,.2f}"))
            self.frictions_recent_table.setItem(r, 6, self._item(f"{float(rec.get('fees', 0.0)):,.2f}"))
            self.frictions_recent_table.setItem(r, 7, self._item(f"{float(rec.get('slippage', 0.0)):,.2f}"))
            self.frictions_recent_table.setItem(r, 8, self._item(f"{float(rec.get('total', 0.0)):,.2f}"))
        self.frictions_recent_table.scrollToBottom()

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

                if self.tabs.currentIndex() == getattr(self, "_trades_tab_index", -1):
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
                else:
                    self._trades_table_needs_rebuild = True

                self._open_trade_by_symbol.pop(sym, None)
                pos = 0

        self._pos_from_fills[sym] = max(0, int(pos))


    def _rebuild_trades_table(self) -> None:
        if not hasattr(self, "trades_table"):
            return
        self.trades_table.setUpdatesEnabled(False)
        self.trades_table.blockSignals(True)
        try:
            self.trades_table.setRowCount(0)
            for t in self._trades:
                self._append_trade_row(
                    entry_ts=str(t.get("entry_ts", "")),
                    symbol=str(t.get("symbol", "")),
                    entry_qty=int(t.get("entry_qty", 0) or 0),
                    entry_vwap=float(t.get("entry_vwap", 0.0) or 0.0),
                    exit_ts=str(t.get("exit_ts", "")),
                    exit_vwap=float(t.get("exit_vwap", 0.0) or 0.0),
                    pnl=float(t.get("pnl", 0.0) or 0.0),
                    duration=str(t.get("duration", "")),
                    max_pos=int(t.get("max_pos", 0) or 0),
                )
        finally:
            self.trades_table.blockSignals(False)
            self.trades_table.setUpdatesEnabled(True)
        self.trades_table.scrollToBottom()

    def _append_trade_row(self, entry_ts: str, symbol: str, entry_qty: int, entry_vwap: float,
                          exit_ts: str, exit_vwap: float, pnl: float,
                          duration: str, max_pos: int) -> None:
        # NOTE: no artificial cap (previously 2000) — user explicitly requested unlimited trades.
        r = self.trades_table.rowCount()
        self.trades_table.insertRow(r)
        self.trades_table.setItem(r, 0, self._item(self._fmt_time(entry_ts)))
        self.trades_table.setItem(r, 1, self._item(str(symbol)))
        self.trades_table.setItem(r, 2, self._item(str(entry_qty)))
        self.trades_table.setItem(r, 3, self._item(f"{entry_vwap:,.4f}"))
        self.trades_table.setItem(r, 4, self._item(self._fmt_time(exit_ts)))
        self.trades_table.setItem(r, 5, self._item(f"{exit_vwap:,.4f}"))
        self.trades_table.setItem(r, 6, self._item(f"{pnl:,.2f}"))
        self.trades_table.setItem(r, 7, self._item(str(duration)))
        self.trades_table.setItem(r, 8, self._item(str(max_pos)))
        if self.trades_table.updatesEnabled():
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
            self.trade_fills_table.setItem(r, 0, self._item(self._fmt_time(str(f.get("ts", "")))))
            self.trade_fills_table.setItem(r, 1, self._item(str(f.get("side", ""))))
            self.trade_fills_table.setItem(r, 2, self._item(str(f.get("qty", ""))))
            self.trade_fills_table.setItem(r, 3, self._item(str(f.get("price", ""))))
            self.trade_fills_table.setItem(r, 4, self._item(str(f.get("fees", ""))))
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
