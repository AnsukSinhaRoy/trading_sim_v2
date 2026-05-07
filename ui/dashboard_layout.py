from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSplitter
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from .axis import DenseTimeAxis
from .theme import create_stat_label, create_info_label, create_table, style_plot, make_pen


def setup_dashboard_ui(self):
    central = QWidget()
    self.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(12)

    # 1. Header Stats
    stats_layout = QHBoxLayout()
    stats_layout.setSpacing(10)
    self.lbl_nav = create_stat_label("NAV: -")
    self.lbl_cash = create_stat_label("Cash: -")
    self.lbl_pnl = create_stat_label("PnL: -")
    self.lbl_ts = create_stat_label("TS: -")
    self.lbl_learning = create_stat_label("Learning: -")

    stats_layout.addWidget(self.lbl_nav)
    stats_layout.addWidget(self.lbl_cash)
    stats_layout.addWidget(self.lbl_pnl)
    stats_layout.addWidget(self.lbl_ts)
    stats_layout.addWidget(self.lbl_learning)
    layout.addLayout(stats_layout)

    # 2. Tabs
    self.tabs = QTabWidget()
    self.tabs.setDocumentMode(False)

    # --- Overview Tab (NAV chart) ---
    overview = QWidget()
    ov_layout = QVBoxLayout(overview)

    dense_axis = DenseTimeAxis(self._get_nav_dt, orientation="bottom")
    self.plot_widget = pg.PlotWidget(axisItems={"bottom": dense_axis})
    style_plot(self.plot_widget, "Live NAV")
    self.nav_curve = self.plot_widget.plot(pen=make_pen("nav", width=2))
    ov_layout.addWidget(self.plot_widget)

    self.ov_fills_table = create_table(["Time", "Symbol", "Side", "Qty", "Price", "Fees"])
    self.ov_fills_table.setMaximumHeight(240)
    ov_layout.addWidget(self.ov_fills_table)
    self._overview_tab_index = self.tabs.addTab(overview, "Overview")

    # --- Backtest Metrics Tab ---
    metrics = QWidget()
    metrics_layout = QVBoxLayout(metrics)

    self.metrics_summary_lbl = create_info_label(
        "Backtest metrics will appear after at least two NAV snapshots. "
        "Sharpe/Sortino use a fixed 4% annual risk-free rate."
    )
    metrics_layout.addWidget(self.metrics_summary_lbl)

    metrics_splitter = QSplitter()
    metrics_splitter.setOrientation(Qt.Orientation.Vertical)

    self.metrics_table = create_table(["Metric", "Value", "Notes"])
    metrics_splitter.addWidget(self.metrics_table)

    dd_axis = DenseTimeAxis(self._get_nav_dt, orientation="bottom")
    self.drawdown_plot = pg.PlotWidget(axisItems={"bottom": dd_axis})
    style_plot(self.drawdown_plot, "Drawdown")
    self.drawdown_curve = self.drawdown_plot.plot(pen=make_pen("drawdown", width=2))
    metrics_splitter.addWidget(self.drawdown_plot)

    metrics_layout.addWidget(metrics_splitter)
    self._metrics_tab_index = self.tabs.addTab(metrics, "Backtest Metrics")

    # --- Return Distribution Tab ---
    returns_tab = QWidget()
    returns_layout = QVBoxLayout(returns_tab)
    self.returns_summary_lbl = create_info_label(
        "Return distribution will appear after enough NAV snapshots. "
        "VaR and CVaR are computed from sampled NAV-to-NAV returns."
    )
    returns_layout.addWidget(self.returns_summary_lbl)

    returns_splitter = QSplitter()
    returns_splitter.setOrientation(Qt.Orientation.Vertical)

    self.return_dist_plot = pg.PlotWidget()
    style_plot(self.return_dist_plot, "Sample Return Distribution", bottom_label="Sample return (%)", left_label="Frequency")
    returns_splitter.addWidget(self.return_dist_plot)
    self.return_dist_bars = None

    self.return_risk_table = create_table(["Metric", "Value", "Notes"])
    returns_splitter.addWidget(self.return_risk_table)

    returns_layout.addWidget(returns_splitter)
    self._returns_tab_index = self.tabs.addTab(returns_tab, "Return Distribution")

    # --- Positions Tab ---
    positions = QWidget()
    pos_layout = QVBoxLayout(positions)
    self.positions_summary_lbl = create_info_label(
        "Positions table shows the current algorithm-visible universe. "
        "Stocks with zero quantity are visible to the algorithm but not currently held."
    )
    pos_layout.addWidget(self.positions_summary_lbl)
    self.pos_table = create_table(["Symbol", "Visible", "Qty", "Price", "Value", "Target W"], show_row_numbers=True)
    pos_layout.addWidget(self.pos_table)
    self._positions_tab_index = self.tabs.addTab(positions, "Positions")

    # --- Fills Tab ---
    fills = QWidget()
    fills_layout = QVBoxLayout(fills)
    self.fills_table = create_table(["Time", "Symbol", "Side", "Qty", "Price", "Fees"])
    fills_layout.addWidget(self.fills_table)
    self._fills_tab_index = self.tabs.addTab(fills, "Fills")

    # --- Online Parameters Tab ---
    learning_params = QWidget()
    learning_params_layout = QVBoxLayout(learning_params)

    self.learn_summary_lbl = create_info_label(
        "No learning telemetry received yet. Parameters, support, target weights, "
        "and scalar diagnostics will appear here."
    )
    learning_params_layout.addWidget(self.learn_summary_lbl)

    params_splitter = QSplitter()
    params_splitter.setOrientation(Qt.Orientation.Vertical)

    self.learn_scalars_table = create_table(["Metric", "Value"])
    params_splitter.addWidget(self.learn_scalars_table)

    self.learn_weights_table = create_table(["Bucket", "Symbol", "Weight"])
    params_splitter.addWidget(self.learn_weights_table)

    self.learn_lists_table = create_table(["Name", "Value"])
    params_splitter.addWidget(self.learn_lists_table)

    learning_params_layout.addWidget(params_splitter)
    self._learning_params_tab_index = self.tabs.addTab(learning_params, "Online Parameters")

    # --- Online Regret Tab ---
    learning_regret = QWidget()
    learning_regret_layout = QVBoxLayout(learning_regret)

    self.regret_summary_lbl = create_info_label(
        "No regret telemetry received yet. Regret plots require the strategy to publish "
        "regret/cum_regret or loss/oracle_loss in learn.scalars."
    )
    learning_regret_layout.addWidget(self.regret_summary_lbl)

    self.ol_regret_plot = pg.PlotWidget()
    style_plot(self.ol_regret_plot, "Online Regret", bottom_label="Tick", left_label="Regret")
    self.ol_regret_curve = self.ol_regret_plot.plot(
        name="regret", pen=make_pen("regret", width=2)
    )
    self.ol_cum_regret_curve = self.ol_regret_plot.plot(
        name="cum_regret", pen=make_pen("cum_regret", width=2)
    )
    self.ol_regret_plot.addLegend()
    learning_regret_layout.addWidget(self.ol_regret_plot)
    self._learning_regret_tab_index = self.tabs.addTab(learning_regret, "Online Regret")

    # --- PnL Tab ---
    pnl = QWidget()
    pnl_layout = QVBoxLayout(pnl)
    self.pnl_table = create_table([
        "Symbol", "Qty", "Avg Cost", "Mark", "Unrealized", "Realized", "Total"
    ])
    pnl_layout.addWidget(self.pnl_table)
    self._pnl_tab_index = self.tabs.addTab(pnl, "PnL")

    # --- Trades Tab ---
    trades = QWidget()
    trades_layout = QVBoxLayout(trades)

    splitter = QSplitter()
    splitter.setOrientation(Qt.Orientation.Vertical)

    self.trades_table = create_table([
        "Entry Time", "Symbol", "Entry Qty", "Entry VWAP",
        "Exit Time", "Exit VWAP", "PnL", "Duration", "Max Pos"
    ], show_row_numbers=True)
    splitter.addWidget(self.trades_table)

    inspector = QWidget()
    insp_layout = QVBoxLayout(inspector)
    self.trade_inspector_lbl = create_info_label("Trade Inspector: select a trade row")
    insp_layout.addWidget(self.trade_inspector_lbl)
    self.trade_fills_table = create_table(["Time", "Side", "Qty", "Price", "Fees"])
    insp_layout.addWidget(self.trade_fills_table)
    splitter.addWidget(inspector)

    trades_layout.addWidget(splitter)
    self._trades_tab_index = self.tabs.addTab(trades, "Trades")

    self.trades_table.itemSelectionChanged.connect(self._on_trade_selected)
    self.tabs.currentChanged.connect(self._on_tab_changed)

    layout.addWidget(self.tabs, stretch=1)

