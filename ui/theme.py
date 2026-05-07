from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QTableWidget, QHeaderView, QAbstractItemView
import pyqtgraph as pg


# Light UI v2 palette. This is intentionally visual-only: no dashboard behavior,
# analytics, or table schemas are changed here.
COLORS = {
    "bg": "#f3f6fb",
    "surface": "#ffffff",
    "surface_2": "#f8fafc",
    "surface_3": "#eef2f7",
    "border": "#d7dee8",
    "text": "#111827",
    "muted": "#64748b",
    "accent": "#2563eb",
    "accent_2": "#16a34a",
    "warning": "#d97706",
    "danger": "#dc2626",
    "purple": "#7c3aed",
    "table_alt": "#f8fafc",
    "selection": "#bfdbfe",
    "selection_text": "#0f172a",
}


APP_QSS = f"""
QMainWindow, QWidget {{
    background: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: Segoe UI, Inter, Arial, sans-serif;
    font-size: 12px;
}}
QLabel {{
    color: {COLORS['text']};
}}
QLabel#StatCard {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #ffffff, stop:1 #eef5ff);
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 16px;
    font-weight: 700;
}}
QLabel#InfoCard {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 13px;
    color: {COLORS['muted']};
}}
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    top: -1px;
    background: {COLORS['surface']};
}}
QTabBar::tab {{
    background: {COLORS['surface_2']};
    color: {COLORS['muted']};
    border: 1px solid {COLORS['border']};
    border-bottom: none;
    padding: 9px 16px;
    margin-right: 4px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}}
QTabBar::tab:selected {{
    background: {COLORS['surface']};
    color: {COLORS['accent']};
    font-weight: 700;
}}
QTabBar::tab:hover {{
    color: {COLORS['text']};
    background: {COLORS['surface_3']};
}}
QTableWidget {{
    background: {COLORS['surface']};
    alternate-background-color: {COLORS['table_alt']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    gridline-color: {COLORS['border']};
    selection-background-color: {COLORS['selection']};
    selection-color: {COLORS['selection_text']};
}}
QTableWidget::item {{
    padding: 4px 6px;
}}
QTableWidget::item:selected {{
    background: {COLORS['selection']};
    color: {COLORS['selection_text']};
}}
QHeaderView::section {{
    background: {COLORS['surface_3']};
    color: {COLORS['text']};
    border: none;
    border-right: 1px solid {COLORS['border']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 7px 8px;
    font-weight: 700;
}}
QTableCornerButton::section {{
    background: {COLORS['surface_3']};
    border: none;
}}
QSplitter::handle {{
    background: {COLORS['border']};
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {COLORS['surface_2']};
    border: none;
    width: 10px;
    height: 10px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #cbd5e1;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: #94a3b8;
}}
QToolTip {{
    background: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    padding: 6px;
}}
"""


PENS = {
    "nav": COLORS["accent"],
    "drawdown": COLORS["danger"],
    "regret": COLORS["warning"],
    "cum_regret": COLORS["accent_2"],
}


def apply_app_theme(app: QApplication) -> None:
    """Apply the dashboard's visual-only light v2 theme."""
    app.setStyle("Fusion")
    pg.setConfigOptions(background=COLORS["surface"], foreground=COLORS["text"])
    app.setStyleSheet(APP_QSS)


def create_stat_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("StatCard")
    label.setMinimumHeight(58)
    label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    return label


def create_info_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("InfoCard")
    label.setWordWrap(True)
    return label


def create_table(headers, *, show_row_numbers: bool = False) -> QTableWidget:
    table = QTableWidget()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    row_header = table.verticalHeader()
    row_header.setVisible(show_row_numbers)
    if show_row_numbers:
        row_header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        row_header.setMinimumWidth(46)
        row_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)

    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setSortingEnabled(False)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setWordWrap(False)
    return table


def style_plot(plot, title: str, *, bottom_label: str = None, left_label: str = None) -> None:
    plot.setBackground(COLORS["surface"])
    plot.setTitle(title, color=COLORS["text"], size="13pt")
    if bottom_label:
        plot.setLabel("bottom", bottom_label, color=COLORS["muted"])
    if left_label:
        plot.setLabel("left", left_label, color=COLORS["muted"])
    plot.showGrid(x=True, y=True, alpha=0.18)
    for axis_name in ("bottom", "left"):
        axis = plot.getAxis(axis_name)
        axis.setPen(pg.mkPen(COLORS["border"]))
        axis.setTextPen(pg.mkPen(COLORS["muted"]))
    plot.setMenuEnabled(False)


def make_pen(name: str, width: int = 2):
    return pg.mkPen(color=PENS.get(name, COLORS["accent"]), width=width)
