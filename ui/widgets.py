from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QTabBar, QTableWidgetItem
import pyqtgraph as pg


class GuardedViewBox(pg.ViewBox):
    """ViewBox that blocks accidental wheel zooms.

    Normal mouse-wheel events are ignored so scrolling does not unexpectedly
    zoom a chart. Hold Ctrl while using the wheel when you deliberately want
    to zoom a plot.
    """

    def wheelEvent(self, ev, axis=None):
        modifiers = QApplication.keyboardModifiers()
        if not (modifiers & Qt.KeyboardModifier.ControlModifier):
            ev.ignore()
            return
        super().wheelEvent(ev, axis=axis)

    def mouseDoubleClickEvent(self, ev):
        try:
            self.autoRange()
            ev.accept()
            return
        except Exception:
            pass
        super().mouseDoubleClickEvent(ev)

class SmartTableItem(QTableWidgetItem):
    """QTableWidgetItem with optional numeric/date-aware sort key.

    This keeps cell text left-aligned and human-readable while allowing the
    Positions header click to sort numbers as numbers rather than strings.
    """

    def __init__(self, text, sort_value=None):
        super().__init__(str(text))
        self.sort_value = sort_value

    def __lt__(self, other):
        a = getattr(self, "sort_value", None)
        b = getattr(other, "sort_value", None)
        if a is not None and b is not None:
            try:
                return a < b
            except Exception:
                pass
        return self.text().lower() < other.text().lower()

class FullWidthTabBar(QTabBar):
    """Tab bar with equal-width tabs spanning the available dashboard width.

    Qt often calculates tab sizes before the top-level window has its final
    width, so the first paint can look compact and only fix itself after a
    click/resize. The dashboard calls ``set_available_width`` after show/resize
    so the first render gets the real width instead of the early placeholder.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._available_width = 0

    def set_available_width(self, width: int) -> None:
        width = max(0, int(width or 0))
        if abs(width - self._available_width) <= 1:
            return
        self._available_width = width
        try:
            self.setFixedWidth(width)
            self.updateGeometry()
            self.update()
        except Exception:
            pass

    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        count = max(1, self.count())
        parent = self.parentWidget()
        available = self._available_width or (parent.width() if parent is not None else self.width())
        if available <= 0:
            available = max(size.width() * count, 900)
        width = max(74, int(available / count) - 1)
        size.setWidth(width)
        size.setHeight(max(size.height(), 42))
        return size

    def minimumTabSizeHint(self, index):
        return self.tabSizeHint(index)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self.updateGeometry()
        except Exception:
            pass
