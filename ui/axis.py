import pyqtgraph as pg


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


