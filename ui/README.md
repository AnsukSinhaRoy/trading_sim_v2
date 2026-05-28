# `ui` — PyQt real-time dashboard

The active dashboard entry point is `qt_dashboard.py`. This file preserves the
latest PyQt application behavior from the uploaded project, including NAV window
controls, chart-type controls, equal-width tabs, asset analyser, return
distribution, online regret, frictions, positions, PnL, fills, and trades.

Small reusable pieces were moved out of the dashboard file:

- `axis.py`: dense trading-time axis, so plots do not show weekend/holiday gaps.
- `listener.py`: background ZMQ subscriber thread.
- `widgets.py`: guarded chart view box, sortable table item, and full-width tab bar.

Run it with:

```bash
python ui/qt_dashboard.py --url tcp://127.0.0.1:5555
```

The dashboard still works as a direct script and as a module:

```bash
python -m ui.qt_dashboard --url tcp://127.0.0.1:5555
```
